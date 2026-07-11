#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import os, time
from random import randint
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import ImageFilter
import torchvision.transforms as transforms

from lpipsPyTorch import lpips
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, websockets, args):
    # 用来打印判断是否正确执行我想要执行的函数的
    flg = True

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    
    # all_edges 用于存储每个训练视角的边缘图
    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        # 把边缘图做归一化，所有值都归到 [0, 1] 中 edges_loss_norm.shape:(1, H, W)
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())
    my_viewpoint_stack = scene.getTrainCameras().copy()
    edges_stack = all_edges.copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Profiling accumulators for user-added components
    from collections import defaultdict
    stage_times = defaultdict(float)
    stage_counts = defaultdict(int)
    stage_memory = defaultdict(float)

    lambda_efre_wl, lambda_efre_wh = opt.lambda_efre_wl, opt.lambda_efre_wh

    init_point_nums = gaussians.get_xyz.shape[0]

    # 根据初始小球数来设置小球场景上限也是 Conf 的一个操作
    """
    if init_point_nums < 120000: opt.budget = 1250000
    else: opt.budget = 1777778"""
    
    psnr_records = []
    for iteration in range(first_iter, opt.iterations + 1):
        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss, 还是正常的 l1 loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        

        # 加入 PAIR Loss (方案 B: 单圆盘 mask + 三段折线半径退火, 与论文公式对齐)
        if opt.lambda_amp_rec > 0:
            r_t = get_multiscale_amp_rec_mask_ratio(
                iteration,
                stage1_iter=opt.amp_rec_stage1_iter,
                stage2_iter=opt.amp_rec_stage2_iter,
                low_ratio=opt.amp_rec_mask_ratio_low,
                mid_ratio=opt.amp_rec_mask_ratio_mid,
                high_ratio=opt.amp_rec_mask_ratio_high,
                final_transition_len=opt.amp_rec_final_transition_len,
            )
            loss += opt.lambda_amp_rec * amplitude_reconstruction_loss(
                image, gt_image, use_log=True, mask_ratio=r_t,
            )



        loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "N_GS": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Per-component stage time logging (in training() scope where accumulators live)
            if tb_writer and opt.profile_components:
                for comp in ['conf', 'rfas', 'fusion']:
                    if stage_counts.get(comp, 0) > 0:
                        tb_writer.add_scalar('stage_time/' + comp,
                            stage_times[comp] / max(stage_counts[comp], 1), iteration)
                        tb_writer.add_scalar('memory/' + comp,
                            stage_memory[comp] / max(stage_counts[comp], 1), iteration)
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration == 300:
                gaussians.only_prune(0.02)

            # Densification
            if opt.densify_from_iter < iteration < opt.densify_until_iter:

                # --- Conf timing probe ---
                if opt.profile_components:
                    torch.cuda.synchronize()
                    start_conf = torch.cuda.Event(enable_timing=True)
                    end_conf = torch.cuda.Event(enable_timing=True)
                    start_conf.record()
                    mem_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()

                # EAS 中计算
                gaussians.add_densification_stats_abs(viewspace_point_tensor, visibility_filter, viewpoint_cam)

                if opt.profile_components:
                    end_conf.record()
                    torch.cuda.synchronize()
                    stage_times['conf'] += start_conf.elapsed_time(end_conf)
                    stage_counts['conf'] += 1
                    stage_memory['conf'] += (torch.cuda.max_memory_allocated() - mem_before)

                if iteration % opt.densification_interval == 0:
                    # 默认用 args.cams 个视角来计算重要性
                    num_cams = args.cams
                    if args.cams == -1 or (iteration % 3000 == 400 and iteration < 9000):
                        num_cams = len(scene.getTrainCameras().copy())
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        if not my_viewpoint_stack:
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            edges_stack = all_edges.copy()
                        camlist.append(my_viewpoint_stack.pop())
                        edge_losses.append(edges_stack.pop())

                    # 计算 EAS score gaussian_importance.shape: (N,)
                    gaussian_importance = compute_edge_score(camlist, edge_losses, gaussians, pipe, bg)
                    #print('gaussian_imortance shape is ',gaussian_importance.shape)

                    # --- RFAS timing probe ---
                    if opt.profile_components:
                        torch.cuda.synchronize()
                        start_rfas = torch.cuda.Event(enable_timing=True)
                        end_rfas = torch.cuda.Event(enable_timing=True)
                        start_rfas.record()
                        mem_before_rfas = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()

                    # 接下来计算 RFAS score
                    gaussian_importance_rf = compute_rf_score1(camlist, gaussians, pipe, bg)
                    #print('gaussian_importance_rf shape is ',gaussian_importance_rf.shape)

                    if opt.profile_components:
                        end_rfas.record()
                        torch.cuda.synchronize()
                        stage_times['rfas'] += start_rfas.elapsed_time(end_rfas)
                        stage_counts['rfas'] += 1
                        stage_memory['rfas'] += (torch.cuda.max_memory_allocated() - mem_before_rfas)

                    # --- Fusion timing probe ---
                    if opt.profile_components:
                        torch.cuda.synchronize()
                        start_fusion = torch.cuda.Event(enable_timing=True)
                        end_fusion = torch.cuda.Event(enable_timing=True)
                        start_fusion.record()
                        mem_before_fusion = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()

                    tt_importance = fuse_importance_scores(
                        gaussian_importance,
                        gaussian_importance_rf,
                        mode='weighted'
                    )

                    # Stop fusion timer BEFORE baseline densification (LAS/pruning)
                    if opt.profile_components:
                        end_fusion.record()
                        torch.cuda.synchronize()
                        stage_times['fusion'] += start_fusion.elapsed_time(end_fusion)
                        stage_counts['fusion'] += 1
                        stage_memory['fusion'] += (torch.cuda.max_memory_allocated() - mem_before_fusion)

                    #tt_importance = gaussian_importance_rf # 只启用 rfas

                    startI = opt.densify_from_iter
                    endI = opt.densify_until_iter - 500
                    # GC 控制实现
                    rate = (iteration - startI) / (endI - startI)
                    if rate >= 1:
                        budget = int(opt.budget)
                    else:
                        budget = int(math.sqrt(rate) * opt.budget)

                    # LAS 入口, 最后再densify_and_prune_Improved 中调用 LAS
                    gaussians.densify_and_prune_Improved(tt_importance, 0.005, budget, opt, iteration, opt.budget)
                
            
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity(0.05)

                # RAP 实现
                if iteration % opt.opacity_reset_interval == 300 and iteration < 9000:
                    gaussians.only_prune(0.2, True)

            # Optimizer step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    # MU 实现
                    if iteration <= 15000:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none=True)
                    elif iteration <= 22500:
                        if iteration % 5 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                    else:
                        if iteration % 20 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

        # Profiler summary + export: run AFTER all densification/optimizer probes
        if opt.profile_components and iteration == opt.iterations:
            with torch.no_grad():
                if iteration not in testing_iterations:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                    iter_start.elapsed_time(iter_end), [iteration],
                                    scene, render, (pipe, background))
            # Print summary (after all probes accumulated)
            total_time = sum(stage_times.values())
            print("\n=== Component Profiling Summary ===")
            print(f"{'Component':<12} {'Time(ms)':<12} {'%Total':<10} {'Calls':<10}")
            print("-" * 46)
            for comp in ['conf', 'rfas', 'fusion']:
                t = stage_times.get(comp, 0.0)
                c = stage_counts.get(comp, 0)
                pct = (t / total_time * 100) if total_time > 0 else 0.0
                print(f"{comp:<12} {t:<12.2f} {pct:<10.2f} {c:<10}")
            print(f"{'TOTAL':<12} {total_time:<12.2f} {'100.00':<10}")
            print("==================================\n")
            # Save to JSON
            import json as _json
            n_gs = gaussians.get_xyz.shape[0]
            prof_data = {
                'n_gaussians': n_gs,
                'psnr': training_report_psnr,
                'conf_time': stage_times.get('conf', 0.0),
                'rfas_time': stage_times.get('rfas', 0.0),
                'fusion_time': stage_times.get('fusion', 0.0),
            }
            prof_path = os.path.join(dataset.model_path, 'profiler_results.json')
            with open(prof_path, 'w') as fp:
                _json.dump(prof_data, fp, indent=True)

def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    return image_edges_tensor


def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)

    return ret_value

#=========================================
# 幅度反重建损失
# 方案 B
def get_multiscale_amp_rec_mask_ratio(
    iteration,
    stage1_iter=5000,
    stage2_iter=12000,
    low_ratio=0.15,
    mid_ratio=0.4,
    high_ratio=0.9,
    final_transition_len=8000
):
    if iteration <= stage1_iter:
        return low_ratio
    if iteration <= stage2_iter:
        t = (iteration - stage1_iter) / max(1, stage2_iter - stage1_iter)
        return low_ratio + t * (mid_ratio - low_ratio)
    
    t = min(1.0, (iteration - stage2_iter) / max(1, final_transition_len))
    return mid_ratio + t * (high_ratio - mid_ratio)

def compute_amplitude_reconstruction_map(
    image,
    use_log=True,
    mask_ratio=None
):
    added_batch =  False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        added_batch = True
    
    if image.dim() != 4:
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {image.shape}")
    
    image = image.float()
    _, _, h, w =  image.shape

    fft = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)

    if use_log:
        amp = torch.log1p(amp)
    
    if mask_ratio is not None:
        amp = torch.fft.fftshift(amp, dim=(-2, -1))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij"
        )
        cy, cx = h // 2, w // 2

        dist = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float())
        max_dist = torch.sqrt(torch.tensor(
            cx ** 2 + cy ** 2, device=image.device,
            dtype=torch.float32
        ))

        mask = (dist <= max_dist * mask_ratio).float().view(1, 1, h, w)
        amp = amp * mask
        amp = torch.fft.ifftshift(amp, dim=(-2, -1))
    
    amp_rec = torch.fft.ifft2(
        amp.to(torch.complex64),
        dim=(-2, -1),
        norm = "ortho"
    ).real

    if added_batch:
        amp_rec = amp_rec.squeeze(0)
    return amp_rec

def amplitude_reconstruction_loss(
    pred,
    gt,
    use_log=True,
    mask_ratio=None
):
    pred_map = compute_amplitude_reconstruction_map(
        pred, use_log=use_log, mask_ratio=mask_ratio
    )
    gt_map = compute_amplitude_reconstruction_map(
        gt, use_log=use_log, mask_ratio=mask_ratio
    )

    return F.l1_loss(pred_map, gt_map)

#=============================================================
# 环带幅度
# 方案 C
def compute_amplitude_reconstruction_ring_map(image, r_lo, r_hi, use_log=True):
    added_batch = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        added_batch = True
    
    image = image.float()
    _, _, h, w = image.shape

    fft = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    if use_log:
        amp = torch.log1p(amp)
    
    amp = torch.fft.fftshift(amp, dim=(-2, -1))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing="ij" 
    )
    cy, cx = h // 2, w // 2
    dist = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float())

    max_dist = torch.sqrt(torch.tensor(
        cx ** 2 + cy ** 2, device=image.device, dtype=torch.float32
    ))
    if r_lo <= 0:
        mask = (dist <= r_hi * max_dist).float()
    else:
        mask = ((dist > r_lo * max_dist) & (dist <= r_hi * max_dist)).float()
    mask = mask.view(1, 1, h, w)
    amp = amp * mask
    amp = torch.fft.ifftshift(amp, dim=(-2, -1))

    amp_rec = torch.fft.ifft2(
        amp.to(torch.complex64), dim=(-2, -1), norm="ortho"
    ).real
    if added_batch:
        amp_rec = amp_rec.squeeze(0)
    return amp_rec

def amplitude_reconstruction_loss_ring(pred, gt, r_lo, r_hi, use_log=True):
    pred_map = compute_amplitude_reconstruction_ring_map(pred, r_lo, r_hi, use_log)
    gt_map  = compute_amplitude_reconstruction_ring_map(gt,   r_lo, r_hi, use_log)
      # 用 mask 内频点数做归一化，让不同环带量纲一致
    return F.l1_loss(pred_map, gt_map)

def compute_amplitude_reconstruction_loss_multiscale(
    pred, gt,
    iteration,
    band_iters=(0, 5000, 12000),
    band_ratios=(0.15, 0.45, 0.9),
    band_weights=(1.0, 1.0, 1.0),
    use_log=True,
  ):
    losses = []
    total_w = 0.0
    for i, (start_iter, r_hi, w) in enumerate(
        zip(band_iters, band_ratios, band_weights)):
        if iteration < start_iter:
            continue
        r_lo = band_ratios[i-1] if i > 0 else 0.0
        l = amplitude_reconstruction_loss_ring(pred, gt, r_lo, r_hi, use_log)
        
        losses.append(w * l)
        total_w += w
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).sum() / total_w

# ==================================== Fre Regulazation and FA ======================================
def compute_frequency_discrepancies(e, iteration, T0, T_end, cutoff_ratio_low=0.15, cutoff_ratio_high=0.5):
    """
    计算低频和高频的幅度和相位差异，带频率退火
    """
    if e.dim() == 3 and e.shape[0] == 3:
        e = 0.299 * e[0] + 0.587 * e[1] + 0.114 * e[2]
    
    H, W = e.shape
    
    # FFT
    F_uv = torch.fft.fft2(e)
    F_uv_shifted = torch.fft.fftshift(F_uv)
    
    # 创建距离矩阵
    center_h, center_w = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=e.device), 
        torch.arange(W, device=e.device), 
        indexing='ij'
    )
    radius = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
    
    # 计算动态高通滤波器的截止频率 D_t 
    D_0 = min(H, W) * cutoff_ratio_low
    D_max = min(H, W) * cutoff_ratio_high
    
    if iteration <= T0:
        # 早期阶段：只使用低频
        D_t = D_0
    else:
        # 渐进式扩展高频带宽 
        t_normalized = (iteration - T0) / (T_end - T0)
        t_normalized = min(t_normalized, 1.0)  # 防止超过 1
        D_t = D_0 + t_normalized * (D_max - D_0)
    
    # 低频滤波器：0 到 D_0 的频率
    H_l = (radius <= D_0).float()
    
    # 高频滤波器：D_0 到 D_t 的频率（动态扩展）
    H_h = ((radius > D_0) & (radius <= D_t)).float()
    
    # 应用滤波器
    LF_uv = F_uv_shifted * H_l
    HF_uv = F_uv_shifted * H_h
    
    # 计算幅度和相位
    LF_amplitude = torch.abs(LF_uv)
    LF_phase = torch.angle(LF_uv)
    
    HF_amplitude = torch.abs(HF_uv)
    HF_phase = torch.angle(HF_uv)
    
    # 归一化
    norm_factor = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=e.device))
    
    d_la = torch.mean(torch.abs(LF_amplitude)) / norm_factor
    d_lp = torch.mean(torch.abs(LF_phase)) / norm_factor
    d_ha = torch.mean(torch.abs(HF_amplitude)) / norm_factor
    d_hp = torch.mean(torch.abs(HF_phase)) / norm_factor
    
    return {
        'd_la': d_la,
        'd_lp': d_lp,
        'd_ha': d_ha,
        'd_hp': d_hp,
        'D_t': D_t 
    }

def compute_frequency_regularization(e_image, iteration, T0=7000, w_l=1.0, w_h=1.0, 
                                     startI=500, endI=25000, 
                                     cutoff_ratio_low=0.15, cutoff_ratio_high=0.5):
    """
    计算频率域正则化损失  + 频率退火
    """
    # 计算频率差异（带动态高通滤波器）
    freq_disc = compute_frequency_discrepancies(
        e_image, iteration, T0, endI, 
        cutoff_ratio_low, cutoff_ratio_high
    )
    
    d_la = freq_disc['d_la']
    d_lp = freq_disc['d_lp']
    d_ha = freq_disc['d_ha']
    d_hp = freq_disc['d_hp']
    
    # 低频始终参与正则化
    L_f = w_l * (d_la + d_lp)
    
    # 高频根据迭代次数逐步引入
    if iteration > T0:
        L_f += w_h * (d_ha + d_hp)
    
    return L_f

#----------------------------------融合 RFAS 和 EAS   ----------------------------------------------

def fuse_importance_scores(eas, rfas, mode='geometric', power=1.0):
    """
    融合 RFAS 和 EAS 分数
    """
    # 归一化到 [0, 1]
    eas_norm = (eas - eas.min()) / (eas.max() - eas.min() + 1e-8)
    rfas_norm = (rfas - rfas.min()) / (rfas.max() - rfas.min() + 1e-8)
    
    if mode == 'geometric':
        # 几何平均
        fused = torch.sqrt((eas_norm + 1e-6) * (rfas_norm + 1e-6))
    elif mode == 'product':
        # 乘积
        fused = eas_norm * rfas_norm
    elif mode == 'weighted':
        # 加权平均
        alpha = 0.4
        fused = alpha * eas_norm + (1 - alpha) * rfas_norm
    
    if power != 1.0:
        fused = torch.pow(fused + 1e-6, power)
    
    return fused
#----------------------------------融合 RFAS 和 EAS   ----------------------------------------------

#----------------------------------加入计算 RFAS 的代码----------------------------------------------
def compute_rf_score(camlist, gaussians, pipe, bg):
    with torch.no_grad():
        """
        计算 RFAS (Residual-Frequency Aware Score)
        结合边缘敏感度 (EAS) 和高频残差信息
        """
        num_points = len(gaussians.get_xyz)
        gaussian_importance_rf = torch.zeros(num_points, device='cuda', dtype=torch.float32)
        
        # 将每个视角的 RFAS 结果累积
        for view_idx, viewpoint_cam in enumerate(camlist):
            gt_image = viewpoint_cam.original_image.cuda()  # [3, H, W]
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            render_image = render_pkg['render']  # [3, H, W]
            
            residual = torch.abs(gt_image - render_image)  # [3, H, W]
            pixel_error = residual.sum(dim=0)  # [H, W] 
            
            h_j = compute_high_freq_residual_log(pixel_error)  # [H, W]
            
            visibility = render_pkg.get('visibility', None)  

            if visibility is not None:
                h_j_flat = h_j.view(1, -1)  
                visibility_flat = visibility.view(num_points, -1)  
                rf_score = (h_j_flat * visibility_flat).sum(dim=1) 
            else:
                rf_score = h_j.sum() * gaussians.get_opacity.squeeze()
            
            # 累加所有视角的 RFAS 得分
            gaussian_importance_rf += rf_score

        gaussian_importance_rf /= len(camlist)

        return gaussian_importance_rf  # 返回 (N,)


def compute_rf_score1(camlist, gaussians, pipe, bg):
    """
    RFAS 的修正实现。
    与 compute_rf_score 不同的是：这里通过 rasterizer 的 pixel_weights 入口
    把 h_j(高频残差图) 当作 per-pixel 权重传入，rasterizer 输出的
    accum_weights 即为 per-Gaussian 的高频残差累积量；这样每个高斯之间的
    RFAS 差异才真正来自 "它在视图里贡献的高频内容多少"，
    而不是退化成 view-level 标量 × opacity。
    用法和 compute_rf_score 完全相同，可直接替换调用点。
    """
    num_points = len(gaussians.get_xyz)
    gaussian_importance_rf = torch.zeros(num_points, device='cuda', dtype=torch.float32)
    visibility_filter_all  = torch.zeros(num_points, device='cuda', dtype=bool)

    with torch.no_grad():
        for view_idx, viewpoint_cam in enumerate(camlist):
            gt_image = viewpoint_cam.original_image.cuda()  # [3, H, W]

            # 第一遍：普通渲染，拿到当前视图的残差与高频残差图
            render_pkg_plain = render(viewpoint_cam, gaussians, pipe, bg)
            render_image = render_pkg_plain['render']                       # [3, H, W]
            residual     = torch.abs(gt_image - render_image)               # [3, H, W]
            pixel_error  = residual.sum(dim=0)                              # [H, W]
            h_j          = compute_high_freq_residual_log(pixel_error)      # [H, W]

            # 第二遍：把 h_j 作为 pixel_weights 传入定制 rasterizer，
            # accum_weights[i] 即为高斯 i 在该视图上"高频残差贡献"的累积值
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, pixel_weights=h_j)
            rf_accum   = normalize(render_pkg["accum_weights"])             # (N,)
            vf         = render_pkg["visibility_filter"].detach()           # 可见高斯索引

            gaussian_importance_rf[vf] += rf_accum[vf] / len(camlist)
            visibility_filter_all[vf]   = True

    return gaussian_importance_rf  # (N,)


def compute_high_freq_residual_log(error_map):
    import torch.nn.functional as F

    error_map_4d = error_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=error_map.device).view(1, 1, 3, 3)
    
    # 应用卷积
    high_freq = F.conv2d(error_map_4d, laplacian_kernel, padding=1)
    high_freq = torch.abs(high_freq.squeeze())  # [H, W]
    
    return high_freq


def compute_high_freq_residual_fft(error_map):
    H, W = error_map.shape
    
    # FFT
    fft = torch.fft.fft2(error_map)
    fft_shifted = torch.fft.fftshift(fft)
    
    # 高通滤波器 (保留高频，抑制低频)
    center_h, center_w = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=error_map.device), 
                          torch.arange(W, device=error_map.device), indexing='ij')
    radius = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
    
    # 距离中心越远权重越大
    cutoff = min(H, W) // 8  # 截止频率
    high_pass_filter = (radius > cutoff).float()
    
    # 应用滤波器
    fft_filtered = fft_shifted * high_pass_filter
    fft_ishifted = torch.fft.ifftshift(fft_filtered)
    high_freq = torch.fft.ifft2(fft_ishifted).real
    high_freq = torch.abs(high_freq)
    return high_freq
#----------------------------------加入计算 RFAS 的代码----------------------------------------------

def compute_edge_score(camlist, edge_losses, gaussians, pipe, bg):
    # nums_points 现在高斯椭球的数量
    num_points = len(gaussians.get_xyz)
    gaussian_importance = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    visibility_filter_all = torch.zeros(num_points, device="cuda", dtype=bool)
    
    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        pixel_weights = edge_losses[view].cuda()
        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights=pixel_weights)

        loss_accum = normalize(render_pkg["accum_weights"])
        visibility_filter = render_pkg["visibility_filter"].detach()

        gaussian_importance[visibility_filter] += loss_accum[visibility_filter] / len(camlist)
        visibility_filter_all[visibility_filter] = True

    gaussian_importance[visibility_filter_all] = gaussian_importance[visibility_filter_all]
    return gaussian_importance


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# Module-level: captured training test PSNR for downstream collection
training_report_psnr = 0.0

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    global training_report_psnr
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test= 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if config['name'] == 'test':
                    training_report_psnr = psnr_test.item() if isinstance(psnr_test, torch.Tensor) else float(psnr_test)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000 * (i + 1) for i in range(10)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--websockets", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations,
        args.debug_from, 
        args.websockets,
        args
    )

    # All done
    print("\nTraining complete.")

"""Conf 候选点标红可视化模块。

在训练 densification 循环中，将 Conf 选中的 Gaussian 3D 中心投影到 2D 相机视图，
以红色实心圆标注，与原始渲染图成对保存。纯 Python 实现，不修改 CUDA 代码。
"""

import json
import os
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage


def project_gaussian_centers(xyz, full_proj_transform, world_view_transform, width, height):
    """将 Gaussian 世界空间中心投影到像素坐标。

    投影链: world → NDC (geom_transform_points) → pixel (ndc_to_pixel)
    同时计算 view-space depth 用于相机后方过滤。

    Args:
        xyz: (N, 3) 世界空间 Gaussian 中心坐标
        full_proj_transform: (4, 4) world→clip 完整投影矩阵 (已转置, 行向量约定)
        world_view_transform: (4, 4) world→view 矩阵 (已转置, 行向量约定)
        width: 图像宽度
        height: 图像高度

    Returns:
        pixel_coords: (N, 2) 像素坐标 (x, y)
        view_depth:   (N,)  view-space z 深度 (正值 = 相机前方)
    """
    from utils.graphics_utils import geom_transform_points, ndc_to_pixel

    N = xyz.shape[0]
    if N == 0:
        return (torch.empty(0, 2, dtype=xyz.dtype, device=xyz.device),
                torch.empty(0, dtype=xyz.dtype, device=xyz.device))

    # world → NDC
    ndc = geom_transform_points(xyz, full_proj_transform)  # (N, 3)
    ndc_xy = ndc[:, :2]

    # NDC → pixel (对齐 CUDA ndc2Pix)
    pixel_coords = ndc_to_pixel(ndc_xy, width, height)

    # view-space depth: xyz_hom @ world_view_transform, 取 z
    ones = torch.ones(N, 1, dtype=xyz.dtype, device=xyz.device)
    xyz_hom = torch.cat([xyz, ones], dim=1)  # (N, 4)
    view_pos = torch.matmul(xyz_hom, world_view_transform.unsqueeze(0))  # (N, 4)
    view_depth = view_pos[:, 2]  # (N,) view-space z

    return pixel_coords, view_depth


def filter_visible_points(pixel_coords, view_depth, width, height,
                          depth_threshold=0.2):
    """过滤不可见点: NaN/Inf、相机后方、图像边界外。

    Args:
        pixel_coords: (N, 2) 像素坐标
        view_depth:   (N,)  view-space z 深度
        width:  图像宽度
        height: 图像高度
        depth_threshold: view-space z 必须大于此值才视为相机前方 (对齐 CUDA in_frustum)

    Returns:
        valid_mask: (N,) bool, True = 可见 (通过了所有过滤)
    """
    N = pixel_coords.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.bool, device=pixel_coords.device)

    # 1. finite check
    finite_coords = torch.isfinite(pixel_coords).all(dim=-1)
    finite_depth = torch.isfinite(view_depth)

    # 2. view-space depth > threshold (对齐 CUDA in_frustum 的 p_view.z <= 0.2f 过滤)
    in_front = view_depth > depth_threshold

    # 3. pixel bounds
    x_in = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width)
    y_in = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)

    valid_mask = finite_coords & finite_depth & in_front & x_in & y_in
    return valid_mask


def draw_conf_overlay(image_pil, pixel_coords, radius=3):
    """在 PIL Image 副本上绘制红色实心圆，返回标注后的新 Image。

    Args:
        image_pil: 原始 PIL Image (RGB)
        pixel_coords: (K, 2) 要绘制红点的像素坐标
        radius: 圆点半径 (默认 3)

    Returns:
        annotated PIL Image (RGB)
    """
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated)

    coords_cpu = pixel_coords.detach().cpu()
    for i in range(coords_cpu.shape[0]):
        x, y = coords_cpu[i].tolist()
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 0, 0)
        )

    return annotated


def save_visualization_pair(render_tensor, annotated_pil, output_dir,
                            filename_prefix, metadata_dict):
    """保存原始渲染图和标注图对，以及 metadata。

    Args:
        render_tensor: (3, H, W) or (1, 3, H, W) 渲染结果 tensor (值域 [0,1])
        annotated_pil: 标注后的 PIL Image
        output_dir: 输出目录
        filename_prefix: 文件名前缀 (不含扩展名)
        metadata_dict: 要写入 metadata.jsonl 的字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 原始渲染图
    to_pil = ToPILImage()
    if render_tensor.dim() == 4:
        render_tensor = render_tensor.squeeze(0)
    original_pil = to_pil(render_tensor.clamp(0.0, 1.0).cpu())

    render_path = os.path.join(output_dir, f"{filename_prefix}_render.png")
    original_pil.save(render_path)

    # 标注图
    annotated_path = os.path.join(output_dir, f"{filename_prefix}_conf.png")
    annotated_pil.save(annotated_path)

    # metadata (jsonl append)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'a') as f:
        f.write(json.dumps(metadata_dict) + '\n')

    return render_path, annotated_path


def save_conf_visualization(gaussians, conf_mask, conf_score,
                            final_mask, selection_score,
                            vis_context, opt):
    """Conf 候选点可视化的顶层入口。

    在 densify_and_prune_Improved() 中 final_mask 确定后、long_axis_split() 前调用。
    快照当前 Gaussian 状态，投影 Conf 选中的中心到指定相机视图，绘制红点并保存。

    Args:
        gaussians: GaussianModel 实例
        conf_mask: (N,) bool — conf_mask_raw & has_enough_views
        conf_score: (N,) — Conf 分数
        final_mask: (N,) bool — selector 输出的 final_mask
        selection_score: (N,) — 用于 multinomial 采样的分数
        vis_context: dict with keys:
            model_path, camera, render_image, iteration
        opt: OptimizationParams (包含可视化配置)
    """
    model_path = vis_context['model_path']
    camera = vis_context['camera']
    render_image = vis_context['render_image']
    iteration = vis_context['iteration']

    mask_type = getattr(opt, 'conf_vis_mask_type', 'conf')
    max_points = getattr(opt, 'conf_vis_max_points', 5000)
    point_radius = getattr(opt, 'conf_vis_point_radius', 3)

    # ---- 快照: Gaussian 数量变化前保存所有数据 ----
    with torch.no_grad():
        num_gaussians = gaussians.get_xyz.shape[0]

        conf_mask_snapshot = conf_mask.detach().bool().clone()
        assert conf_mask_snapshot.ndim == 1
        assert conf_mask_snapshot.shape[0] == num_gaussians

        final_mask_snapshot = final_mask.detach().bool().clone()
        conf_score_snapshot = conf_score.detach().clone()
        selection_score_snapshot = selection_score.detach().clone()

        strategy = getattr(opt, 'candidate_selection_strategy', 'and')

    # ---- 按 mask_type 选择目标 mask 和对应 score ----
    output_dir = os.path.join(model_path, "conf_interval_visualization")
    camera_uid = getattr(camera, 'uid', '0')
    file_prefix = f"iteration_{iteration:06d}_view{camera_uid}"

    mask_types_to_process = []
    if mask_type == 'both':
        mask_types_to_process = ['conf', 'final_candidates']
    else:
        mask_types_to_process = [mask_type]

    for current_type in mask_types_to_process:
        if current_type == 'conf':
            target_mask = conf_mask_snapshot
            target_score = conf_score_snapshot
            type_suffix = 'conf'
        elif current_type == 'final_candidates':
            target_mask = final_mask_snapshot
            target_score = selection_score_snapshot
            type_suffix = 'final_candidates'
        else:
            continue

        selected_count = int(target_mask.sum().item())

        # ---- 获取选中的 xyz 和 indices ----
        with torch.no_grad():
            selected_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
            selected_xyz = gaussians.get_xyz.detach()[target_mask].clone()

        # ---- 投影 ----
        if selected_count == 0:
            # 没有选中任何 Gaussian：保存原图，标注图为原图副本
            to_pil = ToPILImage()
            if render_image.dim() == 4:
                img_tensor = render_image.squeeze(0)
            else:
                img_tensor = render_image
            original_pil = to_pil(img_tensor.clamp(0.0, 1.0).cpu())

            os.makedirs(output_dir, exist_ok=True)
            render_path = os.path.join(output_dir, f"{file_prefix}_render.png")
            annotated_path = os.path.join(output_dir, f"{file_prefix}_{type_suffix}.png")
            original_pil.save(render_path)
            original_pil.save(annotated_path)  # 两张内容一致

            metadata_dict = {
                'iteration': iteration,
                'camera_uid': camera_uid,
                'camera_name': getattr(camera, 'image_name', ''),
                'mask_type': current_type,
                'strategy': strategy,
                'conf_thr': getattr(opt, 'conf_thr', 0.8),
                'conf_min_views': getattr(opt, 'conf_min_views', 3),
                'num_gaussians_before': num_gaussians,
                'mask_count': 0,
                'visible_count': 0,
                'drawn_count': 0,
                'topk_score_name': type_suffix + '_score',
                'same_render': True,
                'trigger_reason': getattr(vis_context, 'trigger_reason', ''),
            }
            metadata_path = os.path.join(output_dir, "metadata.jsonl")
            with open(metadata_path, 'a') as f:
                f.write(json.dumps(metadata_dict) + '\n')

            print(f"[ConfVis] iter={iteration} mask_type={current_type}: "
                  f"0 selected gaussians, saved empty overlay")

        else:
            pixel_coords, view_depth = project_gaussian_centers(
                selected_xyz,
                camera.full_proj_transform.to(selected_xyz.device),
                camera.world_view_transform.to(selected_xyz.device),
                camera.image_width,
                camera.image_height,
            )

            # ---- 过滤 ----
            valid_mask = filter_visible_points(
                pixel_coords, view_depth,
                camera.image_width, camera.image_height,
            )
            visible_count = int(valid_mask.sum().item())

            visible_coords = pixel_coords[valid_mask]
            visible_indices = selected_indices[valid_mask]
            visible_scores = target_score[target_mask][valid_mask]

            # ---- Top-K 裁剪 ----
            drawn_count = visible_count
            if max_points > 0 and visible_count > max_points:
                topk = torch.topk(visible_scores, k=max_points, largest=True)
                visible_coords = visible_coords[topk.indices]
                visible_indices = visible_indices[topk.indices]
                drawn_count = max_points

            # ---- 保存原图 + 绘制并保存标注图 ----
            to_pil = ToPILImage()
            if render_image.dim() == 4:
                img_tensor = render_image.squeeze(0)
            else:
                img_tensor = render_image
            original_pil = to_pil(img_tensor.clamp(0.0, 1.0).cpu())

            # 只在 conf 类型或 only 类型时保存原图 (both 模式一张原图即可)
            if current_type == 'conf' or mask_type != 'both':
                os.makedirs(output_dir, exist_ok=True)
                render_path = os.path.join(output_dir, f"{file_prefix}_render.png")
                original_pil.save(render_path)

            # 绘制标注图
            if drawn_count > 0:
                annotated_pil = draw_conf_overlay(original_pil, visible_coords, point_radius)
            else:
                annotated_pil = original_pil.copy()

            os.makedirs(output_dir, exist_ok=True)
            annotated_path = os.path.join(output_dir, f"{file_prefix}_{type_suffix}.png")
            annotated_pil.save(annotated_path)

            # ---- metadata ----
            metadata_dict = {
                'iteration': iteration,
                'camera_uid': camera_uid,
                'camera_name': getattr(camera, 'image_name', ''),
                'mask_type': current_type,
                'strategy': strategy,
                'conf_thr': getattr(opt, 'conf_thr', 0.8),
                'conf_min_views': getattr(opt, 'conf_min_views', 3),
                'num_gaussians_before': num_gaussians,
                'mask_count': selected_count,
                'visible_count': visible_count,
                'drawn_count': drawn_count,
                'topk_score_name': current_type + '_score',
                'same_render': True,
                'trigger_reason': getattr(vis_context, 'trigger_reason', ''),
            }
            metadata_path = os.path.join(output_dir, "metadata.jsonl")
            with open(metadata_path, 'a') as f:
                f.write(json.dumps(metadata_dict) + '\n')

            # ---- 保存索引 (用于后续对齐验证) ----
            indices_data = {
                'iteration': iteration,
                'selected_indices': selected_indices.cpu().tolist(),
                'drawn_indices': visible_indices.cpu().tolist() if drawn_count > 0 else [],
            }
            indices_path = os.path.join(output_dir, f"{file_prefix}_{type_suffix}_indices.pt")
            torch.save(indices_data, indices_path)

            print(f"[ConfVis] iter={iteration} mask_type={current_type}: "
                  f"N={num_gaussians}, selected={selected_count}, "
                  f"visible={visible_count}, drawn={drawn_count}, "
                  f"strategy={strategy}")

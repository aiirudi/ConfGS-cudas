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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        # 新增计算 Conf 的梯度累积向量
        self.xyz_gradient_vec_accum = torch.empty(0)
        self.xyz_gradient_mag_accum = torch.empty(0)
        # Conf valid-view counter (separate from scalar EAS denom)
        self.xyz_gradient_conf_denom = torch.empty(0)

        # 新增分裂冷却计数器
        #self.split_cooldown = torch.empty(0) 
        
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0


        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        # fused_point_clound.shape: (N, 3) 表示每个高斯椭球的中心点
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # SH 函数，形状一半都是: (N, 3, 16)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Conf world-space gradient accumulators (3D vectors pulled back from NDC)
        self.xyz_gradient_vec_accum = torch.zeros((self.get_xyz.shape[0], 3), device='cuda')
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.xyz_gradient_conf_denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')

        # 初始化冷却计数器 (初始全部为0,表示可以立即参与分裂)
        #self.split_cooldown = torch.zeros((self.get_xyz.shape[0], 1), device='cuda', dtype=torch.int32)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.shfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, min_opacity):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        # 同样对新加入 Conf 的统计量修改
        self.xyz_gradient_vec_accum = self.xyz_gradient_vec_accum[valid_points_mask]
        self.xyz_gradient_mag_accum = self.xyz_gradient_mag_accum[valid_points_mask]
        self.xyz_gradient_conf_denom = self.xyz_gradient_conf_denom[valid_points_mask]

        # 同步修剪冷却计数器
        #self.split_cooldown = self.split_cooldown[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Conf world-space gradient accumulators (3D vectors pulled back from NDC)
        self.xyz_gradient_vec_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_conf_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 重置冷却计数器
        #self.split_cooldown = torch.zeros((self.get_xyz.shape[0],1), device="cuda", dtype=torch.int32)

    # RAP 实现: percent=True 时把 min_opacity 当作 quantile，对不透明度做按比例剪枝；
    # 由 train.py 在每次 opacity reset 后 300 iter（且 iter<9000）以 only_prune(0.2, True) 调用
    def only_prune(self, min_opacity, percent=False):
        if percent is True:
            # opacity_array.shape: (N, 1)
            opacity_array = self.get_opacity.detach().flatten()
            q = min_opacity
            min_opacity = torch.quantile(opacity_array, q)
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()

        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Conf world-space gradient accumulators (3D vectors pulled back from NDC)
        self.xyz_gradient_vec_accum= torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_conf_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 重置冷却计数器
        #self.split_cooldown = torch.zeros((self.get_xyz.shape[0],1), device="cuda", dtype=torch.int32)
    
        torch.cuda.empty_cache()
    
    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration, limitation,residual_offsets=None):
        # grad_vars.shape: (N, 1)
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        
        min_grad = opt.densify_grad_threshold

        # 统一 scores 形状为 (N,)
        if scores is None or iteration > 14500:
            scores = grad_vars.squeeze()  # (N,)
            if self.get_opacity.shape[0] < budget and iteration > 14500:
                min_grad = min_grad / 1.5
        else:
            # 确保传入的 scores 也是 (N,)
            if scores.dim() > 1:
                scores = scores.squeeze()
        
        # grad_qualifiers.shape: (N,)
        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)       
        
        
        # 计算冲突度
        conf = 1.0 - (torch.norm(self.xyz_gradient_vec_accum, dim=-1, keepdim=True)) / (self.xyz_gradient_mag_accum + 1e-6)
        conf[conf.isnan()] = 0.0
        # Zero valid views: no meaningful gradient observations
        zero_views = (self.xyz_gradient_conf_denom <= 0).squeeze(-1)
        conf[zero_views] = 0.0
        conf = conf.squeeze(-1)  # (N,)
        min_views = getattr(opt, "conf_min_views", 3)
        conf_thr = getattr(opt, "conf_thr", 0.6)
        has_enough_views = (self.xyz_gradient_conf_denom.squeeze(-1) >= min_views)  # (N,)
        conf_mask = (conf >= conf_thr)  # (N,)
        
        grad_qualifiers = grad_qualifiers & (has_enough_views & conf_mask)

        total_sum = torch.sum(grad_qualifiers).item()
        curr_points = len(self.get_xyz)
        budget = min(budget, total_sum + curr_points)
        all_budget = budget - curr_points
        
        if all_budget > 0:
            """
            self.long_axis_split(scores.clone(), all_budget, grad_qualifiers, opt.split_distance, opt.opacity_reduction, cooldown_iters=cooldown_iters)
            """
            self.long_axis_split(
                scores.clone(), 
                all_budget, 
                grad_qualifiers, 
                opt.split_distance, 
                opt.opacity_reduction, 
            )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if iteration < 14900:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def _compute_ndc_vjp_world(self, xyz_world, grad_ndc, update_filter, full_proj_transform, eps=1e-8):
        """
        Pull the per-view NDC-space positional gradient back to the
        common world coordinate system using the transpose of the
        camera projection Jacobian.

        Cross-view cancellation is evaluated only after all gradients
        are represented in the same world-space coordinate frame.

        The NDC Jacobian for row-vector convention (clip = xyz_h @ M):
          d(ndc_x)/d(xyz_i) = (M[i,0]*qw - M[i,3]*qx) / qw^2
          d(ndc_y)/d(xyz_i) = (M[i,1]*qw - M[i,3]*qy) / qw^2
        J_ndc ∈ R^{2×3}, g_world = J_ndc^T @ g_ndc ∈ R^3

        Args:
            xyz_world: (N, 3) all Gaussian world positions (detached)
            grad_ndc:  (n, 2) dL/d(NDC_xy) for visible Gaussians (detached)
            update_filter: indices or bool mask selecting n from N
            full_proj_transform: (4, 4) world-to-clip projection matrix
            eps: numerical stability threshold

        Returns:
            g_world: (N, 3) world-space gradient vectors (zero for invalid)
            valid:   (N,)   boolean mask of Gaussians with valid world gradients
        """
        N = xyz_world.shape[0]
        device = xyz_world.device

        # Only transform the visible subset
        xyz_vis = xyz_world[update_filter]  # (n, 3)

        # Row-vector homogeneous transform: clip = xyz_h @ M
        xyz_h = torch.cat([xyz_vis, torch.ones_like(xyz_vis[:, :1])], dim=-1)  # (n, 4)
        clip = xyz_h @ full_proj_transform  # (n, 4)

        qx = clip[:, 0]  # (n,)
        qy = clip[:, 1]  # (n,)
        qw = clip[:, 3]  # (n,)

        # Exclude points at or behind the camera plane (positive clip_w convention)
        safe_qw = torch.where(qw > eps, qw, torch.full_like(qw, eps))

        # Extract matrix columns for the xyz rows (indices 0,1,2) of the
        # projection-matrix rows that participate in ndc_x, ndc_y, and w.
        # full_proj_transform is stored transposed in PyTorch (row-vector
        # convention matches geom_transform_points).
        Mx_xyz = full_proj_transform[:3, 0]  # (3,)  column 0, rows 0..2
        My_xyz = full_proj_transform[:3, 1]  # (3,)  column 1, rows 0..2
        Mw_xyz = full_proj_transform[:3, 3]  # (3,)  column 3, rows 0..2

        # Jacobian rows: d(ndc_x)/d(xyz) and d(ndc_y)/d(xyz)
        jac_x = (Mx_xyz[None, :] * safe_qw[:, None] - Mw_xyz[None, :] * qx[:, None]) / safe_qw[:, None].square()  # (n, 3)
        jac_y = (My_xyz[None, :] * safe_qw[:, None] - Mw_xyz[None, :] * qy[:, None]) / safe_qw[:, None].square()  # (n, 3)

        # J_ndc: (n, 2, 3), then g_world = J_ndc^T @ g_ndc
        jacobian = torch.stack([jac_x, jac_y], dim=1)  # (n, 2, 3)
        g_world_vis = torch.bmm(jacobian.transpose(1, 2), grad_ndc.unsqueeze(-1)).squeeze(-1)  # (n, 3)

        # Validity: front-camera (qw > 0), finite inputs and outputs
        valid_vis = (
            (qw > eps)
            & torch.isfinite(qx) & torch.isfinite(qy) & torch.isfinite(qw)
            & torch.isfinite(grad_ndc).all(dim=-1)
            & torch.isfinite(jacobian).reshape(grad_ndc.shape[0], -1).all(dim=-1)
            & torch.isfinite(g_world_vis).all(dim=-1)
        )

        # Scatter back to full-size tensors
        g_world = torch.zeros((N, 3), device=device, dtype=xyz_world.dtype)
        valid = torch.zeros(N, device=device, dtype=torch.bool)

        g_world[update_filter] = g_world_vis
        valid[update_filter] = valid_vis

        g_world[~valid] = 0.0
        return g_world, valid

    def add_densification_stats(self, viewspace_point_tensor, update_filter, viewpoint_cam=None):
        # Normalize update_filter to 1D indices (nonzero() returns (n,1) in some versions)
        idx = update_filter.view(-1) if update_filter.dim() > 1 else update_filter
        self.xyz_gradient_accum[idx] += torch.norm(viewspace_point_tensor.grad[idx,:2], dim=-1, keepdim=True)

        # Conf: pull per-view NDC-space positional gradient back to world-space
        # via the camera projection Jacobian transpose, then accumulate the
        # 3D world-space vector and its norm.
        if viewpoint_cam is not None:
            g_ndc = viewspace_point_tensor.grad[idx, :2].detach()  # (n,2) dL/d(NDC_xy)
            g_world, valid = self._compute_ndc_vjp_world(
                self.get_xyz.detach(),
                g_ndc,
                idx,
                viewpoint_cam.full_proj_transform,
            )
            # Only accumulate for Gaussians with valid world gradients
            self.xyz_gradient_vec_accum[valid] += g_world[valid]
            self.xyz_gradient_mag_accum[valid] += torch.norm(g_world[valid], dim=-1, keepdim=True)
            self.xyz_gradient_conf_denom[valid] += 1
        else:
            # Fallback: original 2D NDC accumulation (backward compatibility)
            # Pad to 3D with zero z-component to match the (N,3) accumulator
            g = viewspace_point_tensor.grad[idx, :2]  # (n,2)
            g_pad = torch.cat([g, torch.zeros_like(g[:, :1])], dim=-1)  # (n,3)
            self.xyz_gradient_vec_accum[idx] += g_pad
            self.xyz_gradient_mag_accum[idx] += torch.norm(g, dim=-1, keepdim=True)
            self.xyz_gradient_conf_denom[idx] += 1

        # EAS scalar denom unchanged
        self.denom[idx] += 1

    # EAS 中的计算视角绝对值
    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter, viewpoint_cam=None):
        # Normalize update_filter to 1D indices (nonzero() returns (n,1) in some versions)
        idx = update_filter.view(-1) if update_filter.dim() > 1 else update_filter
        self.xyz_gradient_accum[idx] += torch.norm(viewspace_point_tensor.grad[idx,2:], dim=-1, keepdim=True)

        # Conf: pull per-view NDC-space positional gradient back to world-space
        # via the camera projection Jacobian transpose, then accumulate the
        # 3D world-space vector and its norm.
        if viewpoint_cam is not None:
            g_ndc = viewspace_point_tensor.grad[idx, :2].detach()  # (n,2) dL/d(NDC_xy)
            g_world, valid = self._compute_ndc_vjp_world(
                self.get_xyz.detach(),
                g_ndc,
                idx,
                viewpoint_cam.full_proj_transform,
            )
            self.xyz_gradient_vec_accum[valid] += g_world[valid]
            self.xyz_gradient_mag_accum[valid] += torch.norm(g_world[valid], dim=-1, keepdim=True)
            self.xyz_gradient_conf_denom[valid] += 1
        else:
            # Fallback: original 2D NDC accumulation (backward compatibility)
            # Pad to 3D with zero z-component to match the (N,3) accumulator
            g = viewspace_point_tensor.grad[idx, :2]  # (n,2)
            g_pad = torch.cat([g, torch.zeros_like(g[:, :1])], dim=-1)  # (n,3)
            self.xyz_gradient_vec_accum[idx] += g_pad
            self.xyz_gradient_mag_accum[idx] += torch.norm(g, dim=-1, keepdim=True)
            self.xyz_gradient_conf_denom[idx] += 1

        # EAS scalar denom unchanged
        self.denom[idx] += 1

    # LAS 实现: 按 score 加权（multinomial）从可分裂候选中采 budget 个父高斯，
    # 仅沿最长 scaling 轴分裂为两个子高斯（±split_distance·3σ_long），
    # 长轴 rescale 为 (1-rate)/√(1-rate²)、全轴 ×√(1-rate²)，
    # 子高斯 opacity ×= opacity_reduction，最后把父高斯 prune 掉
    def long_axis_split(self, grads, budget, filter, split_distance, opacity_reduction):
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]

        padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
        padded_importance[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")

        num = (padded_importance > 0).sum().item()
        if budget > num:
            budget = num

        # 按照 grads 权重从可以分裂的所有高斯小球中采 budget 个样本
        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        # 更改选择mask
        selected_pts_mask[sampled_indices] = True
        # 获得被选中高斯小球的scaleing: std.shape:(K, 3)
        stds = self.get_scaling[selected_pts_mask]

        # 找出每个高斯所有轴中的最长轴长度以及是哪个轴
        # max_values.shape == max_indices.shape: (K, 1)
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)

        # mask.shape: (N, 3) 将最长轴所在的维度设置为 1
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)

        # samples.shape: (K, 3), 只有最长轴为 3 * std_long
        samples = stds * mask * 3
        reduction = opacity_reduction

        # 子代远离父代的长度
        rate = split_distance
        
        # x1.shape: (K, 3)
        x1 = samples * rate
        rate_w = 1 - rate
        rate_h = math.sqrt(1-rate*rate)
        # x1.shape: (2K, 3) 给每个父高斯生成两个子高斯的偏移（正负方向）
        x1 = torch.cat([x1, -x1], dim=0)

        # rots.shape: (K, 3, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)

        # 返回生成的子代高斯的各个尺寸
        new_scaling = self.scaling_inverse_activation(stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h)
        
        # new_opacity.shape: (2K, 1)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * reduction).repeat(2, 1)
        # new_rotaton.shape: (2K, 4) 直接继续父代的rotaion, SH 函数也是
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

        # 为新生成的子Gaussians设置冷却期
        # 子Gaussians位于末尾 2*K 个位置
        num_children = 2 * selected_pts_mask.sum()
        #self.split_cooldown[-num_children:] = cooldown_iters

        # 将子代的参数加入到 optimizer 后面
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
    
        # 生成 父代 filter
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        #将父代减枝
        self.prune_points(prune_filter)
        
    def update_split_cooldown(self):
        """每次训练迭代调用,递减冷却计数器"""
        self.split_cooldown = torch.clamp(self.split_cooldown - 1, min=0)
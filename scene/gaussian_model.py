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
from utils.candidate_selector import apply_fixed_budget
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
        
        self.conf_world_sum = torch.empty(0)
        self.conf_norm_sum = torch.empty(0)
        self.conf_view_count = torch.empty(0, dtype=torch.int32)
        self.conf_score = torch.empty(0)
        self.conf_history = torch.empty(0)
        self.conf_camera_ids = torch.empty(0, dtype=torch.int64)
        self.conf_window_size = 3
        self.conf_topology_version = 0
        # 候选选择统计（每次 densification 时更新）
        self.candidate_stats = {}

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
            {
                'version': 2,
                'window_size': self.conf_window_size,
                'history': self.conf_history,
                'camera_ids': self.conf_camera_ids,
                'world_sum': self.conf_world_sum,
                'norm_sum': self.conf_norm_sum,
                'view_count': self.conf_view_count,
                'score': self.conf_score,
            },
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) not in (12, 13):
            raise ValueError('Unsupported Gaussian checkpoint format')
        conf_state = model_args[12] if len(model_args) == 13 else None
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
        self.spatial_lr_scale) = model_args[:12]
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if conf_state is not None:
            if not isinstance(conf_state, dict) or conf_state.get('version') not in (1, 2):
                raise ValueError('Unsupported Conf checkpoint state')
            if conf_state['version'] == 1:
                # v1 stored only sums and a global camera set; per-Gaussian
                # history cannot be reconstructed, so start an empty window.
                conf_state = None
        if conf_state is not None:
            n = self.get_xyz.shape[0]
            window = conf_state.get('window_size')
            if not isinstance(window, int) or window < 2:
                raise ValueError('Invalid Conf checkpoint window size')
            if window != self.conf_window_size:
                raise ValueError(f'Conf checkpoint window size {window} differs from configured conf_window_size {self.conf_window_size}')
            specs = [('world_sum', (n, 3), torch.float32),
                     ('norm_sum', (n, 1), torch.float32),
                     ('view_count', (n, 1), torch.int32),
                     ('score', (n, 1), torch.float32),
                     ('history', (n, window, 4), torch.float32),
                     ('camera_ids', (n, window), torch.int64)]
            for key, shape, dtype in specs:
                value = conf_state.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != self.get_xyz.device:
                    raise ValueError(f'Invalid Conf checkpoint tensor: {key}')
            self.conf_world_sum = conf_state['world_sum'].contiguous()
            self.conf_norm_sum = conf_state['norm_sum'].contiguous()
            self.conf_view_count = conf_state['view_count'].contiguous()
            self.conf_score = conf_state['score'].contiguous()
            self.conf_history = conf_state['history'].contiguous()
            self.conf_camera_ids = conf_state['camera_ids'].contiguous()
            self.conf_window_size = window
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
        self.conf_window_size = int(getattr(training_args, 'conf_window_size', 3))
        min_views = int(getattr(training_args, 'conf_min_views', 2))
        if self.conf_window_size < 2 or not 2 <= min_views <= self.conf_window_size:
            raise ValueError('Conf requires conf_window_size >= 2 and 2 <= conf_min_views <= conf_window_size')
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.reset_conf_window()

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

    def reset_conf_window(self):
        """Clear the per-Gaussian rolling view history (initialization only)."""
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        self.conf_world_sum = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self.conf_norm_sum = torch.zeros((n, 1), device=device, dtype=torch.float32)
        self.conf_view_count = torch.zeros((n, 1), device=device, dtype=torch.int32)
        self.conf_score = torch.zeros((n, 1), device=device, dtype=torch.float32)
        self.conf_history = torch.zeros((n, self.conf_window_size, 4), device=device, dtype=torch.float32)
        self.conf_camera_ids = torch.full((n, self.conf_window_size), -1, device=device, dtype=torch.int64)

    @torch.no_grad()
    def add_conf_stats(self, samples, camera_key):
        """Refresh each contributing Gaussian's last-N-view CUDA history."""
        if int(camera_key) < 0:
            raise ValueError('Conf camera key must be nonnegative')
        if not isinstance(samples, torch.Tensor) or samples.shape != (self.get_xyz.shape[0], 4):
            raise ValueError('Conf samples must have shape (N,4) for current topology')
        from diff_gaussian_rasterization import accumulate_conf
        accumulate_conf(samples.detach(), int(camera_key),
                        self.conf_history, self.conf_camera_ids,
                        self.conf_view_count, self.conf_world_sum,
                        self.conf_norm_sum, self.conf_score)

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

        self.conf_world_sum = self.conf_world_sum[valid_points_mask]
        self.conf_norm_sum = self.conf_norm_sum[valid_points_mask]
        self.conf_view_count = self.conf_view_count[valid_points_mask]
        self.conf_score = self.conf_score[valid_points_mask]
        self.conf_history = self.conf_history[valid_points_mask]
        self.conf_camera_ids = self.conf_camera_ids[valid_points_mask]
        self.conf_topology_version += 1

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

        old_n = self.conf_history.shape[0]
        new_n = self.get_xyz.shape[0] - old_n
        if new_n < 0:
            raise RuntimeError('Gaussian topology shrank during append')
        device = self.get_xyz.device
        self.conf_world_sum = torch.cat((self.conf_world_sum, torch.zeros((new_n, 3), device=device)), dim=0)
        self.conf_norm_sum = torch.cat((self.conf_norm_sum, torch.zeros((new_n, 1), device=device)), dim=0)
        self.conf_view_count = torch.cat((self.conf_view_count, torch.zeros((new_n, 1), device=device, dtype=torch.int32)), dim=0)
        self.conf_score = torch.cat((self.conf_score, torch.zeros((new_n, 1), device=device)), dim=0)
        self.conf_history = torch.cat((self.conf_history, torch.zeros((new_n, self.conf_window_size, 4), device=device)), dim=0)
        self.conf_camera_ids = torch.cat((self.conf_camera_ids, torch.full((new_n, self.conf_window_size), -1, device=device, dtype=torch.int64)), dim=0)
        self.conf_topology_version += 1

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
            prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze(-1)

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

        self.conf_world_sum = self.conf_world_sum[valid_points_mask]
        self.conf_norm_sum = self.conf_norm_sum[valid_points_mask]
        self.conf_view_count = self.conf_view_count[valid_points_mask]
        self.conf_score = self.conf_score[valid_points_mask]
        self.conf_history = self.conf_history[valid_points_mask]
        self.conf_camera_ids = self.conf_camera_ids[valid_points_mask]
        self.conf_topology_version += 1

        # 重置冷却计数器
        #self.split_cooldown = torch.zeros((self.get_xyz.shape[0],1), device="cuda", dtype=torch.int32)
    
        torch.cuda.empty_cache()
    
    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration, limitation, residual_offsets=None, rfas_score=None, vis_context=None):
        strategy = getattr(opt, 'candidate_selection_strategy', 'conf_only')
        if strategy != 'conf_only':
            raise ValueError(f"CUDA Conf supports candidate_selection_strategy='conf_only'; got {strategy!r}. Legacy strategies require or bypass the removed gradient gate.")

        n = self.get_xyz.shape[0]
        num_gaussians_before = n
        conf = self.conf_score.reshape(-1)
        min_views = max(2, int(getattr(opt, 'conf_min_views', 2)))
        conf_mask = ((self.conf_view_count.reshape(-1) >= min_views)
                     & (self.conf_norm_sum.reshape(-1) > 0)
                     & torch.isfinite(self.conf_norm_sum.reshape(-1))
                     & torch.isfinite(self.conf_world_sum).all(dim=-1)
                     & torch.isfinite(conf)
                     & (conf >= getattr(opt, 'conf_thr', 0.85)))
        if scores is None:
            selection_score = conf.clone()
        else:
            selection_score = scores.reshape(-1).clone()
            if selection_score.numel() != n:
                raise ValueError('Ranking scores must contain one value per Gaussian')
        finite_score = torch.isfinite(selection_score)
        final_mask = conf_mask & finite_score
        selection_score[~finite_score] = 0.0

        budget_mode = getattr(opt, 'candidate_budget_mode', 'native')
        target_budget = int(final_mask.sum().item())
        actual_budget = target_budget
        if budget_mode == 'fixed':
            reference = getattr(opt, 'candidate_budget_reference', 'match_and')
            if reference == 'match_and':
                raise ValueError('candidate_budget_reference=match_and requires abs-grad; use fixed_number or fixed_ratio with CUDA Conf')
            if reference not in ('fixed_number', 'fixed_ratio'):
                raise ValueError(f'Unsupported Conf budget reference: {reference}')
            final_mask, actual_budget, target_budget = apply_fixed_budget(
                final_mask, selection_score, target_budget, reference, opt,
                strategy='conf_only')
        elif budget_mode != 'native':
            raise ValueError(f'Unsupported Conf budget mode: {budget_mode}')

        # LAS samples by weight; a valid Conf candidate with a zero fused
        # score must remain eligible rather than silently disappearing.
        selection_score[final_mask] = selection_score[final_mask].clamp_min(1e-6)

        def scalar_stats(values):
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return (0.0,) * 5
            return tuple(float(x.item()) for x in (
                values.mean(), values.std(unbiased=False), values.median(),
                values.min(), values.max()))

        conf_stats = scalar_stats(conf)
        rank_stats = scalar_stats(selection_score)
        self.candidate_stats = {
            'strategy': 'conf_only', 'iteration': iteration,
            'n_valid': n, 'n_conf_candidates': int(conf_mask.sum().item()),
            'n_final_candidates': int(final_mask.sum().item()),
            'candidate_ratio': float(final_mask.sum().item()) / max(n, 1),
            'target_budget': target_budget, 'actual_budget': actual_budget,
            'conf_mean': conf_stats[0], 'conf_std': conf_stats[1],
            'conf_median': conf_stats[2], 'conf_min': conf_stats[3],
            'conf_max': conf_stats[4],
            'sel_score_mean': rank_stats[0], 'sel_score_std': rank_stats[1],
            'sel_score_median': rank_stats[2], 'sel_score_min': rank_stats[3],
            'sel_score_max': rank_stats[4],
            'n_nan': int(torch.isnan(conf).sum().item()),
            'n_inf': int(torch.isinf(conf).sum().item()),
        }

        total_sum = torch.sum(final_mask).item()
        curr_points = num_gaussians_before
        budget = min(budget, total_sum + curr_points)
        all_budget = budget - curr_points
        num_split = 0

        # Conf 候选点可视化 hook: 在 interval 累计统计完成后、Gaussian 数量变化前生成快照和标注图
        # 红点来自完整 densification interval 内的累计 Conf 统计，不是单步瞬时结果
        # 可视化在空间多样性选择之前运行，以显示原始 final_mask（不受空间选择影响）
        if vis_context is not None:
            from utils.conf_visualization import save_conf_visualization
            save_conf_visualization(
                self, conf_mask, conf,
                final_mask, selection_score,
                vis_context, opt,
                abs_mask=None, abs_score=None,
            )

        # ---- 空间多样性候选选择器 (后处理模块) ----
        # 在可视化之后、long_axis_split 之前运行，确保可视化显示原始候选
        # 通过 enable_spatial_diversity 控制总开关，spatial_interval 控制执行频率
        if (getattr(opt, 'enable_spatial_diversity', False)
                and all_budget > 0
                and iteration <= 14500
                and iteration % getattr(opt, 'spatial_interval', 100) == 0):
            from utils.spatial_diversity import select_spatially_diverse_candidates

            # 解析体素尺寸: 'auto' 或非正数 → 自适应; 正数 → 固定值
            raw_voxel_size = getattr(opt, 'spatial_voxel_size', 'auto')
            if isinstance(raw_voxel_size, str) and raw_voxel_size == 'auto':
                resolved_voxel_size = None
            else:
                try:
                    val = float(raw_voxel_size)
                    resolved_voxel_size = val if val > 0 else None
                except (ValueError, TypeError):
                    resolved_voxel_size = None

            spatial_mask, spatial_stats = select_spatially_diverse_candidates(
                candidate_mask=final_mask,
                priority_scores=selection_score.clone(),
                xyz=self.get_xyz.detach(),
                budget=all_budget,
                method=getattr(opt, 'spatial_diversity_method', 'voxel'),
                voxel_size=resolved_voxel_size,
                scales=self.get_scaling.detach(),
                max_per_voxel=getattr(opt, 'spatial_max_per_voxel', 1),
                radius_scale=getattr(opt, 'spatial_radius_scale', 1.0),
                suppression_weight=getattr(opt, 'spatial_suppression_weight', 1.0),
                voxel_scale=getattr(opt, 'spatial_voxel_scale', 2.0),
            )

            # 确保空间选中的候选具有严格正的 selection_score
            # (LAS 的 torch.multinomial 只对 padded_importance > 0 抽样)
            # 对全体 spatial_mask 中 score <= 0 或非有限的值 clamp 到安全小量
            if spatial_mask.sum() > 0:
                sel_scores = selection_score[spatial_mask]
                need_clamp = (sel_scores <= 0) | (~torch.isfinite(sel_scores))
                if need_clamp.any():
                    sel_scores[need_clamp] = 1e-6
                    selection_score[spatial_mask] = sel_scores

            # 更新统计信息
            self.candidate_stats['spatial_candidate_count'] = spatial_stats['candidate_count']
            self.candidate_stats['spatial_selected_count'] = spatial_stats['selected_count']
            self.candidate_stats['spatial_occupied_voxels'] = spatial_stats['occupied_voxel_count']
            self.candidate_stats['spatial_voxel_size'] = spatial_stats['voxel_size']
            self.candidate_stats['spatial_runtime_ms'] = spatial_stats['runtime_ms']
            self.candidate_stats['spatial_jaccard'] = spatial_stats['jaccard_vs_original']
            self.candidate_stats['spatial_replaced'] = spatial_stats['replaced_count']
            self.candidate_stats['spatial_method'] = spatial_stats['method']

            # 将空间选择后的掩码替换 final_mask
            final_mask = spatial_mask
            # 更新预算相关统计，确保与 LAS 实际输入一致
            effective_selected = int(final_mask.sum().item())
            self.candidate_stats['n_final_candidates'] = effective_selected
            self.candidate_stats['actual_budget'] = effective_selected

        if all_budget > 0:
            self.long_axis_split(
                selection_score.clone(),
                all_budget,
                final_mask,
                opt.split_distance,
                opt.opacity_reduction,
            )
            num_split = min(all_budget, int(final_mask.sum().item()))

        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        num_pruned = 0

        if iteration < 14900:
            num_pruned = int(prune_mask.sum().item())
            self.prune_points(prune_mask)

        num_gaussians_after = len(self.get_xyz)

        self.candidate_stats['num_gaussians_before'] = num_gaussians_before
        self.candidate_stats['num_gaussians_after'] = num_gaussians_after
        self.candidate_stats['num_split'] = num_split
        self.candidate_stats['num_pruned'] = num_pruned

        torch.cuda.empty_cache()

    def _compute_ndc_vjp_world(self, xyz_world, grad_ndc, update_filter, full_proj_transform, eps=1e-8):
        """
        Pull the per-view NDC-space positional gradient back to the
        common world coordinate system using the transpose of the
        camera projection Jacobian.

        Cross-view cancellation is evaluated only after all gradients
        are represented in the same world-space coordinate frame.

        The NDC Jacobian for row-vector convention (clip = xyz_h @ M):
          r = 1 / (qw + 1e-7)
          d(ndc_x)/d(xyz_i) = M[i,0]*r - M[i,3]*qx*r^2
          d(ndc_y)/d(xyz_i) = M[i,1]*r - M[i,3]*qy*r^2
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

        # Match preprocessCUDA's stabilized projection exactly. Invalid rows
        # use a safe denominator only to keep this reference helper finite.
        reciprocal_w = torch.where(qw > eps, 1.0 / (qw + 1e-7), torch.zeros_like(qw))

        # Extract matrix columns for the xyz rows (indices 0,1,2) of the
        # projection-matrix rows that participate in ndc_x, ndc_y, and w.
        # full_proj_transform is stored transposed in PyTorch (row-vector
        # convention matches geom_transform_points).
        Mx_xyz = full_proj_transform[:3, 0]  # (3,)  column 0, rows 0..2
        My_xyz = full_proj_transform[:3, 1]  # (3,)  column 1, rows 0..2
        Mw_xyz = full_proj_transform[:3, 3]  # (3,)  column 3, rows 0..2

        # Jacobian rows: d(ndc_x)/d(xyz) and d(ndc_y)/d(xyz)
        jac_x = Mx_xyz[None, :] * reciprocal_w[:, None] - Mw_xyz[None, :] * qx[:, None] * reciprocal_w[:, None].square()
        jac_y = My_xyz[None, :] * reciprocal_w[:, None] - Mw_xyz[None, :] * qy[:, None] * reciprocal_w[:, None].square()

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
        raise RuntimeError('Python densification gradient accumulation is obsolete; use CUDA Conf samples')

    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter, viewpoint_cam=None):
        raise RuntimeError('Abs-gradient accumulation was removed; use CUDA Conf samples')

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

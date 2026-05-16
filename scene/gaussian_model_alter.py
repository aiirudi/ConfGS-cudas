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

        # 统计小球最大轴长
        self.max_radii2D = torch.empty(0)
        
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
            self.max_radii2D, # 新加的
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
        self.max_radii2D,  # 新加的
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

        # 初始化小球轴长
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 初始化新增 Conf 向量
        self.xyz_gradient_vec_accum = torch.zeros((self.get_xyz.shape[0], 2), device='cuda')
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')

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

        # 同样对轴长减枝
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if hasattr(self, "tmp_radii") and self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
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

        # 重置新加入计算 Conf 的统计量
        self.xyz_gradient_vec_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 对球的半径也重新统计一下
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

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

        # 对新加入的统计量重置
        self.xyz_gradient_vec_accum= torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.xyz_gradient_mag_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        torch.cuda.empty_cache()
    
    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration):
        # grad_vars.shape: (N, 1) 用来记录每个高斯的梯度
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        
        # min_grad 是否进行致密操作的阈值
        min_grad = opt.densify_grad_threshold

        # 后期训练策：没有 scores 或迭代很后时，退化为用梯度当 scores
        # 这里要随着最后一次致密的iteration 一样
        if scores is None or iteration > 14500:
            print('scores is None')
            scores = grad_vars.squeeze()
            if self.get_opacity.shape[0] < budget and iteration > 14500:
                min_grad = min_grad / 1.5

        # 计算冲突度
        # conf.shpe
        conf = 1.0 - (torch.norm(self.xyz_gradient_vec_accum, dim=-1, keepdim=True)) / (self.xyz_gradient_mag_accum + 1e-6)
        conf[conf.isnan()] = 0.0
        conf = conf.squeeze(-1)

        min_views = getattr(opt, "conf_min_views", 3)      # 建议至少 3 个视角，否则 Conf 不稳定
        conf_thr  = getattr(opt, "conf_thr", 0.6)   
        has_enough_views = (self.denom.squeeze(-1) >= min_views)
        conf_mask = (conf >= conf_thr)
        #print('conf_mask shape is :', conf_mask.shape)

        # 是否 增密mask, grad_qualifiers.shape: (N,)
        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)
        #print('grad_qualifiers shape is :', grad_qualifiers.shape)
        # 检查过了 grad_qualifiers 和 conf_mask 的形状是一样的。

        # 新加入conf 后的条件
        #grad_qualifiers = grad_qualifiers & has_enough_views & conf_mask
    
        # 归一化scores 分数
        """  没加入 RFAS 所以不用归一化
        max_score, min_score = torch.max(scores), torch.min(scores)
        print('scores shape is ', scores.shape)
        scores = (scores - min_score) / (max_score - min_score)
        """

        # total_sum: (N) int 最后增密小球的数量
        total_sum = torch.sum(grad_qualifiers).item()
        # curr_points.shape: (N) int 现有小球数量
        curr_points = len(self.get_xyz)
        # budget 小球数量阈值，若增密数量和现有小球数量相机超过阈值则设置为阈值数量
        budget = min(budget, total_sum + curr_points)

        all_budget = budget - curr_points
        
        # 如果 all_budget <= 说明小球数量此时已经达到上线，此时不新增
        if all_budget > 0:
            self.long_axis_split(scores.clone(), all_budget, grad_qualifiers, opt.split_distance, opt.opacity_reduction)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # 这个固定参数也要改，但是我没改，屮
        if iteration < 14900:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # ----------------------------------------------3DGS 剪枝操作----------------------------------------
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 梯度大的点就会被 clone 或者 split
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 若高斯椭球的 scale 大于 scale_threshold 就会被 split
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # len(selected_pts_mask) == K
        # 这里是把选中的 K 个大高斯每个大高斯都分裂成 N 个小高斯也就是
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # (N * K, 3)
        means =torch.zeros((stds.size(0), 3),device="cuda") # (N * K, 3)  
        samples = torch.normal(mean=means, std=stds) #采样 (N * K. 3)
        
        # (N * K, 3, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 把新生成的点的数量全部加入到一个列表中，此时的列表长度为 N + N * K, 此时还没有剔除大高斯
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))) 
        
        self.prune_points(prune_filter) # 这个函数中就是用来剔除大高斯的
        

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        
        # grad 大说明高斯椭球拟合不好，值得加点来使它变得更加拟合
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 且 clone 点的话我们只 clone 小点不会 clone 大点
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 复制一份 clone 的点的所有属性
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        # 平均梯度
        grads = self.xyz_gradient_accum / self.denom
        # 梯度过大直接设置为 0
        grads[grads.isnan()] = 0.0 # grads.shape: (P, 1)

        # 暂存 高斯椭球 半径 list, 这个过程需要让“与点一一对应的 radii 向量”跟着点数变化，否则形状对不上。
        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        # prune_mask.shape: (P,)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
    # ----------------------------------------------3DGS 剪枝操作----------------------------------------

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

        # 在计算 abs 时候同时统计带方向的梯度向量
        # Conf：带方向的 xy 梯度向量
        g = viewspace_point_tensor.grad[update_filter, :2]  # (n,2)
        self.xyz_gradient_vec_accum[update_filter] += g
        self.xyz_gradient_mag_accum[update_filter] += torch.norm(g, dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    # EAS 中的计算视角绝对值
    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)

        # 在计算 abs 时候同时统计带方向的梯度向量
        # Conf：带方向的 xy 梯度向量
        g = viewspace_point_tensor.grad[update_filter, :2]  # (n,2)
        self.xyz_gradient_vec_accum[update_filter] += g
        self.xyz_gradient_mag_accum[update_filter] += torch.norm(g, dim=-1, keepdim=True)
        
        self.denom[update_filter] += 1

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

        # 将子代的参数加入到 optimizer 后面
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
    
        # 生成 父代 filter
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        #将父代减枝
        self.prune_points(prune_filter)

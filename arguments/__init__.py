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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t in (list, tuple):
                    group.add_argument("--" + key, default=value, nargs='*')
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"  # "cpu"
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00004
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.shfeature_lr = 0.005
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500

        #freq regulation parameter
        self.lambda_efre_wl = 0.05
        self.lambda_efre_wh = 0.1
        self.regulation_convert_iter = 0

        # 幅度反重建损失参数
        self.lambda_amp_rec = 0.5
        self.amp_rec_stage1_iter = 5000
        self.amp_rec_stage2_iter = 12000
        self.amp_rec_mask_ratio_low = 0.15
        self.amp_rec_mask_ratio_mid = 0.4
        self.amp_rec_mask_ratio_high = 0.90
        self.amp_rec_final_transition_len = 8000


        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0003
        self.random_background = False
        self.optimizer_type = "default"
        
        # mip-nerf360
        self.budget = 1300_000
        
        # Tanks & temples
        #self.budget = 125_0000
        self.split_distance = 0.45
        self.opacity_reduction = 0.6

        # Conf parameter
        self.conf_min_views = 3
        self.conf_thr = 0.85

        # ---- 候选点选择策略参数 ----
        # 策略选择: and | or | abs_only | conf_only | weighted_score | soft_fusion | rfas_rank
        self.candidate_selection_strategy = 'and'
        # 预算模式: native | fixed
        self.candidate_budget_mode = 'native'
        # fixed 预算参考: fixed_number | fixed_ratio | match_and
        self.candidate_budget_reference = 'match_and'
        # fixed_number 模式下的固定数量
        self.candidate_fixed_budget = 1000
        # fixed_ratio 模式下的比例
        self.candidate_fixed_ratio = 0.05

        # weighted_score / soft_fusion 融合权重 alpha
        self.candidate_weight_alpha = 0.8
        # 归一化方式: minmax | zscore | percentile
        self.candidate_score_normalization = 'percentile'
        # 连续分数的选择方式: threshold | topk
        self.candidate_score_selection = 'threshold'
        # threshold 模式的阈值
        self.candidate_score_threshold = 0.95
        # topk 模式的绝对数量 (0 = 使用 ratio)
        self.candidate_topk = 0
        # topk 模式的比例 (candidate_topk > 0 时优先使用 absolute)
        self.candidate_topk_ratio = 0.05

        # soft_fusion 参数
        # 融合方式: weighted | soft_and | soft_or
        self.soft_fusion_type = 'weighted'
        self.soft_abs_temperature = 1.0
        self.soft_conf_temperature = 1.0
        self.soft_selection_threshold = 0.5

        # rfas_rank 参数
        self.candidate_rfas_topk = 0
        self.candidate_rfas_topk_ratio = 0.05

        # ---- 空间多样性候选选择器参数 ----
        # 是否启用空间多样性模块
        self.enable_spatial_diversity = True
        # 空间选择方法: voxel | radius_nms | soft_suppression
        self.spatial_diversity_method = 'voxel'
        # 体素尺寸: 'auto' = 自动从 scale 计算; 数值字符串如 '0.5' 也可
        self.spatial_voxel_size = 'auto'
        # 自动体素尺寸系数 (voxel_size = scale * median(max_scale))
        self.spatial_voxel_scale = 2.0
        # 每轮每体素最多选择候选数
        self.spatial_max_per_voxel = 1
        # 预算模式: original | fixed | ratio
        self.spatial_budget_mode = 'original'
        # fixed 模式下的目标预算
        self.spatial_budget = -1
        # ratio 模式下占候选数的比例
        self.spatial_budget_ratio = 1.0
        # Radius NMS 自适应半径系数 (预留)
        self.spatial_radius_scale = 1.0
        # Soft suppression 权重 (预留)
        self.spatial_suppression_weight = 1.0

        # Conf 候选点可视化参数
        self.visualize_conf = False
        self.conf_vis_iterations = []
        self.conf_vis_all_intervals = True
        self.conf_vis_point_radius = 3
        self.conf_vis_max_points = 1000
        self.conf_vis_mask_type = 'both'

        # 统计日志
        self.candidate_stats_enabled = True

        # Profiling flag
        self.profile_components = False

        # 门控冷却 iter 数
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

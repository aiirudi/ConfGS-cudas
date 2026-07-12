"""Conf 候选点可视化端到端验证脚本。

在 Docker rfgs 容器中运行, 验证可视化功能是否满足验收标准。

用法:
    # 基线验证: 关闭可视化, 确认零开销
    python scripts/validate_conf_vis.py --check-artifacts <model_path> --expect-disabled

    # 开启可视化验证: 检查文件完整性
    python scripts/validate_conf_vis.py --check-artifacts <model_path> --expect-mask-types conf
    python scripts/validate_conf_vis.py --check-artifacts <model_path> --expect-mask-types conf final_candidates

    # 投影对齐验证
    python scripts/validate_conf_vis.py --check-projection <model_path> -s <source_path> --iteration 7000
"""

import argparse
import json
import os
import sys


# AC-11 要求的完整 metadata 字段列表
REQUIRED_METADATA_FIELDS = [
    'iteration', 'camera_uid', 'camera_name', 'mask_type',
    'strategy', 'conf_thr', 'conf_min_views',
    'num_gaussians_before', 'mask_count', 'visible_count', 'drawn_count',
    'topk_score_name', 'same_render', 'trigger_reason',
]

# mask_type → 期望的 topk_score_name 映射
SCORE_NAME_MAP = {
    'conf': 'conf_score',
    'final_candidates': 'selection_score',
    'abs': 'abs_score',
}


def check_artifacts(model_path, expect_disabled, expect_mask_types):
    """验证可视化输出文件的完整性。

    Args:
        model_path: 模型输出目录
        expect_disabled: True = 期望可视化已关闭 (AC-1 验证)
        expect_mask_types: 期望的 mask 类型列表, 如 ['conf'] 或 ['conf', 'final_candidates']
    """
    vis_dir = os.path.join(model_path, "conf_interval_visualization")

    if not expect_disabled and not expect_mask_types:
        print("ERROR: --check-artifacts 需要 --expect-disabled 或 --expect-mask-types")
        return False

    if expect_disabled:
        if os.path.isdir(vis_dir):
            print(f"FAIL: {vis_dir} 存在, 但期望关闭可视化 (AC-1)")
            return False
        print(f"PASS: {vis_dir} 不存在 (默认关闭时零开销)")
        return True

    # 期望开启可视化
    if not os.path.isdir(vis_dir):
        print(f"FAIL: {vis_dir} 不存在, 但期望开启可视化")
        return False

    all_ok = True

    # 收集文件
    files = sorted(os.listdir(vis_dir))
    render_files = {f for f in files if f.endswith('_render.png')}
    overlay_files = {f for f in files if f.endswith('_conf.png') or f.endswith('_final_candidates.png') or f.endswith('_abs.png')}
    indices_files = {f for f in files if f.endswith('_indices.pt')}
    metadata_path = os.path.join(vis_dir, "metadata.jsonl")

    print(f"可视化目录: {vis_dir}")
    print(f"  文件总数: {len(files)}")
    print(f"  render: {sorted(render_files)}")
    print(f"  overlay: {sorted(overlay_files)}")
    print(f"  indices: {sorted(indices_files)}")

    # ---- metadata.jsonl (AC-11) ----
    metadata_rows = []
    seen_prefixes = set()
    if not os.path.exists(metadata_path):
        print("FAIL: metadata.jsonl 缺失 (AC-11)")
        all_ok = False
    else:
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
        print(f"  metadata 条目: {len(lines)}")

        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"FAIL: metadata 行 {i} 不是有效 JSON (AC-11)")
                all_ok = False
                continue

            missing = [k for k in REQUIRED_METADATA_FIELDS if k not in entry]
            if missing:
                print(f"FAIL: metadata 行 {i} 缺少字段: {missing} (AC-11)")
                all_ok = False

            mt = entry.get('mask_type', '')
            expected = SCORE_NAME_MAP.get(mt, '')
            actual = entry.get('topk_score_name', '')
            if expected and actual != expected:
                print(f"FAIL: metadata 行 {i} topk_score_name={actual}, 期望={expected} (AC-11)")
                all_ok = False

            if not entry.get('trigger_reason', ''):
                print(f"WARN: metadata 行 {i} trigger_reason 为空")

            metadata_rows.append(entry)

    # ---- enabled 模式守卫: 期望开启时必须有输出 ----
    if expect_mask_types:
        if not metadata_rows:
            print("FAIL: metadata.jsonl 为空, 但期望开启可视化 (AC-11)")
            all_ok = False
        if not render_files:
            print("FAIL: 无 render 文件, 但期望开启可视化 (AC-7)")
            all_ok = False
        if not overlay_files:
            print("FAIL: 无 overlay 文件, 但期望开启可视化 (AC-7)")
            all_ok = False

    # ---- 按 trigger prefix 分组验证 (AC-7) ----
    # 每个 prefix 必须有一张 render + 所有期望的 mask_type overlay/indices
    prefix_map = {}  # prefix -> set of mask_types
    for entry in metadata_rows:
        it = entry.get('iteration', 0)
        uid = entry.get('camera_uid', '')
        mt = entry.get('mask_type', '')
        prefix = f"iteration_{it:06d}_view{uid}"
        seen_prefixes.add(prefix)
        prefix_map.setdefault(prefix, set()).add(mt)

        render_name = f"{prefix}_render.png"
        if render_name not in render_files:
            print(f"FAIL: metadata 引用 {render_name} 但文件不存在")
            all_ok = False

        overlay_name = f"{prefix}_{mt}.png"
        if overlay_name not in overlay_files:
            print(f"FAIL: metadata 引用 {overlay_name} 但文件不存在")
            all_ok = False

        idx_name = f"{prefix}_{mt}_indices.pt"
        if idx_name not in indices_files:
            print(f"FAIL: metadata 引用 {idx_name} 但文件不存在")
            all_ok = False

    # 检查每个 prefix 是否有所有期望的 mask_type
    for prefix, found in sorted(prefix_map.items()):
        missing_types = set(expect_mask_types) - found
        if missing_types:
            print(f"FAIL: {prefix} 缺少期望的 mask 类型: {missing_types} (AC-7)")
            all_ok = False

    # 检查是否有 metadata 未覆盖的孤儿 overlay 文件
    for f in overlay_files:
        matched = False
        for prefix in prefix_map:
            if f.startswith(prefix):
                matched = True
                break
        if not matched:
            print(f"WARN: 孤儿 overlay 文件 (无对应 metadata): {f}")

    # ---- AC-7: each prefix exactly one render ----
    render_count = sum(1 for rf in render_files if any(rf.startswith(p + '_render') for p in prefix_map))
    if render_count != len(prefix_map):
        print(f"FAIL: render 文件数 ({render_count}) != trigger 数 ({len(prefix_map)}) (AC-7)")
        all_ok = False

    # ---- indices.pt 验证 (AC-11) ----
    import torch
    for idx_file in indices_files:
        data = torch.load(os.path.join(vis_dir, idx_file), map_location='cpu')
        required_idx = ['iteration', 'camera_uid', 'mask_type',
                        'selected_indices', 'drawn_indices']
        missing = [k for k in required_idx if k not in data]
        if missing:
            print(f"FAIL: {idx_file} 缺少字段: {missing} (AC-11)")
            all_ok = False

    if all_ok:
        print("\nPASS: 所有 artifact 检查通过")
    else:
        print("\nFAIL: 存在未满足的验收标准")
    return all_ok


def check_projection(model_path, source_path, iteration):
    """投影对齐验证: Python vs CUDA rasterizer gaussian_centers (AC-5).

    使用 render.py 的 Scene 加载模式, 从 cfg_args 读取完整数据集参数。
    """
    import torch
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gaussian_renderer import render, GaussianModel
    from scene import Scene
    from arguments import PipelineParams
    from argparse import ArgumentParser, Namespace
    from utils.conf_visualization import project_gaussian_centers, filter_visible_points

    # 从 cfg_args 读取数据集参数, 构造与 render.py 一致的 args
    import types
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"FAIL: cfg_args 不存在: {cfg_path}")
        return False
    with open(cfg_path) as f:
        cfg_str = f.read()
    args = eval(cfg_str, {'Namespace': Namespace})
    args.iteration = iteration
    args.source_path = source_path or args.source_path

    pp_parser = ArgumentParser()
    pp = PipelineParams(pp_parser).extract(pp_parser.parse_args([]))

    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree, optimizer_type="default")

        # 检查对应 iteration 的 checkpoint 是否存在
        ckpt_path = os.path.join(model_path, "point_cloud",
                                 f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(ckpt_path):
            print(f"FAIL: checkpoint 不存在: {ckpt_path}")
            print(f"  训练默认只保存最终 iteration 和 --save_iterations 中指定的 iteration。")
            print(f"  请将 iteration {args.iteration} 加入 --save_iterations 后重新训练，或换个存在的 iteration。")
            return False

        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # 读取可视化 metadata 获取使用的相机
        vis_dir = os.path.join(model_path, "conf_interval_visualization")
        metadata_path = os.path.join(vis_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            print("FAIL: metadata.jsonl 不存在, 无法确定可视化相机")
            return False

        with open(metadata_path, 'r') as f:
            all_meta = [json.loads(l) for l in f.readlines()]

        # 找所有匹配 iteration 的 metadata 行, 逐一尝试投影对齐
        candidates = [m for m in all_meta if m.get('iteration') == args.iteration]
        if not candidates:
            print(f"FAIL: metadata 中未找到 iteration={args.iteration}")
            return False

        # 按 drawn_count 降序排列, 优先尝试有更多绘制点的行
        candidates.sort(key=lambda m: m.get('drawn_count', 0), reverse=True)

        tested = 0
        for target_meta in candidates:
            if target_meta.get('drawn_count', 0) == 0:
                continue
            tested += 1

            camera_uid = target_meta['camera_uid']
            mask_type = target_meta['mask_type']

            test_cam = None
            for cam in scene.getTrainCameras():
                if str(cam.uid) == str(camera_uid):
                    test_cam = cam
                    break
            if test_cam is None:
                print(f"WARN: 未找到 uid={camera_uid} 的相机, 尝试下一行")
                continue

            idx_file = os.path.join(
                vis_dir,
                f"iteration_{args.iteration:06d}_view{camera_uid}_{mask_type}_indices.pt"
            )
            if not os.path.exists(idx_file):
                print(f"WARN: indices 文件不存在: {idx_file}, 尝试下一行")
                continue

            indices_data = torch.load(idx_file, map_location='cpu')
            drawn_indices = indices_data.get('drawn_indices', [])
            if len(drawn_indices) == 0:
                continue

            drawn_indices = torch.tensor(drawn_indices, dtype=torch.long, device="cuda")

            # 渲染获取 rasterizer gaussian_centers
            render_pkg = render(test_cam, gaussians, pp, bg)
            raster_centers = render_pkg.get("gaussian_centers", None)
            if raster_centers is None:
                print("WARN: rasterizer 未返回 gaussian_centers, 尝试下一行")
                continue

            # Python 投影
            xyz = gaussians.get_xyz.detach()
            python_pix, view_depth = project_gaussian_centers(
                xyz, test_cam.full_proj_transform,
                test_cam.world_view_transform,
                test_cam.image_width, test_cam.image_height,
            )

            py_for_drawn = python_pix[drawn_indices]
            raster_for_drawn = raster_centers[drawn_indices]

            valid_r = (raster_for_drawn[:, 0] > 0) & (raster_for_drawn[:, 1] > 0)
            valid_r &= (raster_for_drawn[:, 0] < test_cam.image_width)
            valid_r &= (raster_for_drawn[:, 1] < test_cam.image_height)
            valid_r &= torch.isfinite(py_for_drawn).all(dim=-1)

            if valid_r.sum() < 10:
                print(f"WARN: camera={camera_uid} mask={mask_type} 仅 {valid_r.sum().item()} 匹配点, 尝试下一行")
                continue

            py_valid = py_for_drawn[valid_r]
            raster_valid = raster_for_drawn[valid_r]
            errors = torch.norm(py_valid - raster_valid, dim=-1)

            median_err = torch.median(errors).item()
            p95_err = torch.quantile(errors, 0.95).item()
            max_err = torch.max(errors).item()

            print(f"\n投影对齐: camera={camera_uid} mask={mask_type} "
                  f"N={valid_r.sum().item()}")
            print(f"  median: {median_err:.3f} px")
            print(f"  p95:    {p95_err:.3f} px")
            print(f"  max:    {max_err:.3f} px")

            if median_err <= 0.5 and p95_err <= 1.0:
                print("PASS: 投影对齐满足精度要求 (AC-5)")
                return True
            else:
                print(f"WARN: 此行误差超阈值, 尝试下一候选行")
                continue

        if tested == 0:
            print(f"FAIL: {len(candidates)} 行 metadata 中无一行有 drawn_count>0 (task-8)")
        else:
            print(f"FAIL: 测试了 {tested} 行, 无一满足投影精度要求 (AC-5)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Conf 可视化验证脚本")
    parser.add_argument("--check-artifacts", type=str, default=None,
                        help="检查指定 model_path 的可视化输出文件")
    parser.add_argument("--expect-disabled", action="store_true",
                        help="期望可视化已关闭 (基线验证, AC-1)")
    parser.add_argument("--expect-mask-types", type=str, nargs='*', default=[],
                        help="期望的 mask 类型: conf final_candidates")
    parser.add_argument("--check-projection", type=str, default=None,
                        help="投影对齐验证 (需要 model_path, 需传 -s source_path)")
    parser.add_argument("-s", "--source_path", type=str, default=None,
                        help="数据源路径 (用于投影验证)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="指定 iteration (用于投影验证)")
    args = parser.parse_args()

    ok = True

    if args.check_artifacts:
        ok = check_artifacts(args.check_artifacts, args.expect_disabled,
                             args.expect_mask_types)

    if args.check_projection:
        if not args.iteration:
            print("ERROR: --check-projection 需要 --iteration")
            sys.exit(1)
        src = args.source_path
        if not src:
            # 尝试从 cfg_args 读取
            cfg_path = os.path.join(args.check_projection, "cfg_args")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg_str = f.read()
                import re
                m = re.search(r"source_path='([^']+)'", cfg_str)
                if m:
                    src = m.group(1)
            if not src:
                print("ERROR: --check-projection 需要 -s <source_path> 或有效的 cfg_args")
                sys.exit(1)
        ok = check_projection(args.check_projection, src, args.iteration) and ok

    if not args.check_artifacts and not args.check_projection:
        print("用法示例:")
        print("  # 基线验证")
        print("  python scripts/validate_conf_vis.py --check-artifacts <model_path> --expect-disabled")
        print("  # 开启可视化验证")
        print("  python scripts/validate_conf_vis.py --check-artifacts <model_path> --expect-mask-types conf")
        print("  # 投影对齐")
        print("  python scripts/validate_conf_vis.py --check-projection <model_path> -s <source_path> --iteration 7000")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

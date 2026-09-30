# Conf CUDA 二次检查与数值修复

检查基线：`d3fdef0`。修复分支：`fix/conf-cuda-numerics-20260930`。

## 方法核对

当前 Conf 的实际定义为：对高斯 i 最近 W 个不同且实际贡献像素的视角，
取屏幕位置的有符号梯度，经投影 Jacobian 转到世界坐标，得到 g_iv，然后计算

```
S_i = sum_v g_iv
M_i = sum_v ||g_iv||_2
Conf_i = clamp(1 - ||S_i||_2 / M_i, 0, 1)
```

至少两个视角且 M_i > 0 时才有有效候选。不同相机的梯度必须先进入同一世界
坐标系，不能直接比较各自的屏幕 x/y。CUDA 使用的位置梯度仅来自投影路径，
没有混入 covariance 或 SH 对最终 xyz 梯度的贡献，也没有重复乘图像尺寸。

这里的“绝对量”是各视角完整有符号梯度的 L2 范数。像素贡献已经在该视角内
求和；逐像素先取绝对值再求和的 AbsGS 统计与这个分母不同。因此，该实现
检测的是**跨视角的梯度抵消**，不能检测已在单个视角内部抵消的像素梯度。
本次保留已有方法定义和 `conf_thr=0.85` 的含义。

窗口按高斯独立更新；重复相机刷新已有条目，不增加视角数；不可见样本不
推进窗口；可见的零梯度是有效样本；新视角使最旧条目过期。幸存高斯在
致密化和剪枝后保留历史，新分裂出的高斯从空历史开始。

## 已复现的问题与修复

1. **float32 累计溢出。** 两个有限梯度 `(2e38,0,0)` 与
   `(-2e38,0,0)` 的真实 Conf 为 1。旧核的 M 溢出为 Inf，得到 Conf=0，
   Python 候选门控也拒绝这个高斯。同向的大梯度还会使 S 溢出。
   现在 S/M 的计算和存储均为 float64，分数仍为 float32。
2. **极小梯度的范数舍入。** float32 历史范数可能无法准确表示次正规数
   向量的长度，破坏尺度一致性。现在从历史 xyz 以 float64 重建范数，
   不把已经舍入的 w 通道作为分母。float32 的有限分量在 float64 中平方
   不会溢出或下溢；其最大窗口长度受 int32 限制，累计量也在 float64 范围内。
   样本/历史 w 仍作为有效性与近似范数标记；超出 float32 范围的范数饱和到
   FLT_MAX，xyz 不改变。这个标记的饱和不影响 Conf 分母。
3. **已提交的二进制不匹配源码。** 检查发现仓库原有 `_C.so` 连
   `accumulate_conf` 都没有。删除 rasterizer 包内已跟踪的旧 `.so` 和缓存
   `.pyc`，添加忽略规则和原生 `conf_api_version=3`，Python 导入时检查版本
   并提供重编译命令。旧文件另外备份到 `/tmp/conf_cuda_stale_artifacts_20260930`。
4. **部分入口可绕过 stream 限制。** 原 Python forward/backward 有检查，但
   `markVisible` 和直接调用原生接口仍可在非默认 stream 上生成/读取数据。
   现在 forward、backward、markVisible、Adam 的原生入口都检查默认 stream，
   并切换到参数所在 CUDA device；临时 rasterizer buffers 也分配在这个 device。
   Conf 累计核继续支持当前 stream。尚未把全部 rasterizer kernel/CUB 操作
   改造成支持任意 stream 的实现。
5. **checkpoint 缓存不校验内容。** 新 version-3 状态保存 double 累计量；
   有稳定相机映射的 version-2 checkpoint 从有序历史重建 S/M/Conf，保留
   相机顺序和下一次淘汰行为，能修复已有的 Inf 缓存。恢复时拒绝越界计数、
   重复/未知相机 ID、非法占用槽和非有限历史。旧版无映射状态仍显式清空历史。

此外，初始化了 CUB 查询临时大小时使用的空指针，消除读取未初始化 C++
指针值的编译警告。

## 验证环境与命令

- NVIDIA A100 80 GB；PyTorch 2.1.2；CUDA 11.8；Python 3.10。
- Docker 镜像：`pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`。
- 本次扩展：`/tmp/conf_cuda_numerics_build_20260930/lib/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so`。
- SHA-256：`7e6c9483380264dfdeeaf59ee7be9f828c83c0ecd5cd7a569b263a47920db25c`。
- 编译日志：`/tmp/conf_cuda_numerics_build.log`、`/tmp/conf_cuda_numerics_rebuild.log`。
- 验证日志目录：`/tmp/conf_cuda_numerics_validation_logs`。

重建扩展后，使用匹配的 Python wrapper。项目 runner 新增 build/checkpoint 模式：

```bash
scripts/run_conf_cuda_validation.sh build
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh kernel
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh checkpoint
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh raster
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh lifecycle
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh sanitizer
CONF_CUDA_LOG_ROOT=/tmp/conf_cuda_numerics_validation_logs scripts/run_conf_cuda_validation.sh garden-default
```

runner 的 baseline、simple-knn/fused-ssim 等临时依赖和 Garden 数据位置与
之前的 [验证记录](conf_cuda_validation.md) 相同，可通过环境变量覆盖。
新 checkout 必须自行安装依赖/编译 baseline，不能假定这些 `/tmp` 文件存在。

## 本次结果

| 检查 | 结果 |
| --- | --- |
| 新增数值回归测试在旧扩展上运行 | 复现 9 个失败子案例；同向/反向/正交及极小/极大梯度暴露舍入或溢出 |
| 新扩展 kernel 测试 | 4/4 通过；包含 24 个尺度/方向子案例，尺度覆盖 1e-45 至 3e38；W=2/3/5 每次更新与独立 float64 deque 对照 |
| 内存布局与 stream | P=0/P=1、dtype、shape、float4 对齐、缓冲区重叠、非默认 stream 累计、原生 markVisible 默认 stream 限制通过 |
| checkpoint/statistics | v3 roundtrip、v2 Inf 缓存修复、相机顺序保留、非法状态拒绝、旧版迁移通过 |
| 原 rasterizer 对照 | 四个 SH/color 与 covariance/scale 分支图像一致；实际参数梯度最大绝对差 1.90735e-6，在既有 float32 reduction 容差内；投影 VJP 与独立解析/有限差分对照通过 |
| 实际贡献标记 | 可见零梯度、完全遮挡但正 radius、全裁剪 B=0、部分 tiles、debug 模式通过 |
| 拓扑生命周期 | 纯剪枝、append、实际 LAS 分裂、零预算/无分裂边界、过期拓扑 generation、序列化恢复后下一次淘汰通过 |
| RFAS/EAS/fusion | helper AST 与检查基线一致；既有 selector 18/18、spatial 21/21、门控/策略兼容 3/3 通过 |
| CUDA compute-sanitizer | 新 rasterizer 场景 memcheck：0 errors |
| 旧二进制加载 | 实际加载备份旧扩展，正确抛出版本不匹配和重编译提示 |
| Garden 短训练 | 128 分辨率、默认 Conf 阈值 0.85，完成 20 次迭代；第 5/10/15/20 次迭代分别有 254/359/373/400 个候选，每次分裂 100 个父高斯；最终数量 138766→139166；第 10/20 次记录的 EMA loss 为 0.18410/0.15216，均有限 |

## 代价与范围

resident Conf 状态变为 `(24*W+40)` bytes/高斯；W=3 时为 112 bytes，比之前
增加 16 bytes/高斯。临时样本仍是 16 bytes/高斯。每次窗口更新包含 float64
运算，未测量性能变化，不声称加速。

测试覆盖正确性和短训练集成，不证明最终 PSNR 或长时间训练的收益。
历史样本来自不同优化时刻，仍可能包含模型随训练变化的影响。CLI 当前
没有 checkpoint 保存/恢复参数；恢复验证针对 GaussianModel 的 checkpoint
API 与序列化状态，未新增训练 CLI 功能。多 GPU device guard 经源码检查，
本机只有一个可用 GPU，未进行双 GPU 运行验证。

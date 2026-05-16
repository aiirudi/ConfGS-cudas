# 3DGS 中的幅度反重建损失设计

## 1. 背景与动机

在 3D Gaussian Splatting（3DGS）中，常用的监督信号主要来自像素域，例如 RGB 重建损失和 SSIM 损失。这类损失能够直接约束渲染图像与真实图像之间的颜色和结构差异，但对于频率分布、纹理强度以及高频细节的约束并不显式。

为了进一步约束渲染图像的频率能量分布，可以引入一种基于傅里叶幅度谱的辅助损失。具体来说，可以分别对 3DGS 渲染图像和真实图像进行傅里叶变换，提取其幅度谱，然后基于幅度谱进行反变换，得到幅度诱导的空间响应，并在空间域中计算二者之间的差异。

该损失可以称为：

**幅度反重建损失（Amplitude-only Reconstruction Loss）**。

其核心思想是：

> 幅度谱主要反映图像中不同频率成分的能量强度。通过比较渲染图像和真实图像的幅度反重建结果，可以促使渲染图像在频率能量分布上更加接近真实图像，从而增强纹理、边缘和高频细节的表达。

---

## 2. 幅度与相位的基本概念

给定一幅图像 $I$，其二维傅里叶变换为：

$$
F(u,v)=\mathcal{F}(I)(u,v)
$$

由于傅里叶变换结果是复数，可以表示为：

$$
F(u,v)=A(u,v)e^{j\phi(u,v)}
$$

其中：

$$
A(u,v)=|F(u,v)|
$$

表示幅度谱，反映频率成分的强弱；

$$
\phi(u,v)=\angle F(u,v)
$$

表示相位谱，反映频率成分的空间对齐关系。

在图像重建中，相位通常对结构位置和轮廓信息更加重要，而幅度主要反映整体频率能量分布。因此，仅使用幅度信息并不能完整恢复原图结构，但可以用于约束图像的频率强度、纹理分布和清晰度倾向。

---

## 3. 损失定义

设 3DGS 渲染图像为：

$$
\hat{I}
$$

真实图像为：

$$
I
$$

首先对二者进行二维傅里叶变换：

$$
\hat{F}=\mathcal{F}(\hat{I}), \quad F=\mathcal{F}(I)
$$

提取幅度谱：

$$
\hat{A}=|\hat{F}|, \quad A=|F|
$$

然后对幅度谱进行反傅里叶变换：

$$
\hat{I}_{A}=\mathcal{F}^{-1}(\hat{A})
$$

$$
I_{A}=\mathcal{F}^{-1}(A)
$$

最后，在空间域计算幅度反重建结果之间的差异：

$$
\mathcal{L}_{amp\_rec}
=
\left\|
\operatorname{Re}(\hat{I}_{A})-
\operatorname{Re}(I_{A})
\right\|_{1}
$$

其中 $\operatorname{Re}(\cdot)$ 表示取复数结果的实部。

---

## 4. 与直接幅度谱损失的关系

该损失和直接幅度谱损失具有一定相似性。

直接幅度谱损失可以写为：

$$
\mathcal{L}_{amp}
=
\left\|
|\mathcal{F}(\hat{I})|-|\mathcal{F}(I)|
\right\|_{1}
$$

而幅度反重建损失为：

$$
\mathcal{L}_{amp\_rec}
=
\left\|
\mathcal{F}^{-1}(|\mathcal{F}(\hat{I})|)
-
\mathcal{F}^{-1}(|\mathcal{F}(I)|)
\right\|_{1}
$$

如果使用 $L_2$ 损失，根据 Parseval 定理，频域差异和空间域差异在能量意义上是近似等价的，只差归一化常数。因此，如果仅使用 MSE，该损失与直接幅度谱 MSE 的区别并不明显。

但是，如果使用 $L_1$、Charbonnier loss，或者在反重建结果上加入频率 mask、局部 patch 约束或退火策略，则幅度反重建损失可以和普通幅度谱损失形成一定区别。

---

## 5. 适合 3DGS 的改进形式

为了提高稳定性，可以对幅度谱使用对数压缩：

$$
A_{log}=\log(1+|\mathcal{F}(I)|)
$$

对应的损失可以写为：

$$
\mathcal{L}_{amp\_rec}
=
\left\|
\mathcal{F}^{-1}\left(\log(1+|\mathcal{F}(\hat{I})|)\right)
-
\mathcal{F}^{-1}\left(\log(1+|\mathcal{F}(I)|)\right)
\right\|_{1}
$$

进一步地，可以引入频率 mask $M$：

$$
\mathcal{L}_{amp\_rec}
=
\left\|
\mathcal{F}^{-1}\left(M \odot \log(1+|\mathcal{F}(\hat{I})|)\right)
-
\mathcal{F}^{-1}\left(M \odot \log(1+|\mathcal{F}(I)|)\right)
\right\|_{1}
$$

其中 $M$ 可以设计为低频 mask、高频 mask，或者从低频逐渐扩展到高频的退火 mask。

这种设计更适合 3DGS 的优化过程，因为 3DGS 训练早期几何和颜色都不稳定，过早强调高频可能会引入噪声或伪影。因此，可以采用由低频到高频的 progressive frequency regularization 策略。

---

## 6. 总体损失函数

该损失不建议单独使用，而应作为 RGB 和 SSIM 损失之外的辅助项。整体训练目标可以写为：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{dssim}\mathcal{L}_{dssim}
+
\lambda_{amp}\mathcal{L}_{amp\_rec}
$$

其中：

- $\mathcal{L}_{rgb}$ 是原始 RGB 重建损失；
- $\mathcal{L}_{dssim}$ 是结构相似性损失；
- $\mathcal{L}_{amp\_rec}$ 是幅度反重建损失；
- $\lambda_{amp}$ 是该损失的权重。

实验中可以先尝试：

$$
\lambda_{amp}=0.01 \sim 0.1
$$

如果出现过度锐化、噪声增强或伪影，可以进一步减小该权重，或者只在训练中后期启用该损失。

---

## 7. PyTorch 实现示例

```python
import torch
import torch.nn.functional as F


def amplitude_reconstruction_loss(pred, gt, use_log=True):
    """
    pred, gt: [B, C, H, W], range [0, 1]
    return: amplitude-only reconstruction loss
    """
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm='ortho')
    gt_fft = torch.fft.fft2(gt, dim=(-2, -1), norm='ortho')

    pred_amp = torch.abs(pred_fft)
    gt_amp = torch.abs(gt_fft)

    if use_log:
        pred_amp = torch.log1p(pred_amp)
        gt_amp = torch.log1p(gt_amp)

    pred_amp_rec = torch.fft.ifft2(pred_amp, dim=(-2, -1), norm='ortho').real
    gt_amp_rec = torch.fft.ifft2(gt_amp, dim=(-2, -1), norm='ortho').real

    loss = F.l1_loss(pred_amp_rec, gt_amp_rec)
    return loss
```

---

## 8. 带频率 Mask 的实现示例

```python
import torch
import torch.nn.functional as F


def build_frequency_mask(h, w, ratio, device):
    """
    构造中心低频 mask。
    ratio: 保留的低频半径比例，范围为 0 到 1。
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    cy, cx = h // 2, w // 2
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, device=device, dtype=torch.float32))

    mask = (dist <= ratio * max_dist).float()
    return mask


def masked_amplitude_reconstruction_loss(pred, gt, ratio=0.5, use_log=True):
    """
    pred, gt: [B, C, H, W]
    ratio: 低频保留比例
    """
    b, c, h, w = pred.shape

    pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm='ortho')
    gt_fft = torch.fft.fft2(gt, dim=(-2, -1), norm='ortho')

    pred_amp = torch.abs(pred_fft)
    gt_amp = torch.abs(gt_fft)

    if use_log:
        pred_amp = torch.log1p(pred_amp)
        gt_amp = torch.log1p(gt_amp)

    pred_amp = torch.fft.fftshift(pred_amp, dim=(-2, -1))
    gt_amp = torch.fft.fftshift(gt_amp, dim=(-2, -1))

    mask = build_frequency_mask(h, w, ratio, pred.device)
    mask = mask.view(1, 1, h, w)

    pred_amp = pred_amp * mask
    gt_amp = gt_amp * mask

    pred_amp = torch.fft.ifftshift(pred_amp, dim=(-2, -1))
    gt_amp = torch.fft.ifftshift(gt_amp, dim=(-2, -1))

    pred_amp_rec = torch.fft.ifft2(pred_amp, dim=(-2, -1), norm='ortho').real
    gt_amp_rec = torch.fft.ifft2(gt_amp, dim=(-2, -1), norm='ortho').real

    loss = F.l1_loss(pred_amp_rec, gt_amp_rec)
    return loss
```

---

## 9. 训练策略建议

### 9.1 不建议训练早期直接使用强高频约束

3DGS 训练早期的几何结构和高斯分布尚不稳定，如果过早加入高频约束，可能会放大噪声或错误纹理。因此，建议采用延迟启用或逐步增强策略。

例如：

$$
\lambda_{amp}(t)=
\lambda_{max}\cdot \min\left(1, \frac{t-t_0}{T}\right)
$$

其中：

- $t$ 表示当前训练迭代；
- $t_0$ 表示开始启用该损失的迭代数；
- $T$ 表示退火长度；
- $\lambda_{max}$ 表示最大损失权重。

### 9.2 可以从低频逐渐扩展到高频

频率 mask 的半径可以随训练迭代逐渐增大：

$$
r(t)=r_{min}+(r_{max}-r_{min})\cdot \min\left(1, \frac{t-t_0}{T}\right)
$$

这样可以让模型先学习整体结构和低频颜色，再逐渐约束纹理和高频细节。

---

## 10. 优点与风险

### 优点

1. 可以显式约束渲染图像和真实图像之间的频率能量分布。
2. 有助于缓解渲染图像偏模糊的问题。
3. 可以增强纹理、边缘和细节表达。
4. 对轻微空间错位不如像素损失敏感。
5. 可以作为 3DGS 中频域正则化的一种辅助设计。

### 风险

1. 幅度谱不能准确约束结构位置。
2. 如果权重过大，可能导致噪声增强或过度锐化。
3. 如果训练早期强调高频，可能引入错误纹理。
4. 如果只使用 $L_2$，该损失与直接幅度谱 MSE 的区别较弱。
5. 需要和 RGB、SSIM 或相位相关约束配合使用。

---

## 11. 论文表述示例

可以在论文中写成：

> To further regularize the frequency-energy distribution of rendered images, we introduce an amplitude-only reconstruction loss. Specifically, we first transform both the rendered image and the ground-truth image into the Fourier domain and extract their magnitude spectra. The magnitude spectra are then transformed back into the spatial domain to obtain amplitude-induced spatial responses. By minimizing the discrepancy between these responses, the proposed loss encourages the rendered image to match the ground-truth image in terms of frequency energy distribution, thereby improving texture fidelity and high-frequency detail reconstruction.

对应中文含义为：

> 为了进一步约束渲染图像的频率能量分布，我们引入幅度反重建损失。具体而言，我们首先将渲染图像和真实图像变换到傅里叶域，并提取其幅度谱。随后，将幅度谱反变换回空间域，得到幅度诱导的空间响应。通过最小化二者之间的差异，该损失促使渲染图像在频率能量分布上更加接近真实图像，从而提升纹理保真度和高频细节重建质量。

---

## 12. 总结

在 3DGS 中使用渲染图像和真实图像的幅度反重建结果来计算损失是可行的。该损失可以作为一种频域辅助正则项，用于约束图像的频率能量分布和纹理强度。

不过，该方法不应替代原始 RGB 或 SSIM 损失，因为幅度信息不能充分保留空间结构和位置关系。更合理的使用方式是将其作为辅助项，与像素域损失、结构损失以及可能的相位或梯度约束共同使用。

推荐最终损失形式为：

$$
\mathcal{L}
=
\mathcal{L}_{rgb}
+
\lambda_{dssim}\mathcal{L}_{dssim}
+
\lambda_{amp}\mathcal{L}_{amp\_rec}
$$

其中 $\lambda_{amp}$ 建议从较小值开始尝试，例如 $0.01$ 到 $0.1$。

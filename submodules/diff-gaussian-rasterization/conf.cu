#include "rasterize_points.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void accumulate_conf_kernel(
    int n, const float4* samples, float3* world_sum, float* norm_sum,
    int32_t* view_count, float* conf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 g = samples[i];
  const float sample_magnitude = hypotf(hypotf(g.x, g.y), g.z);
  float3 sum = world_sum[i];
  float magnitude_sum = norm_sum[i];
  int32_t count = view_count[i];
  if (isfinite(g.x) && isfinite(g.y) && isfinite(g.z) &&
      isfinite(g.w) && g.w > 0.0f &&
      isfinite(sample_magnitude) && sample_magnitude > 0.0f &&
      count < INT32_MAX) {
    sum.x += g.x;
    sum.y += g.y;
    sum.z += g.z;
    magnitude_sum += sample_magnitude;
    ++count;
    world_sum[i] = sum;
    norm_sum[i] = magnitude_sum;
    view_count[i] = count;
  }

  float score = 0.0f;
  if (count >= 2 && isfinite(sum.x) && isfinite(sum.y) &&
      isfinite(sum.z) && isfinite(magnitude_sum) && magnitude_sum > 0.0f) {
    const float sum_magnitude = hypotf(hypotf(sum.x, sum.y), sum.z);
    if (isfinite(sum_magnitude)) {
      score = 1.0f - sum_magnitude / magnitude_sum;
      score = fminf(1.0f, fmaxf(0.0f, score));
    }
  }
  conf_out[i] = score;
}

void check_tensor(const torch::Tensor& tensor, const char* name,
                  torch::ScalarType dtype, int64_t n, int64_t width,
                  const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an incorrect dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == n && tensor.size(1) == width,
              name, " must have shape (", n, ",", width, ")");
}

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.numel() == 0 || b.numel() == 0) return;
  const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data_ptr());
  const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data_ptr());
  const auto a_end = a_begin + a.numel() * a.element_size();
  const auto b_end = b_begin + b.numel() * b.element_size();
  TORCH_CHECK(a_end <= b_begin || b_end <= a_begin,
              "accumulate_conf tensors must not overlap in storage");
}

} // namespace

torch::Tensor AccumulateConfCUDA(
    const torch::Tensor& samples, torch::Tensor& world_sum,
    torch::Tensor& norm_sum, torch::Tensor& view_count,
    torch::Tensor& conf_out) {
  TORCH_CHECK(samples.is_cuda(), "samples must be a CUDA tensor");
  TORCH_CHECK(samples.dim() == 2 && samples.size(1) == 4,
              "samples must have shape (N,4)");
  const int64_t n = samples.size(0);
  TORCH_CHECK(n <= INT32_MAX, "too many Gaussians for CUDA Conf kernel");
  const auto device = samples.device();
  check_tensor(samples, "samples", torch::kFloat32, n, 4, device);
  check_tensor(world_sum, "world_sum", torch::kFloat32, n, 3, device);
  check_tensor(norm_sum, "norm_sum", torch::kFloat32, n, 1, device);
  check_tensor(view_count, "view_count", torch::kInt32, n, 1, device);
  check_tensor(conf_out, "conf_out", torch::kFloat32, n, 1, device);
  const torch::Tensor tensors[] = {samples, world_sum, norm_sum, view_count, conf_out};
  for (int i = 0; i < 5; ++i)
    for (int j = i + 1; j < 5; ++j)
      check_no_overlap(tensors[i], tensors[j]);

  if (n == 0) return conf_out;
  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  accumulate_conf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<int>(n), reinterpret_cast<const float4*>(samples.data_ptr<float>()),
      reinterpret_cast<float3*>(world_sum.data_ptr<float>()),
      norm_sum.data_ptr<float>(), view_count.data_ptr<int32_t>(),
      conf_out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return conf_out;
}

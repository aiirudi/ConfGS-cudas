#include "rasterize_points.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void accumulate_conf_kernel(
    int n, int window, int64_t camera_id, const float4* samples,
    float4* history, int64_t* camera_ids, int32_t* view_count,
    float3* world_sum, float* norm_sum, float* conf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 sample = samples[i];
  if (!(sample.w >= 0.0f) || !isfinite(sample.x) ||
      !isfinite(sample.y) || !isfinite(sample.z) || !isfinite(sample.w))
    return;
  const float magnitude = hypotf(hypotf(sample.x, sample.y), sample.z);
  if (!isfinite(magnitude)) return;

  float4* row = history + static_cast<int64_t>(i) * window;
  int64_t* ids = camera_ids + static_cast<int64_t>(i) * window;
  int32_t count = view_count[i];
  if (count < 0 || count > window) {
    conf_out[i] = 0.0f;
    return;
  }

  int existing = -1;
  for (int j = 0; j < count; ++j) {
    if (ids[j] == camera_id) {
      existing = j;
      break;
    }
  }

  if (existing >= 0) {
    // Refresh a repeated camera: remove its old contribution, then append
    // the current observation as the newest entry without increasing count.
    for (int j = existing; j + 1 < count; ++j) {
      row[j] = row[j + 1];
      ids[j] = ids[j + 1];
    }
  } else if (count == window) {
    // The oldest view leaves this Gaussian's window. Its contribution is
    // removed by recomputing the sufficient statistics from active slots.
    for (int j = 0; j + 1 < window; ++j) {
      row[j] = row[j + 1];
      ids[j] = ids[j + 1];
    }
  } else {
    ++count;
  }

  row[count - 1] = make_float4(sample.x, sample.y, sample.z, magnitude);
  ids[count - 1] = camera_id;
  view_count[i] = count;

  float3 sum = make_float3(0.0f, 0.0f, 0.0f);
  float total_norm = 0.0f;
  for (int j = 0; j < count; ++j) {
    const float4 g = row[j];
    sum.x += g.x;
    sum.y += g.y;
    sum.z += g.z;
    total_norm += g.w;
  }
  world_sum[i] = sum;
  norm_sum[i] = total_norm;

  float score = 0.0f;
  if (count >= 2 && isfinite(sum.x) && isfinite(sum.y) &&
      isfinite(sum.z) && isfinite(total_norm) && total_norm > 0.0f) {
    const float sum_magnitude = hypotf(hypotf(sum.x, sum.y), sum.z);
    if (isfinite(sum_magnitude)) {
      score = 1.0f - sum_magnitude / total_norm;
      score = fminf(1.0f, fmaxf(0.0f, score));
    }
  }
  conf_out[i] = score;
}

void check_tensor(const torch::Tensor& tensor, const char* name,
                  torch::ScalarType dtype, const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an incorrect dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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
    const torch::Tensor& samples, int64_t camera_id,
    torch::Tensor& history, torch::Tensor& camera_ids,
    torch::Tensor& view_count, torch::Tensor& world_sum,
    torch::Tensor& norm_sum, torch::Tensor& conf_out) {
  TORCH_CHECK(camera_id >= 0, "camera_id must be nonnegative");
  TORCH_CHECK(samples.is_cuda(), "samples must be a CUDA tensor");
  TORCH_CHECK(samples.dim() == 2 && samples.size(1) == 4,
              "samples must have shape (N,4)");
  const int64_t n = samples.size(0);
  TORCH_CHECK(n <= INT32_MAX, "too many Gaussians for CUDA Conf kernel");
  TORCH_CHECK(history.dim() == 3 && history.size(0) == n && history.size(2) == 4,
              "history must have shape (N,window,4)");
  const int64_t window = history.size(1);
  TORCH_CHECK(window >= 2 && window <= INT32_MAX, "window must be at least 2");
  const auto device = samples.device();
  check_tensor(samples, "samples", torch::kFloat32, device);
  check_tensor(history, "history", torch::kFloat32, device);
  check_tensor(camera_ids, "camera_ids", torch::kInt64, device);
  check_tensor(view_count, "view_count", torch::kInt32, device);
  check_tensor(world_sum, "world_sum", torch::kFloat32, device);
  check_tensor(norm_sum, "norm_sum", torch::kFloat32, device);
  check_tensor(conf_out, "conf_out", torch::kFloat32, device);
  TORCH_CHECK(camera_ids.dim() == 2 && camera_ids.size(0) == n && camera_ids.size(1) == window,
              "camera_ids must have shape (N,window)");
  TORCH_CHECK(view_count.dim() == 2 && view_count.size(0) == n && view_count.size(1) == 1,
              "view_count must have shape (N,1)");
  TORCH_CHECK(world_sum.dim() == 2 && world_sum.size(0) == n && world_sum.size(1) == 3,
              "world_sum must have shape (N,3)");
  TORCH_CHECK(norm_sum.dim() == 2 && norm_sum.size(0) == n && norm_sum.size(1) == 1,
              "norm_sum must have shape (N,1)");
  TORCH_CHECK(conf_out.dim() == 2 && conf_out.size(0) == n && conf_out.size(1) == 1,
              "conf_out must have shape (N,1)");

  const torch::Tensor tensors[] = {samples, history, camera_ids, view_count,
                                   world_sum, norm_sum, conf_out};
  for (int i = 0; i < 7; ++i)
    for (int j = i + 1; j < 7; ++j)
      check_no_overlap(tensors[i], tensors[j]);

  if (n == 0) return conf_out;
  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  accumulate_conf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<int>(n), static_cast<int>(window), camera_id,
      reinterpret_cast<const float4*>(samples.data_ptr<float>()),
      reinterpret_cast<float4*>(history.data_ptr<float>()),
      camera_ids.data_ptr<int64_t>(), view_count.data_ptr<int32_t>(),
      reinterpret_cast<float3*>(world_sum.data_ptr<float>()),
      norm_sum.data_ptr<float>(), conf_out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return conf_out;
}

// SPDX-License-Identifier: Apache-2.0
//
// torch::Tensor <-> surfel_raster core. Scratch comes from torch::empty on
// the current stream, so torch's caching allocator makes per-frame
// allocation nearly free.
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "core/surfel_raster.h"

namespace sr = surfel_raster;

namespace {

struct TorchScratch {
  torch::Device device;
  std::vector<torch::Tensor> held;
};

void* torch_alloc(void* ctx, size_t bytes) {
  auto* s = static_cast<TorchScratch*>(ctx);
  try {
    s->held.push_back(torch::empty({(int64_t)bytes},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(s->device)));
  } catch (const c10::Error&) {
    return nullptr;
  }
  return s->held.back().data_ptr();
}

void check(const torch::Tensor& t, const char* name, torch::Device dev) {
  TORCH_CHECK(t.device() == dev, name, " must be on ", dev);
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> rasterize(torch::Tensor means, torch::Tensor quats,
                                     torch::Tensor log_scales, torch::Tensor opacity_logits,
                                     torch::Tensor features_dc, torch::Tensor features_rest,
                                     int64_t sh_degree, std::vector<double> w2c,
                                     std::vector<double> intr, int64_t width, int64_t height,
                                     double near_z, std::vector<double> campos, bool lowpass,
                                     bool aux, bool exact_tiles, bool near_crossing,
                                     bool want_rgb8,
                                     std::vector<double> thresholds,
                                     std::vector<double> background) {
  TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
  auto dev = means.device();
  c10::cuda::CUDAGuard guard(dev);
  int64_t n = means.size(0);
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 3 && means.scalar_type() == torch::kFloat32,
              "means must be float32 [N,3]");
  TORCH_CHECK(log_scales.dim() == 2 && log_scales.size(1) == 2,
              "log_scales must be [N,2] (the surfel rasterizer takes exactly 2 per splat), got ",
              log_scales.sizes());
  TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be [N,4]");
  TORCH_CHECK(opacity_logits.numel() == n, "opacity_logits must have N elements");
  TORCH_CHECK(features_dc.numel() == 3 * n, "features_dc must have N*3 elements");
  TORCH_CHECK(features_rest.dim() == 3 && features_rest.size(0) == n && features_rest.size(2) == 3,
              "features_rest must be [N,K,3]");
  TORCH_CHECK(w2c.size() == 12 && intr.size() == 4 && campos.size() == 3 &&
                  thresholds.size() == 4 && background.size() == 3,
              "bad camera/option vector sizes");
  for (auto* t : {&means, &quats, &log_scales, &opacity_logits, &features_dc, &features_rest})
    check(*t, "input", dev);
  auto dt = quats.scalar_type();
  TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kFloat16, "attributes must be float32 or float16");
  for (auto* t : {&log_scales, &opacity_logits, &features_dc, &features_rest})
    TORCH_CHECK(t->scalar_type() == dt, "all attributes except means must share one dtype");

  sr::SurfelInputs in;
  in.n = (int)n;
  in.means = means.data_ptr<float>();
  in.quats = quats.data_ptr();
  in.log_scales = log_scales.data_ptr();
  in.opacity_logits = opacity_logits.data_ptr();
  in.features_dc = features_dc.data_ptr();
  in.features_rest = features_rest.numel() ? features_rest.data_ptr() : nullptr;
  in.rest_stride = (int)features_rest.size(1);
  in.attr_dtype = dt == torch::kFloat16 ? sr::DType::F16 : sr::DType::F32;
  in.sh_degree = (int)sh_degree;

  sr::SurfelCamera cam;
  for (int i = 0; i < 12; ++i) cam.w2c[i] = (float)w2c[i];
  cam.fx = (float)intr[0]; cam.fy = (float)intr[1]; cam.cx = (float)intr[2]; cam.cy = (float)intr[3];
  cam.width = (int)width;
  cam.height = (int)height;
  cam.near_z = (float)near_z;
  for (int i = 0; i < 3; ++i) cam.campos[i] = (float)campos[i];

  sr::SurfelOptions opt;
  opt.lowpass = lowpass;
  opt.aux = aux;
  opt.exact_tiles = exact_tiles;
  opt.near_crossing = near_crossing;
  opt.alpha_min = (float)thresholds[0];
  opt.alpha_max = (float)thresholds[1];
  opt.t_min = (float)thresholds[2];
  opt.max_sigma = (float)thresholds[3];
  for (int i = 0; i < 3; ++i) opt.background[i] = (float)background[i];

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto rgb = torch::empty({3, height, width}, f32);
  auto depth = torch::empty({height, width}, f32);
  auto alpha = torch::empty({height, width}, f32);
  torch::Tensor normal, median, rgb8;
  sr::SurfelOutputs out;
  out.rgb = rgb.data_ptr<float>();
  out.depth = depth.data_ptr<float>();
  out.alpha = alpha.data_ptr<float>();
  if (aux) {
    normal = torch::empty({3, height, width}, f32);
    median = torch::empty({height, width}, f32);
    out.normal = normal.data_ptr<float>();
    out.median_depth = median.data_ptr<float>();
  }
  if (want_rgb8) {
    rgb8 = torch::empty({height, width, 3}, f32.dtype(torch::kUInt8));
    out.rgb8 = rgb8.data_ptr<uint8_t>();
  }

  TorchScratch scratch{dev, {}};
  auto stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();
  long long pairs = sr::surfel_render(in, cam, opt, out, torch_alloc, &scratch, stream);
  if (pairs == sr::kErrCuda) {
    auto err = cudaGetLastError();
    TORCH_CHECK(false, "surfel_render: CUDA error: ", cudaGetErrorString(err));
  }
  TORCH_CHECK(pairs >= 0, "surfel_render failed: ", sr::surfel_error_string(pairs));
  auto pairs_t = torch::tensor({(int64_t)pairs}, torch::kInt64);
  return {rgb, depth, alpha, normal, median, rgb8, pairs_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize", &rasterize, "Render-only surfel rasterizer");
}

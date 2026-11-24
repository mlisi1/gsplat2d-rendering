// SPDX-License-Identifier: Apache-2.0
// Renders one opaque red disc facing the camera through the C++ API, with
// no torch, and checks the centre pixel and a corner. Exit code 0 = pass.
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <surfel_raster.h>
#include <scratch_arena.h>

namespace sr = surfel_raster;

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main() {
  const int W = 64, H = 48;
  // One splat 2 m in front of the camera, identity rotation, 0.3 m scales.
  std::vector<float> means{0.f, 0.f, 2.f}, quats{1.f, 0.f, 0.f, 0.f};
  std::vector<float> ls{std::log(0.3f), std::log(0.3f)}, logit{4.f};
  std::vector<float> rgb{1.f, 0.f, 0.f};
  sr::SurfelInputs in;
  in.n = 1;
  in.means = upload(means);
  in.quats = upload(quats);
  in.log_scales = upload(ls);
  in.opacity_logits = upload(logit);
  in.features_dc = upload(rgb);
  in.sh_degree = -1;

  sr::SurfelCamera cam{};
  float w2c[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 12; ++i) cam.w2c[i] = w2c[i];
  cam.fx = cam.fy = 60.f;
  cam.cx = W / 2.f;
  cam.cy = H / 2.f;
  cam.width = W;
  cam.height = H;
  cam.near_z = 0.01f;

  sr::SurfelOptions opt;
  sr::SurfelOutputs out;
  cudaMalloc(&out.rgb, 3 * W * H * sizeof(float));
  cudaMalloc(&out.depth, W * H * sizeof(float));
  cudaMalloc(&out.alpha, W * H * sizeof(float));

  sr::ScratchArena arena;
  long long pairs = 0;
  for (int frame = 0; frame < 3; ++frame) {  // exercises arena growth + reuse
    arena.reset();
    pairs = sr::surfel_render(in, cam, opt, out, sr::ScratchArena::alloc, &arena, 0);
    cudaStreamSynchronize(0);
    if (pairs < 0) {
      std::printf("surfel_render failed: %s\n", sr::surfel_error_string(pairs));
      return 1;
    }
  }
  std::vector<float> img(3 * W * H), depth(W * H);
  cudaMemcpy(img.data(), out.rgb, img.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(depth.data(), out.depth, depth.size() * sizeof(float), cudaMemcpyDeviceToHost);
  int c = (H / 2) * W + W / 2;
  float r = img[c], g = img[W * H + c], d = depth[c], corner = img[0];
  std::printf("pairs=%lld centre r=%.3f g=%.3f depth=%.3f corner r=%.3f arena=%zu B\n", pairs, r, g,
              d, corner, arena.capacity());
  bool ok = r > 0.9f && g < 1e-6f && std::fabs(d - 2.f) < 1e-3f && corner < 1e-6f;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}

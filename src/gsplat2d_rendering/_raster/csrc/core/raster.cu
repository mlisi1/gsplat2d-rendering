// SPDX-License-Identifier: Apache-2.0
//
// Front-to-back compositing, one 16x16 block per tile, one thread per
// pixel. Splat records are staged through shared memory in batches of 256.
#include "preprocess.cuh"

namespace surfel_raster {
namespace {

template <bool AUX, bool LOWPASS>
__global__ void __launch_bounds__(kBlock)
    raster_kernel(SplatRecords rec, const uint32_t* vals, const uint2* ranges, SurfelCamera cam,
                  SurfelOptions opt, SurfelOutputs out, int tiles_x) {
  __shared__ float4 s_rec[4 * kBlock];
  __shared__ float4 s_blob[LOWPASS ? kBlock : 1];

  int tid = threadIdx.y * kTile + threadIdx.x;
  int px = blockIdx.x * kTile + threadIdx.x, py = blockIdx.y * kTile + threadIdx.y;
  bool inside = px < cam.width && py < cam.height;
  float fx_px = px + 0.5f, fy_px = py + 0.5f;
  Vec3 d = v3((fx_px - cam.cx) / cam.fx, (fy_px - cam.cy) / cam.fy, 1.f);

  uint2 range = ranges[blockIdx.y * tiles_x + blockIdx.x];
  int todo = (int)(range.y - range.x);

  float T = 1.f, cr = 0.f, cg = 0.f, cb = 0.f, dz = 0.f;
  float nx = 0.f, ny = 0.f, nz = 0.f, zmed = 0.f;
  bool done = !inside;
  const float rho_max = opt.max_sigma > 0.f ? opt.max_sigma * opt.max_sigma : INFINITY;

  for (int base = 0; base < todo; base += kBlock) {
    if (__syncthreads_count(done) == kBlock) break;
    int k = base + tid;
    if (k < todo) {
      uint32_t s = vals[range.x + k];
      for (int r = 0; r < 4; ++r) s_rec[4 * tid + r] = rec.rec[4 * s + r];
      if (LOWPASS) s_blob[tid] = rec.blob[s];
    }
    __syncthreads();
    int batch = min(kBlock, todo - base);
    for (int j = 0; j < batch && !done; ++j) {
      float4 qu = s_rec[4 * j], qv = s_rec[4 * j + 1], qn = s_rec[4 * j + 2], qc = s_rec[4 * j + 3];
      float den = d.x * qn.x + d.y * qn.y + d.z * qn.z;
      float rho = INFINITY, tau = 0.f;
      if (fabsf(den) >= 1e-12f) {
        tau = qn.w / den;
        if (tau > cam.near_z) {
          float u = tau * (d.x * qu.x + d.y * qu.y + d.z * qu.z) - qu.w;
          float v = tau * (d.x * qv.x + d.y * qv.y + d.z * qv.z) - qv.w;
          rho = u * u + v * v;
        }
      }
      if (LOWPASS) {
        float4 b = s_blob[j];
        float ex = fx_px - b.x, ey = fy_px - b.y;
        float rho_px = 2.f * (ex * ex + ey * ey);
        if (rho_px < rho) {
          rho = rho_px;
          tau = b.z;
        }
      }
      if (!(rho <= rho_max)) continue;
      float alpha = fminf(opt.alpha_max, qc.w * expf(-0.5f * rho));
      if (alpha < opt.alpha_min) continue;
      float t_next = T * (1.f - alpha);
      if (t_next < opt.t_min) {
        done = true;
        continue;
      }
      float w = alpha * T;
      cr += w * qc.x;
      cg += w * qc.y;
      cb += w * qc.z;
      dz += w * tau;
      if (AUX) {
        nx += w * qn.x;
        ny += w * qn.y;
        nz += w * qn.z;
        if (T > 0.5f) zmed = tau;
      }
      T = t_next;
    }
  }

  if (!inside) return;
  int hw = cam.height * cam.width, pix = py * cam.width + px;
  float a = 1.f - T;
  float r = cr + T * opt.background[0], g = cg + T * opt.background[1],
        b = cb + T * opt.background[2];
  out.rgb[pix] = r;
  out.rgb[hw + pix] = g;
  out.rgb[2 * hw + pix] = b;
  out.alpha[pix] = a;
  out.depth[pix] = a > 0.f ? dz / a : 0.f;
  if (AUX) {
    out.normal[pix] = nx;
    out.normal[hw + pix] = ny;
    out.normal[2 * hw + pix] = nz;
    out.median_depth[pix] = zmed;
  }
  if (out.rgb8) {
    // Truncation, matching torch's .mul(255).byte() the Python path used.
    uint8_t* o8 = out.rgb8 + 3 * pix;
    o8[0] = (uint8_t)(fminf(fmaxf(r, 0.f), 1.f) * 255.f);
    o8[1] = (uint8_t)(fminf(fmaxf(g, 0.f), 1.f) * 255.f);
    o8[2] = (uint8_t)(fminf(fmaxf(b, 0.f), 1.f) * 255.f);
  }
}

template <bool AUX, bool LOWPASS>
void launch(const SplatRecords& rec, const uint32_t* vals, const uint2* ranges,
            const SurfelCamera& cam, const SurfelOptions& opt, const SurfelOutputs& out,
            int tiles_x, int tiles_y, cudaStream_t stream) {
  raster_kernel<AUX, LOWPASS><<<dim3(tiles_x, tiles_y), dim3(kTile, kTile), 0, stream>>>(
      rec, vals, ranges, cam, opt, out, tiles_x);
}

}  // namespace

cudaError_t launch_raster(const SplatRecords& rec, const uint32_t* vals, const uint2* ranges,
                          const SurfelCamera& cam, const SurfelOptions& opt,
                          const SurfelOutputs& out, int tiles_x, int tiles_y, cudaStream_t stream) {
  if (opt.aux) {
    if (opt.lowpass) launch<true, true>(rec, vals, ranges, cam, opt, out, tiles_x, tiles_y, stream);
    else launch<true, false>(rec, vals, ranges, cam, opt, out, tiles_x, tiles_y, stream);
  } else {
    if (opt.lowpass) launch<false, true>(rec, vals, ranges, cam, opt, out, tiles_x, tiles_y, stream);
    else launch<false, false>(rec, vals, ranges, cam, opt, out, tiles_x, tiles_y, stream);
  }
  return cudaGetLastError();
}

}  // namespace surfel_raster

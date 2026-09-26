// CUDA twin of block_thomas_cpu.cc (VULCAN_JAX_SOLVER=ffi on a GPU): the same
// block-Thomas factor and solve on [..., nz, ni, ni] blocks, with the same perm
// convention (perm[i] = source row of permuted row i). Launch: one thread block
// per batch element, 2 ni threads rounded up to a warp, the whole nz sweep inside
// the kernel. Shared memory: two ni x ni double blocks plus O(ni) (about 125 KB
// at ni = 89), opted in to the device maximum. A zero pivot gives inf/nan like
// lax.linalg.lu, not an error. Build: python -m vulcan_jax.solver_fast --cuda.
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

#include "block_thomas_common.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using namespace vulcan_bt;

constexpr int kWarpSize = 32;
constexpr int kLaneMask = kWarpSize - 1;
constexpr unsigned kFullMask = 0xffffffffu;  // every lane of a warp
constexpr int kMaxThreads = 1024;            // CUDA limit per thread block
constexpr int kMaxWarps = kMaxThreads / kWarpSize;
constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();  // CUDA gridDim.x limit

// Sum over q0 <= q < q1 of arow[q] * mcol[q * ms], in four independent
// accumulators so the dependent shared-memory loads overlap. It differs from
// the CPU reference's left-to-right sum by reassociation and by FMA
// contraction (nvcc's default -fmad=true).
__device__ inline double bt_dot4(const double* arow, const double* mcol, int ms,
                                 int q0, int q1) {
  double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
  int q = q0;
  for (; q + 3 < q1; q += 4) {
    a0 += arow[q] * mcol[q * ms];
    a1 += arow[q + 1] * mcol[(q + 1) * ms];
    a2 += arow[q + 2] * mcol[(q + 2) * ms];
    a3 += arow[q + 3] * mcol[(q + 3) * ms];
  }
  for (; q < q1; ++q) a0 += arow[q] * mcol[q * ms];
  return (a0 + a1) + (a2 + a3);
}

// x <- A^{-1} b from one block's LU, run by all lanes of warp 0 and no other
// warp. On entry the shared vector tmp holds b[perm], visible to warp 0; x
// (shared or global) must not overlap tmp. Lane l owns the rows r with
// r % kWarpSize == l, so tmp is thread-local and the only cross-lane traffic is
// the __shfl_sync broadcast of each finished x_i (uniform outer loops, no early
// return). No block barrier: the other warps prefetch meanwhile. The forward
// substitution applies the CPU lu_solve's subtractions in its order, the back
// substitution in reverse order.
__device__ void bt_lu_solve_warp0(const double* lu, int ni, double* tmp, double* x,
                                  int lane) {
  for (int i = 0; i < ni; ++i) {  // unit-lower L: tmp[i] is final
    double xi = 0.0;
    if (lane == (i & kLaneMask)) xi = tmp[i];  // the lane's own writes
    xi = __shfl_sync(kFullMask, xi, i & kLaneMask);
    for (int r = i + 1 + ((lane - i - 1) & kLaneMask); r < ni; r += kWarpSize)
      tmp[r] -= lu[r * ni + i] * xi;
  }
  for (int i = ni - 1; i >= 0; --i) {  // upper U
    double xi = 0.0;
    if (lane == (i & kLaneMask)) {
      xi = tmp[i] / lu[i * ni + i];
      x[i] = xi;
    }
    xi = __shfl_sync(kFullMask, xi, i & kLaneMask);
    for (int r = i - 1 - ((i - 1 - lane) & kLaneMask); r >= 0; r -= kWarpSize)
      tmp[r] -= lu[r * ni + i] * xi;
  }
}

// In-place partial-pivot LU of the ni x ni shared block A by the whole thread
// block (nt a multiple of kWarpSize, at most kMaxThreads); sperm gets the CPU
// lu_inplace permutation. Pivot rule as the CPU kernel's: a row below k wins
// only on a strict |a| > |A[k][k]|, the smallest row wins a tie, and a NaN
// candidate never wins. cand_abs, cand_val and cand_row hold kMaxWarps entries.
// Entry: A visible to every thread. Barriers: one after sperm is initialised,
// then per column (1) each warp's candidate published, (2) the row swap and the
// multipliers written, (3) the trailing update written. Every thread reaches
// every barrier and every full-warp shuffle. Exit: A and sperm visible.
__device__ void bt_lu_block(double* A, int32_t* sperm, double* cand_abs, double* cand_val,
                            int32_t* cand_row, int ni, int tid, int nt) {
  const int lane = tid & kLaneMask, warp = tid / kWarpSize, nwarps = nt / kWarpSize;
  for (int i = tid; i < ni; i += nt) sperm[i] = i;
  __syncthreads();
  for (int k = 0; k < ni; ++k) {
    const double akk = A[k * ni + k];  // read before this column is written
    double best_abs = -1.0, best_val = akk;
    int best_row = ni;  // loses every tie against a real row
    for (int i = tid; i < ni; i += nt) {
      const double a = A[i * ni + k];
      const double av = fabs(a);
      double v = -1.0;
      if (i > k && av == av) v = av;  // NaN never wins
      if (v > best_abs || (v == best_abs && i < best_row)) {
        best_abs = v;
        best_val = a;
        best_row = i;
      }
    }
    for (int off = kWarpSize / 2; off > 0; off /= 2) {
      const double other_abs = __shfl_down_sync(kFullMask, best_abs, off);
      const double other_val = __shfl_down_sync(kFullMask, best_val, off);
      const int other_row = __shfl_down_sync(kFullMask, best_row, off);
      if (other_abs > best_abs || (other_abs == best_abs && other_row < best_row)) {
        best_abs = other_abs;
        best_val = other_val;
        best_row = other_row;
      }
    }
    if (lane == 0) {
      cand_abs[warp] = best_abs;
      cand_val[warp] = best_val;
      cand_row[warp] = best_row;
    }
    __syncthreads();  // (1)
    // Every thread reduces the warp candidates itself, so the swap and the
    // scale share one phase: the signed pivot rides the reduction, and column
    // k belongs to the scale, so no thread re-reads A after a write.
    double win_abs = cand_abs[0], win_val = cand_val[0];
    int win_row = cand_row[0];
    for (int q = 1; q < nwarps; ++q) {
      if (cand_abs[q] > win_abs || (cand_abs[q] == win_abs && cand_row[q] < win_row)) {
        win_abs = cand_abs[q];
        win_val = cand_val[q];
        win_row = cand_row[q];
      }
    }
    const int p = (win_abs > fabs(akk)) ? win_row : k;  // -1 and NaN never beat |akk|
    const double piv = (p != k) ? win_val : akk;
    if (p != k) {
      for (int j = tid; j < ni; j += nt) {  // full rows, L columns included
        if (j == k) continue;               // column k is the scale's below
        const double swp = A[k * ni + j];
        A[k * ni + j] = A[p * ni + j];
        A[p * ni + j] = swp;
      }
      if (tid == 0) {
        const int32_t t = sperm[k];
        sperm[k] = sperm[p];
        sperm[p] = t;
      }
    }
    if (tid == 0) A[k * ni + k] = piv;  // the swap's column-k entry
    for (int i = k + 1 + tid; i < ni; i += nt) {
      const double num = (i == p) ? akk : A[i * ni + k];  // post-swap column k
      A[i * ni + k] = num / piv;
    }
    __syncthreads();  // (2)
    const int w = ni - k - 1;  // rank-1 update, threads strided over the elements
    if (w > 0) {
      const int row_step = nt / w, col_step = nt % w;
      int r = tid / w, c = tid % w;
      for (int e = tid; e < w * w; e += nt) {
        A[(k + 1 + r) * ni + k + 1 + c] -=
            A[(k + 1 + r) * ni + k] * A[k * ni + k + 1 + c];
        r += row_step;
        c += col_step;
        if (c >= w) {  // col_step < w and c < w, so one correction is enough
          c -= w;
          ++r;
        }
      }
    }
    __syncthreads();  // (3)
  }
}

// Factors one batch element: A'_0 = D_0, A'_j = D_j - (c_j b_{j-1}^T) .*
// inv(A'_{j-1}) with b = sup, c = sub, each A'_j LU-factored in shared memory
// and written to lu and perm. Shared: A'_j and inv(A'_{j-1}) (ni^2 doubles
// each), the pivot candidates (2 kMaxWarps doubles, kMaxWarps ints) and sperm
// (ni ints). Per layer j >= 1 the barriers are: before the inverse (A'_{j-1}
// and sperm visible), after it (inv complete), after the Schur assembly (A'_j
// complete), then bt_lu_block's.
__global__ void bt_factor_kernel(const double* __restrict__ diag,
                                 const double* __restrict__ sup,
                                 const double* __restrict__ sub, double* __restrict__ lu,
                                 int32_t* __restrict__ perm, int nz, int ni) {
  extern __shared__ double bt_smem[];
  const int blk = ni * ni;  // fits an int: the block lives in shared memory
  double* A = bt_smem;
  double* Minv = A + blk;
  double* cand_abs = Minv + blk;
  double* cand_val = cand_abs + kMaxWarps;
  int32_t* cand_row = reinterpret_cast<int32_t*>(cand_val + kMaxWarps);
  int32_t* sperm = cand_row + kMaxWarps;

  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t b = blockIdx.x;
  const double* D = diag + b * nz * blk;
  const double* S = sup + b * (nz - 1) * ni;
  const double* C = sub + b * (nz - 1) * ni;
  double* L = lu + b * nz * blk;
  int32_t* P = perm + b * nz * ni;

  for (int i = tid; i < blk; i += nt) A[i] = D[i];
  __syncthreads();
  bt_lu_block(A, sperm, cand_abs, cand_val, cand_row, ni, tid, nt);
  for (int i = tid; i < blk; i += nt) L[i] = A[i];
  for (int i = tid; i < ni; i += nt) P[i] = sperm[i];

  const int lane = tid & kLaneMask;
  const int half = tid & 1;    // which half of a column's dot products
  const int npair = nt / 2;    // columns in flight
  const int num_rounds = (ni + npair - 1) / npair;
  for (int j = 1; j < nz; ++j) {
    __syncthreads();
    // inv(A'_{j-1}) column by column, a lane pair per column: lanes 2m and
    // 2m+1 split every dot product and combine it with one __shfl_xor_sync.
    // Column s solves A'_{j-1} x = e_s with the rhs permuted (e_s[perm]) and
    // the column itself as lu_solve's tmp. The even lane owns the column's
    // shared slot and __syncwarp publishes it to its partner. Whole pairs drop
    // out together when s >= ni, and pair_mask names the lanes that remain.
    for (int round = 0; round < num_rounds; ++round) {
      const int s = tid / 2 + round * npair;
      if (s >= ni) continue;
      const int active_pairs = ni - (s - lane / 2);  // pairs of this warp with a column
      const unsigned pair_mask =
          (active_pairs >= kWarpSize / 2) ? kFullMask : ((1u << (2 * active_pairs)) - 1u);
      double* inv_col = Minv + s;
      if (half == 0)
        for (int i = 0; i < ni; ++i) inv_col[i * ni] = (sperm[i] == s) ? 1.0 : 0.0;
      for (int i = 0; i < ni; ++i) {  // unit-lower L
        const int mid = i / 2;
        double part = bt_dot4(A + i * ni, inv_col, ni, half ? mid : 0, half ? i : mid);
        part += __shfl_xor_sync(pair_mask, part, 1);
        if (half == 0) inv_col[i * ni] -= part;
        __syncwarp(pair_mask);
      }
      for (int i = ni - 1; i >= 0; --i) {  // upper U
        const int mid = i + 1 + (ni - i - 1) / 2;
        double part = bt_dot4(A + i * ni, inv_col, ni, half ? mid : i + 1, half ? ni : mid);
        part += __shfl_xor_sync(pair_mask, part, 1);
        if (half == 0) inv_col[i * ni] = (inv_col[i * ni] - part) / A[i * ni + i];
        __syncwarp(pair_mask);
      }
    }
    __syncthreads();
    // A'_j = D_j - (c b^T) .* inv(A'_{j-1}), threads strided over the elements
    // so the reads of D_j and inv are coalesced and independent.
    const double* Dj = D + static_cast<int64_t>(j) * blk;
    const double* c = C + static_cast<int64_t>(j - 1) * ni;
    const double* bb = S + static_cast<int64_t>(j - 1) * ni;
    const int row_step = nt / ni, col_step = nt % ni;
    int r = tid / ni, col = tid % ni;
    for (int e = tid; e < blk; e += nt) {
      A[e] = Dj[e] - (c[r] * bb[col]) * Minv[e];
      r += row_step;
      col += col_step;
      if (col >= ni) {
        col -= ni;
        ++r;
      }
    }
    __syncthreads();
    bt_lu_block(A, sperm, cand_abs, cand_val, cand_row, ni, tid, nt);
    double* Lj = L + static_cast<int64_t>(j) * blk;
    for (int i = tid; i < blk; i += nt) Lj[i] = A[i];
    for (int i = tid; i < ni; i += nt) P[static_cast<int64_t>(j) * ni + i] = sperm[i];
  }
}

// Copies one layer's LU block into shared memory with threads t0, t0 + n0, ...
// The caller barriers. While warp 0 substitutes, the other warps call this as
// the prefetch, so the global latency hides behind the solve.
__device__ inline void bt_load_block(double* A, const double* L, int blk, int t0, int n0) {
  for (int i = t0; i < blk; i += n0) A[i] = L[i];
}

__device__ inline void bt_swap(double*& a, double*& b) {
  double* t = a;
  a = b;
  b = t;
}

// Solves one rhs batch element on the factors bt_index maps it to: the forward
// sweep r'_j = r_j - c_j .* (A'_{j-1}^{-1} r'_{j-1}), then k_last = A'^{-1}
// r'_last and k_j = A'_j^{-1} (r'_j - b_j .* k_{j+1}); x holds r' until k
// overwrites it. Warp 0 runs every substitution while the other warps prefetch
// the next layer's LU into the second shared buffer (a one-warp block does both
// in turn). Shared: the current and the prefetched LU (ni^2 doubles each), t
// and u (ni each). Per layer the barriers are: u filled and the spare buffer
// free; the substitution done and the prefetch staged; in the forward sweep
// also r'_j complete before its permuted read.
__global__ void bt_solve_kernel(const double* __restrict__ lu,
                                const int32_t* __restrict__ perm,
                                const double* __restrict__ sup,
                                const double* __restrict__ sub,
                                const double* __restrict__ rhs, double* __restrict__ x,
                                int nz, int ni, BtBcast s) {
  extern __shared__ double bt_smem[];
  const int blk = ni * ni;  // fits an int: the block lives in shared memory
  double* Ac = bt_smem;     // the layer being solved
  double* An = Ac + blk;    // the layer being prefetched
  double* t = An + blk;     // one layer's solution, and
  double* u = t + ni;       // the permuted rhs the substitution consumes

  const int tid = threadIdx.x, nt = blockDim.x;
  const int warp = tid / kWarpSize, lane = tid & kLaneMask;
  const int64_t b = blockIdx.x;
  const int64_t band = static_cast<int64_t>(nz - 1) * ni;
  const double* L = lu + bt_index(b, s, kBtLu) * nz * blk;
  const int32_t* P = perm + bt_index(b, s, kBtPerm) * nz * ni;
  const double* S = sup + bt_index(b, s, kBtSup) * band;
  const double* C = sub + bt_index(b, s, kBtSub) * band;
  const double* R = rhs + b * nz * ni;
  double* X = x + b * nz * ni;
  const bool prefetches = (nt == kWarpSize) || (warp != 0);
  const int prefetch_tid = (nt > kWarpSize) ? tid - kWarpSize : tid;
  const int prefetch_nt = (nt > kWarpSize) ? nt - kWarpSize : nt;

  for (int i = tid; i < ni; i += nt) X[i] = R[i];
  bt_load_block(Ac, L, blk, tid, nt);
  __syncthreads();
  for (int i = tid; i < ni; i += nt) u[i] = X[P[i]];  // r'_0, permuted
  for (int j = 1; j < nz; ++j) {
    __syncthreads();  // u filled, An free
    if (prefetches)
      bt_load_block(An, L + static_cast<int64_t>(j) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0(Ac, ni, u, t, lane);
    __syncthreads();  // t = A'_{j-1}^{-1} r'_{j-1}, An staged
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      X[o] = R[o] - C[static_cast<int64_t>(j - 1) * ni + i] * t[i];
    }
    __syncthreads();  // r'_j complete
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) u[i] = X[static_cast<int64_t>(j) * ni + Pj[i]];
    bt_swap(Ac, An);
  }
  __syncthreads();  // u filled, An free
  if (prefetches && nz > 1)
    bt_load_block(An, L + static_cast<int64_t>(nz - 2) * blk, blk, prefetch_tid, prefetch_nt);
  if (warp == 0)  // u already holds r'_last, so x may overwrite it
    bt_lu_solve_warp0(Ac, ni, u, X + static_cast<int64_t>(nz - 1) * ni, lane);
  __syncthreads();
  bt_swap(Ac, An);
  for (int j = nz - 2; j >= 0; --j) {
    // r'_j - b_j .* k_{j+1}, gathered straight into permuted order
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + Pj[i];
      u[i] = X[o] - S[o] * X[o + ni];
    }
    __syncthreads();  // u filled, An free
    if (prefetches && j > 0)
      bt_load_block(An, L + static_cast<int64_t>(j - 1) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0(Ac, ni, u, X + static_cast<int64_t>(j) * ni, lane);
    __syncthreads();  // k_j written, An staged
    bt_swap(Ac, An);
  }
}

// Threads per block: 2 ni rounded up to a warp, so the factor's inverse gives
// every column a lane pair in one round (192 at ni = 89).
int bt_threads(int ni) {
  const int t = (2 * ni + kWarpSize - 1) / kWarpSize * kWarpSize;
  return t > kMaxThreads ? kMaxThreads : (t < kWarpSize ? kWarpSize : t);
}

// Dynamic shared memory of each kernel; the layouts are in the kernels.
size_t bt_factor_shmem(int ni) {
  return (2 * static_cast<size_t>(ni) * ni + 2 * kMaxWarps) * sizeof(double) +
         (kMaxWarps + static_cast<size_t>(ni)) * sizeof(int32_t);
}

size_t bt_solve_shmem(int ni) {
  return (2 * static_cast<size_t>(ni) * ni + 2 * static_cast<size_t>(ni)) * sizeof(double);
}

// Per device: the most dynamic shared memory both kernels may use, or the CUDA
// error that stopped the opt-in.
struct BtOptIn {
  cudaError_t status;
  int limit;
};

// Dynamic shared memory above the 48 KB default needs a per-kernel opt-in.
// Both kernels are opted in once per device, to the device's maximum, so a
// call changes no attribute: two differently sized calls cannot race on one,
// and only the launch happens while XLA records a command buffer. XLA makes
// the device's context current before calling a handler, so
// cudaFuncSetAttribute acts on `device`.
BtOptIn bt_opt_in(int device) {
  static std::mutex mu;
  static std::unordered_map<int, BtOptIn> done;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = done.find(device);
  if (it != done.end()) return it->second;
  int optin = 0;
  cudaError_t r = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  int limit = optin;
  for (const void* k : {reinterpret_cast<const void*>(bt_factor_kernel),
                        reinterpret_cast<const void*>(bt_solve_kernel)}) {
    cudaFuncAttributes a = {};
    if (r == cudaSuccess) r = cudaFuncGetAttributes(&a, k);
    // the opt-in maximum bounds static plus dynamic shared memory
    const int dyn = optin - static_cast<int>(a.sharedSizeBytes);
    if (r == cudaSuccess && dyn < limit) limit = dyn;
    if (r == cudaSuccess) r = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  }
  done[device] = BtOptIn{r, limit};
  return done[device];
}

// An error unless `bytes` of dynamic shared memory fit the opted-in limit.
ffi::Error bt_shared_fits(size_t bytes, const BtOptIn& opt) {
  if (opt.status != cudaSuccess) {
    return ffi::Error::Internal(std::string("block-Thomas shared-memory opt-in failed: ") +
                                cudaGetErrorString(opt.status));
  }
  if (bytes > static_cast<size_t>(opt.limit)) {
    return ffi::Error::InvalidArgument("block-Thomas needs " + std::to_string(bytes) +
                                       " B of shared memory per block; this device allows " +
                                       std::to_string(opt.limit));
  }
  return ffi::Error::Success();
}

// The launch's own error, if any; execution errors surface on the stream.
ffi::Error bt_launch_status() {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error FactorImplCuda(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::F64> diag,
                          ffi::Buffer<ffi::F64> sup, ffi::Buffer<ffi::F64> sub,
                          ffi::ResultBuffer<ffi::F64> lu, ffi::ResultBuffer<ffi::S32> perm) {
  BtShape shape;
  ffi::Error err = bt_check_factor(diag, sup, sub, *lu, *perm, &shape);
  if (err.failure()) return err;
  if (shape.batch == 0 || shape.ni == 0) return ffi::Error::Success();
  if (shape.batch > kMaxGridX) return ffi::Error::InvalidArgument("batch exceeds the CUDA grid limit");
  const size_t shmem = bt_factor_shmem(shape.ni);
  err = bt_shared_fits(shmem, bt_opt_in(device));
  if (err.failure()) return err;
  bt_factor_kernel<<<static_cast<unsigned int>(shape.batch), bt_threads(shape.ni), shmem, stream>>>(
      diag.typed_data(), sup.typed_data(), sub.typed_data(), lu->typed_data(),
      perm->typed_data(), shape.nz, shape.ni);
  return bt_launch_status();
}

ffi::Error SolveImplCuda(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::F64> lu,
                         ffi::Buffer<ffi::S32> perm, ffi::Buffer<ffi::F64> sup,
                         ffi::Buffer<ffi::F64> sub, ffi::Buffer<ffi::F64> rhs,
                         ffi::ResultBuffer<ffi::F64> x) {
  BtShape shape;
  BtBcast s;
  ffi::Error err = bt_check_solve(lu, perm, sup, sub, rhs, *x, &shape, &s);
  if (err.failure()) return err;
  if (shape.batch == 0 || shape.ni == 0) return ffi::Error::Success();
  if (shape.batch > kMaxGridX) return ffi::Error::InvalidArgument("batch exceeds the CUDA grid limit");
  const size_t shmem = bt_solve_shmem(shape.ni);
  err = bt_shared_fits(shmem, bt_opt_in(device));
  if (err.failure()) return err;
  bt_solve_kernel<<<static_cast<unsigned int>(shape.batch), bt_threads(shape.ni), shmem, stream>>>(
      lu.typed_data(), perm.typed_data(), sup.typed_data(), sub.typed_data(),
      rhs.typed_data(), x->typed_data(), shape.nz, shape.ni, s);
  return bt_launch_status();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtFactorCuda, FactorImplCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtSolveCuda, SolveImplCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>(),
    {ffi::Traits::kCmdBufferCompatible});

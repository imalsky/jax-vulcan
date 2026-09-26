// CUDA twin of block_thomas_cpu.cc (VULCAN_JAX_SOLVER=ffi on a GPU): the same
// block-Thomas factor and solve on [..., nz, ni, ni] blocks, with the same perm
// convention (perm[i] = source row of permuted row i). Launch: one thread block
// per batch element, 2 ni threads rounded up to a warp, the whole nz sweep inside
// the kernel. Each kernel has two variants with bitwise equal results: two
// ni x ni double blocks of shared memory plus O(ni) (about 125 KB at ni = 89),
// or one (about 63 KB), which fits more blocks on an SM. A call takes two
// buffers when they fit the device and need no more waves than one
// (bt_choose); VULCAN_JAX_BT_BUFFERS=1 or 2 forces a variant. Shared memory is
// opted in to the device maximum. The solve keeps
// each lane's substitution rows in registers, so it takes ni <= 128 and refuses
// more. A zero pivot gives inf/nan like lax.linalg.lu, not an error. Build:
// python -m vulcan_jax.solver_fast --cuda.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
// Rows per lane the solve's substitution holds in registers: ni <= kMaxNb
// kWarpSize (ni 97, the DMS network, takes 4). The solve kernel is compiled
// for every NB up to kMaxNb and launched with NB = ceil(ni / kWarpSize).
constexpr int kMaxNb = 4;
// Minimum resident blocks per SM in the NB solve kernel's __launch_bounds__
// (its maximum is 2 NB warps, the most threads bt_threads gives that NB). They
// hold the kernels near the shared-memory solve's 72-76 registers: 68-80 on
// sm_89 and sm_90 for NB 1-3, without spills, so the NB 3 class (ni 65-96)
// keeps the 3 blocks per SM shared memory allows at ni 89 on sm_90. NB 4 takes
// 116-125, which still allows the 2 blocks shared memory allows at ni 97.
constexpr int kSolveMinBlocks[kMaxNb + 1] = {0, 12, 6, 4, 2};

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
// warp. On entry the shared vector tmp holds b[perm], visible to warp 0; tmp is
// not written, and x (shared or global) must not overlap it. Lane l owns the
// rows r with r % kWarpSize == l and keeps them in registers, t[q] for row
// l + q kWarpSize (NB >= ceil(ni / kWarpSize) slots, indexed only by unrolled
// constants so they stay in registers); the only cross-lane traffic is the
// __shfl_sync broadcast of each finished x_i (uniform outer loops, no early
// return). No block barrier: the other warps prefetch meanwhile. The forward
// substitution applies the CPU lu_solve's subtractions in its order, the back
// substitution in reverse order, each row's in the same order and with the
// same operations as a shared-memory tmp would take them.
template <int NB>
__device__ void bt_lu_solve_warp0(const double* lu, int ni, const double* tmp, double* x,
                                  int lane) {
  double t[NB];
#pragma unroll
  for (int q = 0; q < NB; ++q) {
    const int r = lane + kWarpSize * q;
    t[q] = (r < ni) ? tmp[r] : 0.0;
  }
  // Row i sits in slot i / kWarpSize of lane i % kWarpSize. The loops over that
  // slot are unrolled, so every index into t is a constant; a row below i is
  // in a slot >= i's, a row above in a slot <= i's.
#pragma unroll
  for (int qi = 0; qi < NB; ++qi) {  // unit-lower L: row i is final
    for (int i = kWarpSize * qi; i < min(ni, kWarpSize * (qi + 1)); ++i) {
      const double xi = __shfl_sync(kFullMask, t[qi], i & kLaneMask);
#pragma unroll
      for (int q = qi; q < NB; ++q) {
        const int r = lane + kWarpSize * q;
        if (r > i && r < ni) t[q] -= lu[r * ni + i] * xi;
      }
    }
  }
#pragma unroll
  for (int qi = NB - 1; qi >= 0; --qi) {  // upper U
    for (int i = min(ni, kWarpSize * (qi + 1)) - 1; i >= kWarpSize * qi; --i) {
      double xi = 0.0;
      if (lane == (i & kLaneMask)) {
        xi = t[qi] / lu[i * ni + i];
        x[i] = xi;
      }
      xi = __shfl_sync(kFullMask, xi, i & kLaneMask);
#pragma unroll
      for (int q = 0; q <= qi; ++q) {
        const int r = lane + kWarpSize * q;
        if (r < i) t[q] -= lu[r * ni + i] * xi;
      }
    }
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
// and written to lu and perm. Shared: A'_j and, with two buffers,
// inv(A'_{j-1}) (ni^2 doubles each), the pivot candidates (2 kMaxWarps
// doubles, kMaxWarps ints) and sperm (ni ints). With one buffer (kOneBuf) the
// inverse reads the factors of A'_{j-1} back from lu, stored before the layer
// barrier, builds inv in A itself and the Schur update runs in place: the same
// operations in the same order. Per layer j >= 1 the barriers are: before the
// inverse (A'_{j-1}, sperm and, with one buffer, its LU in lu visible), after
// it (inv complete), after the Schur assembly (A'_j complete), then
// bt_lu_block's.
template <bool kOneBuf>
__global__ void bt_factor_kernel(const double* __restrict__ diag,
                                 const double* __restrict__ sup,
                                 const double* __restrict__ sub, double* __restrict__ lu,
                                 int32_t* __restrict__ perm, int nz, int ni) {
  extern __shared__ double bt_smem[];
  const int blk = ni * ni;  // fits an int: the block lives in shared memory
  double* A = bt_smem;
  double* Minv = kOneBuf ? A : A + blk;
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
    const double* F = kOneBuf ? L + static_cast<int64_t>(j - 1) * blk : A;  // LU of A'_{j-1}
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
        double part = bt_dot4(F + i * ni, inv_col, ni, half ? mid : 0, half ? i : mid);
        part += __shfl_xor_sync(pair_mask, part, 1);
        if (half == 0) inv_col[i * ni] -= part;
        __syncwarp(pair_mask);
      }
      for (int i = ni - 1; i >= 0; --i) {  // upper U
        const int mid = i + 1 + (ni - i - 1) / 2;
        double part = bt_dot4(F + i * ni, inv_col, ni, half ? mid : i + 1, half ? ni : mid);
        part += __shfl_xor_sync(pair_mask, part, 1);
        if (half == 0) inv_col[i * ni] = (inv_col[i * ni] - part) / F[i * ni + i];
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
// overwrites it. Warp 0 runs every substitution. With two buffers the other
// warps prefetch the next layer's LU into the second one meanwhile (a one-warp
// block does both in turn); with one (kOneBuf) all threads load it after the
// substitution. Shared: the current and the prefetched LU (ni^2 doubles each;
// one with kOneBuf), t and u (ni each). Per layer the barriers are: u filled
// and the spare buffer free (one buffer: the layer loaded); the substitution
// done and the prefetch staged; in the forward sweep also r'_j complete (and
// the next layer loaded) before its permuted read.
template <bool kOneBuf, int NB>
__global__ void __launch_bounds__(2 * kWarpSize * NB, kSolveMinBlocks[NB])
bt_solve_kernel(const double* __restrict__ lu,
                                const int32_t* __restrict__ perm,
                                const double* __restrict__ sup,
                                const double* __restrict__ sub,
                                const double* __restrict__ rhs, double* __restrict__ x,
                                int nz, int ni, BtBcast s) {
  extern __shared__ double bt_smem[];
  const int blk = ni * ni;  // fits an int: the block lives in shared memory
  double* Ac = bt_smem;     // the layer being solved
  double* An = kOneBuf ? Ac : Ac + blk;  // the layer being prefetched (two buffers)
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
    if (!kOneBuf && prefetches)
      bt_load_block(An, L + static_cast<int64_t>(j) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0<NB>(Ac, ni, u, t, lane);
    __syncthreads();  // t = A'_{j-1}^{-1} r'_{j-1}, An staged
    if (kOneBuf) bt_load_block(Ac, L + static_cast<int64_t>(j) * blk, blk, tid, nt);
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      X[o] = R[o] - C[static_cast<int64_t>(j - 1) * ni + i] * t[i];
    }
    __syncthreads();  // r'_j complete
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) u[i] = X[static_cast<int64_t>(j) * ni + Pj[i]];
    if (!kOneBuf) bt_swap(Ac, An);
  }
  __syncthreads();  // u filled, An free
  if (!kOneBuf && prefetches && nz > 1)
    bt_load_block(An, L + static_cast<int64_t>(nz - 2) * blk, blk, prefetch_tid, prefetch_nt);
  if (warp == 0)  // u already holds r'_last, so x may overwrite it
    bt_lu_solve_warp0<NB>(Ac, ni, u, X + static_cast<int64_t>(nz - 1) * ni, lane);
  __syncthreads();
  if (!kOneBuf) bt_swap(Ac, An);
  for (int j = nz - 2; j >= 0; --j) {
    if (kOneBuf) bt_load_block(Ac, L + static_cast<int64_t>(j) * blk, blk, tid, nt);
    // r'_j - b_j .* k_{j+1}, gathered straight into permuted order
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + Pj[i];
      u[i] = X[o] - S[o] * X[o + ni];
    }
    __syncthreads();  // u filled, An free
    if (!kOneBuf && prefetches && j > 0)
      bt_load_block(An, L + static_cast<int64_t>(j - 1) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0<NB>(Ac, ni, u, X + static_cast<int64_t>(j) * ni, lane);
    __syncthreads();  // k_j written, An staged
    if (!kOneBuf) bt_swap(Ac, An);
  }
}

// Threads per block: 2 ni rounded up to a warp, so the factor's inverse gives
// every column a lane pair in one round (192 at ni = 89).
int bt_threads(int ni) {
  const int t = (2 * ni + kWarpSize - 1) / kWarpSize * kWarpSize;
  return t > kMaxThreads ? kMaxThreads : (t < kWarpSize ? kWarpSize : t);
}

// Dynamic shared memory of each kernel; the layouts are in the kernels.
size_t bt_factor_shmem(int ni, bool one_buf) {
  return ((one_buf ? 1 : 2) * static_cast<size_t>(ni) * ni + 2 * kMaxWarps) * sizeof(double) +
         (kMaxWarps + static_cast<size_t>(ni)) * sizeof(int32_t);
}

size_t bt_solve_shmem(int ni, bool one_buf) {
  return ((one_buf ? 1 : 2) * static_cast<size_t>(ni) * ni + 2 * static_cast<size_t>(ni)) *
         sizeof(double);
}

using BtSolveKernel = void (*)(const double*, const int32_t*, const double*, const double*,
                               const double*, double*, int, int, BtBcast);

// The solve kernel with one or two buffers and NB register rows per lane.
BtSolveKernel bt_solve_kernel_for(bool one_buf, int nb) {
  static const BtSolveKernel k[2][kMaxNb] = {
      {bt_solve_kernel<false, 1>, bt_solve_kernel<false, 2>, bt_solve_kernel<false, 3>,
       bt_solve_kernel<false, 4>},
      {bt_solve_kernel<true, 1>, bt_solve_kernel<true, 2>, bt_solve_kernel<true, 3>,
       bt_solve_kernel<true, 4>}};
  return k[one_buf][nb - 1];
}

// Per device: the most dynamic shared memory every kernel may use and the SM
// count, or the CUDA error that stopped the opt-in.
struct BtOptIn {
  cudaError_t status;
  int limit;
  int sms;
};

// Dynamic shared memory above the 48 KB default needs a per-kernel opt-in.
// All four kernels are opted in once per device, to the device's maximum, so a
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
  int optin = 0, sms = 0;
  cudaError_t r = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (r == cudaSuccess) r = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  int limit = optin;
  const void* kernels[2 + 2 * kMaxNb] = {reinterpret_cast<const void*>(bt_factor_kernel<false>),
                                         reinterpret_cast<const void*>(bt_factor_kernel<true>)};
  for (int b = 0; b < 2; ++b)
    for (int nb = 1; nb <= kMaxNb; ++nb)
      kernels[2 + b * kMaxNb + nb - 1] = reinterpret_cast<const void*>(bt_solve_kernel_for(b, nb));
  for (const void* k : kernels) {
    cudaFuncAttributes a = {};
    if (r == cudaSuccess) r = cudaFuncGetAttributes(&a, k);
    // the opt-in maximum bounds static plus dynamic shared memory
    const int dyn = optin - static_cast<int>(a.sharedSizeBytes);
    if (r == cudaSuccess && dyn < limit) limit = dyn;
    if (r == cudaSuccess) r = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  }
  done[device] = BtOptIn{r, limit, sms};
  return done[device];
}

// An error unless `bytes` of dynamic shared memory fit the opted-in limit.
ffi::Error bt_shared_fits(size_t bytes, const BtOptIn& opt) {
  if (bytes > static_cast<size_t>(opt.limit)) {
    return ffi::Error::InvalidArgument("block-Thomas needs " + std::to_string(bytes) +
                                       " B of shared memory per block; this device allows " +
                                       std::to_string(opt.limit));
  }
  return ffi::Error::Success();
}

// VULCAN_JAX_BT_BUFFERS, read once per process: unset or empty picks the
// variant per call (bt_choose); "1" or "2" forces the one- or two-buffer
// kernels, for the A/B and the byte-identity test.
const char* bt_forced_buffers() {
  static const char* const v = [] {
    const char* e = std::getenv("VULCAN_JAX_BT_BUFFERS");
    return (e == nullptr) ? "" : e;
  }();
  return v;
}

// Sets *one_buf to the variant a call of `batch` blocks launches. Two buffers
// when they fit the device and need no more waves than one, a wave being the
// SM count times the blocks of that variant resident on an SM (the occupancy
// calculator, so the device's shared memory per SM, registers and block limit
// all count): at equal waves the two-buffer kernels are as fast or faster
// (the solve prefetches, the factor's inverse reads shared memory). Returns
// the error that forbids the launch, if any.
ffi::Error bt_choose(int device, int64_t batch, const void* two, size_t shmem2, const void* one,
                     size_t shmem1, int threads, bool* one_buf) {
  const BtOptIn opt = bt_opt_in(device);
  if (opt.status != cudaSuccess) {
    return ffi::Error::Internal(std::string("block-Thomas shared-memory opt-in failed: ") +
                                cudaGetErrorString(opt.status));
  }
  const char* forced = bt_forced_buffers();
  if (std::strcmp(forced, "1") == 0 || std::strcmp(forced, "2") == 0) {
    *one_buf = forced[0] == '1';
  } else if (forced[0] != '\0') {
    return ffi::Error::InvalidArgument(std::string("VULCAN_JAX_BT_BUFFERS=") + forced +
                                       ": expected 1, 2 or unset");
  } else if (shmem2 > static_cast<size_t>(opt.limit)) {
    *one_buf = true;
  } else {
    int per_sm2 = 0, per_sm1 = 0;
    cudaError_t r = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm2, two, threads, shmem2);
    if (r == cudaSuccess)
      r = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm1, one, threads, shmem1);
    if (r != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(r));
    const auto waves = [&](int per_sm) {  // no resident block: never chosen
      const int64_t w = static_cast<int64_t>(per_sm) * opt.sms;
      return (w > 0) ? (batch + w - 1) / w : std::numeric_limits<int64_t>::max();
    };
    *one_buf = waves(per_sm1) < waves(per_sm2);
  }
  return bt_shared_fits(*one_buf ? shmem1 : shmem2, opt);
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
  const int threads = bt_threads(shape.ni);
  bool one_buf = false;
  err = bt_choose(device, shape.batch, reinterpret_cast<const void*>(bt_factor_kernel<false>),
                  bt_factor_shmem(shape.ni, false),
                  reinterpret_cast<const void*>(bt_factor_kernel<true>),
                  bt_factor_shmem(shape.ni, true), threads, &one_buf);
  if (err.failure()) return err;
  const auto kernel = one_buf ? bt_factor_kernel<true> : bt_factor_kernel<false>;
  kernel<<<static_cast<unsigned int>(shape.batch), threads, bt_factor_shmem(shape.ni, one_buf),
           stream>>>(diag.typed_data(), sup.typed_data(), sub.typed_data(), lu->typed_data(),
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
  const int nb = (shape.ni + kWarpSize - 1) / kWarpSize;
  if (nb > kMaxNb) {
    return ffi::Error::InvalidArgument("block-Thomas solve holds at most " +
                                       std::to_string(kMaxNb * kWarpSize) +
                                       " rows per block in registers; ni = " +
                                       std::to_string(shape.ni));
  }
  const int threads = bt_threads(shape.ni);
  bool one_buf = false;
  err = bt_choose(device, shape.batch, reinterpret_cast<const void*>(bt_solve_kernel_for(false, nb)),
                  bt_solve_shmem(shape.ni, false),
                  reinterpret_cast<const void*>(bt_solve_kernel_for(true, nb)),
                  bt_solve_shmem(shape.ni, true), threads, &one_buf);
  if (err.failure()) return err;
  bt_solve_kernel_for(one_buf, nb)<<<static_cast<unsigned int>(shape.batch), threads,
                                     bt_solve_shmem(shape.ni, one_buf), stream>>>(
      lu.typed_data(), perm.typed_data(), sup.typed_data(), sub.typed_data(), rhs.typed_data(),
      x->typed_data(), shape.nz, shape.ni, s);
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

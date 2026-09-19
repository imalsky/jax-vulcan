// GPU twin of block_thomas_cpu.cc: the fused block-Thomas factor / solve for
// VULCAN_JAX_SOLVER=ffi on a CUDA device. Same math, same [..., nz, ni, ni]
// layout, same perm convention (perm[i] = source row of permuted row i, so
// x = b[perm]), which is what lets solver_fast's transpose_solve keep running
// the JAX sweep on these factors. One thread block per lane (a batch element):
// the ni x ni block and inv(A'_{j-1}) live in dynamic shared memory, pivoting
// happens in-block, and the whole nz loop runs inside the kernel, so a Ros2
// step is one factor launch and two solve launches. The solve takes a stack of
// right-hand sides on shared factors: one block per rhs element, the factors'
// leading dimensions broadcasting against the rhs's (equal, or 1). A zero pivot
// gives inf/nan like lax.linalg.lu, not an error.
// Both kernels are latency-bound, not throughput-bound: at these shared-memory
// sizes an SM holds one block, so the structure is built around short barrier
// and dependency chains. The LU's pivot search reduces inside each warp with
// __shfl_down_sync and publishes one candidate per warp, which leaves three
// barriers per matrix column; its rank-1 update and the Schur assembly stride
// threads over the ELEMENTS of a block, so their FMAs are independent; the
// inverse gives each column a lane pair that splits its dot products; and the
// solve's triangular substitutions run inside warp 0 with no block barrier at
// all, while the other warps prefetch the next layer's LU into the second
// shared buffer. A block is 2 ni threads rounded up to a warp (bt_threads),
// which is what the lane pairs and the element loops spend.
// Build: python -m vulcan_jax.solver_fast --cuda (nvcc, on the GPU host); the
// library registers under platform="CUDA".
#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// sum over q0 <= q < q1 of arow[q] * mcol[q * ms], in four independent
// accumulators so the dependent shared loads overlap instead of queueing behind
// one fp chain. The reassociation is the only accuracy change against the CPU
// reference's left-to-right sum.
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

// x <- A^{-1} b from one block's LU, by WARP 0 alone. Lane l owns the rows
// r with r % 32 == l, so every read and write of the running vector `tmp` is
// thread-local and the only cross-lane traffic is the broadcast of the finished
// x_i (__shfl_sync, all 32 lanes, uniform loop bounds, no early return). No
// block barrier inside: that is what lets the other warps prefetch the next
// layer while this runs. `tmp` must be shared and must already hold b[perm]
// (the caller loads it block-parallel, which is also what makes b safe to alias
// x: b is consumed before x is written); x may be shared or global.
// Column-oriented: once x_i is final its owner broadcasts it and every lane
// updates its own rows. Each element takes exactly the subtractions the
// row-oriented lu_solve in block_thomas_cpu.cc gives it, in the same order
// forward (increasing i) and in reverse backward (decreasing i), so only the
// back substitution differs from the CPU reference, by that reassociation.
__device__ void bt_lu_solve_warp0(const double* lu, int ni, double* tmp, double* x,
                                  int lane) {
  for (int i = 0; i < ni; ++i) {  // unit-lower L: tmp[i] is final
    double xi = 0.0;
    if (lane == (i & 31)) xi = tmp[i];  // its own writes: no sync needed
    xi = __shfl_sync(0xffffffffu, xi, i & 31);
    for (int r = i + 1 + ((lane - i - 1) & 31); r < ni; r += 32)
      tmp[r] -= lu[r * ni + i] * xi;
  }
  for (int i = ni - 1; i >= 0; --i) {  // upper U
    double xi = 0.0;
    if (lane == (i & 31)) {
      xi = tmp[i] / lu[i * ni + i];
      x[i] = xi;
    }
    xi = __shfl_sync(0xffffffffu, xi, i & 31);
    for (int r = i - 1 - ((i - 1 - lane) & 31); r >= 0; r -= 32)
      tmp[r] -= lu[r * ni + i] * xi;
  }
}

// In-place partial-pivot LU of the ni x ni shared-memory block A, by the whole
// thread block; sperm gets lu_inplace's permutation. The pivot is the CPU
// kernel's: rows k+1.. beat |A[k][k]| only on a strict >, so the smallest index
// wins a tie, and a NaN candidate below row k loses (masked to -1) exactly as
// the CPU scan's `v > best` skips it. Three barriers per column: each thread
// scans its strided rows, the warps reduce with __shfl_down_sync and lane 0
// publishes their candidates (barrier 1); every thread repeats the <= 32 warp
// candidates, so the row swap and the column scale can run in one phase
// (barrier 2) -- the signed pivot rides the reduction, and column k belongs to
// the scale, so nothing has to re-read A after a write; then the rank-1 update
// strides threads over the trailing block's elements (barrier 3). nt is a
// multiple of 32 and no thread returns early, so every barrier and every
// full-warp shuffle is reached by all threads. wval/wsv/widx hold 32 entries.
__device__ void bt_lu_block(double* A, int32_t* sperm, double* wval, double* wsv,
                            int32_t* widx, int ni, int tid, int nt) {
  const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
  for (int i = tid; i < ni; i += nt) sperm[i] = i;
  __syncthreads();
  for (int k = 0; k < ni; ++k) {
    const double akk = A[k * ni + k];  // read before this column is written
    double bv = -1.0, bs = akk;
    int bi = ni;  // sentinel: loses every tie against a real row
    for (int i = tid; i < ni; i += nt) {
      const double a = A[i * ni + k];
      const double av = fabs(a);
      double v = -1.0;
      if (i > k && av == av) v = av;  // NaN never wins
      if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bs = a;
        bi = i;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const double ov = __shfl_down_sync(0xffffffffu, bv, off);
      const double os = __shfl_down_sync(0xffffffffu, bs, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bs = os;
        bi = oi;
      }
    }
    if (lane == 0) {
      wval[warp] = bv;
      wsv[warp] = bs;
      widx[warp] = bi;
    }
    __syncthreads();  // 1: the warp candidates are published
    double pv = wval[0], ps = wsv[0];
    int pi = widx[0];
    for (int q = 1; q < nw; ++q) {
      const double ov = wval[q], os = wsv[q];
      const int oi = widx[q];
      if (ov > pv || (ov == pv && oi < pi)) {
        pv = ov;
        ps = os;
        pi = oi;
      }
    }
    const int p = (pv > fabs(akk)) ? pi : k;  // -1 never beats |akk|, nor does NaN
    const double piv = (p != k) ? ps : akk;
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
    __syncthreads();  // 2: the swap and the multipliers are visible
    const int w = ni - k - 1;  // rank-1 update, threads over the block's elements
    if (w > 0) {
      const int dr = nt / w, dc = nt % w;
      int r = tid / w, c = tid % w;
      for (int e = tid; e < w * w; e += nt) {
        A[(k + 1 + r) * ni + k + 1 + c] -=
            A[(k + 1 + r) * ni + k] * A[k * ni + k + 1 + c];
        r += dr;
        c += dc;
        if (c >= w) {  // dc < w and c < w, so one correction is enough
          c -= w;
          ++r;
        }
      }
    }
    __syncthreads();  // 3: the trailing block is updated
  }
}

__global__ void bt_factor_kernel(const double* diag, const double* sup, const double* sub,
                                 double* lu, int32_t* perm, int nz, int ni) {
  extern __shared__ double bt_smem[];
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  double* A = bt_smem;          // A'_j, factored in place
  double* Minv = A + blk;       // inv(A'_{j-1})
  double* wval = Minv + blk;    // one pivot candidate per warp: |value|,
  double* wsv = wval + 32;      // the signed value,
  int32_t* widx = reinterpret_cast<int32_t*>(wsv + 32);  // and the row
  int32_t* sperm = widx + 32;

  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t b = blockIdx.x;
  const double* D = diag + b * nz * blk;
  const double* S = sup + b * static_cast<int64_t>(nz - 1) * ni;
  const double* C = sub + b * static_cast<int64_t>(nz - 1) * ni;
  double* L = lu + b * nz * blk;
  int32_t* P = perm + b * static_cast<int64_t>(nz) * ni;

  for (int64_t i = tid; i < blk; i += nt) A[i] = D[i];
  __syncthreads();
  bt_lu_block(A, sperm, wval, wsv, widx, ni, tid, nt);
  for (int64_t i = tid; i < blk; i += nt) L[i] = A[i];
  for (int i = tid; i < ni; i += nt) P[i] = sperm[i];

  const int lane = tid & 31;
  const int hlf = tid & 1;    // which half of a column's dot products
  const int npair = nt >> 1;  // columns in flight
  const int nrd = (ni + npair - 1) / npair;
  for (int j = 1; j < nz; ++j) {
    __syncthreads();
    // inv(A'_{j-1}) column by column, a LANE PAIR per column: lanes 2m and
    // 2m+1 split every dot product and combine it with one __shfl_xor_sync, so
    // the chain per row is half as long as one thread per column. Column s is
    // the solve of A'_{j-1} x = e_s, with the rhs permuted (e_s[perm]) like
    // lu_solve's, and the column doubling as lu_solve's tmp before it holds x
    // (the CPU kernel's b-aliases-x case). The even lane owns the column's
    // shared slot and __syncwarp publishes it to its partner. Whole pairs drop
    // out together when s >= ni, so the mask names exactly the lanes that run.
    for (int rd = 0; rd < nrd; ++rd) {
      const int s = (tid >> 1) + rd * npair;
      if (s >= ni) continue;
      const int nact = ni - (s - (lane >> 1));  // active pairs in this warp
      const unsigned msk = (nact >= 16) ? 0xffffffffu : ((1u << (2 * nact)) - 1u);
      double* col = Minv + s;
      if (hlf == 0)
        for (int i = 0; i < ni; ++i) col[i * ni] = (sperm[i] == s) ? 1.0 : 0.0;
      for (int i = 0; i < ni; ++i) {  // unit-lower L
        const int mid = i >> 1;
        double part = bt_dot4(A + i * ni, col, ni, hlf ? mid : 0, hlf ? i : mid);
        part += __shfl_xor_sync(msk, part, 1);
        if (hlf == 0) col[i * ni] -= part;
        __syncwarp(msk);
      }
      for (int i = ni - 1; i >= 0; --i) {  // upper U
        const int mid = i + 1 + ((ni - i - 1) >> 1);
        double part = bt_dot4(A + i * ni, col, ni, hlf ? mid : i + 1, hlf ? ni : mid);
        part += __shfl_xor_sync(msk, part, 1);
        if (hlf == 0) col[i * ni] = (col[i * ni] - part) / A[i * ni + i];
        __syncwarp(msk);
      }
    }
    __syncthreads();
    // A'_j = D_j - (c b^T) .* inv(A'_{j-1}), threads over the elements: the
    // reads of D_j and of Minv are coalesced and each thread's are independent.
    const double* Dj = D + static_cast<int64_t>(j) * blk;
    const double* c = C + static_cast<int64_t>(j - 1) * ni;
    const double* bb = S + static_cast<int64_t>(j - 1) * ni;
    const int dr = nt / ni, ds = nt % ni;
    int r = tid / ni, sc = tid % ni;
    for (int64_t e = tid; e < blk; e += nt) {
      A[e] = Dj[e] - (c[r] * bb[sc]) * Minv[e];
      r += dr;
      sc += ds;
      if (sc >= ni) {
        sc -= ni;
        ++r;
      }
    }
    __syncthreads();
    bt_lu_block(A, sperm, wval, wsv, widx, ni, tid, nt);
    double* Lj = L + static_cast<int64_t>(j) * blk;
    for (int64_t i = tid; i < blk; i += nt) Lj[i] = A[i];
    for (int i = tid; i < ni; i += nt) P[static_cast<int64_t>(j) * ni + i] = sperm[i];
  }
}

// The solve's batch is the product of the rhs leading dimensions; each factor
// operand's leading dimensions broadcast against them numpy-style (equal, or 1
// -> stride 0), which is what vmap_method="expand_dims" hands the handler when
// only the rhs is batched at a vmap level.
static constexpr int kBtMaxBatch = 4;

struct BtBcast {
  int nd;                             // number of leading dimensions
  int64_t dim[kBtMaxBatch];           // the rhs leading dimensions
  int64_t stride[4][kBtMaxBatch];     // per factor operand: lu, perm, sup, sub
};

// Strides of one factor operand into `s`; `trail` is its per-element count.
static ffi::Error bt_leading(BtBcast* s, int op, ffi::AnyBuffer::Dimensions dims,
                             int ntrail, int64_t trail, int64_t count, const char* name) {
  const int nb = static_cast<int>(dims.size()) - ntrail;
  if (nb != s->nd) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must have the rhs's number of leading dimensions");
  }
  int64_t stride = 1;
  for (int k = nb - 1; k >= 0; --k) {
    if (dims[k] != s->dim[k] && dims[k] != 1) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        std::string(name) + " leading dimensions must equal the rhs's or be 1");
    }
    s->stride[op][k] = (dims[k] == 1) ? 0 : stride;
    stride *= dims[k];
  }
  if (count != stride * trail) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " has the wrong trailing shape");
  }
  return ffi::Error::Success();
}

// Element of factor operand `op` that rhs batch element `b` reads.
__device__ static int64_t bt_index(int64_t b, const BtBcast& s, int op) {
  int64_t idx = 0;
  for (int k = s.nd - 1; k >= 0; --k) {
    idx += (b % s.dim[k]) * s.stride[op][k];
    b /= s.dim[k];
  }
  return idx;
}

// Stage one layer's LU block in shared memory. The caller barriers; when a
// substitution is running this is the prefetch, issued by the warps that are
// not in it, so the global latency hides behind the solve.
__device__ inline void bt_load_block(double* A, const double* L, int64_t blk,
                                     int t0, int n0) {
  for (int64_t i = t0; i < blk; i += n0) A[i] = L[i];
}

__global__ void bt_solve_kernel(const double* lu, const int32_t* perm, const double* sup,
                                const double* sub, const double* rhs, double* x,
                                int nz, int ni, BtBcast s) {
  extern __shared__ double bt_smem[];
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  double* Ac = bt_smem;        // the layer being solved
  double* An = Ac + blk;       // the layer being prefetched
  double* t = An + blk;        // one layer's solution, and
  double* u = t + ni;          // the permuted rhs the substitution consumes

  const int tid = threadIdx.x, nt = blockDim.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int64_t b = blockIdx.x;  // one block per rhs element; its factors are mapped
  const int64_t bnd = static_cast<int64_t>(nz - 1) * ni;
  const double* L = lu + bt_index(b, s, 0) * nz * blk;
  const int32_t* P = perm + bt_index(b, s, 1) * static_cast<int64_t>(nz) * ni;
  const double* S = sup + bt_index(b, s, 2) * bnd;
  const double* C = sub + bt_index(b, s, 3) * bnd;
  const double* R = rhs + b * static_cast<int64_t>(nz) * ni;
  double* X = x + b * static_cast<int64_t>(nz) * ni;
  // warp 0 substitutes, the rest prefetch; with a single warp it does both
  const bool pfme = (nt == 32) || (warp != 0);
  const int pft = (nt > 32) ? tid - 32 : tid;
  const int pfn = (nt > 32) ? nt - 32 : nt;

  for (int i = tid; i < ni; i += nt) X[i] = R[i];  // X holds r' while sweeping
  bt_load_block(Ac, L, blk, tid, nt);
  __syncthreads();
  for (int i = tid; i < ni; i += nt) u[i] = X[P[i]];  // r'_0, permuted
  for (int j = 1; j < nz; ++j) {
    __syncthreads();  // u is filled, An is free
    if (pfme) bt_load_block(An, L + static_cast<int64_t>(j) * blk, blk, pft, pfn);
    if (warp == 0) bt_lu_solve_warp0(Ac, ni, u, t, lane);
    __syncthreads();  // t holds A'_{j-1}^{-1} r'_{j-1}, An is staged
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      X[o] = R[o] - C[static_cast<int64_t>(j - 1) * ni + i] * t[i];
    }
    __syncthreads();  // r'_j is complete, so it can be read permuted
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) u[i] = X[static_cast<int64_t>(j) * ni + Pj[i]];
    double* sw = Ac;
    Ac = An;
    An = sw;
  }
  __syncthreads();  // u is filled, An is free
  if (pfme && nz > 1)
    bt_load_block(An, L + static_cast<int64_t>(nz - 2) * blk, blk, pft, pfn);
  if (warp == 0)  // b aliases x, as in the CPU sweep: u already holds it
    bt_lu_solve_warp0(Ac, ni, u, X + static_cast<int64_t>(nz - 1) * ni, lane);
  __syncthreads();
  {
    double* sw = Ac;
    Ac = An;
    An = sw;
  }
  for (int j = nz - 2; j >= 0; --j) {
    // r'_j - b_j .* k_{j+1}, gathered straight into permuted order
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + Pj[i];
      u[i] = X[o] - S[o] * X[o + ni];
    }
    __syncthreads();  // u is filled, An is free
    if (pfme && j > 0)
      bt_load_block(An, L + static_cast<int64_t>(j - 1) * blk, blk, pft, pfn);
    if (warp == 0) bt_lu_solve_warp0(Ac, ni, u, X + static_cast<int64_t>(j) * ni, lane);
    __syncthreads();  // k_j is written, An is staged
    double* sw = Ac;
    Ac = An;
    An = sw;
  }
}

// One warp per 16 matrix columns: the factor kernel's inverse gives every column
// a lane pair, and both kernels' element loops and prefetches spend the threads,
// so a block is 2 ni rounded up to a warp (192 at ni = 89).
static int bt_threads(int ni) {
  const int t = ((2 * ni + 31) / 32) * 32;
  return t > 1024 ? 1024 : (t < 32 ? 32 : t);
}

// Opt in to more than the 48 KB default of dynamic shared memory per block
// (a GH200 allows 228 KB): the factor kernel needs 2 ni^2 doubles and the solve
// the same for its double-buffered LU, 125 KB at ni = 89. A block too large for
// the device fails here, not silently.
static ffi::Error bt_shared_opt_in(const void* kernel, size_t bytes) {
  if (bytes <= 48 * 1024) return ffi::Error::Success();
  const cudaError_t e = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
  if (e != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "block-Thomas needs " + std::to_string(bytes) +
                          " B of shared memory per block: " + cudaGetErrorString(e));
  }
  return ffi::Error::Success();
}

static ffi::Error bt_launched() {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
  return ffi::Error::Success();
}

static ffi::Error FactorImplCuda(cudaStream_t stream, ffi::Buffer<ffi::F64> diag,
                                 ffi::Buffer<ffi::F64> sup, ffi::Buffer<ffi::F64> sub,
                                 ffi::ResultBuffer<ffi::F64> lu,
                                 ffi::ResultBuffer<ffi::S32> perm) {
  auto dims = diag.dimensions();
  const int nd = static_cast<int>(dims.size());
  if (nd < 3 || dims[nd - 1] != dims[nd - 2]) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "diag must be [..., nz, ni, ni]");
  }
  const int nz = static_cast<int>(dims[nd - 3]);
  const int ni = static_cast<int>(dims[nd - 1]);
  int64_t batch = 1;
  for (int i = 0; i < nd - 3; ++i) batch *= dims[i];
  if (sup.element_count() != batch * (nz - 1) * ni || sub.element_count() != sup.element_count()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sup/sub must be [..., nz-1, ni]");
  }
  if (batch == 0) return ffi::Error::Success();
  const size_t shmem = (2 * static_cast<size_t>(ni) * ni + 64) * sizeof(double) +
                       (32 + static_cast<size_t>(ni)) * sizeof(int32_t);
  ffi::Error err = bt_shared_opt_in(reinterpret_cast<const void*>(bt_factor_kernel), shmem);
  if (err.failure()) return err;
  bt_factor_kernel<<<static_cast<unsigned int>(batch), bt_threads(ni), shmem, stream>>>(
      diag.typed_data(), sup.typed_data(), sub.typed_data(), lu->typed_data(),
      perm->typed_data(), nz, ni);
  return bt_launched();
}

static ffi::Error SolveImplCuda(cudaStream_t stream, ffi::Buffer<ffi::F64> lu,
                                ffi::Buffer<ffi::S32> perm, ffi::Buffer<ffi::F64> sup,
                                ffi::Buffer<ffi::F64> sub, ffi::Buffer<ffi::F64> rhs,
                                ffi::ResultBuffer<ffi::F64> x) {
  auto dims = lu.dimensions();
  const int nd = static_cast<int>(dims.size());
  if (nd < 3 || dims[nd - 1] != dims[nd - 2]) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lu must be [..., nz, ni, ni]");
  }
  const int nz = static_cast<int>(dims[nd - 3]);
  const int ni = static_cast<int>(dims[nd - 1]);
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  auto rd = rhs.dimensions();
  const int rnd = static_cast<int>(rd.size());
  if (rnd < 2 || rd[rnd - 1] != ni || rd[rnd - 2] != nz) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "rhs must be [..., nz, ni]");
  }
  BtBcast s = {};
  s.nd = rnd - 2;
  if (s.nd > kBtMaxBatch) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "rhs has more than 4 leading dimensions");
  }
  int64_t batch = 1;
  for (int k = 0; k < s.nd; ++k) {
    s.dim[k] = rd[k];
    batch *= rd[k];
  }
  if (x->element_count() != batch * nz * ni) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "the result must have the rhs's shape");
  }
  const int64_t bnd = static_cast<int64_t>(nz - 1) * ni;
  ffi::Error err = bt_leading(&s, 0, dims, 3, nz * blk, lu.element_count(), "lu");
  if (err.failure()) return err;
  err = bt_leading(&s, 1, perm.dimensions(), 2, static_cast<int64_t>(nz) * ni,
                   perm.element_count(), "perm");
  if (err.failure()) return err;
  err = bt_leading(&s, 2, sup.dimensions(), 2, bnd, sup.element_count(), "sup");
  if (err.failure()) return err;
  err = bt_leading(&s, 3, sub.dimensions(), 2, bnd, sub.element_count(), "sub");
  if (err.failure()) return err;
  if (batch == 0) return ffi::Error::Success();
  const size_t shmem = (2 * static_cast<size_t>(ni) * ni + 2 * ni) * sizeof(double);
  err = bt_shared_opt_in(reinterpret_cast<const void*>(bt_solve_kernel), shmem);
  if (err.failure()) return err;
  bt_solve_kernel<<<static_cast<unsigned int>(batch), bt_threads(ni), shmem, stream>>>(
      lu.typed_data(), perm.typed_data(), sup.typed_data(), sub.typed_data(),
      rhs.typed_data(), x->typed_data(), nz, ni, s);
  return bt_launched();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtFactorCuda, FactorImplCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtSolveCuda, SolveImplCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

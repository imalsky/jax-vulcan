// GPU twin of block_thomas_cpu.cc: the fused block-Thomas factor / solve for
// VULCAN_JAX_SOLVER=ffi on a CUDA device. Same math, same [..., nz, ni, ni]
// layout, same perm convention (perm[i] = source row of permuted row i, so
// x = b[perm]), which is what lets solver_fast's transpose_solve keep running
// the JAX sweep on these factors. One thread block per lane (a batch element):
// the ni x ni block and inv(A'_{j-1}) live in dynamic shared memory, pivoting
// happens in-block, and the whole nz loop runs inside the kernel, so a Ros2
// step is one factor launch and two solve launches. Each layer's triangular
// substitutions are column-oriented, so the whole thread block works on them
// instead of one thread. A zero pivot gives inf/nan like lax.linalg.lu, not an
// error.
// Build: python -m vulcan_jax.solver_fast --cuda (nvcc, on the GPU host); the
// library registers under platform="CUDA".
#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// x <- A^{-1} b from one block's LU and perm, by the whole thread block.
// Column-oriented: once x_i is final every thread updates its own rows. Each
// element takes exactly the subtractions the row-oriented lu_solve in
// block_thomas_cpu.cc gives it, in the same order forward (increasing i) and
// in reverse backward (decreasing i), so only the back substitution differs
// from the CPU reference, by that reassociation. tmp must be shared, x may be
// shared or global, and b may alias x: b is consumed into tmp before x is
// written. Every barrier is reached by all threads (uniform loop bounds, no
// early return) and no thread reads what another writes in the same phase.
__device__ void bt_lu_solve_block(const double* lu, const int32_t* perm, int ni,
                                  const double* b, double* x, double* tmp,
                                  int tid, int nt) {
  for (int i = tid; i < ni; i += nt) tmp[i] = b[perm[i]];
  __syncthreads();
  for (int i = 0; i < ni; ++i) {  // unit-lower L: tmp[i] is final
    const double xi = tmp[i];
    for (int r = i + 1 + tid; r < ni; r += nt) tmp[r] -= lu[r * ni + i] * xi;
    __syncthreads();
  }
  for (int i = ni - 1; i >= 0; --i) {  // upper U
    const double xi = tmp[i] / lu[i * ni + i];
    if (tid == 0) x[i] = xi;
    for (int r = tid; r < i; r += nt) tmp[r] -= lu[r * ni + i] * xi;
    __syncthreads();
  }
}

// In-place partial-pivot LU of the ni x ni shared-memory block A, by the whole
// thread block; sperm gets lu_inplace's permutation. The pivot is the CPU
// kernel's: rows k+1.. beat |A[k][k]| only on a strict >, so the smallest index
// wins a tie, and a NaN candidate below row k loses (masked to -1) exactly as
// the CPU scan's `v > best` skips it. Every parallel section is a strided loop,
// so the kernel is also correct when run serially with one thread.
__device__ void bt_lu_block(double* A, int32_t* sperm, double* rval, int32_t* ridx,
                            int ni, int tid, int nt) {
  for (int i = tid; i < ni; i += nt) sperm[i] = i;
  __syncthreads();
  for (int k = 0; k < ni; ++k) {
    for (int i = tid; i < ni; i += nt) {
      double v = -1.0;
      if (i > k) {
        const double a = fabs(A[i * ni + k]);
        if (a == a) v = a;  // NaN never wins
      }
      rval[i] = v;
      ridx[i] = i;
    }
    __syncthreads();
    for (int n = ni; n > 1;) {  // argmax over rows > k, smallest index on a tie
      const int half = (n + 1) / 2;
      for (int i = tid; i < n - half; i += nt) {
        const double vb = rval[i + half];
        if (vb > rval[i] || (vb == rval[i] && ridx[i + half] < ridx[i])) {
          rval[i] = vb;
          ridx[i] = ridx[i + half];
        }
      }
      __syncthreads();
      n = half;
    }
    const int p = (rval[0] > fabs(A[k * ni + k])) ? ridx[0] : k;
    if (p != k) {
      for (int j = tid; j < ni; j += nt) {  // full rows, L columns included
        const double t = A[k * ni + j];
        A[k * ni + j] = A[p * ni + j];
        A[p * ni + j] = t;
      }
      if (tid == 0) {
        const int32_t t = sperm[k];
        sperm[k] = sperm[p];
        sperm[p] = t;
      }
    }
    __syncthreads();
    const double piv = A[k * ni + k];
    for (int i = k + 1 + tid; i < ni; i += nt) A[i * ni + k] /= piv;
    __syncthreads();
    for (int i = k + 1; i < ni; ++i) {  // rank-1 update, columns over threads
      const double l = A[i * ni + k];
      for (int j = k + 1 + tid; j < ni; j += nt) A[i * ni + j] -= l * A[k * ni + j];
    }
    __syncthreads();
  }
}

__global__ void bt_factor_kernel(const double* diag, const double* sup, const double* sub,
                                 double* lu, int32_t* perm, int nz, int ni) {
  extern __shared__ double bt_smem[];
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  double* A = bt_smem;          // A'_j, factored in place
  double* Minv = A + blk;       // inv(A'_{j-1})
  double* rval = Minv + blk;    // pivot-search scratch
  int32_t* ridx = reinterpret_cast<int32_t*>(rval + ni);
  int32_t* sperm = ridx + ni;

  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t b = blockIdx.x;
  const double* D = diag + b * nz * blk;
  const double* S = sup + b * static_cast<int64_t>(nz - 1) * ni;
  const double* C = sub + b * static_cast<int64_t>(nz - 1) * ni;
  double* L = lu + b * nz * blk;
  int32_t* P = perm + b * static_cast<int64_t>(nz) * ni;

  for (int64_t i = tid; i < blk; i += nt) A[i] = D[i];
  __syncthreads();
  bt_lu_block(A, sperm, rval, ridx, ni, tid, nt);
  for (int64_t i = tid; i < blk; i += nt) L[i] = A[i];
  for (int i = tid; i < ni; i += nt) P[i] = sperm[i];

  for (int j = 1; j < nz; ++j) {
    __syncthreads();
    // inv(A'_{j-1}) column by column, one column per thread: column s is the
    // solve of A'_{j-1} x = e_s, with the rhs permuted (e_s[perm]) like
    // lu_solve's, and the column doubling as lu_solve's tmp before it holds x
    // (the CPU kernel's b-aliases-x case).
    for (int s = tid; s < ni; s += nt) {
      for (int i = 0; i < ni; ++i) Minv[i * ni + s] = (sperm[i] == s) ? 1.0 : 0.0;
      for (int i = 0; i < ni; ++i) {
        double v = Minv[i * ni + s];
        for (int q = 0; q < i; ++q) v -= A[i * ni + q] * Minv[q * ni + s];
        Minv[i * ni + s] = v;
      }
      for (int i = ni - 1; i >= 0; --i) {
        double v = Minv[i * ni + s];
        for (int q = i + 1; q < ni; ++q) v -= A[i * ni + q] * Minv[q * ni + s];
        Minv[i * ni + s] = v / A[i * ni + i];
      }
    }
    __syncthreads();
    const double* Dj = D + static_cast<int64_t>(j) * blk;
    const double* c = C + static_cast<int64_t>(j - 1) * ni;
    const double* bb = S + static_cast<int64_t>(j - 1) * ni;
    for (int r = 0; r < ni; ++r) {  // A'_j = D_j - (c b^T) .* inv(A'_{j-1})
      const double cr = c[r];
      const int64_t row = static_cast<int64_t>(r) * ni;
      for (int s = tid; s < ni; s += nt)
        A[row + s] = Dj[row + s] - (cr * bb[s]) * Minv[row + s];
    }
    __syncthreads();
    bt_lu_block(A, sperm, rval, ridx, ni, tid, nt);
    double* Lj = L + static_cast<int64_t>(j) * blk;
    for (int64_t i = tid; i < blk; i += nt) Lj[i] = A[i];
    for (int i = tid; i < ni; i += nt) P[static_cast<int64_t>(j) * ni + i] = sperm[i];
  }
}

// Stage one layer's LU block and permutation in shared memory.
__device__ void bt_load_block(double* A, int32_t* sp, const double* L, const int32_t* P,
                              int ni, int tid, int nt) {
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  for (int64_t i = tid; i < blk; i += nt) A[i] = L[i];
  for (int i = tid; i < ni; i += nt) sp[i] = P[i];
  __syncthreads();
}

__global__ void bt_solve_kernel(const double* lu, const int32_t* perm, const double* sup,
                                const double* sub, const double* rhs, double* x,
                                int nz, int ni) {
  extern __shared__ double bt_smem[];
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  double* A = bt_smem;  // the current layer's LU block
  double* t = A + blk;
  double* u = t + ni;
  int32_t* sp = reinterpret_cast<int32_t*>(u + ni);

  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t b = blockIdx.x;
  const double* L = lu + b * nz * blk;
  const int32_t* P = perm + b * static_cast<int64_t>(nz) * ni;
  const double* S = sup + b * static_cast<int64_t>(nz - 1) * ni;
  const double* C = sub + b * static_cast<int64_t>(nz - 1) * ni;
  const double* R = rhs + b * static_cast<int64_t>(nz) * ni;
  double* X = x + b * static_cast<int64_t>(nz) * ni;

  for (int i = tid; i < ni; i += nt) X[i] = R[i];  // X holds r' while sweeping
  __syncthreads();
  for (int j = 1; j < nz; ++j) {
    bt_load_block(A, sp, L + static_cast<int64_t>(j - 1) * blk,
                  P + static_cast<int64_t>(j - 1) * ni, ni, tid, nt);
    bt_lu_solve_block(A, sp, ni, X + static_cast<int64_t>(j - 1) * ni, t, u, tid, nt);
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      X[o] = R[o] - C[static_cast<int64_t>(j - 1) * ni + i] * t[i];
    }
    __syncthreads();
  }
  bt_load_block(A, sp, L + static_cast<int64_t>(nz - 1) * blk,
                P + static_cast<int64_t>(nz - 1) * ni, ni, tid, nt);
  double* Xl = X + static_cast<int64_t>(nz - 1) * ni;
  bt_lu_solve_block(A, sp, ni, Xl, Xl, u, tid, nt);  // b aliases x, as in the CPU sweep
  for (int j = nz - 2; j >= 0; --j) {
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      t[i] = X[o] - S[o] * X[o + ni];
    }
    bt_load_block(A, sp, L + static_cast<int64_t>(j) * blk,
                  P + static_cast<int64_t>(j) * ni, ni, tid, nt);
    bt_lu_solve_block(A, sp, ni, t, X + static_cast<int64_t>(j) * ni, u, tid, nt);
  }
}

static int bt_threads(int ni) {
  const int t = ((ni + 31) / 32) * 32;
  return t > 1024 ? 1024 : (t < 32 ? 32 : t);
}

// Opt in to more than the 48 KB default of dynamic shared memory per block
// (a GH200 allows 228 KB): the factor kernel needs 2 ni^2 doubles, 126 KB at
// ni = 89. A block too large for the device fails here, not silently.
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
  const size_t shmem = (2 * static_cast<size_t>(ni) * ni + ni) * sizeof(double) +
                       2 * static_cast<size_t>(ni) * sizeof(int32_t);
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
  int64_t batch = 1;
  for (int i = 0; i < nd - 3; ++i) batch *= dims[i];
  if (rhs.element_count() != batch * nz * ni || perm.element_count() != batch * nz * ni) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "rhs/perm must be [..., nz, ni]");
  }
  if (batch == 0) return ffi::Error::Success();
  const size_t shmem = (static_cast<size_t>(ni) * ni + 2 * ni) * sizeof(double) +
                       static_cast<size_t>(ni) * sizeof(int32_t);
  ffi::Error err = bt_shared_opt_in(reinterpret_cast<const void*>(bt_solve_kernel), shmem);
  if (err.failure()) return err;
  bt_solve_kernel<<<static_cast<unsigned int>(batch), bt_threads(ni), shmem, stream>>>(
      lu.typed_data(), perm.typed_data(), sup.typed_data(), sub.typed_data(),
      rhs.typed_data(), x->typed_data(), nz, ni);
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

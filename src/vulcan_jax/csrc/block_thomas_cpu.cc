// CPU reference of the fused block-Thomas factor / solve (solver_fast.py,
// VULCAN_JAX_SOLVER=ffi). Same math as solver.py, layer by layer:
//   factor:  A'_0 = A_0;  A'_j = A_j - (c_j b_{j-1}^T) .* inv(A'_{j-1});  pivoted LU of each A'_j
//   solve:   r'_0 = r_0;  r'_j = r_j - c_j .* (A'_{j-1}^{-1} r'_{j-1})
//            k_last = A'^{-1} r'_last;  k_j = A'_j^{-1} (r'_j - b_j .* k_{j+1})
// with b = sup_d (nz-1, ni), c = sub_d (nz-1, ni). Leading dimensions are a
// batch; in the solve the factors' leading dimensions broadcast against the
// rhs's (equal, or 1: one factorisation, a stack of right-hand sides). Plain
// O(nz ni^3) loops: this is the reference the GPU kernel is checked against,
// not a fast CPU solver. A zero pivot gives inf/nan like lax.linalg.lu, not an
// error.
// Build: python -m vulcan_jax.solver_fast
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// In-place partial-pivot LU of an n x n row-major block. perm[i] = source row
// of permuted row i (lax.linalg.lu's convention: x = b[perm]).
static void lu_inplace(double* a, int32_t* perm, int n) {
  for (int i = 0; i < n; ++i) perm[i] = i;
  for (int k = 0; k < n; ++k) {
    int p = k;
    double best = std::fabs(a[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double v = std::fabs(a[i * n + k]);
      if (v > best) { best = v; p = i; }
    }
    if (p != k) {
      for (int j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
      std::swap(perm[k], perm[p]);
    }
    const double piv = a[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      const double l = (a[i * n + k] /= piv);
      for (int j = k + 1; j < n; ++j) a[i * n + j] -= l * a[k * n + j];
    }
  }
}

// x <- A^{-1} b from the block's LU and perm. b may alias x: b is consumed
// into tmp before x is written.
static void lu_solve(const double* lu, const int32_t* perm, int n,
                     const double* b, double* x, double* tmp) {
  for (int i = 0; i < n; ++i) tmp[i] = b[perm[i]];
  for (int i = 0; i < n; ++i) {
    double s = tmp[i];
    for (int j = 0; j < i; ++j) s -= lu[i * n + j] * tmp[j];
    tmp[i] = s;
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = tmp[i];
    for (int j = i + 1; j < n; ++j) s -= lu[i * n + j] * x[j];
    x[i] = s / lu[i * n + i];
  }
}

static ffi::Error FactorImpl(ffi::Buffer<ffi::F64> diag, ffi::Buffer<ffi::F64> sup,
                             ffi::Buffer<ffi::F64> sub, ffi::ResultBuffer<ffi::F64> lu,
                             ffi::ResultBuffer<ffi::S32> perm) {
  auto dims = diag.dimensions();
  const int nd = static_cast<int>(dims.size());
  if (nd < 3 || dims[nd - 1] != dims[nd - 2]) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "diag must be [..., nz, ni, ni]");
  }
  const int nz = static_cast<int>(dims[nd - 3]);
  const int ni = static_cast<int>(dims[nd - 1]);
  const int64_t blk = static_cast<int64_t>(ni) * ni;
  int64_t batch = 1;
  for (int i = 0; i < nd - 3; ++i) batch *= dims[i];
  if (sup.element_count() != batch * (nz - 1) * ni || sub.element_count() != sup.element_count()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sup/sub must be [..., nz-1, ni]");
  }
  std::vector<double> inv(blk), e(ni), col(ni), tmp(ni);
  for (int64_t b = 0; b < batch; ++b) {
    const double* D = diag.typed_data() + b * nz * blk;
    const double* S = sup.typed_data() + b * (nz - 1) * ni;
    const double* C = sub.typed_data() + b * (nz - 1) * ni;
    double* L = lu->typed_data() + b * nz * blk;
    int32_t* P = perm->typed_data() + b * nz * ni;
    for (int64_t i = 0; i < blk; ++i) L[i] = D[i];
    lu_inplace(L, P, ni);
    for (int j = 1; j < nz; ++j) {
      const double* Lp = L + (j - 1) * blk;
      const int32_t* Pp = P + (j - 1) * ni;
      for (int s = 0; s < ni; ++s) {  // inv(A'_{j-1}) column by column
        for (int i = 0; i < ni; ++i) e[i] = (i == s) ? 1.0 : 0.0;
        lu_solve(Lp, Pp, ni, e.data(), col.data(), tmp.data());
        for (int r = 0; r < ni; ++r) inv[r * ni + s] = col[r];
      }
      double* Lj = L + j * blk;
      const double* Dj = D + j * blk;
      const double* c = C + (j - 1) * ni;
      const double* bb = S + (j - 1) * ni;
      for (int r = 0; r < ni; ++r)
        for (int s = 0; s < ni; ++s)
          Lj[r * ni + s] = Dj[r * ni + s] - (c[r] * bb[s]) * inv[r * ni + s];
      lu_inplace(Lj, P + j * ni, ni);
    }
  }
  return ffi::Error::Success();
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
static inline int64_t bt_index(int64_t b, const BtBcast& s, int op) {
  int64_t idx = 0;
  for (int k = s.nd - 1; k >= 0; --k) {
    idx += (b % s.dim[k]) * s.stride[op][k];
    b /= s.dim[k];
  }
  return idx;
}

static ffi::Error SolveImpl(ffi::Buffer<ffi::F64> lu, ffi::Buffer<ffi::S32> perm,
                            ffi::Buffer<ffi::F64> sup, ffi::Buffer<ffi::F64> sub,
                            ffi::Buffer<ffi::F64> rhs, ffi::ResultBuffer<ffi::F64> x) {
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
  std::vector<double> t(ni), tmp(ni);
  for (int64_t b = 0; b < batch; ++b) {
    const double* L = lu.typed_data() + bt_index(b, s, 0) * nz * blk;
    const int32_t* P = perm.typed_data() + bt_index(b, s, 1) * nz * ni;
    const double* S = sup.typed_data() + bt_index(b, s, 2) * bnd;
    const double* C = sub.typed_data() + bt_index(b, s, 3) * bnd;
    const double* R = rhs.typed_data() + b * nz * ni;
    double* X = x->typed_data() + b * nz * ni;
    for (int i = 0; i < ni; ++i) X[i] = R[i];  // X holds r' during the forward sweep
    for (int j = 1; j < nz; ++j) {
      lu_solve(L + (j - 1) * blk, P + (j - 1) * ni, ni, X + (j - 1) * ni, t.data(), tmp.data());
      for (int i = 0; i < ni; ++i) X[j * ni + i] = R[j * ni + i] - C[(j - 1) * ni + i] * t[i];
    }
    lu_solve(L + (nz - 1) * blk, P + (nz - 1) * ni, ni, X + (nz - 1) * ni, X + (nz - 1) * ni, tmp.data());
    for (int j = nz - 2; j >= 0; --j) {
      for (int i = 0; i < ni; ++i) t[i] = X[j * ni + i] - S[j * ni + i] * X[(j + 1) * ni + i];
      lu_solve(L + j * blk, P + j * ni, ni, t.data(), X + j * ni, tmp.data());
    }
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtFactor, FactorImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VulcanBtSolve, SolveImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

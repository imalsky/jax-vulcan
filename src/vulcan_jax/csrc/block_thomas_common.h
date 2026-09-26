// Shape checks and rhs broadcasting shared by block_thomas_cpu.cc and
// block_thomas_cuda.cu. Each library is one translation unit.
#ifndef VULCAN_JAX_CSRC_BLOCK_THOMAS_COMMON_H_
#define VULCAN_JAX_CSRC_BLOCK_THOMAS_COMMON_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

#include "xla/ffi/api/ffi.h"

#ifdef __CUDACC__
#define BT_HOST_DEVICE __host__ __device__
#else
#define BT_HOST_DEVICE
#endif

namespace vulcan_bt {

namespace ffi = xla::ffi;

// Most rhs leading (batch) dimensions a solve takes. BtBcast is passed to the
// CUDA kernel by value, so its arrays have a fixed size.
inline constexpr int kBtMaxBatch = 4;

// The factor operands of a solve, in handler order.
enum BtOperand { kBtLu, kBtPerm, kBtSup, kBtSub, kBtNumOperands };

// The rhs leading dimensions and, per factor operand, the stride of each one
// in operand elements (0 where the operand has size 1 and broadcasts).
struct BtBcast {
  int nd;
  int64_t dim[kBtMaxBatch];
  int64_t stride[kBtNumOperands][kBtMaxBatch];
};

// One call: `batch` independent systems of nz layers of ni x ni blocks.
struct BtShape {
  int nz;
  int ni;
  int64_t batch;
};

using Dims = ffi::Span<const int64_t>;

// True if `dims` is `lead` followed by `trail`.
inline bool bt_shape_is(Dims dims, Dims lead, std::initializer_list<int64_t> trail) {
  return dims.size() == lead.size() + trail.size() && dims.first(lead.size()) == lead &&
         std::equal(trail.begin(), trail.end(), dims.begin() + lead.size());
}

// nz and ni of a [..., nz, ni, ni] operand, narrowed to int after a range check.
inline ffi::Error bt_blocks(Dims dims, const char* name, BtShape* shape) {
  const size_t nd = dims.size();
  if (nd < 3 || dims[nd - 1] != dims[nd - 2] || dims[nd - 3] < 1) {
    return ffi::Error::InvalidArgument(std::string(name) + " must be [..., nz, ni, ni], nz >= 1");
  }
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  if (dims[nd - 3] > kIntMax || dims[nd - 1] > kIntMax) {
    return ffi::Error::InvalidArgument(std::string(name) + ": nz and ni must fit in an int");
  }
  shape->nz = static_cast<int>(dims[nd - 3]);
  shape->ni = static_cast<int>(dims[nd - 1]);
  return ffi::Error::Success();
}

// Checks every factor operand: diag and lu [..., nz, ni, ni], sup and sub
// [..., nz-1, ni], perm [..., nz, ni], one set of leading dimensions (the
// batch). A mismatch is an error, never an out-of-bounds access.
inline ffi::Error bt_check_factor(const ffi::Buffer<ffi::F64>& diag,
                                  const ffi::Buffer<ffi::F64>& sup,
                                  const ffi::Buffer<ffi::F64>& sub,
                                  const ffi::Buffer<ffi::F64>& lu,
                                  const ffi::Buffer<ffi::S32>& perm, BtShape* shape) {
  const Dims dims = diag.dimensions();
  ffi::Error err = bt_blocks(dims, "diag", shape);
  if (err.failure()) return err;
  const int64_t nz = shape->nz, ni = shape->ni;
  const Dims lead = dims.first(dims.size() - 3);
  if (!bt_shape_is(sup.dimensions(), lead, {nz - 1, ni}) ||
      !bt_shape_is(sub.dimensions(), lead, {nz - 1, ni})) {
    return ffi::Error::InvalidArgument("sup/sub must be [..., nz-1, ni]");
  }
  if (!(lu.dimensions() == dims)) {
    return ffi::Error::InvalidArgument("lu must have diag's shape");
  }
  if (!bt_shape_is(perm.dimensions(), lead, {nz, ni})) {
    return ffi::Error::InvalidArgument("perm must be [..., nz, ni]");
  }
  shape->batch = 1;
  for (const int64_t d : lead) shape->batch *= d;
  return ffi::Error::Success();
}

// Checks factor operand `op` of a solve: its leading dimensions (as many as
// the rhs has) each equal the rhs's or are 1, followed by `trail`. Records the
// strides in `s`.
inline ffi::Error bt_leading(BtBcast* s, BtOperand op, Dims dims,
                             std::initializer_list<int64_t> trail, const char* name) {
  const size_t nb = static_cast<size_t>(s->nd);
  if (dims.size() != nb + trail.size()) {
    return ffi::Error::InvalidArgument(std::string(name) +
                                       " must have the rhs's number of leading dimensions");
  }
  if (!std::equal(trail.begin(), trail.end(), dims.begin() + nb)) {
    return ffi::Error::InvalidArgument(std::string(name) + " has the wrong trailing shape");
  }
  int64_t stride = 1;
  for (int k = s->nd - 1; k >= 0; --k) {
    if (dims[k] != s->dim[k] && dims[k] != 1) {
      return ffi::Error::InvalidArgument(std::string(name) +
                                         " leading dimensions must equal the rhs's or be 1");
    }
    s->stride[op][k] = (dims[k] == 1) ? 0 : stride;
    stride *= dims[k];
  }
  return ffi::Error::Success();
}

// Checks every solve operand: lu [..., nz, ni, ni], perm [..., nz, ni], sup and
// sub [..., nz-1, ni], rhs and x [..., nz, ni]. The batch is the product of the
// rhs leading dimensions; the factors' leading dimensions broadcast against
// them (equal, or 1), which is what vmap_method="expand_dims" produces when
// only the rhs is batched at a vmap level.
inline ffi::Error bt_check_solve(const ffi::Buffer<ffi::F64>& lu,
                                 const ffi::Buffer<ffi::S32>& perm,
                                 const ffi::Buffer<ffi::F64>& sup,
                                 const ffi::Buffer<ffi::F64>& sub,
                                 const ffi::Buffer<ffi::F64>& rhs,
                                 const ffi::Buffer<ffi::F64>& x, BtShape* shape, BtBcast* s) {
  ffi::Error err = bt_blocks(lu.dimensions(), "lu", shape);
  if (err.failure()) return err;
  const int64_t nz = shape->nz, ni = shape->ni;
  const Dims rd = rhs.dimensions();
  if (rd.size() < 2 || rd[rd.size() - 2] != nz || rd[rd.size() - 1] != ni) {
    return ffi::Error::InvalidArgument("rhs must be [..., nz, ni]");
  }
  if (rd.size() - 2 > static_cast<size_t>(kBtMaxBatch)) {
    return ffi::Error::InvalidArgument("rhs has more than " + std::to_string(kBtMaxBatch) +
                                       " leading dimensions");
  }
  if (!(x.dimensions() == rd)) {
    return ffi::Error::InvalidArgument("the result must have the rhs's shape");
  }
  *s = BtBcast{};
  s->nd = static_cast<int>(rd.size() - 2);
  shape->batch = 1;
  for (int k = 0; k < s->nd; ++k) {
    s->dim[k] = rd[k];
    shape->batch *= rd[k];
  }
  err = bt_leading(s, kBtLu, lu.dimensions(), {nz, ni, ni}, "lu");
  if (err.failure()) return err;
  err = bt_leading(s, kBtPerm, perm.dimensions(), {nz, ni}, "perm");
  if (err.failure()) return err;
  err = bt_leading(s, kBtSup, sup.dimensions(), {nz - 1, ni}, "sup");
  if (err.failure()) return err;
  return bt_leading(s, kBtSub, sub.dimensions(), {nz - 1, ni}, "sub");
}

// Element of factor operand `op` that rhs batch element `b` reads, in units of
// that operand's per-system size.
BT_HOST_DEVICE inline int64_t bt_index(int64_t b, const BtBcast& s, BtOperand op) {
  int64_t idx = 0;
  for (int k = s.nd - 1; k >= 0; --k) {
    idx += (b % s.dim[k]) * s.stride[op][k];
    b /= s.dim[k];
  }
  return idx;
}

}  // namespace vulcan_bt

#endif  // VULCAN_JAX_CSRC_BLOCK_THOMAS_COMMON_H_

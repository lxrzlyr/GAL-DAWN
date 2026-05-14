#pragma once

#include <dawn/common/status.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/memory/buffer.hxx>

namespace DAWN {
namespace Adapters {

struct KernelCsrGraph {
  HostBuffer<int> row_ptr;
  HostBuffer<int> col;
  HostBuffer<float> val;
  int rows = 0;
  int nnz = 0;
};

StatusOr<KernelCsrGraph> MakeKernelCsrGraph(const CsrView& view);

}  // namespace Adapters
}  // namespace DAWN

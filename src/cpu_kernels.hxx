#pragma once

#include <dawn/common/status.hxx>
#include <dawn/common/types.hxx>
#include <dawn/graph/csr_graph.hxx>

#include <vector>

namespace DAWN {
namespace Kernel {
namespace CPU {

struct BfsOptions {
  VertexId source = 0;
  int thread_count = 1;
};

struct BfsResult {
  std::vector<int> distances;
};

struct SsspOptions {
  VertexId source = 0;
  int thread_count = 1;
};

struct SsspResult {
  std::vector<Weight> distances;
};

struct BcOptions {
  int thread_count = 1;
};

struct BcResult {
  std::vector<float> centrality;
};

StatusOr<BfsResult> RunBfs(CsrView graph, const BfsOptions& options);
StatusOr<SsspResult> RunSssp(CsrView graph, const SsspOptions& options);
StatusOr<BcResult> RunBc(CsrView graph, const BcOptions& options);

}  // namespace CPU
}  // namespace Kernel
}  // namespace DAWN

#pragma once

#include <string>

#include <dawn/common/status.hxx>
#include <dawn/graph/csr_graph.hxx>

namespace DAWN {
namespace IO {

struct MatrixMarketOptions {
  bool ignore_self_loops = true;
};

struct DegreeSummary {
  EdgeId min_degree = 0;
  EdgeId max_degree = 0;
  double average_degree = 0.0;
};

StatusOr<HostCsrGraph> ReadMatrixMarket(
    const std::string& input_path,
    const MatrixMarketOptions& options = MatrixMarketOptions());

DegreeSummary ComputeDegreeSummary(CsrView view);

}  // namespace IO
}  // namespace DAWN

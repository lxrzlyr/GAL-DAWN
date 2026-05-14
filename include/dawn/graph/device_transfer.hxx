#pragma once

#include <dawn/common/status.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/runtime/handle.hxx>

namespace DAWN {

StatusOr<DeviceCsrGraph> to_device(dawnHandle_t handle,
                                   const HostCsrGraph& graph);

}  // namespace DAWN

#pragma once

#include <string>
#include <vector>

#include <dawn/common/config.hxx>
#include <dawn/common/status.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/runtime/handle.hxx>

namespace DAWN {

struct DeviceProfile {
  BackendKind backend = BackendKind::kCpu;
  int device_id = 0;
  std::string name = "cpu";
  int64_t total_memory_bytes = 0;
  int64_t free_memory_bytes = 0;
};

struct SelectedDevice {
  BackendKind backend = BackendKind::kCpu;
  int device_id = 0;
  std::string reason;
};

std::vector<DeviceProfile> EnumerateDevices();
StatusOr<SelectedDevice> SelectDevice(dawnHandle_t handle,
                                      const HostCsrGraph& graph);

}  // namespace DAWN

#include <dawn/graph/device_transfer.hxx>

#include <cuda_runtime.h>

namespace DAWN {

StatusOr<DeviceCsrGraph> to_device(dawnHandle_t, const HostCsrGraph&) {
  return Status(
      StatusCode::kUnsupported,
      "CUDA graph transfer is reserved for the CUDA Stage2 build path");
}

}  // namespace DAWN

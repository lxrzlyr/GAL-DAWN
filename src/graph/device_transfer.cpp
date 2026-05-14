#include <dawn/graph/device_transfer.hxx>

namespace DAWN {

StatusOr<DeviceCsrGraph> to_device(dawnHandle_t, const HostCsrGraph&) {
  return Status(StatusCode::kUnavailable,
                "DAWN was built without CUDA device graph transfer support");
}

}  // namespace DAWN

#include <dawn/runtime/device_selector.hxx>

#if DAWN_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace DAWN {

std::vector<DeviceProfile> EnumerateDevices() {
  std::vector<DeviceProfile> devices;
  devices.push_back(DeviceProfile{BackendKind::kCpu, 0, "cpu", 0, 0});
#if DAWN_HAS_CUDA
  int count = 0;
  if (cudaGetDeviceCount(&count) == cudaSuccess) {
    for (int i = 0; i < count; ++i) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
        devices.push_back(
            DeviceProfile{BackendKind::kCuda, i, prop.name,
                          static_cast<int64_t>(prop.totalGlobalMem), 0});
      }
    }
  }
#endif
  return devices;
}

StatusOr<SelectedDevice> SelectDevice(dawnHandle_t handle,
                                      const HostCsrGraph&) {
  const BackendPolicy policy = dawnGetBackendPolicy(handle);
#if DAWN_HAS_CUDA
  const int requested_device = dawnGetDeviceId(handle);
#endif

  switch (policy) {
    case BackendPolicy::kCpu:
      return SelectedDevice{BackendKind::kCpu, 0, "explicit cpu backend"};
    case BackendPolicy::kAuto:
    case BackendPolicy::kBest:
#if DAWN_HAS_CUDA
      if (requested_device >= 0) {
        int count = 0;
        if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
          return SelectedDevice{BackendKind::kCuda, requested_device,
                                "cuda available"};
        }
      }
#endif
      return SelectedDevice{BackendKind::kCpu, 0,
                            "cpu selected by auto policy"};
    case BackendPolicy::kCuda:
#if DAWN_HAS_CUDA
    {
      int count = 0;
      if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return Status(StatusCode::kUnavailable,
                      "CUDA backend requested but no CUDA device is available");
      }
      if (requested_device < 0 || requested_device >= count) {
        return Status(StatusCode::kOutOfRange,
                      "requested CUDA device id is invalid");
      }
      return SelectedDevice{BackendKind::kCuda, requested_device,
                            "explicit cuda backend"};
    }
#else
      (void)handle;
      return Status(StatusCode::kUnavailable,
                    "CUDA backend requested but DAWN was built without CUDA");
#endif
  }
  return Status(StatusCode::kInvalidArgument, "unknown backend policy");
}

}  // namespace DAWN

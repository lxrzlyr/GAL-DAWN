#include <dawn/runtime/handle.hxx>

namespace DAWN {

struct dawnContext {
  RuntimeOptions options;
};

StatusOr<dawnHandle_t> dawnCreate(const RuntimeOptions& options) {
  dawnContext* context = new dawnContext;
  context->options = options;
  if (context->options.cpu_threading.thread_count <= 0) {
    context->options.cpu_threading.thread_count = 1;
  }
  return context;
}

Status dawnDestroy(dawnHandle_t handle) {
  delete handle;
  return Status::OK();
}

Status dawnSetBackendPolicy(dawnHandle_t handle, BackendPolicy policy) {
  if (handle == nullptr) {
    return Status(StatusCode::kInvalidArgument, "runtime handle is null");
  }
  handle->options.device_policy.backend_policy = policy;
  return Status::OK();
}

Status dawnSetDeviceId(dawnHandle_t handle, int device_id) {
  if (handle == nullptr) {
    return Status(StatusCode::kInvalidArgument, "runtime handle is null");
  }
  if (device_id < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "device id must be non-negative");
  }
  handle->options.device_policy.device_id = device_id;
  handle->options.device_policy.explicit_device_id = true;
  return Status::OK();
}

BackendPolicy dawnGetBackendPolicy(dawnHandle_t handle) {
  if (handle == nullptr) {
    return BackendPolicy::kAuto;
  }
  return handle->options.device_policy.backend_policy;
}

int dawnGetDeviceId(dawnHandle_t handle) {
  if (handle == nullptr) {
    return 0;
  }
  return handle->options.device_policy.device_id;
}

int dawnGetCpuThreadCount(dawnHandle_t handle) {
  if (handle == nullptr) {
    return 1;
  }
  return handle->options.cpu_threading.thread_count;
}

StatusOr<BackendPolicy> ParseBackendPolicy(const std::string& name) {
  if (name == "auto") {
    return BackendPolicy::kAuto;
  }
  if (name == "best") {
    return BackendPolicy::kBest;
  }
  if (name == "cpu") {
    return BackendPolicy::kCpu;
  }
  if (name == "cuda" || name == "gpu") {
    return BackendPolicy::kCuda;
  }
  return Status(StatusCode::kInvalidArgument,
                "unknown backend policy: " + name);
}

const char* BackendPolicyName(BackendPolicy policy) {
  switch (policy) {
    case BackendPolicy::kAuto:
      return "auto";
    case BackendPolicy::kBest:
      return "best";
    case BackendPolicy::kCpu:
      return "cpu";
    case BackendPolicy::kCuda:
      return "cuda";
  }
  return "auto";
}

const char* BackendKindName(BackendKind backend) {
  switch (backend) {
    case BackendKind::kCpu:
      return "cpu";
    case BackendKind::kCuda:
      return "cuda";
  }
  return "cpu";
}

}  // namespace DAWN

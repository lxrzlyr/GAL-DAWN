#include <dawn/runtime/handle.hxx>
#include <dawn/runtime/device_selector.hxx>
#include <dawn/runtime/kernel_registry.hxx>

#if DAWN_HAS_CUDA
#include <cuda_runtime.h>
#endif

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

namespace DAWN {

GraphProfile ProfileGraph(CsrView view, int source_count) {
  GraphProfile profile;
  profile.vertex_count = view.num_vertices;
  profile.edge_count = view.num_edges;
  profile.degree_summary = IO::ComputeDegreeSummary(view);
  profile.weighted = view.weighted;
  profile.source_count = source_count;
  return profile;
}

std::vector<KernelDescriptor> BuildDefaultKernelRegistry() {
  std::vector<KernelDescriptor> registry;
  registry.push_back({"cpu_bfs", "bfs", BackendKind::kCpu, false, true});
  registry.push_back({"cpu_sssp", "sssp", BackendKind::kCpu, true, false});
  registry.push_back({"cpu_mssp", "mssp", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_apsp", "apsp", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_cc", "cc", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_bc", "bc", BackendKind::kCpu, false, true});
#if DAWN_HAS_CUDA
  registry.push_back({"cuda_bfs", "bfs", BackendKind::kCuda, false, true});
  registry.push_back({"cuda_sssp", "sssp", BackendKind::kCuda, true, false});
#endif
  return registry;
}

std::vector<KernelDescriptor> ListKernelsForAlgorithm(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm) {
  std::vector<KernelDescriptor> out;
  for (const KernelDescriptor& descriptor : registry) {
    if (algorithm.empty() || descriptor.algorithm == algorithm) {
      out.push_back(descriptor);
    }
  }
  return out;
}

StatusOr<KernelDescriptor> SelectKernel(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm,
    BackendKind backend,
    const GraphProfile& profile,
    const std::string& explicit_kernel) {
  for (const KernelDescriptor& descriptor : registry) {
    if (descriptor.algorithm != algorithm || descriptor.backend != backend) {
      continue;
    }
    if (!explicit_kernel.empty() && descriptor.name != explicit_kernel) {
      continue;
    }
    if (profile.weighted && !descriptor.supports_weighted) {
      continue;
    }
    if (!profile.weighted && !descriptor.supports_unweighted) {
      continue;
    }
    return descriptor;
  }

  if (!explicit_kernel.empty()) {
    return Status(
        StatusCode::kNotFound,
        "kernel override not found or incompatible: " + explicit_kernel);
  }
  return Status(StatusCode::kNotFound,
                "no compatible kernel registered for algorithm " + algorithm);
}

}  // namespace DAWN

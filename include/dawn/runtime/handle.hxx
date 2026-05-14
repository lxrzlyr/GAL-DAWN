#pragma once

#include <memory>
#include <string>

#include <dawn/common/status.hxx>

namespace DAWN {

enum class BackendKind { kCpu, kCuda };

enum class BackendPolicy { kAuto, kBest, kCpu, kCuda };

struct CpuThreadingPolicy {
  int thread_count = 1;
};

struct DevicePolicy {
  BackendPolicy backend_policy = BackendPolicy::kAuto;
  int device_id = 0;
  bool explicit_device_id = false;
};

struct RuntimeOptions {
  DevicePolicy device_policy;
  CpuThreadingPolicy cpu_threading;
};

struct dawnContext;
using dawnHandle_t = dawnContext*;

StatusOr<dawnHandle_t> dawnCreate(const RuntimeOptions& options);
Status dawnDestroy(dawnHandle_t handle);
Status dawnSetBackendPolicy(dawnHandle_t handle, BackendPolicy policy);
Status dawnSetDeviceId(dawnHandle_t handle, int device_id);

BackendPolicy dawnGetBackendPolicy(dawnHandle_t handle);
int dawnGetDeviceId(dawnHandle_t handle);
int dawnGetCpuThreadCount(dawnHandle_t handle);

StatusOr<BackendPolicy> ParseBackendPolicy(const std::string& name);
const char* BackendPolicyName(BackendPolicy policy);
const char* BackendKindName(BackendKind backend);

}  // namespace DAWN

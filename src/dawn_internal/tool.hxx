/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <dawn_internal/openmp_compat.hxx>

namespace DAWN {
namespace Tool {

float average(int* result, int n);  // average value of a list

float average(float* result, int n);

void infoprint(int entry,
               int total,
               int interval,
               int thread,
               float elapsed_time);  // Print current task progress

}  // namespace Tool
}  // namespace DAWN

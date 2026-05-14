#pragma once

#include <dawn/common/config.hxx>

#if DAWN_HAS_OPENMP
#include <omp.h>
#else
inline int omp_get_num_threads() {
  return 1;
}

inline int omp_get_max_threads() {
  return 1;
}

inline int omp_get_thread_num() {
  return 0;
}

inline void omp_set_dynamic(int) {}
#endif

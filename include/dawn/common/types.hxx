#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <dawn/common/status.hxx>

namespace DAWN {

using VertexId = int32_t;
using EdgeId = int64_t;
using Weight = float;

template <typename To, typename From>
StatusOr<To> checked_downcast(From value, const char* field_name) {
  static_assert(std::is_integral<To>::value, "To must be integral");
  static_assert(std::is_integral<From>::value, "From must be integral");

  const int64_t v = static_cast<int64_t>(value);
  if (v < static_cast<int64_t>(std::numeric_limits<To>::min()) ||
      v > static_cast<int64_t>(std::numeric_limits<To>::max())) {
    return Status(StatusCode::kOutOfRange,
                  std::string(field_name) + " cannot be represented by int");
  }
  return static_cast<To>(value);
}

inline StatusOr<int> checked_int_downcast(EdgeId value,
                                          const char* field_name) {
  return checked_downcast<int>(value, field_name);
}

inline StatusOr<int> checked_int_downcast(VertexId value,
                                          const char* field_name) {
  return checked_downcast<int>(value, field_name);
}

}  // namespace DAWN

#pragma once
#include <string>

namespace DAWN {

enum class ParseStatus { Success = 0, Help = 1, Error = 2 };

struct RunResult {
  bool success;
  std::string error_msg;
  float elapsed_time;
  float value;

  static RunResult ok(float elapsed, float val = 0.0f) {
    return {true, "", elapsed, val};
  }

  static RunResult err(const std::string& msg) {
    return {false, msg, 0.0f, 0.0f};
  }
};

}  // namespace DAWN

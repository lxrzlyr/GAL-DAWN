#include <dawn/common/status.hxx>

namespace DAWN {

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }

  const char* code = "internal";
  switch (code_) {
    case StatusCode::kOk:
      code = "ok";
      break;
    case StatusCode::kInvalidArgument:
      code = "invalid argument";
      break;
    case StatusCode::kOutOfRange:
      code = "out of range";
      break;
    case StatusCode::kNotFound:
      code = "not found";
      break;
    case StatusCode::kUnavailable:
      code = "unavailable";
      break;
    case StatusCode::kUnsupported:
      code = "unsupported";
      break;
    case StatusCode::kInternal:
      code = "internal";
      break;
  }
  return message_.empty() ? code : std::string(code) + ": " + message_;
}

}  // namespace DAWN

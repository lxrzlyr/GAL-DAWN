#pragma once

#include <memory>
#include <string>
#include <utility>

namespace DAWN {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kNotFound,
  kUnavailable,
  kUnsupported,
  kInternal
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status OK() { return Status(); }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  std::string ToString() const;

 private:
  StatusCode code_;
  std::string message_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(const Status& status) : status_(status) {}
  StatusOr(Status&& status) : status_(std::move(status)) {}
  StatusOr(const T& value) : status_(Status::OK()), value_(new T(value)) {}
  StatusOr(T&& value)
      : status_(Status::OK()), value_(new T(std::move(value))) {}

  StatusOr(StatusOr&&) noexcept = default;
  StatusOr& operator=(StatusOr&&) noexcept = default;

  StatusOr(const StatusOr&) = delete;
  StatusOr& operator=(const StatusOr&) = delete;

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  T& value() { return *value_; }
  const T& value() const { return *value_; }

  T&& ConsumeValueOrDie() { return std::move(*value_); }

  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }
  T& operator*() { return value(); }
  const T& operator*() const { return value(); }

 private:
  Status status_;
  std::unique_ptr<T> value_;
};

}  // namespace DAWN

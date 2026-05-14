#pragma once

#include <cstddef>

namespace DAWN {

template <typename T>
class Span {
 public:
  Span() : data_(nullptr), size_(0) {}
  Span(T* data, size_t size) : data_(data), size_(size) {}

  T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](size_t index) const { return data_[index]; }
  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

template <typename T>
using ConstSpan = Span<const T>;

}  // namespace DAWN

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>

#include <dawn/common/config.hxx>
#include <dawn/memory/span.hxx>

#if DAWN_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace DAWN {

enum class MemorySpace { kHost, kDevice, kPinned };

template <typename T, MemorySpace Space>
class Buffer;

template <typename T>
class Buffer<T, MemorySpace::kHost> {
 public:
  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size)
      : data_(size ? new T[size] : nullptr), size_(size) {}
  Buffer(size_t size, const T& value) : Buffer(size) { fill(value); }

  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() { reset(); }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

  void fill(const T& value) { std::fill(data_, data_ + size_, value); }

  void reset() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
  }

  T* release() {
    T* out = data_;
    data_ = nullptr;
    size_ = 0;
    return out;
  }

  Span<T> span() { return Span<T>(data_, size_); }
  ConstSpan<T> span() const { return ConstSpan<T>(data_, size_); }

 private:
  T* data_;
  size_t size_;
};

#if DAWN_HAS_CUDA
template <typename T>
class Buffer<T, MemorySpace::kDevice> {
 public:
  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size) : data_(nullptr), size_(size) {
    if (size_) {
      cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * size_);
    }
  }
  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { reset(); }
  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  void reset() {
    if (data_) {
      cudaFree(data_);
    }
    data_ = nullptr;
    size_ = 0;
  }
  T* release() {
    T* out = data_;
    data_ = nullptr;
    size_ = 0;
    return out;
  }

 private:
  T* data_;
  size_t size_;
};

template <typename T>
class Buffer<T, MemorySpace::kPinned> {
 public:
  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size) : data_(nullptr), size_(size) {
    if (size_) {
      cudaMallocHost(reinterpret_cast<void**>(&data_), sizeof(T) * size_);
    }
  }
  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { reset(); }
  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  void reset() {
    if (data_) {
      cudaFreeHost(data_);
    }
    data_ = nullptr;
    size_ = 0;
  }
  T* release() {
    T* out = data_;
    data_ = nullptr;
    size_ = 0;
    return out;
  }

 private:
  T* data_;
  size_t size_;
};
#endif

template <typename T>
using HostBuffer = Buffer<T, MemorySpace::kHost>;

void dawn_memory_anchor();

}  // namespace DAWN

#pragma once

namespace DAWN {
namespace Matrix {

struct Csr_t {
 public:
  int* row_ptr;
  int* col;
  float* val;
};

struct Coo_t {
 public:
  int* row;
  int* col;
  float* val;
};

void coo2Csr(int n, int nnz, Csr_t& csr, Coo_t& coo);
void coo2Csr_Weighted(int n, int nnz, Csr_t& csr, Coo_t& coo);
void transpose(int nnz, Coo_t& coo);
void transpose_Weighted(int nnz, Coo_t& coo);

}  // namespace Matrix
}  // namespace DAWN

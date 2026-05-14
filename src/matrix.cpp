#include <dawn_internal/matrix.hxx>

#include <algorithm>
#include <utility>
#include <vector>

void DAWN::Matrix::transpose_Weighted(int nnz, DAWN::Matrix::Coo_t& coo) {
  std::vector<std::pair<int, std::pair<int, float>>> tmp;
  for (int i = 0; i < nnz; i++) {
    tmp.push_back({coo.row[i], {coo.col[i], coo.val[i]}});
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 0; i < nnz; i++) {
    coo.row[i] = tmp[i].first;
    coo.col[i] = tmp[i].second.first;
    coo.val[i] = tmp[i].second.second;
  }
}

void DAWN::Matrix::transpose(int nnz, DAWN::Matrix::Coo_t& coo) {
  std::vector<std::pair<int, int>> tmp;
  for (int i = 0; i < nnz; i++) {
    tmp.push_back({coo.row[i], coo.col[i]});
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 0; i < nnz; i++) {
    coo.row[i] = tmp[i].first;
    coo.col[i] = tmp[i].second;
  }
}

void DAWN::Matrix::coo2Csr_Weighted(int n,
                                    int nnz,
                                    DAWN::Matrix::Csr_t& csr,
                                    DAWN::Matrix::Coo_t& coo) {
  csr.val = new float[nnz];
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];

  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
      csr.val[j] = coo.val[j];
    }
  }
  delete[] row_count;
}

void DAWN::Matrix::coo2Csr(int n,
                           int nnz,
                           DAWN::Matrix::Csr_t& csr,
                           DAWN::Matrix::Coo_t& coo) {
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];

  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
    }
  }
  delete[] row_count;
}

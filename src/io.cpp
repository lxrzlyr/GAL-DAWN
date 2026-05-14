#include <dawn_internal/io.hxx>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool is_dawn_finite_output(float value) {
  return std::isfinite(value) && value != 0.0f &&
         value < std::numeric_limits<float>::max();
}

}  // namespace

void DAWN::IO::outfile(int n,
                       int* result,
                       int source,
                       std::string& output_path) {
  std::ofstream outfile(output_path, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && (result[j] > 0))
      outfile << source << " " << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::IO::outfile(int n,
                       float* result,
                       int source,
                       std::string& output_path) {
  std::ofstream outfile(output_path, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && is_dawn_finite_output(result[j]))
      outfile << source << " " << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::IO::outfile(int n, int* result, std::string& output_path) {
  std::ofstream outfile(output_path, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if (result[j] > 0)
      outfile << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::IO::outfile(int n, float* result, std::string& output_path) {
  std::ofstream outfile(output_path, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if (is_dawn_finite_output(result[j]))
      outfile << j << " " << std::fixed << std::setprecision(6) << result[j]
              << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

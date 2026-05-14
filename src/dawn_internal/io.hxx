#pragma once

#include <string>

namespace DAWN {
namespace IO {

void outfile(int n, int* result, int source, std::string& output_path);
void outfile(int n, float* result, int source, std::string& output_path);
void outfile(int n, int* result, std::string& output_path);
void outfile(int n, float* result, std::string& output_path);

}  // namespace IO
}  // namespace DAWN

#include <dawn/algorithms/operators.hxx>
#include <dawn/common/config.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/device_selector.hxx>
#include <dawn/runtime/kernel_registry.hxx>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  dawn run --algorithm bfs --input graph.mtx --source 0 [options]\n"
      << "  dawn inspect --input graph.mtx\n"
      << "  dawn kernels [--algorithm bfs]\n"
      << "  dawn version\n"
      << "\n"
      << "Run options:\n"
      << "  --algorithm bfs|sssp|mssp|apsp|cc|bc\n"
      << "  --input PATH\n"
      << "  --output PATH\n"
      << "  --source N\n"
      << "  --source-list PATH\n"
      << "  --output-source N\n"
      << "  --backend auto|best|cpu|cuda\n"
      << "  --device N\n"
      << "  --kernel NAME\n"
      << "  --weighted true|false\n"
      << "  --format matrix-market\n"
      << "\n"
      << "Backends: auto, best, cpu, cuda\n";
}

std::string ValueAfter(int* index, int argc, char** argv) {
  if (*index + 1 >= argc) {
    return "";
  }
  ++(*index);
  return argv[*index];
}

bool ParseBool(const std::string& text, bool* out) {
  if (text == "true" || text == "1") {
    *out = true;
    return true;
  }
  if (text == "false" || text == "0") {
    *out = false;
    return true;
  }
  return false;
}

DAWN::StatusOr<std::vector<DAWN::VertexId>> ReadSourceList(
    const std::string& path) {
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    return DAWN::Status(DAWN::StatusCode::kNotFound,
                        "failed to open source list " + path);
  }
  std::vector<DAWN::VertexId> sources;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '%') {
      continue;
    }
    std::stringstream ss(line);
    int source = 0;
    if (!(ss >> source)) {
      return DAWN::Status(DAWN::StatusCode::kInvalidArgument,
                          "malformed source list entry");
    }
    sources.push_back(source);
  }
  return sources;
}

int Inspect(int argc, char** argv) {
  std::string input_path;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" || arg == "-i") {
      input_path = ValueAfter(&i, argc, argv);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown inspect option: " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  if (input_path.empty()) {
    std::cerr << "inspect requires --input\n";
    return EXIT_FAILURE;
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket(input_path);
  if (!graph.ok()) {
    std::cerr << graph.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  DAWN::CsrView view = graph->view();
  DAWN::IO::DegreeSummary degree = DAWN::IO::ComputeDegreeSummary(view);

  std::cout << "vertices: " << view.num_vertices << "\n"
            << "edges: " << view.num_edges << "\n"
            << "directed: " << (view.directed ? "true" : "false") << "\n"
            << "weighted: " << (view.weighted ? "true" : "false") << "\n"
            << "format: csr\n"
            << "degree_min: " << degree.min_degree << "\n"
            << "degree_max: " << degree.max_degree << "\n"
            << "degree_avg: " << std::fixed << std::setprecision(3)
            << degree.average_degree << "\n";
  return EXIT_SUCCESS;
}

int Kernels(int argc, char** argv) {
  std::string algorithm;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--algorithm") {
      algorithm = ValueAfter(&i, argc, argv);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown kernels option: " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  std::vector<DAWN::KernelDescriptor> registry =
      DAWN::BuildDefaultKernelRegistry();
  std::vector<DAWN::KernelDescriptor> kernels =
      DAWN::ListKernelsForAlgorithm(registry, algorithm);
  if (kernels.empty()) {
    std::cerr << "No kernels registered";
    if (!algorithm.empty()) {
      std::cerr << " for algorithm " << algorithm;
    }
    std::cerr << "\n";
    return EXIT_FAILURE;
  }

  for (const DAWN::KernelDescriptor& kernel : kernels) {
    std::cout << kernel.name << " algorithm=" << kernel.algorithm
              << " backend=" << DAWN::BackendKindName(kernel.backend)
              << " weighted=" << (kernel.supports_weighted ? "true" : "false")
              << " unweighted="
              << (kernel.supports_unweighted ? "true" : "false") << "\n";
  }
  return EXIT_SUCCESS;
}

int Run(int argc, char** argv) {
  std::string algorithm;
  std::string input_path;
  std::string output_path;
  std::string source_list_path;
  std::string backend = "auto";
  std::string kernel_name;
  std::string format = "matrix-market";
  int source = 0;
  int output_source = -1;
  int device = 0;
  bool explicit_device = false;
  bool weighted_override = false;
  bool has_weighted_override = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--algorithm") {
      algorithm = ValueAfter(&i, argc, argv);
    } else if (arg == "--input" || arg == "-i") {
      input_path = ValueAfter(&i, argc, argv);
    } else if (arg == "--output" || arg == "-o") {
      output_path = ValueAfter(&i, argc, argv);
    } else if (arg == "--source") {
      source = std::atoi(ValueAfter(&i, argc, argv).c_str());
    } else if (arg == "--source-list" || arg == "--sourceList") {
      source_list_path = ValueAfter(&i, argc, argv);
    } else if (arg == "--output-source") {
      output_source = std::atoi(ValueAfter(&i, argc, argv).c_str());
    } else if (arg == "--backend") {
      backend = ValueAfter(&i, argc, argv);
    } else if (arg == "--device") {
      device = std::atoi(ValueAfter(&i, argc, argv).c_str());
      explicit_device = true;
    } else if (arg == "--kernel") {
      kernel_name = ValueAfter(&i, argc, argv);
    } else if (arg == "--weighted") {
      std::string value = ValueAfter(&i, argc, argv);
      if (!ParseBool(value, &weighted_override)) {
        std::cerr << "--weighted expects true or false\n";
        return EXIT_FAILURE;
      }
      has_weighted_override = true;
    } else if (arg == "--format") {
      format = ValueAfter(&i, argc, argv);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown run option: " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  if (algorithm.empty() || input_path.empty()) {
    std::cerr << "run requires --algorithm and --input\n";
    return EXIT_FAILURE;
  }
  if (format != "matrix-market" && format != "mtx") {
    std::cerr << "unsupported format: " << format << "\n";
    return EXIT_FAILURE;
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket(input_path);
  if (!graph.ok()) {
    std::cerr << graph.status().ToString() << "\n";
    return EXIT_FAILURE;
  }
  if (has_weighted_override && weighted_override != graph->weighted()) {
    std::cerr << "--weighted does not match input graph metadata\n";
    return EXIT_FAILURE;
  }

  std::vector<DAWN::VertexId> sources;
  if (!source_list_path.empty()) {
    DAWN::StatusOr<std::vector<DAWN::VertexId>> source_list =
        ReadSourceList(source_list_path);
    if (!source_list.ok()) {
      std::cerr << source_list.status().ToString() << "\n";
      return EXIT_FAILURE;
    }
    sources = std::move(source_list.value());
  }

  if (!output_path.empty()) {
    std::remove(output_path.c_str());
  }

  DAWN::StatusOr<DAWN::BackendPolicy> backend_policy =
      DAWN::ParseBackendPolicy(backend);
  if (!backend_policy.ok()) {
    std::cerr << backend_policy.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  DAWN::RuntimeOptions runtime_options;
  runtime_options.device_policy.backend_policy = backend_policy.value();
  runtime_options.device_policy.device_id = device;
  runtime_options.device_policy.explicit_device_id = explicit_device;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle = DAWN::dawnCreate(runtime_options);
  if (!handle.ok()) {
    std::cerr << handle.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  DAWN::StatusOr<DAWN::Algorithms::AlgorithmResult> result =
      DAWN::Algorithms::RunAlgorithm(algorithm, graph.value(), source, sources,
                                     output_path, kernel_name, output_source,
                                     handle.value());
  DAWN::dawnDestroy(handle.value());

  if (!result.ok()) {
    std::cerr << result.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "algorithm: " << algorithm << "\n"
            << "backend: " << DAWN::BackendKindName(result->backend) << "\n"
            << "kernel: " << result->kernel_name << "\n"
            << "elapsed_seconds: " << std::fixed << std::setprecision(6)
            << result->elapsed_seconds << "\n";
  if (algorithm == "cc") {
    std::cout << "value: " << std::fixed << std::setprecision(6)
              << result->value << "\n";
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return EXIT_FAILURE;
  }

  std::string command = argv[1];
  if (command == "run") {
    return Run(argc, argv);
  }
  if (command == "inspect") {
    return Inspect(argc, argv);
  }
  if (command == "kernels") {
    return Kernels(argc, argv);
  }
  if (command == "version") {
    std::cout << DAWN_VERSION_STRING << "\n";
    return EXIT_SUCCESS;
  }
  if (command == "--help" || command == "-h") {
    PrintUsage();
    return EXIT_SUCCESS;
  }

  std::cerr << "Unknown command: " << command << "\n";
  PrintUsage();
  return EXIT_FAILURE;
}

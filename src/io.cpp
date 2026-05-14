#include <dawn/io/matrix_market.hxx>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace DAWN {
namespace IO {

namespace {

struct Entry {
  VertexId row;
  VertexId col;
  Weight weight;
};

bool IsCommentOrEmpty(const std::string& line) {
  return line.empty() || line[0] == '%';
}

Status ParseHeader(const std::string& line, bool* weighted, bool* directed) {
  std::stringstream ss(line);
  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  ss >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" ||
      format != "coordinate") {
    return Status(StatusCode::kInvalidArgument,
                  "malformed MatrixMarket coordinate header");
  }
  if (field == "pattern") {
    *weighted = false;
  } else if (field == "real" || field == "integer") {
    *weighted = true;
  } else {
    return Status(
        StatusCode::kUnsupported,
        "only pattern, real, and integer MatrixMarket fields are supported");
  }
  if (symmetry == "symmetric") {
    *directed = false;
  } else if (symmetry == "general") {
    *directed = true;
  } else {
    return Status(
        StatusCode::kUnsupported,
        "only general and symmetric MatrixMarket graphs are supported");
  }
  return Status::OK();
}

}  // namespace

StatusOr<HostCsrGraph> ReadMatrixMarket(const std::string& input_path,
                                        const MatrixMarketOptions& options) {
  std::ifstream file(input_path.c_str());
  if (!file.is_open()) {
    return Status(StatusCode::kNotFound, "failed to open " + input_path);
  }

  std::string line;
  if (!std::getline(file, line)) {
    return Status(StatusCode::kInvalidArgument, "empty MatrixMarket file");
  }

  bool weighted = false;
  bool directed = true;
  Status header_status = ParseHeader(line, &weighted, &directed);
  if (!header_status.ok()) {
    return header_status;
  }

  VertexId rows = 0;
  VertexId cols = 0;
  EdgeId declared_entries = 0;
  bool found_size = false;
  while (std::getline(file, line)) {
    if (IsCommentOrEmpty(line)) {
      continue;
    }
    std::stringstream ss(line);
    if (!(ss >> rows >> cols >> declared_entries)) {
      return Status(StatusCode::kInvalidArgument,
                    "malformed MatrixMarket size line");
    }
    found_size = true;
    break;
  }
  if (!found_size || rows < 0 || cols < 0 || declared_entries < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid MatrixMarket size line");
  }
  if (rows != cols) {
    return Status(StatusCode::kUnsupported,
                  "DAWN CSR graph loader requires a square matrix");
  }

  std::vector<Entry> entries;
  entries.reserve(static_cast<size_t>(declared_entries) * (directed ? 1u : 2u));

  EdgeId seen_entries = 0;
  while (std::getline(file, line)) {
    if (IsCommentOrEmpty(line)) {
      continue;
    }
    std::stringstream ss(line);
    int64_t row_one_based = 0;
    int64_t col_one_based = 0;
    Weight weight = 1.0f;
    if (!(ss >> row_one_based >> col_one_based)) {
      return Status(StatusCode::kInvalidArgument,
                    "malformed MatrixMarket entry");
    }
    if (weighted && !(ss >> weight)) {
      return Status(StatusCode::kInvalidArgument,
                    "weighted MatrixMarket entry is missing a value");
    }
    ++seen_entries;

    if (row_one_based <= 0 || row_one_based > rows || col_one_based <= 0 ||
        col_one_based > cols) {
      return Status(StatusCode::kOutOfRange,
                    "MatrixMarket entry index is outside declared dimensions");
    }

    const VertexId row = static_cast<VertexId>(row_one_based - 1);
    const VertexId col = static_cast<VertexId>(col_one_based - 1);
    if (row == col && options.ignore_self_loops) {
      continue;
    }
    entries.push_back({row, col, weight});
    if (!directed && row != col) {
      entries.push_back({col, row, weight});
    }
  }

  if (seen_entries != declared_entries) {
    return Status(StatusCode::kInvalidArgument,
                  "MatrixMarket entry count does not match size line");
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return std::tie(a.row, a.col, a.weight) < std::tie(b.row, b.col, b.weight);
  });

  GraphMetadata metadata;
  metadata.num_vertices = rows;
  metadata.num_edges = static_cast<EdgeId>(entries.size());
  metadata.directed = directed;
  metadata.weighted = weighted;
  metadata.storage_format = GraphStorageFormat::kCsr;

  HostBuffer<EdgeId> row_offsets(static_cast<size_t>(rows) + 1, 0);
  HostBuffer<VertexId> column_indices(entries.size());
  HostBuffer<Weight> weights;
  if (weighted) {
    weights = HostBuffer<Weight>(entries.size());
  }

  for (const Entry& entry : entries) {
    ++row_offsets[entry.row + 1];
  }
  for (VertexId row = 1; row <= rows; ++row) {
    row_offsets[row] += row_offsets[row - 1];
  }

  HostBuffer<EdgeId> cursor(static_cast<size_t>(rows), 0);
  for (VertexId row = 0; row < rows; ++row) {
    cursor[row] = row_offsets[row];
  }
  for (const Entry& entry : entries) {
    const EdgeId out = cursor[entry.row]++;
    column_indices[out] = entry.col;
    if (weighted) {
      weights[out] = entry.weight;
    }
  }

  return HostCsrGraph::Create(metadata, std::move(row_offsets),
                              std::move(column_indices), std::move(weights));
}

DegreeSummary ComputeDegreeSummary(CsrView view) {
  DegreeSummary summary;
  if (view.num_vertices == 0) {
    return summary;
  }
  summary.min_degree = view.num_edges;
  EdgeId total = 0;
  for (VertexId v = 0; v < view.num_vertices; ++v) {
    const EdgeId degree = view.row_offsets[v + 1] - view.row_offsets[v];
    summary.min_degree = std::min(summary.min_degree, degree);
    summary.max_degree = std::max(summary.max_degree, degree);
    total += degree;
  }
  summary.average_degree =
      static_cast<double>(total) / static_cast<double>(view.num_vertices);
  return summary;
}

}  // namespace IO
}  // namespace DAWN

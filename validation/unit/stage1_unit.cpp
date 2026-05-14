#include <dawn/common/types.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/memory/buffer.hxx>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

int Fail(const std::string& message) {
  std::cerr << message << std::endl;
  return EXIT_FAILURE;
}

int CheckStatusAndDowncast() {
  DAWN::Status ok = DAWN::Status::OK();
  if (!ok.ok()) {
    return Fail("OK status is not ok");
  }

  DAWN::StatusOr<int> good =
      DAWN::checked_int_downcast(static_cast<DAWN::EdgeId>(123), "edge");
  if (!good.ok() || good.value() != 123) {
    return Fail("valid downcast failed");
  }

  DAWN::StatusOr<int> bad = DAWN::checked_int_downcast(
      static_cast<DAWN::EdgeId>(std::numeric_limits<int>::max()) + 1, "edge");
  if (bad.ok()) {
    return Fail("out-of-range downcast unexpectedly succeeded");
  }
  return EXIT_SUCCESS;
}

int CheckBuffer() {
  DAWN::HostBuffer<bool> flags(3, false);
  flags[1] = true;
  bool* raw = flags.data();
  if (raw == nullptr || flags.size() != 3 || !flags[1]) {
    return Fail("HostBuffer<bool> did not expose addressable storage");
  }

  DAWN::HostBuffer<int> values(2, 7);
  DAWN::Span<int> span = values.span();
  if (span.size() != 2 || span[0] != 7 || span[1] != 7) {
    return Fail("HostBuffer span/fill failed");
  }
  int* released = values.release();
  if (values.size() != 0 || values.data() != nullptr) {
    delete[] released;
    return Fail("HostBuffer release did not clear owner");
  }
  delete[] released;
  return EXIT_SUCCESS;
}

int CheckGraphContract() {
  DAWN::GraphMetadata metadata;
  metadata.num_vertices = 2;
  metadata.num_edges = 1;
  metadata.directed = true;
  metadata.weighted = false;

  DAWN::HostBuffer<DAWN::EdgeId> row_offsets(3, 0);
  row_offsets[0] = 0;
  row_offsets[1] = 1;
  row_offsets[2] = 1;
  DAWN::HostBuffer<DAWN::VertexId> cols(1, 1);
  DAWN::HostBuffer<DAWN::Weight> weights;
  DAWN::StatusOr<DAWN::HostCsrGraph> graph = DAWN::HostCsrGraph::Create(
      metadata, std::move(row_offsets), std::move(cols), std::move(weights));
  if (!graph.ok()) {
    return Fail("valid HostCsrGraph rejected: " + graph.status().ToString());
  }

  DAWN::GraphMetadata bad_metadata = metadata;
  bad_metadata.num_edges = 2;
  DAWN::HostBuffer<DAWN::EdgeId> bad_rows(3, 0);
  bad_rows[0] = 0;
  bad_rows[1] = 1;
  bad_rows[2] = 1;
  DAWN::HostBuffer<DAWN::VertexId> bad_cols(1, 1);
  DAWN::StatusOr<DAWN::HostCsrGraph> bad_graph = DAWN::HostCsrGraph::Create(
      bad_metadata, std::move(bad_rows), std::move(bad_cols),
      DAWN::HostBuffer<DAWN::Weight>());
  if (bad_graph.ok()) {
    return Fail("invalid CSR terminal row offset accepted");
  }
  return EXIT_SUCCESS;
}

int CheckMatrixMarket() {
  const std::string prefix = "validation/fixtures/";

  DAWN::StatusOr<DAWN::HostCsrGraph> unweighted =
      DAWN::IO::ReadMatrixMarket(prefix + "tiny_unweighted.mtx");
  if (!unweighted.ok()) {
    return Fail("unweighted graph failed: " + unweighted.status().ToString());
  }
  if (unweighted->num_vertices() != 4 || unweighted->num_edges() != 6 ||
      unweighted->directed() || unweighted->weighted()) {
    return Fail("unweighted symmetric metadata/edge expansion is wrong");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> weighted =
      DAWN::IO::ReadMatrixMarket(prefix + "tiny_weighted.mtx");
  if (!weighted.ok() || !weighted->weighted() || weighted->num_edges() != 6) {
    return Fail("weighted symmetric graph failed");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> directed =
      DAWN::IO::ReadMatrixMarket(prefix + "tiny_directed.mtx");
  if (!directed.ok() || !directed->directed() || directed->num_edges() != 2) {
    return Fail("directed graph handling failed");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> tail =
      DAWN::IO::ReadMatrixMarket(prefix + "tiny_tail_isolated.mtx");
  if (!tail.ok() || tail->num_vertices() != 5 ||
      tail->row_offsets().size() != 6 ||
      tail->row_offsets()[5] != tail->num_edges()) {
    return Fail("tail isolated vertex was not preserved");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> loops =
      DAWN::IO::ReadMatrixMarket(prefix + "tiny_self_loops_only.mtx");
  if (!loops.ok() || loops->num_vertices() != 3 || loops->num_edges() != 0 ||
      loops->row_offsets()[3] != 0) {
    return Fail("self-loop-only graph was not represented as zero edges");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> zero =
      DAWN::IO::ReadMatrixMarket(prefix + "zero_effective_edges.mtx");
  if (!zero.ok() || zero->num_vertices() != 4 || zero->num_edges() != 0) {
    return Fail("zero-edge graph failed");
  }

  if (DAWN::IO::ReadMatrixMarket(prefix + "invalid_index.mtx").ok()) {
    return Fail("invalid index graph was accepted");
  }
  if (DAWN::IO::ReadMatrixMarket(prefix + "malformed_header.mtx").ok()) {
    return Fail("malformed header graph was accepted");
  }

  return EXIT_SUCCESS;
}

}  // namespace

int main() {
  if (CheckStatusAndDowncast() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckBuffer() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckGraphContract() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckMatrixMarket() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: DAWN stage1 unit tests" << std::endl;
  return EXIT_SUCCESS;
}

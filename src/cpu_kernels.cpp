#include "cpu_kernels.hxx"

#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace DAWN {
namespace Kernel {
namespace CPU {

StatusOr<BfsResult> RunBfs(CsrView graph, const BfsOptions& options) {
  if (options.source < 0 || options.source >= graph.num_vertices) {
    return Status(StatusCode::kOutOfRange, "BFS source is outside graph");
  }
  if (graph.num_vertices == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "BFS requires a non-empty graph");
  }

  BfsResult result;
  result.distances.assign(static_cast<size_t>(graph.num_vertices), 0);
  result.distances[static_cast<size_t>(options.source)] = 0;

  std::vector<VertexId> frontier;
  std::vector<VertexId> next_frontier;
  frontier.push_back(options.source);

  int depth = 0;
  while (!frontier.empty()) {
    ++depth;
    next_frontier.clear();

    for (VertexId vertex : frontier) {
      const EdgeId edge_begin = graph.row_offsets[vertex];
      const EdgeId edge_end = graph.row_offsets[vertex + 1];
      for (EdgeId edge = edge_begin; edge < edge_end; ++edge) {
        const VertexId neighbor = graph.column_indices[edge];
        if (neighbor == options.source) {
          continue;
        }
        int& distance = result.distances[static_cast<size_t>(neighbor)];
        if (distance == 0) {
          distance = depth;
          next_frontier.push_back(neighbor);
        }
      }
    }

    frontier.swap(next_frontier);
  }

  return result;
}

StatusOr<SsspResult> RunSssp(CsrView graph, const SsspOptions& options) {
  if (!graph.weighted || graph.weights == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "SSSP requires a weighted graph");
  }
  if (options.source < 0 || options.source >= graph.num_vertices) {
    return Status(StatusCode::kOutOfRange, "SSSP source is outside graph");
  }
  if (graph.num_vertices == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "SSSP requires a non-empty graph");
  }

  const Weight inf = std::numeric_limits<Weight>::max();
  SsspResult result;
  result.distances.assign(static_cast<size_t>(graph.num_vertices), inf);
  result.distances[static_cast<size_t>(options.source)] = 0.0f;

  typedef std::pair<Weight, VertexId> QueueEntry;
  std::priority_queue<QueueEntry,
                      std::vector<QueueEntry>,
                      std::greater<QueueEntry>>
      queue;
  queue.push(QueueEntry(0.0f, options.source));

  while (!queue.empty()) {
    const QueueEntry entry = queue.top();
    queue.pop();
    const Weight distance = entry.first;
    const VertexId vertex = entry.second;
    if (distance != result.distances[static_cast<size_t>(vertex)]) {
      continue;
    }

    const EdgeId edge_begin = graph.row_offsets[vertex];
    const EdgeId edge_end = graph.row_offsets[vertex + 1];
    for (EdgeId edge = edge_begin; edge < edge_end; ++edge) {
      const VertexId neighbor = graph.column_indices[edge];
      const Weight weight = graph.weights[edge];
      if (weight < 0.0f) {
        return Status(StatusCode::kInvalidArgument,
                      "SSSP requires non-negative edge weights");
      }
      const Weight next_distance = distance + weight;
      Weight& current = result.distances[static_cast<size_t>(neighbor)];
      if (next_distance < current) {
        current = next_distance;
        queue.push(QueueEntry(next_distance, neighbor));
      }
    }
  }

  return result;
}

namespace {

void AccumulateSourceDependency(CsrView graph,
                                VertexId source,
                                std::vector<float>* centrality) {
  const VertexId n = graph.num_vertices;
  int step = 1;
  bool is_converged = false;
  std::vector<std::vector<VertexId>> predecessors(static_cast<size_t>(n));
  std::vector<bool> alpha(static_cast<size_t>(n), false);
  std::vector<bool> beta(static_cast<size_t>(n), false);
  std::vector<int> distance(static_cast<size_t>(n), -1);
  std::vector<float> visit(static_cast<size_t>(n), 0.0f);
  std::vector<float> dependency(static_cast<size_t>(n), 0.0f);
  std::vector<VertexId> prequeue(static_cast<size_t>(n), 0);

  distance[static_cast<size_t>(source)] = 0;
  visit[static_cast<size_t>(source)] = 1.0f;
  VertexId index = 0;
  prequeue[static_cast<size_t>(index)] = source;

  for (EdgeId edge = graph.row_offsets[source];
       edge < graph.row_offsets[source + 1];
       ++edge) {
    const VertexId neighbor = graph.column_indices[edge];
    alpha[static_cast<size_t>(neighbor)] = true;
    distance[static_cast<size_t>(neighbor)] = 1;
    visit[static_cast<size_t>(neighbor)] +=
        visit[static_cast<size_t>(source)];
  }

  while (step < n) {
    ++step;
    std::vector<bool>& current = (step % 2 == 0) ? alpha : beta;
    std::vector<bool>& next = (step % 2 == 0) ? beta : alpha;
    is_converged = true;

    for (VertexId vertex = 0; vertex < n; ++vertex) {
      if (!current[static_cast<size_t>(vertex)]) {
        continue;
      }
      current[static_cast<size_t>(vertex)] = false;
      ++index;
      prequeue[static_cast<size_t>(index)] = vertex;

      for (EdgeId edge = graph.row_offsets[vertex];
           edge < graph.row_offsets[vertex + 1];
           ++edge) {
        const VertexId neighbor = graph.column_indices[edge];
        if (distance[static_cast<size_t>(neighbor)] < 0) {
          next[static_cast<size_t>(neighbor)] = true;
          distance[static_cast<size_t>(neighbor)] = step;
          is_converged = false;
        }
        if (distance[static_cast<size_t>(neighbor)] ==
            distance[static_cast<size_t>(vertex)] + 1) {
          predecessors[static_cast<size_t>(neighbor)].push_back(vertex);
          visit[static_cast<size_t>(neighbor)] +=
              visit[static_cast<size_t>(vertex)];
        }
      }
    }

    if (is_converged) {
      break;
    }
  }

  for (VertexId queue_index = index; queue_index >= 0; --queue_index) {
    const VertexId vertex = prequeue[static_cast<size_t>(queue_index)];
    for (VertexId predecessor :
         predecessors[static_cast<size_t>(vertex)]) {
      const float vertex_visit = visit[static_cast<size_t>(vertex)];
      if (vertex_visit != 0.0f) {
        dependency[static_cast<size_t>(predecessor)] +=
            (visit[static_cast<size_t>(predecessor)] / vertex_visit) *
            (1.0f + dependency[static_cast<size_t>(vertex)]);
      }
    }
    if (vertex != source) {
      (*centrality)[static_cast<size_t>(vertex)] +=
          dependency[static_cast<size_t>(vertex)];
    }
    if (queue_index == 0) {
      break;
    }
  }
}

}  // namespace

StatusOr<BcResult> RunBc(CsrView graph, const BcOptions& options) {
  (void)options;
  if (graph.weighted) {
    return Status(StatusCode::kInvalidArgument,
                  "BC requires an unweighted graph");
  }
  if (graph.num_vertices < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "BC requires a valid vertex count");
  }

  BcResult result;
  result.centrality.assign(static_cast<size_t>(graph.num_vertices), 0.0f);
  if (graph.num_vertices == 0) {
    return result;
  }

  for (VertexId source = 0; source < graph.num_vertices; ++source) {
    AccumulateSourceDependency(graph, source, &result.centrality);
  }

  if (graph.num_vertices > 2) {
    const float scale = static_cast<float>((graph.num_vertices - 1) *
                                           (graph.num_vertices - 2));
    for (float& value : result.centrality) {
      if (!graph.directed) {
        value /= 2.0f;
      }
      value /= scale;
    }
  } else {
    for (float& value : result.centrality) {
      value = 0.0f;
    }
  }

  return result;
}

}  // namespace CPU
}  // namespace Kernel
}  // namespace DAWN

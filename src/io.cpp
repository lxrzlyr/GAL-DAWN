#include <dawn/io.hxx>
void DAWN::IO::readGraph(std::string& input_path, DAWN::Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;
  std::priority_queue<std::pair<int, int>> s;
  int rows, cols;

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols;
    rows--;
    cols--;
    if (rows != cols) {
      s.push({rows, cols});
      s.push({cols, rows});
    }
  }
  file.close();

  graph.nnz = s.size();
  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  int i = graph.nnz - 1;

  while (!s.empty()) {
    graph.coo.row[i] = s.top().first;
    graph.coo.col[i] = s.top().second;
    --i;
    s.pop();
  }
  coo2Csr(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void DAWN::IO::readGraph_Directed(std::string& input_path,
                                  DAWN::Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;
  int rows, cols;
  int i = 0;

  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols;
    rows--;
    cols--;
    if (rows != cols) {
      graph.coo.row[i] = rows;
      graph.coo.col[i] = cols;
      ++i;
    }
  }
  file.close();

  graph.nnz = i;
  transpose(graph.nnz, graph.coo);
  coo2Csr(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void DAWN::IO::readGraph_Weighted(std::string& input_path,
                                  DAWN::Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  std::priority_queue<std::tuple<int, int, float>> s;

  int rows, cols;
  float vals;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> vals;
    rows--;
    cols--;
    if (rows != cols) {
      s.push({rows, cols, vals});
      s.push({cols, rows, vals});
    }
  }
  file.close();

  graph.nnz = s.size();
  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  graph.coo.val = new float[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  std::fill_n(graph.coo.val, graph.nnz, 0.0f);
  int i = graph.nnz - 1;

  while (!s.empty()) {
    graph.coo.row[i] = std::get<0>(s.top());
    graph.coo.col[i] = std::get<1>(s.top());
    graph.coo.val[i] = std::get<2>(s.top());
    --i;
    s.pop();
  }

  coo2Csr_Weighted(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void DAWN::IO::readGraph_Directed_Weighted(std::string& input_path,
                                           DAWN::Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  graph.coo.val = new float[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  std::fill_n(graph.coo.val, graph.nnz, 0.0f);

  int rows, cols;
  float vals;
  int i = 0;
  // std::cout << "nnz: " << graph.nnz << std::endl;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> vals;
    rows--;
    cols--;
    if (rows != cols) {
      graph.coo.row[i] = rows;
      graph.coo.col[i] = cols;
      graph.coo.val[i] = vals;
      ++i;
    }
  }
  file.close();

  graph.nnz = i;

  transpose_Weighted(graph.nnz, graph.coo);
  coo2Csr_Weighted(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void DAWN::IO::readList(std::string& input_path, DAWN::Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  int source;
  int i = 0;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> source;
    graph.msource.push_back(source);
    i++;
  }
  file.close();
}

void DAWN::IO::outfile(int n,
                       int* result,
                       int source,
                       std::string& output_path) {
  std::ofstream outfile(output_path);
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
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  int INF = 0xfffffff;
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && (result[j] != 0))
      outfile << source << " " << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::IO::outfile(int n, int* result, std::string& output_path) {
  std::ofstream outfile(output_path);
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
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if (result[j] != 0)
      outfile << j << " " << std::fixed << std::setprecision(6) << result[j]
              << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

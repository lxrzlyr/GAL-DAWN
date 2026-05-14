# GAL-DAWN Validation

This directory contains production validation assets.

- `unit/`: focused C++ unit and quality tests.
- `fixtures/`: small hand-written MatrixMarket fixtures.
- `correctness/`: generated graph correctness regression harness.
- `downstream/`: install/export consumer smoke project.

## Generated Correctness Regression

Run the generated correctness suite:

```bash
ctest --test-dir build -R dawn_correctness_generated --output-on-failure
```

The harness creates deterministic small-to-medium graphs covering 100-1000
vertices:

- random graphs;
- power-law-like graphs;
- grid/path/star-like graphs;
- directed and undirected graphs;
- weighted and unweighted graphs.

Python computes reference results for BFS, SSSP, MSSP, APSP, CC, and BC.
The normal CTest target records all per-algorithm results and reports every
failure it finds. The strict target fails the build on any mismatch:

```bash
ctest --test-dir build -R dawn_correctness_generated_strict --output-on-failure
```

MSSP and APSP output is intentionally bounded in correctness tests: they do not
write all-pairs output by default, and `--output-source N` writes only one source
row for comparison.

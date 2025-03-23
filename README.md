# GAL-DAWN: An Novel Graph Algorithm Library based on DAWN, CUDA/C++

GAL-DAWN is a novel Graph Algorithm Library, with all algorithms developed on the DAWN, utilizing CUDA and C++ for enhanced performance and efficiency.

DAWN is a novel shortest path algorithm that enhances parallelism through the use of matrix operations, thus eliminating the need for priority queues. Prior to this, mainstream research on optimizing shortest path algorithms primarily concentrated on improving the parallel performance of priority queues. For a more in-depth understanding of DAWN, please refer to the papers listed below.

DAWN and [Gunrock](https://github.com/gunrock/gunrock) have a well-established collaborative relationship. DAWN is included in the main branch of Gunrock and since the releases V2.1. When using the API for single-source shortest path tasks in Gunrock, the default implementation employs DAWN. If you prefer to use the original implementations in Gunrock, please follow the provided instructions to modify the source files accordingly.

If you intend to use DAWN as a baseline algorithm and aim to surpass its performance on the GPU, we recommend utilizing the DAWN API available in the Gunrock repository rather than the GPU functions in this repository. The GPU functions here are subject to modifications for testing optimization feasibilities, which may occasionally lead to suboptimal performance and potential misinterpretations of results. Gunrock is a specialized library for graph computing within CUDA-X, and its components are consistently reliable. 

For CPU-based work, the above recommendation does not apply; you can directly use the code from this repository, and also can use the cpu reference implementation of DAWN in the Gunrock.

| [**Examples**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/algorithm) | [**Documentation**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/document) | [**Test**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/test) |[**Gunrock**](https://github.com/gunrock/gunrock/tree/main/include/gunrock/algorithms)|
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |-------------------------------------------------------------------------------------- |
- Examples: Demonstrate the usage of algorithms in GAL-DAWN.
- Document: Provides the detailed description of GAL-DAWN, include the **Quick_Start** and **Code_Guide**. **Quick_Start** provides the guide for quickly working with GAL-DAWN. **Code_Guide** provides the detailed description for how to implement own algorithm in GAL-DAWN.
- Test: Provides the detailed information of Testing.
- Gunrock: Implementations in the Gunrock.

We welcome any interest and ideas related to DAWN and its applications. If you are interested in DAWN algorithms and their applications, please feel free to share your thoughts via [email:1289539524@qq.com], and we will do our best to assist you in your research based on DAWN.


## Development Status

The betweenness centrality implementation based on DAWN is currently under development, and CPU version has appeared in the dev branch. We have utilized an accumulation technique and graph traversal approach of DAWN that differs from the Brandes algorithm, which is a novel algorithm with lower time and space complexity, tentatively named BCDAWN. Moving forward, our focus will be on parallelizing the current CPU version and developing a GPU version. We plan to release a paper describing the technical details of the weighted version of DAWN and the BCDAWN algorithm at an appropriate time, and these implementations will be made available for early access. We still encourage colleagues to complete the implementation of the Brandes algorithm.

The repository contributors and paper authors hold all rights to the code currently in the repository and the corresponding publications. If you are interested in work related to the DAWN algorithm and are developing new algorithms, please be mindful of not replicating the technical implementations currently in our repository. If you are working on engineering optimizations for algorithms like DAWN, please promptly create a fork and raise issues in this repository to avoid instances of plagiarism and preemptive paper publication.

| Algorithm           | Release |
| ------------------- | ------- |
| APSP                | V2.1    |
| MSSP                | V2.1    |
| SSSP                | V2.1    |
| BFS                 | V2.1    |
| BC                  | CPU Done; GPU Doing|
| CC                  | V2.3    |
| Cluster Analysis    | Future  |
| Community Detection | Future  |

Further applications of these algorithms, such as community detection, clustering, and path planning, are also on our agenda. We will release new features of DAWN and the application algorithms based on DAWN on this repository. If the algorithms are also needed by Gunrock, we will contribute them to the Gunrock repository later.

## How to Cite DAWN
Thank you for citing our work. 

```bibtex
@inproceedings{dawn,
author = {Feng, Yelai and Wang, Huaixi and Zhu, Yining and Liu, Xiandong and Lu, Hongyi and Liu, Qing},
title = {DAWN: Matrix Operation-Optimized Algorithm for Shortest Paths Problem on Unweighted Graphs},
year = {2024},
isbn = {9798400706103},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3650200.3656600}, 
url = {https://doi.org/10.1145/3650200.3656600},
booktitle = {Proceedings of the 38th {ACM} International Conference on Supercomputing,  {ICS} 2024, Kyoto, Japan, June 4-7, 2024},
pages = {1–13},
series = {ICS '24}
}
```
This paper is nominated for the **Best Paper Award** at the 38th ACM International Conference on Supercomputing (ICS) 2024.
## Copyright & License

All source code are released under [Apache 2.0](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/blob/4266d98053678ce76e34be64477ac2364f0f4291/LICENSE).

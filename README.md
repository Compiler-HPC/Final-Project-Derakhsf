In the realm of high-performance computing, the quest for efficient computational methods remains a critical challenge. This study addresses the performance potential of Python, a language traditionally not synonymous with high computational efficiency, in the context of the Polyhedral Benchmark (Polybench) suite. We explore the application of Numba, a Just-In-Time (JIT) compiler, to optimize Python code through parallelization, vectorization and loop unrolling, and transformation techniques. Our methodology involves the implementation of selected Polybench benchmarks in Python, followed by their optimization using Numba's features. We systematically measure and compare the performance of the optimized Python code against its unoptimized counterparts and Corresponding C++ implementations, focusing on execution time and computational efficiency. The results indicate significant performance improvements, highlighting the effectiveness of Numba in bridging the gap between Python's ease of use and the demands of high-performance computing. This study not only showcases the feasibility of Python as a viable option in performance-critical applications but also contributes to the growing body of knowledge in Python-based scientific computing. Our findings open avenues for further research in optimizing Python for high-performance tasks, potentially expanding its adoption in computational-intensive domains.
Key Words: High-Performance-Computing, Polybench, Numba, Vectorization, Loop-Unrolling
In order to recreate the graphs and delve into the specifics of methods applied across all three benchmarks, I've included both the C++ and Python code for each benchmark. Furthermore, I've provided three scripts tailored for the Niagara Server, each corresponding to a specific array size. These scripts facilitate the reproduction of experimental results. For each script, I've incorporated the output from Niagara, and to streamline analysis, an Excel file has been included. This file encompasses the table derived from the Niagara result files, along with a Pivot table and Pivot chart essential for generating the graphs. It's noteworthy that all graphs have been generated from the same Pivot table, leveraging various filtrations for distinct visualizations. The Excel file also includes a separate sheet that contains a summary of the main table; I have used this sheet to exploit the speed-up factor. (The evaluation file has been also included)

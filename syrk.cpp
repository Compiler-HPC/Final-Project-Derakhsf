#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

// Array initialization
void init_array(int n, int m, std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[i][j] = (i * j + 1) % n / static_cast<double>(n);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = (i * j + 2) % m / static_cast<double>(m);
        }
    }
}

// Main computational kernel
void kernel_syrk(int n, int m, double alpha, double beta, std::vector<std::vector<double>>& C, const std::vector<std::vector<double>>& A) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            C[i][j] *= beta;
            for (int k = 0; k < m; ++k) {
                C[i][j] += alpha * A[i][k] * A[j][k];
            }
        }
    }
}

// Main computational kernel with parallelization on the most outer loop
void kernel_syrk_parallel(int n, int m, double alpha, double beta, std::vector<std::vector<double>>& C, const std::vector<std::vector<double>>& A) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            C[i][j] *= beta;
            for (int k = 0; k < m; ++k) {
                C[i][j] += alpha * A[i][k] * A[j][k];
            }
        }
    }
}

// Print the resulting array
void print_array(const std::vector<std::vector<double>>& C) {
    for (const auto& row : C) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " m n" << std::endl;
        return 1;
    }

    double alpha = 1.5;
    double beta = 1.2;

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    //int k = std::atoi(argv[3]);
    int num_runs=20;

    std::vector<std::vector<double>> A(n, std::vector<double>(m, 0.0));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

    // Warm-up runs
    // Run kernel
    kernel_syrk(n, m, alpha, beta, C, A);

    clock_t start = clock();
    for (int run = 0; run < num_runs; ++run) {
        kernel_syrk(n, m, alpha, beta, C, A);
    }
     clock_t end = clock();

    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Execution Time: " << elapsed_time << " seconds" << std::endl;
    
    
    kernel_syrk_parallel(n, m, alpha, beta, C, A);
    // Measure execution time
    start = clock();
    for (int run = 0; run < 20; ++run) {
        kernel_syrk_parallel(n, m, alpha, beta, C, A);
    }
    end = clock();

    double elapsed_time_parallel = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Parallel Execution Time: " << elapsed_time_parallel << " seconds" << std::endl;

    // Print the resulting array (commented out for large sizes)
    // print_array(C);

    
    return 0;
}


#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

// Array initialization
void init_array(int m, int n, double& alpha, std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    alpha = 1.5;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = (i + j) % m / static_cast<double>(m);
        }
        A[i][i] = 1.0;

        for (int j = 0; j < n; ++j) {
            B[i][j] = (n + (i - j)) % n / static_cast<double>(n);
        }
    }
}

// Main computational kernel
void kernel_trmm(int m, int n, double alpha, std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = i + 1; k < m; ++k) {
                B[i][j] += A[k][i] * B[k][j];
            }
            B[i][j] = alpha * B[i][j];
        }
    }
}

// Main computational kernel with parallelization on the most outer loop
void kernel_trmm_parallel(int m, int n, double alpha, std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = i + 1; k < m; ++k) {
                B[i][j] += A[k][i] * B[k][j];
            }
            B[i][j] = alpha * B[i][j];
        }
    }
}


// Print the resulting array
void print_array(const std::vector<std::vector<double>>& B) {
    int m = B.size();
    int n = B[0].size();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i * m + j % 20 == 0) {
                std::cout << std::endl;
            }
            std::cout << B[i][j] << " ";
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " m n " << std::endl;
        return 1;
    }

    double alpha;
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);

    std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
    std::vector<std::vector<double>> B(m, std::vector<double>(n, 0.0));

      
    
    int num_runs = 20;
    
    //naive
    kernel_trmm(m, n, alpha, A, B);

    clock_t start = clock();
    for (int run = 0; run < num_runs; ++run) {
        kernel_trmm(m, n, alpha, A, B);
    }
    clock_t end = clock();
    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Execution Time: " << elapsed_time << " seconds" << std::endl;
    
	//parallel
    kernel_trmm_parallel(m, n, alpha, A, B);

    start = clock();
    for (int run = 0; run < num_runs; ++run) {
        kernel_trmm_parallel(m, n, alpha, A, B);
    }
    end = clock();
    double parallel_elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Parallel Execution Time: " << parallel_elapsed_time << " seconds" << std::endl;
    
    return 0;
}


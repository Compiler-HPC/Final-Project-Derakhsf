#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

// Array initialization
void init_array(int m, int n, double& alpha, double& beta, double**& C, double**& A, double**& B) {
    alpha = 1.5;
    beta = 1.2;
    C = new double*[m];
    A = new double*[m];
    B = new double*[m];

    for (int i = 0; i < m; ++i) {
        C[i] = new double[n];
        A[i] = new double[m];
        B[i] = new double[n];
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = fmod((i + j), 100) / m;
            B[i][j] = fmod((n + i - j), 100) / m;
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            A[i][j] = fmod((i + j), 100) / m;
        }
        for (int j = i + 1; j < m; ++j) {
            A[i][j] = -999; // regions of arrays that should not be used
        }
    }
}

// Main computational kernel
void kernel_symm(int m, int n, double alpha, double beta, double** C, double** A, double** B) {
    for (int i = 0; i < m; ++i) {
        double temp2 = 0;
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < i; ++k) {
                C[k][j] += alpha * B[i][j] * A[i][k];
                temp2 += B[k][j] * A[i][k];
            }
            C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
    }
}

// Main computational kernel with parallelization
void kernel_symm_parallelized(int m, int n, double alpha, double beta, double** C, double** A, double** B) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        double temp2 = 0;
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < i; ++k) {
                #pragma omp atomic
                C[k][j] += alpha * B[i][j] * A[i][k];
                temp2 += B[k][j] * A[i][k];
            }
            C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
    }
}

// Print the resulting array
void print_array(int m, int n, double** C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((i * m + j) % 20 == 0) std::cout << std::endl;
            std::cout << C[i][j] << " ";
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./symm m n" << std::endl;
        return 1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);

    double alpha, beta;
    double** C, ** A, ** B;

    init_array(m, n, alpha, beta, C, A, B);
    
    int num_runs = 20;

    // Warm-up runs
    kernel_symm(m, n, alpha, beta, C, A, B);    

    clock_t start = clock();
    for (int run = 0; run < num_runs; ++run) {
        kernel_symm(m, n, alpha, beta, C, A, B);
    }
    clock_t end = clock();

    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Execution Time: " << elapsed_time << " seconds" << std::endl;
    
    //parallel
    // Warm-up runs
    kernel_symm_parallelized(m, n, alpha, beta, C, A, B); 
    
    start = clock();
    for (int run = 0; run < num_runs; ++run) {
        kernel_symm_parallelized(m, n, alpha, beta, C, A, B);
    }
    end = clock();

    double elapsed_time_parallel = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Print the execution time
    std::cout << "Parallel Execution Time: " << elapsed_time_parallel << " seconds" << std::endl;

    // Print the resulting array
   // print_array(m, n, C);

    // Free allocated memory
    for (int i = 0; i < m; ++i) {
        delete[] C[i];
        delete[] A[i];
        delete[] B[i];
    }
    delete[] C;
    delete[] A;
    delete[] B;

    return 0;
}


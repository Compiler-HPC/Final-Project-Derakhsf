import sys
import time  
import numpy as np
import subprocess
from numba import njit, prange, vectorize, float64
from numba.core.compiler import DefaultPassBuilder, CompilerBase,  IRProcessing
from numba.core.typed_passes import NopythonTypeInference, DeadCodeElimination
from numba.core import ir
from types import FunctionType, BuiltinFunctionType
from numba.core.compiler_machinery import FunctionPass, register_pass


# Array initialization
def init_array(n, m):
    A = np.zeros((n, m), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            A[i][j] = ((i * j + 1) % n) / n
    for i in range(n):
        for j in range(n):
            C[i][j] = ((i * j + 2) % m) / m
    return A, C

# Dummy vectorized function for multiplication
@vectorize(['float64(float64, float64)'])
def dummy_vectorized_multiply(lhs, rhs):
    return lhs * rhs

# Function pass for vectorization
@register_pass(mutates_CFG=False, analysis_only=False)
class VectorizationPass(FunctionPass):
    _name = "vectorization_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def vectorize(self, stmt):
     # Create a call to the vectorized multiply function
        print('hllo')
        lhs, rhs = stmt.value.lhs, stmt.value.rhs
        return ir.Assign(
            value=ir.Expr.call(dummy_vectorized_multiply, [lhs, rhs], (), stmt.loc),
            target=stmt.target,
            loc=stmt.loc
        )
        
    def is_vectorizable(self, stmt):
        # Check if the operation is a multiplication
        return ( isinstance(stmt.value, ir.Expr)
            and stmt.value.op == 'binop'             
            and isinstance(stmt.value.fn, BuiltinFunctionType)
            and str(stmt.value.fn) == '<built-in function mul>'
            and isinstance(stmt.value.lhs, ir.Var) 
            and isinstance(stmt.value.rhs, ir.Var)  
            and stmt.value.lhs.is_temp
            and stmt.value.rhs.is_temp
             
                )

    def run_pass(self, state):
        mutate = False
        for block in state.func_ir.blocks.values():
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and self.is_vectorizable(stmt):
                    # Vectorize the assignment if it meets the conditions
                    new_stmt = stmt
                    if new_stmt:
                        print("Vectorizable:", new_stmt)
                        mutate = True
                        new_body.append(new_stmt)
                    else:
                        new_body.append(stmt)
                else:
                    new_body.append(stmt)

            block.body = new_body

        return mutate



# Main computational kernel
def kernel_syrk(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i][j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i][j] += alpha * A[i][k] * A[j][k]
 
#loop_unrolled
def syrk_unrolled(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(0, m, 4):  # Unroll by 4
                C[i, j] += alpha * A[i, k] * A[j, k]
                C[i, j] += alpha * A[i, k + 1] * A[j, k + 1]
                C[i, j] += alpha * A[i, k + 2] * A[j, k + 2]
                C[i, j] += alpha * A[i, k + 3] * A[j, k + 3]

#loop_parallelized_unrolled
def syrk_punrolled(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(0, m, 4):  # Unroll by 4
                C[i, j] += alpha * A[i, k] * A[j, k]
                C[i, j] += alpha * A[i, k + 1] * A[j, k + 1]
                C[i, j] += alpha * A[i, k + 2] * A[j, k + 2]
                C[i, j] += alpha * A[i, k + 3] * A[j, k + 3]
                               
# Vectorized inner loop
@vectorize(['float64(float64, float64, float64)'])
def vectorized_inner_loop(alpha, A_ik, A_jk):
    return alpha * A_ik * A_jk

# Vectorized syrk function
def syrk_vectorized(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i, j] += vectorized_inner_loop(alpha, A[i, k], A[j, k])
            
#loop_unrolled_vectorized
@njit(parallel=True)
def syrk_parallelized(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(m):
                C[i, j] += alpha * A[i, k] * A[j, k]
              
# Combined loop unrolling and vectorization and parallelization
@njit(parallel=True)
def syrk_combined(n, m, alpha, beta, C, A):
    for i in prange(n):
        for j in range(i + 1):
            C[i, j] *= beta
            for k in range(0, m, 4):  # Unroll by 4 
                C[i, j] += vectorized_inner_loop(alpha , A[i, k] , A[j, k])
                C[i, j] += vectorized_inner_loop(alpha , A[i, k + 1] , A[j, k + 1])
                C[i, j] += vectorized_inner_loop(alpha , A[i, k + 2] , A[j, k + 2])
                C[i, j] += vectorized_inner_loop(alpha , A[i, k + 3] , A[j, k + 3])

# Define the compilation pipeline with transformation pass
class MyCompiler(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(DeadCodeElimination, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with transformation
@njit(pipeline_class=MyCompiler)
def syrk_trans(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i][j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i][j] += alpha * A[i][k] * A[j][k]

# Define the compilation pipeline with auto vectorization pass
class MyCompiler2(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(VectorizationPass, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with auto vectorization pass
@njit(pipeline_class=MyCompiler2)
def syrk_auto_vector(n, m, alpha, beta, C, A):
    for i in range(n):
        for j in range(i + 1):
            C[i][j] *= beta
        for k in range(m):
            for j in range(i + 1):
                C[i][j] += alpha * A[i][k] * A[j][k]

#cpp 
def run_cpp_code(m, n):
    # Compile the C++ code
    #compile_command = ['g++', 'syrk.cpp', '-fopenmp', '-o', 'syrk']
    compile_command = ['icc', 'syrk.cpp', '-fopenmp', '-o', 'syrk']
    subprocess.run(compile_command, check=True)

    # Run the compiled C++ executable with the specified arguments
    run_command = ['./syrk', str(m), str(n)]
    subprocess.run(run_command, check=True)

# Print the resulting array
def print_array(C):
    for i in range(n):
        for j in range(n):
            if (i * n + j) % 20 == 0:
                print()
            print(f"{C[i][j]:.2f}", end=" ")

#print_array(C)


# main entry
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python syrk.py m n')
        sys.exit(1)
    
    alpha = 1.5
    beta = 1.2

    m, n = map(int, sys.argv[1:3])
    
    A, C = init_array(n, m)

        
    num_runs = 20
    
    # Warm-up runs
    # Run kernel
    kernel_syrk(n, m, alpha, beta, C, A)
       
    start = time.time()
    for _ in range(num_runs):
        kernel_syrk(n, m, alpha, beta, C, A)
    end = time.time()
    naive_time = end - start

    #unrolled
    syrk_unrolled(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_unrolled(n, m, alpha, beta, C, A)
    end = time.time()
    unrolled_time = end - start

    #parallelized_unrolled
    syrk_punrolled(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_punrolled(n, m, alpha, beta, C, A)
    end = time.time()
    parallelized_unrolled_time = end - start

    #vectorized    
    syrk_vectorized(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_vectorized(n, m, alpha, beta, C, A)
    end = time.time()
    vectorized_time = end - start
    
     #parallelized     
    syrk_parallelized(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_parallelized(n, m, alpha, beta, C, A)
    end = time.time()
    parallelized_time = end - start
    
    #transformation     
    syrk_trans(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_trans(n, m, alpha, beta, C, A)
    end = time.time()
    transformation_time = end - start
    
    #auto_vectorization     
    syrk_auto_vector(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_auto_vector(n, m, alpha, beta, C, A)
    end = time.time()
    auto_vector_time = end - start
    
    #combined
    syrk_combined(n, m, alpha, beta, C, A)
    
    start = time.time()
    for _ in range(num_runs):
        syrk_combined(n, m, alpha, beta, C, A)
    end = time.time()
    combined_time = end - start
    

    print('naive time:         {}'.format(naive_time))
    print('unrolled time:     {}'.format(unrolled_time))
    print('vectorized time:  {}'.format(vectorized_time))
    print('parallelized time:{}'.format(parallelized_time))
    print('transformation time:{}'.format(transformation_time))
    print('auto vectorized time:{}'.format(auto_vector_time))
    print('punrolled time:    {}'.format(parallelized_unrolled_time))
    print('combined time:   {}'.format(combined_time))
    
    run_cpp_code(m, n)
 

import sys
import time
import numpy as np
import subprocess
from numba import njit, prange, float64, vectorize
from numba.core.compiler import DefaultPassBuilder, CompilerBase,  IRProcessing
from numba.core.typed_passes import NopythonTypeInference, DeadCodeElimination
from numba.core import ir
from types import FunctionType, BuiltinFunctionType
from numba.core.compiler_machinery import FunctionPass, register_pass


# Array initialization
def init_array(m, n):
    alpha = 1.5
    beta = 1.2
    C = np.zeros((m, n), dtype=np.float64)
    A = np.zeros((m, m), dtype=np.float64)
    B = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            C[i, j] = (i + j) % 100 / m
            B[i, j] = (n + i - j) % 100 / m

    for i in range(m):
        for j in range(i + 1):
            A[i, j] = (i + j) % 100 / m
        for j in range(i + 1, m):
            A[i, j] = -999  # regions of arrays that should not be used

    return alpha, beta, C, A, B

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
def kernel_symm(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2
            
# Main computational kernel with parallelization
@njit(parallel=True)
def symm_paralleized(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

# Main computational kernel with loop unrolling
def symm_unrolled(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(0, i, 4):  # Unroll by 4
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
                C[k + 1, j] += alpha * B[i, j] * A[i, k + 1]
                temp2 += B[k + 1, j] * A[i, k + 1]
                C[k + 2, j] += alpha * B[i, j] * A[i, k + 2]
                temp2 += B[k + 2, j] * A[i, k + 2]
                C[k + 3, j] += alpha * B[i, j] * A[i, k + 3]
                temp2 += B[k + 3, j] * A[i, k + 3]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2           


# Vectorized inner loop
@vectorize(['float64(float64, float64, float64)'])
def vectorized_inner_loop(alpha, B_ij, A_ik):
    return alpha * B_ij * A_ik

# Vectorized symm function
def symm_vectorized(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):  # Vectorized inner loop
                C[k, j] += vectorized_inner_loop(alpha, B[i, j], A[i, k])
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2
           
# Main computational kernel with loop unrolling and parallelization
@njit(parallel=True)
def symm_punrolled(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(0, i, 4):  # Unroll by 4
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
                C[k + 1, j] += alpha * B[i, j] * A[i, k + 1]
                temp2 += B[k + 1, j] * A[i, k + 1]
                C[k + 2, j] += alpha * B[i, j] * A[i, k + 2]
                temp2 += B[k + 2, j] * A[i, k + 2]
                C[k + 3, j] += alpha * B[i, j] * A[i, k + 3]
                temp2 += B[k + 3, j] * A[i, k + 3]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2           


# Combined loop unrolling and vectorization and parallelization
@njit(parallel=True)
def symm_combined(m, n, alpha, beta, C, A, B):
    for i in prange(m):
        temp2 = 0
        for j in range(n):
            for k in range(0, i, 4):  # Unroll by 4
                C[k, j] += vectorized_inner_loop(alpha , B[i, j] , A[i, k])
                temp2 += B[k, j] * A[i, k]
                C[k + 1, j] += vectorized_inner_loop(alpha , B[i, j] , A[i, k+1])
                temp2 += B[k + 1, j] * A[i, k + 1]
                C[k + 2, j] += vectorized_inner_loop(alpha , B[i, j] , A[i, k+2])
                temp2 += B[k + 2, j] * A[i, k + 2]
                C[k + 3, j] += vectorized_inner_loop(alpha , B[i, j] , A[i, k+3])
                temp2 += B[k + 3, j] * A[i, k + 3]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

# Define the compilation pipeline with transformation pass
class MyCompiler(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(DeadCodeElimination, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with transformation pass
@njit(pipeline_class=MyCompiler)
def symm_trans(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2


# Define the compilation pipeline with auto vectorization pass
class MyCompiler2(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(VectorizationPass, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with auto vectorization pass
@njit(pipeline_class=MyCompiler2)
def symm_auto_vector(m, n, alpha, beta, C, A, B):
    for i in range(m):
        temp2 = 0
        for j in range(n):
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2

#cpp 
def run_cpp_code(m, n):
    # Compile the C++ code
    #compile_command = ['g++', 'symm.cpp', '-fopenmp',  '-o', 'symm']
    compile_command = ['icc', 'symm.cpp', '-fopenmp',  '-o', 'symm']
    subprocess.run(compile_command, check=True)

    # Run the compiled C++ executable with the specified arguments
    run_command = ['./symm', str(m), str(n)]
    subprocess.run(run_command, check=True)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python symm.py m n')
        sys.exit(1)

    m, n = map(int, sys.argv[1:3])

    alpha, beta, C, A, B = init_array(m, n)
    
    num_runs=20

    #naive
    #warm_up
    kernel_symm(m, n, alpha, beta, C, A, B)
  
    start = time.time()
    for _ in range(num_runs):
        kernel_symm(m, n, alpha, beta, C, A, B)
    end = time.time()
    naive_time = end - start
    
    #unrolled
    symm_unrolled(m, n, alpha, beta, C, A, B)

    start = time.time()
    for _ in range(num_runs):
        symm_unrolled(m, n, alpha, beta, C, A, B)
    end = time.time()
    unrolled_time = end - start
    
    #vectorized
    symm_vectorized(m, n, alpha, beta, C, A, B)

    start = time.time()
    for _ in range(num_runs):
        symm_vectorized(m, n, alpha, beta, C, A, B)
    end = time.time()
    vectorized_time = end - start
    
    #parallelized
    symm_paralleized(m, n, alpha, beta, C, A, B)
    
    start = time.time()
    for _ in range(num_runs):
        symm_paralleized(m, n, alpha, beta, C, A, B)
    end = time.time()
    parallelizd_time = end - start
    
    #transformed
    symm_trans(m, n, alpha, beta, C, A, B)
    
    start = time.time()
    for _ in range(num_runs):
        symm_trans(m, n, alpha, beta, C, A, B)
    end = time.time()
    trans_time = end - start
    
    #auto_vector
    symm_trans(m, n, alpha, beta, C, A, B)
    
    start = time.time()
    for _ in range(num_runs):
        symm_auto_vector(m, n, alpha, beta, C, A, B)
    end = time.time()
    auto_vector_time = end - start
    
    #parallelized_unrolled    
    symm_punrolled(m, n, alpha, beta, C, A, B)

    start = time.time()
    for _ in range(num_runs):
        symm_punrolled(m, n, alpha, beta, C, A, B)
    end = time.time()
    punrolled_time = end - start
    
    #combined
    symm_combined(m, n, alpha, beta, C, A, B)

    start = time.time()
    for _ in range(num_runs):
        symm_combined(m, n, alpha, beta, C, A, B)
    end = time.time()
    combined_time = end - start

    print('naive time:      {}'.format(naive_time))
    print('unrolled time:   {}'.format(unrolled_time))
    print('vectorized time: {}'.format(vectorized_time))
    print('parallelized time: {}'.format(parallelizd_time))
    print('transformed time: {}'.format(trans_time))
    print('auto vectorized time: {}'.format(auto_vector_time))
    print('punrolled time: {}'.format(punrolled_time))
    print('combined time:   {}'.format(combined_time))
    
    run_cpp_code(m, n)


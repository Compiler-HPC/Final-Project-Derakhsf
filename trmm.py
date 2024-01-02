import sys
import time  
import numpy as np
import subprocess
from numba import njit, prange,vectorize, float64
from numba.core.compiler import DefaultPassBuilder, CompilerBase,  IRProcessing
from numba.core.typed_passes import NopythonTypeInference, DeadCodeElimination
from numba.core import ir
from types import FunctionType, BuiltinFunctionType
from numba.core.compiler_machinery import FunctionPass, register_pass



# Array initialization
def init_array(m, n):
    alpha = 1.5
    A = np.zeros((m, m), dtype=np.float64)
    B = np.zeros((m, n), dtype=np.float64)
    
    for i in range(m):
        for j in range(i):
            A[i, j] = ((i + j) % m) / m
        A[i, i] = 1.0
        for j in range(n):
            B[i, j] = ((n + (i - j)) % n) / n
    
    return alpha, A, B
    
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
def kernel_trmm(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]

# Main computational kernel with parallelization
@njit(parallel=True)
def kernel_trmm_parallelized(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            for k in range(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]


@vectorize(['float64(float64, float64)'])
def vectorized_kernel(acc, a_b):
    return acc + a_b

# Main computational kernel with loop vectorization
def kernel_trmm_vectorized(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            acc = 0.0

            for k in range(i + 1, m):
                acc = vectorized_kernel(acc, A[k, i] * B[k, j])

            B[i, j] = alpha * (B[i, j] + acc)

# Main computational kernel with manual loop unrolling
def kernel_trmm_unrolled(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0

            for k in range(i + 1, m, 4):
                if k<32:            	
                    acc0 += A[k, i] * B[k, j]
                if k+1<32:
                    acc1 += A[k + 1, i] * B[k + 1, j]
                if k+2<32:
                    acc2 += A[k + 2, i] * B[k + 2, j]
                if k+3<32:
                    acc3 += A[k + 3, i] * B[k + 3, j]

            B[i, j] = alpha * (B[i, j] + acc0 + acc1 + acc2 + acc3)
            
# Main computational kernel with manual loop unrolling and parallelization
@njit(parallel=True)
def kernel_trmm_punrolled(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0

            for k in range(i + 1, m, 4):
                if k<32:            	
                    acc0 += A[k, i] * B[k, j]
                if k+1<32:
                    acc1 += A[k + 1, i] * B[k + 1, j]
                if k+2<32:
                    acc2 += A[k + 2, i] * B[k + 2, j]
                if k+3<32:
                    acc3 += A[k + 3, i] * B[k + 3, j]

            B[i, j] = alpha * (B[i, j] + acc0 + acc1 + acc2 + acc3)

# Main computational kernel with a combination of vectorization and unrolling and parallelization
@njit(parallel=True)
def kernel_trmm_combined(m, n, alpha, A, B):
    for i in prange(m):
        for j in range(n):
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0

            for k in range(i + 1, m, 4):
            
                if k<32:            	
                    acc0 = vectorized_kernel(acc0, A[k, i] * B[k, j])
                if k+1<32:
                    acc1 = vectorized_kernel(acc1, A[k + 1, i] * B[k + 1, j])
                if k+2<32:
                    acc2 = vectorized_kernel(acc2, A[k + 2, i] * B[k + 2, j])
                if k+3<32:
                    acc3 = vectorized_kernel(acc3, A[k + 3, i] * B[k + 3, j])              
                
             

            B[i, j] = alpha * (B[i, j] + acc0 + acc1 + acc2 + acc3)

# Define the compilation pipeline with transformation pass
class MyCompiler(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(DeadCodeElimination, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with transformation
@njit(pipeline_class=MyCompiler)
def kernel_trmm_trans(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]
            
 # Define the compilation pipeline with auto vectorization pass
class MyCompiler2(CompilerBase):
    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(VectorizationPass, NopythonTypeInference)
        pm.finalize()
        return [pm]


# Main computational kernel with auto vectorization pass
@njit(pipeline_class=MyCompiler2)
def kernel_trmm_auto_vector(m, n, alpha, A, B):
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]

#cpp 
def run_cpp_code(m, n):
    # Compile the C++ code
    #compile_command = ['g++', 'trmm.cpp', '-fopenmp',  '-o', 'trmm']
    compile_command = ['icc', 'trmm.cpp', '-fopenmp',  '-o', 'trmm']
    subprocess.run(compile_command, check=True)

    # Run the compiled C++ executable with the specified arguments
    run_command = ['./trmm', str(m), str(n)]
    subprocess.run(run_command, check=True)

# Print the resulting array
def print_array(B):
    m, n = B.shape
    for i in range(m):
        for j in range(n):
            if (i * m + j) % 20 == 0:
                print()
            print(f"{B[i, j]:.2f}", end=" ")

if __name__ == '__main__':
    # Retrieve problem size.
    if len(sys.argv) != 3:
        print('Usage: python trmm.py m n ')
        sys.exit(1)
    m, n = map(int, sys.argv[1:3])

    # Initialize array(s).
    alpha, A, B = init_array(m, n)

    num_runs = 20
    
    # Warm-up run
    # Run kernel
    kernel_trmm(m, n, alpha, A, B)
    
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm(m, n, alpha, A, B)
    end = time.time()
    naive_time = end - start
    
    
    #unrolled
    kernel_trmm_unrolled(m, n, alpha, A, B)

    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_unrolled(m, n, alpha, A, B)
    end = time.time()
    unrolled_time = end - start
    
    
    #parallelized_unrolled
    kernel_trmm_punrolled(m, n, alpha, A, B)

    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_punrolled(m, n, alpha, A, B)
    end = time.time()
    parallelized_unrolled_time = end - start
    

    #vectorized
    kernel_trmm_vectorized(m, n, alpha, A, B)
	
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_vectorized(m, n, alpha, A, B)
    end = time.time()
    vectorized_time = end - start  

    #transformation
    kernel_trmm_trans(m, n, alpha, A, B)
	
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_trans(m, n, alpha, A, B)
    end = time.time()
    trans_time = end - start      

    #auto_vector
    kernel_trmm_auto_vector(m, n, alpha, A, B)
	
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_auto_vector(m, n, alpha, A, B)
    end = time.time()
    auto_vector_time = end - start          
    
    
    #parallelized
    kernel_trmm_parallelized(m, n, alpha, A, B)
    
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_parallelized(m, n, alpha, A, B)
    end = time.time()
    parallelized_time = end - start
    
    
    #combined
    kernel_trmm_combined(m, n, alpha, A, B)
     
    start = time.time()
    for _ in range(num_runs):
        kernel_trmm_combined(m, n, alpha, A, B)
    end = time.time()
    combined_time = end - start

    

    print('naive time:      {}'.format(naive_time))
    print('unrolled time:   {}'.format(unrolled_time))
    print('vectorized time: {}'.format(vectorized_time))
    print('parallelized time: {}'.format(parallelized_time))
    print('transformation time: {}'.format(trans_time))
    print('auto vectorized time: {}'.format(auto_vector_time))
    print('punrolled time:   {}'.format(parallelized_unrolled_time))
    print('combined time:   {}'.format(combined_time))
    
    run_cpp_code(m, n)


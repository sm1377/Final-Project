import numpy as np
import time
import cupy as cp
import json

# Define the matrix sizes
matrix_size = 500
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

# Define the number of times to run the code
num_runs = int(input("Enter the number of times to run the code: "))

# Define the output file path
output_file_path = "output.json"

# CPU Matrix Multiplication
def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using CPU
    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# GPU Matrix Multiplication
def gpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using GPU
    gpu_matrix_a = cp.asarray(matrix_a)
    gpu_matrix_b = cp.asarray(matrix_b)
    gpu_result = cp.dot(gpu_matrix_a, gpu_matrix_b)
    cpu_result = cp.asnumpy(gpu_result)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# Run the code multiple times and record the results
cpu_total_time = 0
gpu_total_time = 0
results = []
for i in range(num_runs):
    # Run CPU Matrix Multiplication
    cpu_result, cpu_execution_time = cpu_matrix_multiplication(matrix_a, matrix_b)
    cpu_total_time += cpu_execution_time

    # Run GPU Matrix Multiplication
    gpu_result, gpu_execution_time = gpu_matrix_multiplication(matrix_a, matrix_b)
    gpu_total_time += gpu_execution_time

    # Compare the results (optional)
    result = {
        "run": i+1,
        "cpu_result": cpu_result.tolist(),
        "gpu_result": gpu_result.tolist(),
        "cpu_execution_time": cpu_execution_time,
        "gpu_execution_time": gpu_execution_time
    }
    results.append(result)

# Calculate the average execution time
avg_cpu_execution_time = cpu_total_time / num_runs
avg_gpu_execution_time = gpu_total_time / num_runs

# Add the average execution time to the results
results.append({
    "average_cpu_execution_time": avg_cpu_execution_time,
    "average_gpu_execution_time": avg_gpu_execution_time
})

# Write the results to a JSON file
with open(output_file_path, "w") as f:
    json.dump(results, f)
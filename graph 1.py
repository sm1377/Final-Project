import numpy as np
import time
import json
import matplotlib.pyplot as plt

# Ask the user for the number of times to run the program
num_runs = int(input("Enter the number of times to run the program: "))

# Create lists to store the execution times
cpu_execution_times = []
gpu_execution_times = []

# Loop over each run
for i in range(num_runs):
    # Generate random matrices
    matrix_size = np.random.randint(100, 1000)
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    # Perform matrix multiplication on the CPU and record the execution time
    cpu_start_time = time.time()
    np.dot(matrix_a, matrix_b)
    cpu_execution_time = time.time() - cpu_start_time
    cpu_execution_times.append(cpu_execution_time)

    # Perform matrix multiplication on the GPU (if available) and record the execution time
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        matrix_a_gpu = cp.asarray(matrix_a)
        matrix_b_gpu = cp.asarray(matrix_b)
        gpu_start_time = time.time()
        cp.dot(matrix_a_gpu, matrix_b_gpu)
        gpu_execution_time = time.time() - gpu_start_time
        gpu_execution_times.append(gpu_execution_time)
    except:
        pass

# Print the execution times
print("CPU execution times:", cpu_execution_times)
if gpu_execution_times:
    print("GPU execution times:", gpu_execution_times)

# Plot the execution times
fig, ax = plt.subplots()
ax.plot(cpu_execution_times, label="CPU")
if gpu_execution_times:
    ax.plot(gpu_execution_times, label="GPU")
ax.set_xlabel("Run number")
ax.set_ylabel("Execution time (s)")
ax.set_title("Matrix multiplication execution times")
ax.legend()

# Add text labels to the plot
cpu_mean_time = np.mean(cpu_execution_times)
gpu_mean_time = np.mean(gpu_execution_times) if gpu_execution_times else None
ax.text(0.05, 0.95, f"Mean CPU time: {cpu_mean_time:.3f} s", transform=ax.transAxes, va="top")
if gpu_mean_time:
    ax.text(0.05, 0.9, f"Mean GPU time: {gpu_mean_time:.3f} s", transform=ax.transAxes, va="top")

# Save the plot to a file
fig.savefig("graph.png", dpi=300)

# Show the plot
plt.show()
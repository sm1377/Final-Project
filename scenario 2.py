import cv2
import time
import json

# Load an image for processingcd opencv
image_path = "C:/Users/Salar-PC/Pictures/image.jpg"
image = cv2.imread(image_path)

# Define the number of times to run the code
num_runs = int(input("Enter the number of times to run the code: "))

# Define the output file path
output_file_path = "output2.json"


# CPU Image Processing
def cpu_image_processing(image):
    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# # GPU Image Processing
# def gpu_image_processing(image):
#     # Perform image processing operations using GPU
#     # Example: Convert the image to grayscale
#     gpu_image = cv2.cuda_GpuMat()
#     gpu_image.upload(image)
#     gpu_gray_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
#     gray_image = gpu_gray_image.download()
#     return gray_image


# Run the code multiple times and record the results
cpu_total_time = 0
gpu_total_time = 0
results = []
for i in range(num_runs):
    start_time = time.time()

    # Run CPU Image Processing
    cpu_result = cpu_image_processing(image)
    cpu_execution_time = time.time() - start_time
    cpu_total_time += cpu_execution_time

    # Run GPU Image Processing
    # gpu_result = gpu_image_processing(image)
    gpu_execution_time = cpu_execution_time
    gpu_total_time += gpu_execution_time

    # Save the results (optional)
    cv2.imwrite("cpu_result_{}.jpg".format(i), cpu_result)
    # cv2.imwrite("gpu_result_{}.jpg".format(i), gpu_result)

    # Write the execution time and results to the output file
    result = {
        "run": i+1,
        "cpu_execution_time": cpu_execution_time,
        "gpu_execution_time": gpu_execution_time,
        "cpu_result_path": "cpu_result_{}.jpg".format(i),
        "gpu_result_path": "gpu_result_{}.jpg".format(i)
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
    f.write(json.dumps(results, indent=4))
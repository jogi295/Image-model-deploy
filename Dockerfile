# Stage 1: Build Stage
FROM tensorflow/tensorflow:2.9.1 as builder 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the TFLite model, labels, and Python script to the container
COPY mobilenet_v2.tflite .
COPY labels.txt .
COPY predict.py .

# Freeze the requirements into a new requirements file
RUN pip freeze > /app/frozen_requirements.txt

# Stage 2: Runtime Stage
FROM python:3.10-slim  

# Set the working directory inside the container
WORKDIR /app

# Copy the frozen requirements file from the build stage
COPY --from=builder /app/frozen_requirements.txt .

# Install required Python packages from the frozen requirements
RUN pip install --no-cache-dir -r frozen_requirements.txt

# Copy required files from the builder stage
COPY --from=builder /app/mobilenet_v2.tflite .
COPY --from=builder /app/labels.txt .
COPY --from=builder /app/predict.py .

# Install required system dependencies for running TFLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose the port if needed
EXPOSE 5000

# Run the prediction script
CMD ["python", "predict.py"]

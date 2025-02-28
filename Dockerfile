# Use an official Python runtime as a parent image (adjust as needed)
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your application
CMD ["python", "main.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# Copy the current directory contents into the container at /app
COPY --chown=user . .

# Ensure permissions for logs and other potential write targets
RUN mkdir -p logs && chown -R user:user logs

# Build argument for versioning
ARG MODEL_VERSION=1.0.0
ENV MODEL_VERSION=${MODEL_VERSION}

# Switch to non-root user
USER user

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Define environment variable
ENV PYTHONPATH=/app
ENV PORT=7860

# Command to run the application
CMD exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}

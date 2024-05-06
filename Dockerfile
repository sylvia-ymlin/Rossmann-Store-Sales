# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure logs and models directories exist
RUN mkdir -p logs models data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run the Streamlit portfolio dashboard
CMD ["streamlit", "run", "streamlit_portfolio/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]

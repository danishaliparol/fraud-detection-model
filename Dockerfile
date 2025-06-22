# Use a specific, stable Python 3.10 image for consistency
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first. This optimizes Docker caching:
# if requirements.txt doesn't change, this layer is rebuilt, speeding up subsequent builds.
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes fraud_detection.py, api.py, train.csv, score.csv, etc.
COPY . .

# Expose the port where the Flask API will listen for real-time predictions
EXPOSE 8000

# Command to run the Gunicorn server for your Flask API
# This runs the 'app' object inside the 'api.py' module
# '-w 4': 4 worker processes (adjust based on CPU cores for production)
# '-b 0.0.0.0:8000': Bind to all network interfaces on port 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api:app"]
# Start with lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
# We copy requirements.txt first to use Docker's cache
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . /app

# Tell Docker to expose port 8081
# This is the port your app.py runs on
EXPOSE 8081

# Health check - Kubernetes uses this to verify container is alive
# We'll skip the complex healthcheck from the README for now
# to keep it simple. The CMD is the most important part.

# Run the FastAPI app
# Note: This is different from 'python app.py'
# This is the production-ready way to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]

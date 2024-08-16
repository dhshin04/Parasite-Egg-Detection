# Run "docker build -t myapp:latest ." to build docker
# Run "docker run -p 5000:5000 myapp:latest" to launch docker on port 5000

# Python Base Image
FROM python:3.12.5-slim-bookworm

# Set working directory
WORKDIR /app

# Install OpenGL library (with updates) from OpenCV - needed to use cv2 on Docker
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy requirements.txt for Python dependencies into /app (COPY source dest)
COPY requirements.txt ./

# Install Python dependencies without storing cache (which can slow down build)
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to container's workspace (COPY source dest)
COPY . .

# Hint which port Flask app runs on - for local host testing
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
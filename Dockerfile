FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV (headless)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Ensure captured_faces directory exists
RUN mkdir -p captured_faces

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

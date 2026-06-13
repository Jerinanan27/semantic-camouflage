# The recipe Hugging Face Spaces follows to run your app.
# Each line is a step the build computer executes top to bottom.

FROM python:3.11-slim

WORKDIR /app

# Install the Python libraries first (cached unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the web page.
COPY src/ ./src/
COPY frontend/ ./frontend/

# Models are downloaded at runtime; give them a writable cache directory.
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Spaces sends web traffic to port 7860.
EXPOSE 7860

# Start the web server.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]

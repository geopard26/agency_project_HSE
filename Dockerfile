# Use official Python slim image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-app.txt ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Copy the entire project into the container
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.headless=true"]

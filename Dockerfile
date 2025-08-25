# Base Python image
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /src

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only necessary files
COPY interface.py /src/

# Expose default Streamlit port
EXPOSE 8502

# Run the Streamlit app
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.enableCORS=false"]

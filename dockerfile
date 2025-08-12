# Use the official Python 3.10 base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Optional: Check contents of the app folder inside the container (debugging only)
RUN ls -la /app

# Set Streamlit to run the app
CMD ["streamlit", "run", "docker_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.9-slim
	
	

# Install system dependencies for building C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
	
	
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Set environment variable to indicate the app is in development
ENV FLASK_APP=app.py
ENV FLASK_ENV=development


# Command to run the app
CMD ["flask", "run", "--host=0.0.0.0"]
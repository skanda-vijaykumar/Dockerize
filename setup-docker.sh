#!/bin/bash
set -e

echo "Setting up fully dockerized Nicomatic Chatbot..."

# Create necessary directories
mkdir -p data
mkdir -p static/js
mkdir -p templates
mkdir -p extracted_best

# Check if .env file exists, if not create from template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file to add your API keys."
fi

# Make sure all files have the right permissions
chmod +x *.sh

# Check if we have the required files
for file in "docker-compose.yml" "Dockerfile" "requirements.txt" "postgres-init.sql"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file $file is missing"
        exit 1
    fi
done

# Check if route18.py exists and modify if necessary
if [ -f "route18.py" ]; then
    echo "✅ Found route18.py"
    
    # Check if environment variables are already set in route18.py
    if ! grep -q "POSTGRES_CONNECTION_STRING" route18.py; then
        echo "⚠️ Warning: route18.py does not contain connection string variables."
        echo "You will need to modify route18.py to use environment variables for connections."
        echo "See the connection-updates.py file for details."
    fi
    
    if ! grep -q "OLLAMA_BASE_URL" route18.py; then
        echo "⚠️ Warning: route18.py does not contain Ollama base URL configuration."
        echo "You will need to modify route18.py to use environment variables for Ollama connections."
        echo "See the connection-updates.py file for details."
    fi
else
    echo "❌ Error: route18.py not found"
    exit 1
fi

# Stop any possibly running containers
echo "Stopping any existing containers..."
docker-compose down || true

# Start with a clean build
echo "Building containers..."
docker-compose build

# Start the services
echo "Starting services..."
docker-compose up -d postgres ollama

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
echo "This may take a minute or two..."
attempt=0
max_attempts=30
until docker-compose ps | grep postgres | grep -q "(healthy)" && docker-compose ps | grep ollama | grep -q "(healthy)" || [ $attempt -ge $max_attempts ]; do
    attempt=$((attempt+1))
    echo "Waiting... ($attempt/$max_attempts)"
    sleep 10
done

if [ $attempt -ge $max_attempts ]; then
    echo "❌ Timed out waiting for services to be healthy"
    echo "Please check logs with: docker-compose logs"
    exit 1
fi

echo "✅ Services are healthy"

# Start model initialization
echo "Starting model initialization (this will download the models, it may take a while)..."
docker-compose up -d model-init

# Start the application
echo "Starting application..."
docker-compose up -d app

echo "Checking if services are running correctly..."
if docker-compose ps | grep -q "Up" ; then
    echo "✅ All services are running!"
else
    echo "❌ Some services failed to start"
    echo "Please check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "You can access the application at: http://localhost:8000"
echo ""
echo "To view logs, run:"
echo "  docker-compose logs -f app"
echo ""
echo "To see Ollama model download progress, run:"
echo "  docker-compose logs -f model-init"
echo ""
echo "To stop all services, run:"
echo "  docker-compose down"
echo ""
echo "Note: The first startup may take some time as the Ollama models need to be downloaded."
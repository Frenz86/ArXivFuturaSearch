#!/bin/bash
# Locust load testing startup script for Linux/Mac

set -e

echo "===================================="
echo "ArXivFuturaSearch Load Testing"
echo "===================================="
echo ""

# Check if application is running
echo "Checking if application is running on http://localhost:8000..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Application is not running!"
    echo "Please start the application first:"
    echo "  uvicorn app.main:app --reload"
    echo ""
    exit 1
fi

echo "Application is running!"
echo ""

# Create results directory if it doesn't exist
mkdir -p locust/results

# Display menu
echo "Choose load test intensity:"
echo "  1. Light (50 users, 10/s spawn)"
echo "  2. Medium (200 users, 20/s spawn)"
echo "  3. Heavy (500 users, 50/s spawn)"
echo "  4. Custom"
echo "  5. Web UI mode (interactive)"
echo ""

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        USERS=50
        SPAWN=10
        ;;
    2)
        USERS=200
        SPAWN=20
        ;;
    3)
        USERS=500
        SPAWN=50
        ;;
    4)
        read -p "Enter number of users: " USERS
        read -p "Enter spawn rate (users per second): " SPAWN
        ;;
    5)
        echo ""
        echo "Starting Locust web interface..."
        echo "Open http://localhost:8089 in your browser to configure and run tests."
        echo "Press Ctrl+C to stop."
        echo ""
        python -m locust -f locust/locustfile.py --host=http://localhost:8000
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "Starting load test with $USERS users (spawn rate: $SPAWN/s)..."
echo "Press Ctrl+C to stop."
echo ""

python -m locust -f locust/locustfile.py \
    --headless \
    --host=http://localhost:8000 \
    --users $USERS \
    --spawn-rate $SPAWN \
    --run-time 300s \
    --csv locust/results/headless

echo ""
echo "Load test completed! Results saved to locust/results/"

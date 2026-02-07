"""Locust configuration for load testing.

This file allows you to customize load testing parameters without modifying locustfile.py.
"""

# Target host (can be overridden with --host flag)
HOST = "http://localhost:8000"

# User distribution for multi-user scenarios
# Format: (user_class, weight)
USER_CLASSES = [
    ("locustfile.QuickTestUser", 1),   # 10% - Quick tests
    ("locustfile.NormalUser", 5),      # 50% - Normal users
    ("locustfile.PowerUser", 3),       # 30% - Power users
    ("locustfile.AuthUser", 1),        # 10% - Authenticated users
]

# Test duration (seconds)
RUN_TIME = 300  # 5 minutes

# User spawn settings
USERS = 100
SPAWN_RATE = 10  # Users per second

# Custom queries to test (overrides default queries)
CUSTOM_QUERIES = [
    "What is machine learning?",
    "Explain neural networks",
    "How does deep learning work?",
]

# Endpoint weights for targeted testing
ENDPOINT_WEIGHTS = {
    "/ask": 10,           # Main search endpoint
    "/ask (with filters)": 3,
    "/ask (stream)": 1,
    "/health": 1,
}

# Request timeouts (seconds)
TIMEOUTS = {
    "ask": 30,      # /ask endpoint
    "health": 5,    # /health endpoint
    "auth": 10,     # Auth endpoints
}

# Failure rate threshold (percentage) - triggers warnings if exceeded
FAILURE_RATE_THRESHOLD = 5

# Response time thresholds (milliseconds)
RESPONSE_TIME_THRESHOLDS = {
    "avg": 2000,      # Average should be under 2s
    "p95": 5000,      # 95th percentile under 5s
    "p99": 10000,     # 99th percentile under 10s
}

# Logging
LOG_SLOW_REQUESTS = True  # Log requests exceeding timeout
SLOW_REQUEST_THRESHOLD = 5000  # ms

# Output settings
OUTPUT_DIR = "results"
OUTPUT_PREFIX = "load_test"

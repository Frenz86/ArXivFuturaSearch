"""Locust load testing file for ArXivFuturaSearch.

This file defines load testing scenarios for stress testing the RAG application.

Run with:
    locust -f locust/locustfile.py --host=http://localhost:8000
Or:
    python -m locust -f locust/locustfile.py --host=http://localhost:8000

For headless mode:
    locust -f locust/locustfile.py --headless --host=http://localhost:8000 --users 100 --spawn-rate 10
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import random


# Sample queries for load testing the /ask endpoint
SAMPLE_QUERIES = [
    "What is transformer architecture?",
    "Explain attention mechanism in neural networks",
    "How does BERT work?",
    "What is the difference between CNN and RNN?",
    "Explain gradient descent optimization",
    "What are the latest advances in computer vision?",
    "How does GPT generate text?",
    "What is reinforcement learning?",
    "Explain the concept of overfitting",
    "What are word embeddings?",
    "How does batch normalization work?",
    "What is transfer learning?",
    "Explain the vanishing gradient problem",
    "What are autoencoders used for?",
    "How does sentiment analysis work?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of backpropagation",
    "What are generative adversarial networks?",
    "How does LSTM solve the vanishing gradient problem?",
    "What is the role of activation functions?",
]


class QuickTestUser(HttpUser):
    """Quick test user - simulates light load with short queries."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task(3)
    def ask_question(self):
        """Test the main /ask endpoint with various queries."""
        query = random.choice(SAMPLE_QUERIES)

        payload = {
            "question": query,
            "top_k": random.choice([3, 5, 10]),
            "stream": False,
        }

        with self.client.post(
            "/ask",
            json=payload,
            catch_response=True,
            name="/ask"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure
                    if "answer" not in data:
                        response.failure("Missing 'answer' in response")
                    elif "sources" not in data:
                        response.failure("Missing 'sources' in response")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                # Rate limited - mark as success but log it
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def health_check(self):
        """Test the health endpoint."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")


class NormalUser(HttpUser):
    """Normal user - simulates typical user behavior."""

    wait_time = between(2, 5)  # Wait 2-5 seconds between tasks

    @task(5)
    def ask_question(self):
        """Ask a question with filters."""
        query = random.choice(SAMPLE_QUERIES)

        payload = {
            "question": query,
            "top_k": 5,
            "stream": False,
        }

        # Sometimes add filters
        if random.random() > 0.7:
            payload["filters"] = {
                "categories": ["cs.AI", "cs.LG", "cs.CV"][random.randint(0, 2):]
            }

        with self.client.post("/ask", json=payload, catch_response=True, name="/ask (with filters)") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" not in data:
                        response.failure("Missing 'answer' in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code != 429:
                response.failure(f"Got status {response.status_code}")

    @task(2)
    def ask_streaming(self):
        """Test streaming endpoint (short request to avoid hanging)."""
        payload = {
            "question": random.choice(SAMPLE_QUERIES[:3]),  # Use simpler queries
            "top_k": 3,
            "stream": True,
        }

        # Note: streaming responses may not be fully supported in Locust
        # This tests the endpoint initialization
        with self.client.post("/ask", json=payload, catch_response=True, name="/ask (stream)", timeout=5) as response:
            if response.status_code not in [200, 206]:
                response.failure(f"Streaming failed: {response.status_code}")

    @task(1)
    def health_check(self):
        """Health check."""
        self.client.get("/health", name="/health")


class PowerUser(HttpUser):
    """Power user - simulates heavy usage scenario."""

    wait_time = between(0.5, 2)  # Very short wait time

    @task(8)
    def rapid_questions(self):
        """Rapid fire questions."""
        payload = {
            "question": random.choice(SAMPLE_QUERIES),
            "top_k": random.choice([5, 10, 15]),
            "stream": False,
        }

        self.client.post("/ask", json=payload, name="/ask (rapid)")

    @task(2)
    def concurrent_searches(self):
        """Simulate concurrent search patterns."""
        # Send multiple requests quickly
        for _ in range(3):
            payload = {
                "question": random.choice(SAMPLE_QUERIES),
                "top_k": 5,
                "stream": False,
            }
            self.client.post("/ask", json=payload, name="/ask (concurrent)")


class AuthUser(HttpUser):
    """User that also tests authentication endpoints."""

    wait_time = between(3, 8)

    def on_start(self):
        """Register/login on start."""
        # Try to login first
        login_payload = {
            "email": f"loadtest_{random.randint(1000, 9999)}@test.com",
            "password": "TestPassword123!",
        }

        response = self.client.post("/api/auth/login", json=login_payload, name="/api/auth/login")
        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                self.token = data["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
            else:
                self.token = None
                self.headers = {}
        else:
            # Try to register
            register_payload = {
                **login_payload,
                "username": f"loadtest_{random.randint(1000, 9999)}",
            }
            response = self.client.post("/api/auth/register", json=register_payload, name="/api/auth/register")
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            else:
                self.token = None
                self.headers = {}

    @task(3)
    def authenticated_ask(self):
        """Ask with authentication."""
        if hasattr(self, 'headers') and self.headers:
            payload = {
                "question": random.choice(SAMPLE_QUERIES),
                "top_k": 5,
                "stream": False,
            }
            self.client.post("/ask", json=payload, headers=self.headers, name="/ask (authenticated)")

    @task(1)
    def check_profile(self):
        """Check user profile."""
        if hasattr(self, 'headers') and self.headers:
            self.client.get("/api/auth/me", headers=self.headers, name="/api/auth/me")


# Event handlers for additional metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log slow requests."""
    if response_time > 5000:  # Log requests taking more than 5 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary when test stops."""
    if isinstance(environment.runner, MasterRunner):
        print("\n=== Load Test Completed ===")
        print(f"Total requests: {environment.runner.stats.total.num_requests}")
        print(f"Failure rate: {environment.runner.stats.total.fail_ratio:.2%}")
        print(f"Average response time: {environment.runner.stats.total.avg_response_time:.0f}ms")
        print(f"Min response time: {environment.runner.stats.total.min_response_time:.0f}ms")
        print(f"Max response time: {environment.runner.stats.total.max_response_time:.0f}ms")
        print(f"Requests/s: {environment.runner.stats.total.total_rps:.2f}")

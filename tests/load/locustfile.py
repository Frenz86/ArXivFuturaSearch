"""
Load tests using Locust.

Tests the performance and scalability of the application under load.

Run with:
    locust -f tests/load/locustfile.py
"""

import os
import time
import json
from typing import Dict, Any

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


class ArXivSearchUser(HttpUser):
    """
    Simulates a user searching for academic papers.
    """

    # Wait time between tasks (in seconds)
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a user starts."""
        # Login (if auth is enabled)
        self.client.headers.update({
            "Content-Type": "application/json",
        })

    @task(5)
    def search_papers(self):
        """Search for papers (high frequency task)."""
        queries = [
            "machine learning",
            "neural networks",
            "quantum computing",
            "computer vision",
            "natural language processing",
            "reinforcement learning",
            "transformer models",
        ]

        query = self.environment.runner.user.random.choice(queries)

        with self.client.post(
            "/api/search",
            json={"query": query},
            catch_response=True,
            name="/api/search",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    response.success()
                else:
                    response.failure("Missing answer in response")
            elif response.status_code == 429:
                # Rate limiting - expected under load
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def ask_question(self):
        """Ask a specific question."""
        questions = [
            "What are the latest advances in transformer models?",
            "Explain the attention mechanism in neural networks.",
            "How do GANs work?",
            "What is the difference between CNN and RNN?",
        ]

        question = self.environment.runner.user.random.choice(questions)

        self.client.post(
            "/api/ask",
            json={"question": question},
            name="/api/ask",
        )

    @task(1)
    def get_paper_details(self):
        """Get details of a specific paper."""
        # Simulate paper IDs
        paper_ids = [
            "2301.07041",
            "2305.14314",
            "2307.12345",
        ]

        paper_id = self.environment.runner.user.random.choice(paper_ids)

        self.client.get(
            f"/api/papers/{paper_id}",
            name="/api/papers/{paper_id}",
        )


class AuthUser(HttpUser):
    """
    Simulates an authenticated user.
    """

    wait_time = between(2, 10)

    def on_start(self):
        """Login and store token."""
        # Try to login
        response = self.client.post(
            "/api/auth/login",
            json={
                "email": "test@example.com",
                "password": "testpassword123",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}",
            })
        else:
            self.token = None

    @task(3)
    def get_my_alerts(self):
        """Get user's alerts."""
        if not self.token:
            return

        self.client.get(
            "/api/alerts",
            name="/api/alerts",
        )

    @task(2)
    def create_alert(self):
        """Create a new alert."""
        if not self.token:
            return

        keywords = ["transformer", "attention", "gan", "reinforcement learning"]
        keyword = self.environment.runner.user.random.choice(keywords)

        self.client.post(
            "/api/alerts",
            json={
                "name": f"Alert for {keyword}",
                "keywords": [keyword],
                "categories": ["cs.AI", "cs.LG"],
                "notification_method": "email",
            },
            name="/api/alerts [POST]",
        )

    @task(1)
    def get_collections(self):
        """Get user's collections."""
        if not self.token:
            return

        self.client.get(
            "/api/collections",
            name="/api/collections",
        )


class AdminUser(HttpUser):
    """
    Simulates an admin user.
    """

    wait_time = between(5, 15)

    def on_start(self):
        """Login as admin."""
        response = self.client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.com",
                "password": "adminpassword123",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}",
            })

    @task(3)
    def get_audit_log(self):
        """Get audit logs."""
        if not self.token:
            return

        self.client.get(
            "/api/audit/logs",
            params={"limit": 50},
            name="/api/audit/logs",
        )

    @task(1)
    def get_system_health(self):
        """Check system health."""
        self.client.get(
            "/health",
            name="/health",
        )


# =============================================================================
# TEST EVENTS
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print(f"\n{'='*60}")
    print("Load Test Starting")
    print(f"Target: {BASE_URL}")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print(f"\n{'='*60}")
    print("Load Test Complete")
    print(f"{'='*60}\n")

    if isinstance(environment.runner, MasterRunner):
        print("\nMaster Runner Statistics:")
        print(f"  Workers: {environment.runner.worker_count}")
        print(f"  Users: {environment.runner.target_user_count}")

    stats = environment.runner.stats

    print("\nOverall Statistics:")
    print(f"  Total Requests: {stats.total.num_requests}")
    print(f"  Failures: {stats.total.fail_ratio:.2%}")
    print(f"  RPS: {stats.total.total_rps:.2f}")
    print(f"  Avg Response Time: {stats.total.avg_response_time:.0f}ms")
    print(f"  Median Response Time: {stats.total.median_response_time:.0f}ms")
    print(f"  95th Percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")


# =============================================================================
# CUSTOM METRICS
# =============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Track custom metrics for each request.
    """
    # Log slow requests
    if response_time > 5000:  # 5 seconds
        print(f"\n[WARNING] Slow request detected:")
        print(f"  Endpoint: {name}")
        print(f"  Time: {response_time:.0f}ms")
        print(f"  Type: {request_type}")


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

# Example command line usage:
#
# Single process:
#   locust -f tests/load/locustfile.py
#
# With master/worker:
#   locust -f tests/load/locustfile.py --master
#   locust -f tests/load/locustfile.py --worker --master-host=localhost
#
# Headless mode:
#   locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 1m
#
# Options:
#   -u, --users: Number of users
#   -r, --spawn-rate: Users spawned per second
#   -t, --run-time: Run time (e.g., 1m, 5m, 1h)
#   --headless: Run without web UI
#   --host: Target host URL

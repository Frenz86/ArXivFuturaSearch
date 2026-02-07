# Load Testing with Locust

This directory contains load testing configurations for stress testing the ArXivFuturaSearch application.

## Prerequisites

Install Locust:

```bash
pip install locust
```

Or add to dev dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "locust>=2.0.0",
    # ... other dev dependencies
]
```

## Starting the Application

Make sure the application is running:

```bash
# Development mode
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running Load Tests

### Web UI Mode (Recommended for Testing)

```bash
locust -f locust/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser.

### Headless Mode (For Automated Testing)

```bash
# Light load: 50 users, spawn 10 per second
locust -f locust/locustfile.py --headless --host=http://localhost:8000 --users 50 --spawn-rate 10 --run-time 60s

# Medium load: 200 users, spawn 20 per second
locust -f locust/locustfile.py --headless --host=http://localhost:8000 --users 200 --spawn-rate 20 --run-time 120s

# Heavy load: 500 users, spawn 50 per second
locust -f locust/locustfile.py --headless --host=http://localhost:8000 --users 500 --spawn-rate 50 --run-time 300s
```

### Distributed Mode (Multiple Workers)

For high-load testing, use multiple machines:

**Master:**
```bash
locust -f locust/locustfile.py --master --host=http://localhost:8000 --expect-workers 4
```

**Workers (on each machine):**
```bash
locust -f locust/locustfile.py --worker --master-host=<master-ip>
```

## Test Scenarios

### User Types

| User Type | Description | Wait Time | Tasks |
|-----------|-------------|-----------|-------|
| `QuickTestUser` | Light load, quick tests | 1-3s | Health checks, simple queries |
| `NormalUser` | Typical user behavior | 2-5s | Questions with/without filters, streaming |
| `PowerUser` | Heavy usage | 0.5-2s | Rapid fire questions, concurrent searches |
| `AuthUser` | Authenticated requests | 3-8s | Login/register, authenticated queries |

### Default Weights

In the web UI, you can specify the mix of user types. Example distribution:

- 70% NormalUser
- 20% PowerUser
- 10% AuthUser

## Key Metrics to Monitor

### Response Times
- **Average**: Should be < 2 seconds for /ask endpoint
- **95th percentile**: Should be < 5 seconds
- **99th percentile**: Should be < 10 seconds

### Error Rates
- **Target**: < 1% failure rate
- **Acceptable**: < 5% failure rate
- **Critical**: > 10% failure rate

### Throughput
- **Requests/sec**: Monitor how RPS affects response times
- **Concurrent users**: Find the breaking point

## Interpreting Results

### Good Performance
- Avg response time < 2s
- 95th percentile < 5s
- Failure rate < 1%
- Stable RPS under load

### Needs Optimization
- Avg response time > 3s
- 95th percentile > 8s
- Failure rate > 5%
- Response times degrading with more users

### Critical Issues
- Timeouts or connection errors
- Memory exhaustion
- Database connection pool exhaustion
- Response times > 30s

## Common Issues and Solutions

### Rate Limiting (429 errors)
- Reduce spawn rate or number of users
- Check `RATE_LIMIT_*` settings in `.env`

### Connection Errors
- Check database connection pool size
- Verify Redis connection (if caching enabled)
- Check network bandwidth

### Slow Response Times
- Enable caching (`CACHE_ENABLED=true`)
- Reduce retrieval size (`RETRIEVAL_K`)
- Disable reranking (`RERANK_ENABLED=false`)

### Memory Issues
- Reduce batch sizes
- Check for memory leaks in embedding models
- Monitor with memory profiler

## Advanced Options

### Custom Queries

Edit `SAMPLE_QUERIES` in `locustfile.py` to test domain-specific queries.

### Different Hosts

```bash
# Test staging environment
locust -f locust/locustfile.py --host=https://staging.example.com

# Test production (use with caution!)
locust -f locust/locustfile.py --host=https://api.example.com
```

### Export Results

```bash
# Export to CSV
locust -f locust/locustfile.py --headless --host=http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 60s \
    --csv results/load_test

# Export to HTML
locust -f locust/locustfile.py --headless --host=http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 60s \
    --html results/report.html
```

## CI/CD Integration

Add to `.github/workflows/load-test.yml`:

```yaml
name: Load Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install locust
      - run: |
          uvicorn app.main:app &
          sleep 10
          locust -f locust/locustfile.py --headless \
            --host=http://localhost:8000 \
            --users 50 --spawn-rate 5 --run-time 30s \
            --csv results/load_test
      - uses: actions/upload-artifact@v4
        with:
          name: load-test-results
          path: results/
```

## Notes

- Streaming responses (`stream: true`) may not be fully tested due to SSE limitations
- Always run load tests in a staging environment first
- Monitor system resources (CPU, memory, disk) during tests
- Consider using docker-compose for a realistic environment setup

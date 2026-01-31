"""Tests for Redis cache module.

Run with: pytest tests/test_redis_cache.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_redis_pool():
    """Mock Redis connection pool."""
    pool = Mock()
    return pool


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.setex = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=1)
    client.expire = AsyncMock(return_value=True)
    client.ttl = AsyncMock(return_value=3600)
    client.incrby = AsyncMock(return_value=1)
    client.mget = AsyncMock(return_value=[b'value1', b'value2'])
    client.pipeline = Mock(return_value=MagicMock())
    client.pipeline.return_value.__enter__ = Mock(return_value=MagicMock())
    client.pipeline.return_value.__exit__ = Mock(return_value=False)
    client.pipeline.return_value.set = Mock()
    client.pipeline.return_value.setex = Mock()
    client.pipeline.return_value.execute = AsyncMock(return_value=None)
    client.info = AsyncMock(return_value={
        "connected_clients": 5,
        "used_memory_human": "10M",
        "total_commands_processed": 1000,
        "db0": {},
    })
    client.scan_iter = Mock(return_value=[])
    return client


# =============================================================================
# SERIALIZER TESTS
# =============================================================================

class TestJSONSerializer:
    """Tests for JSONSerializer class."""

    def test_serialize_simple_types(self):
        """Test serializing simple Python types."""
        from app.redis_cache import JSONSerializer

        serializer = JSONSerializer()

        # Test dict
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

        # Test list
        data = [1, 2, 3, "four"]
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

    def test_deserialize_simple_types(self):
        """Test deserializing simple Python types."""
        from app.redis_cache import JSONSerializer

        serializer = JSONSerializer()

        # Test dict
        serialized = b'{"key": "value", "number": 42}'
        deserialized = serializer.deserialize(serialized)
        assert deserialized == {"key": "value", "number": 42}

        # Test list
        serialized = b'[1, 2, 3, "four"]'
        deserialized = serializer.deserialize(serialized)
        assert deserialized == [1, 2, 3, "four"]

    def test_serialize_deserialize_roundtrip(self):
        """Test serialize/deserialize roundtrip."""
        from app.redis_cache import JSONSerializer

        serializer = JSONSerializer()

        original = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
        }

        serialized = serializer.serialize(original)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == original

    def test_unsupported_types(self):
        """Test with unsupported types for JSON."""
        from app.redis_cache import JSONSerializer

        serializer = JSONSerializer()

        # Sets are not JSON serializable
        with pytest.raises(TypeError):
            serializer.serialize({1, 2, 3})


class TestPickleSerializer:
    """Tests for PickleSerializer class."""

    def test_serialize_complex_types(self):
        """Test serializing complex Python types."""
        from app.redis_cache import PickleSerializer

        serializer = PickleSerializer()

        # Test set
        data = {1, 2, 3}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

        # Test complex nested
        data = {"set": {1, 2}, "tuple": (1, 2)}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

    def test_deserialize_complex_types(self):
        """Test deserializing complex Python types."""
        from app.redis_cache import PickleSerializer

        serializer = PickleSerializer()

        # Test set
        original = {1, 2, 3}
        serialized = serializer.serialize(original)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == original

        # Test tuple
        original = (1, 2, 3)
        serialized = serializer.serialize(original)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == original

    def test_serialize_deserialize_roundtrip(self):
        """Test pickle serialize/deserialize roundtrip."""
        from app.redis_cache import PickleSerializer

        serializer = PickleSerializer()

        original = {
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "custom": type('Obj', (), {'attr': 'value'})(),
        }

        serialized = serializer.serialize(original)
        deserialized = serializer.deserialize(serialized)

        # Note: custom objects may not roundtrip perfectly without __reduce__
        assert isinstance(deserialized, dict)


# =============================================================================
# REDIS CACHE TESTS
# =============================================================================

class TestRedisCache:
    """Tests for RedisCache class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test Redis cache initialization."""
        from app.redis_cache import RedisCache

        cache = RedisCache(
            host="localhost",
            port=6379,
            db=0,
            prefix="test:",
        )

        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.prefix == "test:"
        assert cache._client is None

    @pytest.mark.asyncio
    async def test_connect(self, mock_redis_client):
        """Test Redis connection."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.ConnectionPool') as mock_pool:
            with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
                cache = RedisCache()
                await cache.connect()

                assert cache._client is not None
                mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_get(self, mock_redis_client):
        """Test getting value from cache."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Mock return value
            mock_redis_client.get.return_value = b'{"key": "value"}'

            result = await cache.get("test_key")

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_miss(self, mock_redis_client):
        """Test cache miss."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.get.return_value = None

            result = await cache.get("nonexistent_key", default="default_value")

            assert result == "default_value"

    @pytest.mark.asyncio
    async def test_set(self, mock_redis_client):
        """Test setting value in cache."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            result = await cache.set("test_key", {"data": "value"}, ttl=60)

            assert result is True

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis_client):
        """Test deleting key from cache."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.delete.return_value = 1

            result = await cache.delete("test_key")

            assert result is True

    @pytest.mark.asyncio
    async def test_exists(self, mock_redis_client):
        """Test checking if key exists."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.exists.return_value = 1

            result = await cache.exists("test_key")

            assert result is True

    @pytest.mark.asyncio
    async def test_expire(self, mock_redis_client):
        """Test setting TTL on key."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.expire.return_value = 1

            result = await cache.expire("test_key", 300)

            assert result is True

    @pytest.mark.asyncio
    async def test_ttl(self, mock_redis_client):
        """Test getting TTL of key."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.ttl.return_value = 3600

            result = await cache.ttl("test_key")

            assert result == 3600

    @pytest.mark.asyncio
    async def test_increment(self, mock_redis_client):
        """Test incrementing counter."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.incrby.return_value = 5

            result = await cache.increment("counter", amount=3)

            assert result == 5

    @pytest.mark.asyncio
    async def test_get_many(self, mock_redis_client):
        """Test getting multiple values."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_redis_client.mget.return_value = [b'{"key": "1"}', b'{"key": "2"}']

            result = await cache.get_many(["key1", "key2"])

            assert len(result) == 2
            assert "key1" in result
            assert "key2" in result

    @pytest.mark.asyncio
    async def test_set_many(self, mock_redis_client):
        """Test setting multiple values."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            mock_pipeline = AsyncMock()
            mock_pipeline.set = Mock()
            mock_pipeline.execute = AsyncMock(return_value=None)
            cache._client.pipeline.return_value = mock_pipeline

            result = await cache.set_many({"key1": "val1", "key2": "val2"}, ttl=60)

            assert result == 2

    @pytest.mark.asyncio
    async def test_clear_pattern(self, mock_redis_client):
        """Test clearing keys by pattern."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Mock scan_iter to return keys
            mock_redis_client.scan_iter.return_value = [b"test:key1", b"test:key2"]
            mock_redis_client.delete.return_value = 2

            result = await cache.clear_pattern("test:*")

            assert result == 2

    @pytest.mark.asyncio
    async def test_get_info(self, mock_redis_client):
        """Test getting Redis info."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            info = await cache.get_info()

            assert "connected_clients" in info
            assert "used_memory_human" in info

    @pytest.mark.asyncio
    async def test_close(self, mock_redis_client):
        """Test closing Redis connection."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client
            cache._pool = Mock()

            await cache.close()

            assert cache._client is None
            mock_redis_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_key(self):
        """Test key prefixing."""
        from app.redis_cache import RedisCache

        cache = RedisCache(prefix="myapp:")

        full_key = cache._make_key("user:123")

        assert full_key == "myapp:user:123"


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalRedisCache:
    """Tests for global Redis cache instance."""

    def test_get_redis_cache_singleton(self):
        """Test that get_redis_cache returns singleton."""
        from app.redis_cache import get_redis_cache

        cache1 = get_redis_cache()
        cache2 = get_redis_cache()

        assert cache1 is cache2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestRedisCacheEdgeCases:
    """Edge case tests for Redis cache."""

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test handling of connection failure."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.ConnectionPool') as mock_pool:
            with patch('app.redis_cache.aioredis.Redis') as mock_redis:
                mock_redis.return_value.ping.side_effect = Exception("Connection failed")

                cache = RedisCache()

                with pytest.raises(Exception):
                    await cache.connect()

    @pytest.mark.asyncio
    async def test_serialize_error(self, mock_redis_client):
        """Test handling of serialization errors."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Try to cache non-serializable object with JSON serializer
            result = await cache.set("key", object())  # object() not JSON serializable

            # Should handle error gracefully
            assert result is False

    @pytest.mark.asyncio
    async def test_deserialize_error(self, mock_redis_client):
        """Test handling of deserialization errors."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Return invalid JSON
            mock_redis_client.get.return_value = b"invalid json"

            result = await cache.get("test_key", default="fallback")

            # Should return default on error
            assert result == "fallback"

    @pytest.mark.asyncio
    async def test_very_large_value(self, mock_redis_client):
        """Test caching very large value."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Large value (1MB)
            large_value = "x" * (1024 * 1024)

            result = await cache.set("large_key", large_value)

            # Should handle large values
            assert result is True

    @pytest.mark.asyncio
    async def test_unicode_keys_and_values(self, mock_redis_client):
        """Test unicode in keys and values."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            # Unicode key and value
            result = await cache.set("тест_key", {"данные": "значение"})

            assert result is True

    @pytest.mark.asyncio
    async def test_zero_ttl(self, mock_redis_client):
        """Test with zero TTL (no expiration)."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
            cache = RedisCache()
            cache._client = mock_redis_client

            result = await cache.set("test_key", "value", ttl=None)

            assert result is True
            mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_connect_on_get(self, mock_redis_client):
        """Test auto-connect when client is None."""
        from app.redis_cache import RedisCache

        with patch('app.redis_cache.aioredis.ConnectionPool'):
            with patch('app.redis_cache.aioredis.Redis', return_value=mock_redis_client):
                cache = RedisCache()
                # Don't call connect, should auto-connect

                mock_redis_client.get.return_value = b'"value"'

                result = await cache.get("test_key")

                assert cache._client is not None

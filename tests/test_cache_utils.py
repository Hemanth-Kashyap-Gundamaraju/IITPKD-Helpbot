"""
Description: Unit tests for backend.utils.cache_utils.ResponseCache.
"""
import time
from backend.utils.cache_utils import ResponseCache

def test_cache_put_and_get():
    """Test that a basic put and get works correctly."""
    cache = ResponseCache()
    cache.put("What is IITPKD?", "An engineering institute.")
    
    assert cache.get("What is IITPKD?") == "An engineering institute."
    assert cache.get("Unknown question") is None
    
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1

def test_cache_normalization():
    """Test that different spacings/casings hit the same cache entry."""
    cache = ResponseCache()
    cache.put("  What is   IITPKD?  ", "An engineering institute.")
    
    # These should all hit the same normalized key
    assert cache.get("what is iitpkd?") == "An engineering institute."
    assert cache.get("What is IITPKD?") == "An engineering institute."

def test_cache_expiry():
    """Test that cache entries expire after TTL."""
    cache = ResponseCache(ttl_seconds=1)
    cache.put("Test expiry", "This will expire.")
    
    assert cache.get("Test expiry") == "This will expire."
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    assert cache.get("Test expiry") is None
    assert cache.stats()["size"] == 0

def test_cache_eviction():
    """Test that cache evicts oldest entry when max_size is reached."""
    cache = ResponseCache(max_size=2)
    
    cache.put("Q1", "A1")
    time.sleep(0.01) # Ensure different timestamps
    cache.put("Q2", "A2")
    time.sleep(0.01)
    
    assert cache.stats()["size"] == 2
    
    # This should evict Q1
    cache.put("Q3", "A3")
    
    assert cache.stats()["size"] == 2
    assert cache.get("Q1") is None
    assert cache.get("Q2") == "A2"
    assert cache.get("Q3") == "A3"

def test_cache_clear():
    """Test that clear() empties the cache and resets stats."""
    cache = ResponseCache()
    cache.put("Q1", "A1")
    cache.get("Q1") # hit
    cache.get("Q2") # miss
    
    cache.clear()
    
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0
    assert cache.get("Q1") is None

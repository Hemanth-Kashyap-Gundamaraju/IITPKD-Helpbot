"""
Description: Implements response caching to avoid redundant LLM/embedding API
    calls for repeated or identical questions. Supports exact-match caching
    with TTL-based expiry. This directly addresses the PRD requirement:
    "Use caching to save API credits."
Inputs: none (this file defines classes and functions).
Outputs: none.
Dependencies: none (pure Python, uses only stdlib).
Utilities: used by chain_builder.py and asgi.py.
"""

import hashlib
import time
import threading


class ResponseCache:
    """
    Description: Thread-safe in-memory cache for LLM responses. Maps a
        normalized question hash to the cached answer, with TTL-based
        expiry. This is the ONE place caching logic lives — every other
        module uses get() and put() without knowing the internals.
    Inputs (constructor): ttl_seconds (how long entries stay valid,
        default 24 hours), max_size (max entries before eviction,
        default 1000).
    Utilities: used by build_rag_chain_with_cache() in chain_builder.py.
    """

    def __init__(self, ttl_seconds=86400, max_size=1000):
        self._cache = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize_question(question):
        """
        Description: Normalizes a question for consistent cache key
            generation — lowercases, strips whitespace, and collapses
            multiple spaces. This ensures "What is IITPKD?" and
            "  what   is   iitpkd?  " hit the same cache entry.
        Inputs: question (string). No globals read.
        Outputs: returns a normalized string. No globals changed.
        Dependencies: none.
        Utilities: called by _make_key().
        """
        return " ".join(question.lower().strip().split())

    @staticmethod
    def _make_key(question):
        """
        Description: Produces a fixed-length cache key from a question
            string by normalizing it and then hashing with SHA-256.
        Inputs: question (string). No globals read.
        Outputs: returns a hex string. No globals changed.
        Dependencies: calls _normalize_question().
        Utilities: called by get() and put().
        """
        normalized = ResponseCache._normalize_question(question)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question):
        """
        Description: Looks up a cached answer for the given question.
            Returns the cached answer string if found and not expired,
            or None if not cached / expired. Thread-safe.
        Inputs: question (string). Reads self._cache, self._ttl_seconds.
        Outputs: returns a string or None. Updates self._hits or
            self._misses.
        Dependencies: calls _make_key().
        Utilities: called before invoking the RAG chain.
        """
        key = self._make_key(question)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            answer, timestamp = entry
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return answer

    def put(self, question, answer):
        """
        Description: Stores an answer in the cache, keyed by the
            normalized question hash. If the cache is full, evicts the
            oldest entry first. Thread-safe.
        Inputs: question (string), answer (string). Reads self._max_size.
        Outputs: no return value. Mutates self._cache.
        Dependencies: calls _make_key().
        Utilities: called after getting a response from the RAG chain.
        """
        key = self._make_key(question)
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (answer, time.time())

    def stats(self):
        """
        Description: Returns cache statistics for monitoring/analytics.
        Inputs: none. Reads self._hits, self._misses, self._cache.
        Outputs: returns a dict with hits, misses, size, and hit_rate.
        Dependencies: none.
        Utilities: called by analytics endpoints.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            }

    def clear(self):
        """
        Description: Empties the entire cache. Useful after re-ingestion
            when the underlying knowledge base has changed.
        Inputs: none.
        Outputs: no return value. Clears self._cache, resets counters.
        Dependencies: none.
        Utilities: called after update_database.py runs.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# One shared instance for the whole app.
_response_cache = ResponseCache()


def get_response_cache():
    """
    Description: Returns the shared ResponseCache instance. Thin accessor
        so other modules don't need to know about the module-level variable.
    Inputs: none.
    Outputs: returns the ResponseCache singleton.
    Dependencies: none.
    Utilities: called by chain_builder.py, asgi.py.
    """
    return _response_cache

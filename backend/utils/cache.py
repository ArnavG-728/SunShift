"""
In-Memory Caching Utility for SunShift
Provides TTL-based caching for external API responses to reduce redundant calls
and improve concurrent request handling.
"""
import time
import threading
from typing import Any, Optional, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe TTL cache for API responses."""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with TTL."""
        with self._lock:
            expiry = time.time() + (ttl or self.default_ttl)
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup(self) -> int:
        """Remove expired entries. Returns count of removed items."""
        with self._lock:
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if now > exp]
            for k in expired:
                del self._cache[k]
            return len(expired)


# Global cache instances with appropriate TTLs
weather_cache = TTLCache(default_ttl=300)      # 5 minutes for current weather
forecast_cache = TTLCache(default_ttl=1800)    # 30 minutes for forecasts
currency_cache = TTLCache(default_ttl=3600)    # 1 hour for currency rates


def cached(cache: TTLCache, key_func=None, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache: TTLCache instance to use
        key_func: Function to generate cache key from args/kwargs
        ttl: Optional override for TTL
    
    Example:
        @cached(weather_cache, key_func=lambda lat, lon: f"{lat:.2f},{lon:.2f}")
        def get_weather(lat, lon):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT for {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache MISS for {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def location_key(lat: float, lon: float) -> str:
    """Generate a normalized cache key from coordinates (2 decimal precision)."""
    return f"{lat:.2f},{lon:.2f}"


# Async version of cached decorator
def async_cached(cache: TTLCache, key_func=None, ttl: Optional[int] = None):
    """Async version of the cached decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT for {cache_key}")
                return cached_value
            
            logger.debug(f"Cache MISS for {cache_key}")
            result = await func(*args, **kwargs)
            
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

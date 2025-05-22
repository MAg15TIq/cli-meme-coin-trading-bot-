"""
Performance optimization utilities for the Solana Memecoin Trading Bot.
Implements parallel processing and caching mechanisms.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from cachetools import TTLCache, cached
import logging

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

class PerformanceOptimizer:
    def __init__(self, max_workers: int = 4, cache_ttl: int = 300):
        """
        Initialize the performance optimizer.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            cache_ttl: Time-to-live for cached results in seconds
        """
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self._cache_lock = threading.Lock()
        
    def parallel_map(self, func: Callable[[T], R], items: List[T], 
                    use_processes: bool = False) -> List[R]:
        """
        Execute a function in parallel on a list of items.
        
        Args:
            func: Function to execute
            items: List of items to process
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of results
        """
        pool = self.process_pool if use_processes else self.thread_pool
        return list(pool.map(func, items))
    
    async def async_parallel_map(self, func: Callable[[T], R], items: List[T],
                               use_processes: bool = False) -> List[R]:
        """
        Execute a function in parallel on a list of items asynchronously.
        
        Args:
            func: Function to execute
            items: List of items to process
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of results
        """
        loop = asyncio.get_event_loop()
        pool = self.process_pool if use_processes else self.thread_pool
        
        async def _process_item(item: T) -> R:
            return await loop.run_in_executor(pool, func, item)
        
        tasks = [_process_item(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def cached_call(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """
        Call a function with caching.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached result if available, otherwise new result
        """
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        
        with self._cache_lock:
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return self.cache[cache_key]
            
            result = func(*args, **kwargs)
            self.cache[cache_key] = result
            return result
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        with self._cache_lock:
            self.cache.clear()
    
    def shutdown(self) -> None:
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown()
        self.process_pool.shutdown()

# Global instance
performance_optimizer = PerformanceOptimizer()

# Decorator for caching function results
def cache_result(ttl: int = 300):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live for cached results in seconds
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return performance_optimizer.cached_call(func, *args, **kwargs)
        return wrapper
    return decorator

# Decorator for parallel processing
def parallel_processing(use_processes: bool = False):
    """
    Decorator to enable parallel processing for list operations.
    
    Args:
        use_processes: Whether to use processes instead of threads
    """
    def decorator(func: Callable[[List[T]], List[R]]) -> Callable[[List[T]], List[R]]:
        @functools.wraps(func)
        def wrapper(items: List[T]) -> List[R]:
            return performance_optimizer.parallel_map(func, items, use_processes)
        return wrapper
    return decorator 
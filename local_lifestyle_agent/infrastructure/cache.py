"""缓存模块

提供 LRU + TTL 缓存实现，用于缓存 API 响应以提高性能。

特性：
- LRU（最近最少使用）淘汰策略
- TTL（生存时间）过期检查
- 缓存统计（命中率、大小、淘汰次数）
- 线程安全

验证需求：4.1, 4.2, 4.3, 4.4, 4.5
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Optional
from pydantic import BaseModel, Field


class CacheStats(BaseModel):
    """缓存统计信息
    
    Attributes:
        hits: 缓存命中次数
        misses: 缓存未命中次数
        hit_rate: 缓存命中率（0.0-1.0）
        size: 当前缓存大小
        max_size: 最大缓存大小
        evictions: 缓存淘汰次数
    """
    
    hits: int = Field(default=0, description="缓存命中次数")
    misses: int = Field(default=0, description="缓存未命中次数")
    hit_rate: float = Field(default=0.0, description="缓存命中率")
    size: int = Field(default=0, description="当前缓存大小")
    max_size: int = Field(default=0, description="最大缓存大小")
    evictions: int = Field(default=0, description="缓存淘汰次数")


class Cache:
    """LRU + TTL 缓存实现
    
    使用 OrderedDict 实现 LRU（最近最少使用）淘汰策略，
    同时支持 TTL（生存时间）过期检查。
    
    线程安全：使用 threading.Lock 保护内部状态。
    
    Attributes:
        max_size: 最大缓存条目数
        ttl: 缓存条目生存时间（秒）
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """初始化缓存
        
        Args:
            max_size: 最大缓存条目数（默认 1000）
            ttl: 缓存条目生存时间（秒，默认 3600 = 1 小时）
        """
        self.max_size = max_size
        self.ttl = ttl
        
        # 缓存存储：key -> value
        self._cache: OrderedDict[str, Any] = OrderedDict()
        
        # 时间戳存储：key -> timestamp
        self._timestamps: dict[str, float] = {}
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # 线程锁
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        如果缓存命中且未过期，返回缓存值并更新 LRU 顺序。
        如果缓存未命中或已过期，返回 None。
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值（如果存在且未过期），否则返回 None
        
        验证需求：4.3, 4.5
        """
        with self._lock:
            # 检查缓存是否存在
            if key not in self._cache:
                self._misses += 1
                return None
            
            # 检查是否过期
            if self._is_expired(key):
                # 删除过期条目
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            
            # 缓存命中：移动到末尾（最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            
            return self._cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存值
        
        如果缓存已满，淘汰最久未使用的条目（LRU）。
        
        Args:
            key: 缓存键
            value: 缓存值
        
        验证需求：4.1, 4.2, 4.4
        """
        with self._lock:
            # 如果键已存在，先删除（稍后重新添加到末尾）
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
            
            # 如果缓存已满，淘汰最久未使用的条目
            elif len(self._cache) >= self.max_size:
                # OrderedDict 的第一个条目是最久未使用的
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
                self._evictions += 1
            
            # 添加新条目（添加到末尾）
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def invalidate(self, key: str):
        """使缓存失效（删除指定条目）
        
        Args:
            key: 缓存键
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息
        
        Returns:
            CacheStats: 缓存统计信息
        
        验证需求：4.5
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                hit_rate=hit_rate,
                size=len(self._cache),
                max_size=self.max_size,
                evictions=self._evictions
            )
    
    def _is_expired(self, key: str) -> bool:
        """检查缓存条目是否过期
        
        Args:
            key: 缓存键
        
        Returns:
            bool: 如果过期返回 True，否则返回 False
        
        验证需求：4.3
        """
        if key not in self._timestamps:
            return True
        
        timestamp = self._timestamps[key]
        age = time.time() - timestamp
        
        return age > self.ttl
    
    def __len__(self) -> int:
        """返回当前缓存大小"""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """检查缓存是否包含指定键（不检查过期）"""
        with self._lock:
            return key in self._cache

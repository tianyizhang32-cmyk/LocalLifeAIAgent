"""缓存模块单元测试

测试 Cache 类的功能：
- LRU 淘汰策略
- TTL 过期检查
- 缓存统计
- 线程安全

验证需求：4.1, 4.2, 4.3, 4.4, 4.5
"""

import time
import threading
import pytest
from local_lifestyle_agent.infrastructure.cache import Cache, CacheStats


class TestCache:
    """Cache 类单元测试"""
    
    def test_cache_initialization(self):
        """测试缓存初始化"""
        cache = Cache(max_size=100, ttl=60)
        
        assert cache.max_size == 100
        assert cache.ttl == 60
        assert len(cache) == 0
    
    def test_cache_set_and_get(self):
        """测试缓存设置和获取
        
        验证需求：4.1, 4.2
        """
        cache = Cache(max_size=10, ttl=3600)
        
        # 设置缓存
        cache.set("key1", "value1")
        cache.set("key2", {"data": "value2"})
        
        # 获取缓存
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == {"data": "value2"}
        assert cache.get("nonexistent") is None
    
    def test_cache_hit_and_miss(self):
        """测试缓存命中和未命中统计
        
        验证需求：4.5
        """
        cache = Cache(max_size=10, ttl=3600)
        
        # 初始统计
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        
        # 设置并获取（命中）
        cache.set("key1", "value1")
        cache.get("key1")
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.hit_rate == 1.0
        
        # 获取不存在的键（未命中）
        cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
    
    def test_cache_ttl_expiration(self):
        """测试 TTL 过期
        
        验证需求：4.3
        """
        cache = Cache(max_size=10, ttl=1)  # 1 秒 TTL
        
        # 设置缓存
        cache.set("key1", "value1")
        
        # 立即获取（未过期）
        assert cache.get("key1") == "value1"
        
        # 等待过期
        time.sleep(1.5)
        
        # 获取过期缓存（应返回 None）
        assert cache.get("key1") is None
        
        # 统计应显示未命中
        stats = cache.get_stats()
        assert stats.misses == 1
    
    def test_cache_lru_eviction(self):
        """测试 LRU 淘汰策略
        
        验证需求：4.4
        """
        cache = Cache(max_size=3, ttl=3600)
        
        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache) == 3
        
        # 添加第 4 个条目，应淘汰 key1（最久未使用）
        cache.set("key4", "value4")
        
        assert len(cache) == 3
        assert cache.get("key1") is None  # 已被淘汰
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        
        # 统计应显示 1 次淘汰
        stats = cache.get_stats()
        assert stats.evictions == 1
    
    def test_cache_lru_order_update(self):
        """测试 LRU 顺序更新
        
        验证需求：4.4
        """
        cache = Cache(max_size=3, ttl=3600)
        
        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # 访问 key1（更新为最近使用）
        cache.get("key1")
        
        # 添加第 4 个条目，应淘汰 key2（现在是最久未使用）
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # 未被淘汰
        assert cache.get("key2") is None  # 已被淘汰
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_cache_update_existing_key(self):
        """测试更新现有键"""
        cache = Cache(max_size=10, ttl=3600)
        
        # 设置初始值
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # 更新值
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"
        
        # 缓存大小不变
        assert len(cache) == 1
    
    def test_cache_invalidate(self):
        """测试缓存失效"""
        cache = Cache(max_size=10, ttl=3600)
        
        # 设置缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert len(cache) == 2
        
        # 使 key1 失效
        cache.invalidate("key1")
        
        assert len(cache) == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_cache_clear(self):
        """测试清空缓存"""
        cache = Cache(max_size=10, ttl=3600)
        
        # 设置多个缓存条目
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache) == 3
        
        # 清空缓存
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None
    
    def test_cache_stats(self):
        """测试缓存统计信息
        
        验证需求：4.5
        """
        cache = Cache(max_size=5, ttl=3600)
        
        # 初始统计
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.size == 0
        assert stats.max_size == 5
        assert stats.evictions == 0
        
        # 添加缓存条目
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 命中和未命中
        cache.get("key1")  # 命中
        cache.get("key2")  # 命中
        cache.get("key3")  # 未命中
        
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 2 / 3
        assert stats.size == 2
        
        # 触发淘汰
        for i in range(3, 8):
            cache.set(f"key{i}", f"value{i}")
        
        stats = cache.get_stats()
        assert stats.evictions == 2  # 淘汰了 key1 和 key2
        assert stats.size == 5
    
    def test_cache_thread_safety(self):
        """测试线程安全"""
        cache = Cache(max_size=100, ttl=3600)
        errors = []
        
        def worker(thread_id: int):
            """工作线程"""
            try:
                for i in range(100):
                    key = f"thread{thread_id}_key{i}"
                    cache.set(key, f"value{i}")
                    value = cache.get(key)
                    assert value == f"value{i}" or value is None
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查是否有错误
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_cache_contains(self):
        """测试 __contains__ 方法"""
        cache = Cache(max_size=10, ttl=3600)
        
        cache.set("key1", "value1")
        
        assert "key1" in cache
        assert "key2" not in cache
    
    def test_cache_with_complex_values(self):
        """测试缓存复杂数据类型"""
        cache = Cache(max_size=10, ttl=3600)
        
        # 字典
        cache.set("dict", {"a": 1, "b": 2})
        assert cache.get("dict") == {"a": 1, "b": 2}
        
        # 列表
        cache.set("list", [1, 2, 3])
        assert cache.get("list") == [1, 2, 3]
        
        # 嵌套结构
        cache.set("nested", {"list": [1, 2], "dict": {"x": "y"}})
        assert cache.get("nested") == {"list": [1, 2], "dict": {"x": "y"}}
    
    def test_cache_zero_ttl(self):
        """测试 TTL 为 0（立即过期）"""
        cache = Cache(max_size=10, ttl=0)
        
        cache.set("key1", "value1")
        
        # 即使立即获取，也应该过期
        time.sleep(0.1)
        assert cache.get("key1") is None
    
    def test_cache_stats_model(self):
        """测试 CacheStats 模型"""
        stats = CacheStats(
            hits=10,
            misses=5,
            hit_rate=0.67,
            size=50,
            max_size=100,
            evictions=3
        )
        
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.hit_rate == 0.67
        assert stats.size == 50
        assert stats.max_size == 100
        assert stats.evictions == 3
        
        # 测试默认值
        default_stats = CacheStats()
        assert default_stats.hits == 0
        assert default_stats.misses == 0
        assert default_stats.hit_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

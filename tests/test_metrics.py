"""单元测试：指标收集模块

测试 Counter、Histogram、Gauge 和 MetricsCollector 的功能。
"""

import time
import pytest
from local_lifestyle_agent.infrastructure.metrics import (
    Counter,
    Histogram,
    Gauge,
    MetricsCollector
)


class TestCounter:
    """测试 Counter 指标"""
    
    def test_counter_basic_increment(self):
        """测试基本的计数器增加"""
        counter = Counter("test_counter", "Test counter")
        
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc()
        assert counter.get() == 2.0
        
        counter.inc(amount=5.0)
        assert counter.get() == 7.0
    
    def test_counter_with_labels(self):
        """测试带标签的计数器"""
        counter = Counter("test_counter", "Test counter", ["status"])
        
        counter.inc({"status": "200"})
        counter.inc({"status": "200"})
        counter.inc({"status": "404"})
        
        assert counter.get({"status": "200"}) == 2.0
        assert counter.get({"status": "404"}) == 1.0
        assert counter.get({"status": "500"}) == 0.0
    
    def test_counter_multiple_labels(self):
        """测试多个标签的计数器"""
        counter = Counter("test_counter", "Test counter", ["method", "status"])
        
        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "POST", "status": "201"})
        
        assert counter.get({"method": "GET", "status": "200"}) == 2.0
        assert counter.get({"method": "POST", "status": "201"}) == 1.0
    
    def test_counter_prometheus_export(self):
        """测试 Prometheus 格式导出"""
        counter = Counter("test_counter", "Test counter", ["status"])
        counter.inc({"status": "200"}, amount=10.0)
        counter.inc({"status": "404"}, amount=2.0)
        
        output = counter.export_prometheus()
        
        assert "# HELP test_counter Test counter" in output
        assert "# TYPE test_counter counter" in output
        assert 'test_counter{status="200"} 10.0' in output
        assert 'test_counter{status="404"} 2.0' in output


class TestHistogram:
    """测试 Histogram 指标"""
    
    def test_histogram_basic_observe(self):
        """测试基本的直方图观测"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=[0.1, 0.5, 1.0]
        )
        
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        
        assert histogram.get_count() == 3
        assert histogram.get_sum() == pytest.approx(1.15, rel=1e-6)
    
    def test_histogram_with_labels(self):
        """测试带标签的直方图"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            labels=["api"],
            buckets=[0.1, 0.5, 1.0]
        )
        
        histogram.observe(0.2, {"api": "openai"})
        histogram.observe(0.3, {"api": "openai"})
        histogram.observe(0.7, {"api": "google"})
        
        assert histogram.get_count({"api": "openai"}) == 2
        assert histogram.get_count({"api": "google"}) == 1
        assert histogram.get_sum({"api": "openai"}) == pytest.approx(0.5, rel=1e-6)
    
    def test_histogram_buckets(self):
        """测试直方图分桶"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=[0.1, 0.5, 1.0]
        )
        
        # 观测值分布：0.05 (<=0.1), 0.3 (<=0.5), 0.8 (<=1.0), 1.5 (>1.0)
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(1.5)
        
        output = histogram.export_prometheus()
        
        # 验证累计分桶计数
        assert 'test_histogram_bucket{le="0.1"} 1' in output
        assert 'test_histogram_bucket{le="0.5"} 2' in output
        assert 'test_histogram_bucket{le="1.0"} 3' in output
        assert 'test_histogram_bucket{le="+Inf"} 4' in output
    
    def test_histogram_prometheus_export(self):
        """测试 Prometheus 格式导出"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=[0.5, 1.0]
        )
        
        histogram.observe(0.3)
        histogram.observe(0.7)
        
        output = histogram.export_prometheus()
        
        assert "# HELP test_histogram Test histogram" in output
        assert "# TYPE test_histogram histogram" in output
        assert "test_histogram_sum" in output
        assert "test_histogram_count" in output


class TestGauge:
    """测试 Gauge 指标"""
    
    def test_gauge_set(self):
        """测试仪表盘设置值"""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(10.0)
        assert gauge.get() == 10.0
        
        gauge.set(5.0)
        assert gauge.get() == 5.0
    
    def test_gauge_inc_dec(self):
        """测试仪表盘增加和减少"""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(10.0)
        gauge.inc()
        assert gauge.get() == 11.0
        
        gauge.inc(amount=5.0)
        assert gauge.get() == 16.0
        
        gauge.dec()
        assert gauge.get() == 15.0
        
        gauge.dec(amount=3.0)
        assert gauge.get() == 12.0
    
    def test_gauge_with_labels(self):
        """测试带标签的仪表盘"""
        gauge = Gauge("test_gauge", "Test gauge", ["instance"])
        
        gauge.set(10.0, {"instance": "server1"})
        gauge.set(20.0, {"instance": "server2"})
        
        assert gauge.get({"instance": "server1"}) == 10.0
        assert gauge.get({"instance": "server2"}) == 20.0
    
    def test_gauge_prometheus_export(self):
        """测试 Prometheus 格式导出"""
        gauge = Gauge("test_gauge", "Test gauge", ["instance"])
        gauge.set(10.0, {"instance": "server1"})
        gauge.set(20.0, {"instance": "server2"})
        
        output = gauge.export_prometheus()
        
        assert "# HELP test_gauge Test gauge" in output
        assert "# TYPE test_gauge gauge" in output
        assert 'test_gauge{instance="server1"} 10.0' in output
        assert 'test_gauge{instance="server2"} 20.0' in output


class TestMetricsCollector:
    """测试 MetricsCollector"""
    
    def test_record_request(self):
        """测试记录请求"""
        collector = MetricsCollector()
        
        collector.record_request(0.5, 200)
        collector.record_request(1.2, 200)
        collector.record_request(0.3, 404)
        
        assert collector.request_total.get({"status": "200"}) == 2.0
        assert collector.request_total.get({"status": "404"}) == 1.0
        assert collector.request_duration_seconds.get_count() == 3
    
    def test_record_api_call(self):
        """测试记录 API 调用"""
        collector = MetricsCollector()
        
        collector.record_api_call("openai", 1.5, 200)
        collector.record_api_call("openai", 2.0, 200)
        collector.record_api_call("google_places", 0.8, 200)
        
        assert collector.api_call_total.get({"api": "openai", "status": "200"}) == 2.0
        assert collector.api_call_total.get({"api": "google_places", "status": "200"}) == 1.0
        assert collector.api_call_duration_seconds.get_count({"api": "openai"}) == 2
    
    def test_record_error(self):
        """测试记录错误"""
        collector = MetricsCollector()
        
        collector.record_error("API_TIMEOUT")
        collector.record_error("API_TIMEOUT")
        collector.record_error("VALIDATION_ERROR")
        
        assert collector.error_total.get({"error_type": "API_TIMEOUT"}) == 2.0
        assert collector.error_total.get({"error_type": "VALIDATION_ERROR"}) == 1.0
    
    def test_cache_metrics(self):
        """测试缓存指标"""
        collector = MetricsCollector()
        
        # 记录缓存命中和未命中
        collector.record_cache_hit()
        collector.record_cache_hit()
        collector.record_cache_miss()
        
        assert collector.cache_hit_total.get() == 2.0
        assert collector.cache_miss_total.get() == 1.0
        assert collector.cache_hit_rate.get() == pytest.approx(2.0 / 3.0, rel=1e-6)
        
        # 更新缓存大小
        collector.update_cache_size(100)
        assert collector.cache_size.get() == 100.0
    
    def test_active_requests(self):
        """测试活跃请求数"""
        collector = MetricsCollector()
        
        collector.update_active_requests(5)
        assert collector.active_requests.get() == 5.0
        
        collector.update_active_requests(10)
        assert collector.active_requests.get() == 10.0
    
    def test_resource_usage(self):
        """测试资源使用情况"""
        collector = MetricsCollector()
        
        collector.update_resource_usage()
        
        # 验证内存和 CPU 指标已更新
        assert collector.memory_usage_bytes.get() > 0
        assert collector.cpu_usage_percent.get() >= 0
    
    def test_prometheus_export(self):
        """测试 Prometheus 格式导出"""
        collector = MetricsCollector()
        
        # 记录一些指标
        collector.record_request(0.5, 200)
        collector.record_api_call("openai", 1.0, 200)
        collector.record_error("API_TIMEOUT")
        collector.record_cache_hit()
        collector.update_active_requests(3)
        
        output = collector.export_prometheus()
        
        # 验证输出包含所有指标
        assert "request_total" in output
        assert "api_call_total" in output
        assert "error_total" in output
        assert "cache_hit_total" in output
        assert "request_duration_seconds" in output
        assert "api_call_duration_seconds" in output
        assert "active_requests" in output
        assert "cache_hit_rate" in output
        assert "memory_usage_bytes" in output
        assert "cpu_usage_percent" in output
        
        # 验证格式正确
        assert "# HELP" in output
        assert "# TYPE" in output


class TestEdgeCases:
    """测试边缘情况"""
    
    def test_counter_zero_increment(self):
        """测试计数器零增量"""
        counter = Counter("test", "Test")
        counter.inc(amount=0.0)
        assert counter.get() == 0.0
    
    def test_histogram_empty(self):
        """测试空直方图"""
        histogram = Histogram("test", "Test")
        assert histogram.get_count() == 0
        assert histogram.get_sum() == 0.0
    
    def test_gauge_negative_values(self):
        """测试仪表盘负值"""
        gauge = Gauge("test", "Test")
        gauge.set(-10.0)
        assert gauge.get() == -10.0
        
        gauge.inc(amount=5.0)
        assert gauge.get() == -5.0
    
    def test_cache_hit_rate_zero_total(self):
        """测试缓存命中率（总数为零）"""
        collector = MetricsCollector()
        assert collector.cache_hit_rate.get() == 0.0
    
    def test_missing_label_values(self):
        """测试缺失标签值"""
        counter = Counter("test", "Test", ["label1", "label2"])
        
        # 只提供部分标签
        counter.inc({"label1": "value1"})
        assert counter.get({"label1": "value1"}) == 1.0
        
        # 标签顺序不同
        counter.inc({"label2": "value2", "label1": "value1"})
        assert counter.get({"label1": "value1", "label2": "value2"}) == 1.0


class TestConcurrency:
    """测试并发安全性"""
    
    def test_counter_concurrent_increment(self):
        """测试计数器并发增加"""
        import threading
        
        counter = Counter("test", "Test")
        
        def increment():
            for _ in range(1000):
                counter.inc()
        
        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert counter.get() == 10000.0
    
    def test_histogram_concurrent_observe(self):
        """测试直方图并发观测"""
        import threading
        
        histogram = Histogram("test", "Test")
        
        def observe():
            for _ in range(100):
                histogram.observe(0.5)
        
        threads = [threading.Thread(target=observe) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert histogram.get_count() == 1000
        assert histogram.get_sum() == pytest.approx(500.0, rel=1e-6)

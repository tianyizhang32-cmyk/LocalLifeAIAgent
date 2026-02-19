"""指标收集模块

提供系统指标收集和导出能力，支持：
- 计数器（Counter）：累计值，如请求总数、错误总数
- 直方图（Histogram）：分布统计，如响应时间分布
- 仪表盘（Gauge）：瞬时值，如活跃请求数、内存使用量
- Prometheus 格式导出

验证需求：3.8, 3.9, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8
"""

import time
import psutil
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock


class Counter:
    """计数器指标
    
    用于记录累计值，只能增加不能减少。
    支持标签（labels）用于区分不同维度的计数。
    
    示例：
        request_total = Counter("request_total", "Total requests", ["status"])
        request_total.inc({"status": "200"})
    """
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """初始化计数器
        
        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表（可选）
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = Lock()
    
    def inc(self, label_values: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """增加计数器值
        
        Args:
            label_values: 标签值字典（如 {"status": "200"}）
            amount: 增加的数量（默认 1.0）
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            self._values[label_key] += amount
    
    def get(self, label_values: Optional[Dict[str, str]] = None) -> float:
        """获取计数器值
        
        Args:
            label_values: 标签值字典
        
        Returns:
            计数器当前值
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            return self._values[label_key]
    
    def _make_label_key(self, label_values: Optional[Dict[str, str]]) -> tuple:
        """生成标签键
        
        Args:
            label_values: 标签值字典
        
        Returns:
            标签键元组
        """
        if not label_values:
            return ()
        
        # 按标签名称排序，确保一致性
        return tuple(
            label_values.get(label, "")
            for label in sorted(self.labels)
        )
    
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式
        
        Returns:
            Prometheus 格式的指标字符串
        """
        lines = []
        
        # 添加 HELP 和 TYPE
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} counter")
        
        # 添加指标值
        with self._lock:
            for label_key, value in self._values.items():
                if self.labels and label_key:
                    # 构建标签字符串
                    label_str = ",".join(
                        f'{label}="{label_key[i]}"'
                        for i, label in enumerate(sorted(self.labels))
                    )
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        
        return "\n".join(lines)


class Histogram:
    """直方图指标
    
    用于记录值的分布，如响应时间分布。
    自动计算总和、计数和分桶统计。
    
    示例：
        request_duration = Histogram(
            "request_duration_seconds",
            "Request duration",
            buckets=[0.1, 0.5, 1.0, 5.0]
        )
        request_duration.observe(0.3)
    """
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        """初始化直方图
        
        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表（可选）
            buckets: 分桶边界列表（可选，默认使用标准分桶）
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        # 存储每个标签组合的统计数据
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)
        self._buckets: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {bucket: 0 for bucket in self.buckets}
        )
        self._lock = Lock()
    
    def observe(self, value: float, label_values: Optional[Dict[str, str]] = None):
        """记录一个观测值
        
        Args:
            value: 观测值
            label_values: 标签值字典
        """
        label_key = self._make_label_key(label_values)
        
        with self._lock:
            # 更新总和和计数
            self._sum[label_key] += value
            self._count[label_key] += 1
            
            # 更新分桶计数（只增加第一个满足条件的桶）
            for bucket in self.buckets:
                if value <= bucket:
                    self._buckets[label_key][bucket] += 1
                    break
    
    def get_sum(self, label_values: Optional[Dict[str, str]] = None) -> float:
        """获取总和
        
        Args:
            label_values: 标签值字典
        
        Returns:
            观测值总和
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            return self._sum[label_key]
    
    def get_count(self, label_values: Optional[Dict[str, str]] = None) -> int:
        """获取计数
        
        Args:
            label_values: 标签值字典
        
        Returns:
            观测次数
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            return self._count[label_key]
    
    def _make_label_key(self, label_values: Optional[Dict[str, str]]) -> tuple:
        """生成标签键"""
        if not label_values:
            return ()
        
        return tuple(
            label_values.get(label, "")
            for label in sorted(self.labels)
        )
    
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式
        
        Returns:
            Prometheus 格式的指标字符串
        """
        lines = []
        
        # 添加 HELP 和 TYPE
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} histogram")
        
        with self._lock:
            # 导出每个标签组合的数据
            for label_key in self._sum.keys():
                label_str = self._format_labels(label_key)
                
                # 导出分桶计数
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += self._buckets[label_key][bucket]
                    lines.append(
                        f'{self.name}_bucket{{{label_str}le="{bucket}"}} {cumulative}'
                    )
                
                # 添加 +Inf 桶
                lines.append(
                    f'{self.name}_bucket{{{label_str}le="+Inf"}} {self._count[label_key]}'
                )
                
                # 导出总和
                lines.append(f"{self.name}_sum{{{label_str[:-1]}}} {self._sum[label_key]}")
                
                # 导出计数
                lines.append(f"{self.name}_count{{{label_str[:-1]}}} {self._count[label_key]}")
        
        return "\n".join(lines)
    
    def _format_labels(self, label_key: tuple) -> str:
        """格式化标签字符串
        
        Args:
            label_key: 标签键元组
        
        Returns:
            格式化的标签字符串（如 'status="200",'）
        """
        if not self.labels or not label_key:
            return ""
        
        label_pairs = [
            f'{label}="{label_key[i]}"'
            for i, label in enumerate(sorted(self.labels))
        ]
        return ",".join(label_pairs) + ","


class Gauge:
    """仪表盘指标
    
    用于记录瞬时值，可以增加、减少或直接设置。
    
    示例：
        active_requests = Gauge("active_requests", "Active requests")
        active_requests.inc()
        active_requests.dec()
        active_requests.set(10)
    """
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """初始化仪表盘
        
        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表（可选）
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = Lock()
    
    def set(self, value: float, label_values: Optional[Dict[str, str]] = None):
        """设置仪表盘值
        
        Args:
            value: 新值
            label_values: 标签值字典
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, label_values: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """增加仪表盘值
        
        Args:
            label_values: 标签值字典
            amount: 增加的数量（默认 1.0）
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            self._values[label_key] += amount
    
    def dec(self, label_values: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """减少仪表盘值
        
        Args:
            label_values: 标签值字典
            amount: 减少的数量（默认 1.0）
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            self._values[label_key] -= amount
    
    def get(self, label_values: Optional[Dict[str, str]] = None) -> float:
        """获取仪表盘值
        
        Args:
            label_values: 标签值字典
        
        Returns:
            仪表盘当前值
        """
        label_key = self._make_label_key(label_values)
        with self._lock:
            return self._values[label_key]
    
    def _make_label_key(self, label_values: Optional[Dict[str, str]]) -> tuple:
        """生成标签键"""
        if not label_values:
            return ()
        
        return tuple(
            label_values.get(label, "")
            for label in sorted(self.labels)
        )
    
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式
        
        Returns:
            Prometheus 格式的指标字符串
        """
        lines = []
        
        # 添加 HELP 和 TYPE
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} gauge")
        
        # 添加指标值
        with self._lock:
            for label_key, value in self._values.items():
                if self.labels and label_key:
                    # 构建标签字符串
                    label_str = ",".join(
                        f'{label}="{label_key[i]}"'
                        for i, label in enumerate(sorted(self.labels))
                    )
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        
        return "\n".join(lines)


class MetricsCollector:
    """指标收集器
    
    集中管理所有系统指标，提供统一的指标收集和导出接口。
    
    验证需求：11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8
    """
    
    def __init__(self):
        """初始化指标收集器"""
        # 计数器指标
        self.request_total = Counter(
            "request_total",
            "Total number of requests",
            ["status"]
        )
        self.api_call_total = Counter(
            "api_call_total",
            "Total number of API calls",
            ["api", "status"]
        )
        self.error_total = Counter(
            "error_total",
            "Total number of errors",
            ["error_type"]
        )
        self.cache_hit_total = Counter(
            "cache_hit_total",
            "Total number of cache hits"
        )
        self.cache_miss_total = Counter(
            "cache_miss_total",
            "Total number of cache misses"
        )
        
        # 直方图指标
        self.request_duration_seconds = Histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        self.api_call_duration_seconds = Histogram(
            "api_call_duration_seconds",
            "API call duration in seconds",
            labels=["api"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # 仪表盘指标
        self.active_requests = Gauge(
            "active_requests",
            "Number of active requests"
        )
        self.cache_size = Gauge(
            "cache_size",
            "Current cache size"
        )
        self.cache_hit_rate = Gauge(
            "cache_hit_rate",
            "Cache hit rate (0.0 to 1.0)"
        )
        self.memory_usage_bytes = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes"
        )
        self.cpu_usage_percent = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage"
        )
        
        # 进程对象（用于资源监控）
        self._process = psutil.Process()
    
    def record_request(self, duration: float, status: int):
        """记录请求
        
        Args:
            duration: 请求耗时（秒）
            status: HTTP 状态码
            
        验证需求：11.2, 11.3
        """
        # 记录请求总数
        status_str = str(status)
        self.request_total.inc({"status": status_str})
        
        # 记录请求延迟
        self.request_duration_seconds.observe(duration)
    
    def record_api_call(self, api: str, duration: float, status: int):
        """记录 API 调用
        
        Args:
            api: API 名称（如 "openai", "google_places"）
            duration: 调用耗时（秒）
            status: HTTP 状态码
            
        验证需求：11.4
        """
        # 记录 API 调用总数
        status_str = str(status)
        self.api_call_total.inc({"api": api, "status": status_str})
        
        # 记录 API 调用延迟
        self.api_call_duration_seconds.observe(duration, {"api": api})
    
    def record_error(self, error_type: str):
        """记录错误
        
        Args:
            error_type: 错误类型（如 "API_TIMEOUT", "VALIDATION_ERROR"）
            
        验证需求：11.6
        """
        self.error_total.inc({"error_type": error_type})
    
    def record_cache_hit(self):
        """记录缓存命中
        
        验证需求：11.5
        """
        self.cache_hit_total.inc()
        self._update_cache_hit_rate()
    
    def record_cache_miss(self):
        """记录缓存未命中
        
        验证需求：11.5
        """
        self.cache_miss_total.inc()
        self._update_cache_hit_rate()
    
    def update_cache_size(self, size: int):
        """更新缓存大小
        
        Args:
            size: 当前缓存大小
            
        验证需求：11.5
        """
        self.cache_size.set(float(size))
    
    def update_active_requests(self, count: int):
        """更新活跃请求数
        
        Args:
            count: 当前活跃请求数
            
        验证需求：11.7
        """
        self.active_requests.set(float(count))
    
    def update_resource_usage(self):
        """更新资源使用情况（内存、CPU）
        
        验证需求：11.8
        """
        # 更新内存使用
        memory_info = self._process.memory_info()
        self.memory_usage_bytes.set(float(memory_info.rss))
        
        # 更新 CPU 使用（非阻塞）
        cpu_percent = self._process.cpu_percent(interval=None)
        self.cpu_usage_percent.set(cpu_percent)
    
    def _update_cache_hit_rate(self):
        """更新缓存命中率"""
        hits = self.cache_hit_total.get()
        misses = self.cache_miss_total.get()
        total = hits + misses
        
        if total > 0:
            hit_rate = hits / total
            self.cache_hit_rate.set(hit_rate)
        else:
            self.cache_hit_rate.set(0.0)
    
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式的所有指标
        
        Returns:
            Prometheus 格式的指标字符串
            
        验证需求：11.1
        """
        # 更新资源使用情况
        self.update_resource_usage()
        
        # 收集所有指标
        metrics = [
            # 计数器
            self.request_total.export_prometheus(),
            self.api_call_total.export_prometheus(),
            self.error_total.export_prometheus(),
            self.cache_hit_total.export_prometheus(),
            self.cache_miss_total.export_prometheus(),
            
            # 直方图
            self.request_duration_seconds.export_prometheus(),
            self.api_call_duration_seconds.export_prometheus(),
            
            # 仪表盘
            self.active_requests.export_prometheus(),
            self.cache_size.export_prometheus(),
            self.cache_hit_rate.export_prometheus(),
            self.memory_usage_bytes.export_prometheus(),
            self.cpu_usage_percent.export_prometheus(),
        ]
        
        return "\n\n".join(metrics) + "\n"

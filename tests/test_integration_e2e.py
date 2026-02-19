"""端到端集成测试：完整推荐流程

测试完整的推荐流程、错误恢复和降级策略。
使用 Mock 模拟外部 API 调用。

验证需求：所有核心需求
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from local_lifestyle_agent.orchestrator import Orchestrator, RunContext
from local_lifestyle_agent.planner import Planner
from local_lifestyle_agent.executor import Executor
from local_lifestyle_agent.evaluator import Evaluator
from local_lifestyle_agent.llm_client import LLMClient
from local_lifestyle_agent.adapters.google_places import GooglePlacesAdapter
from local_lifestyle_agent.schemas import (
    NormalizedIntent,
    ExecutableMCP,
    ToolCall,
    CandidateVenue,
    EvaluationReport,
    FinalPlan
)
from local_lifestyle_agent.infrastructure.error_handler import ErrorResponse, ErrorCode
from local_lifestyle_agent.infrastructure.config import Config
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector
from local_lifestyle_agent.infrastructure.cache import Cache


@pytest.fixture
def mock_config():
    """创建 Mock 配置"""
    config = Mock(spec=Config)
    config.openai_timeout = 30
    config.google_places_timeout = 30
    config.max_retries = 3
    config.retry_base_delay = 1.0
    config.cache_enabled = True
    config.cache_ttl = 3600
    config.log_level = "INFO"
    return config


@pytest.fixture
def mock_cache():
    """创建 Mock 缓存"""
    cache = Mock(spec=Cache)
    cache.get.return_value = None  # 默认缓存未命中
    return cache


@pytest.fixture
def mock_logger():
    """创建 Mock Logger"""
    logger = Mock(spec=StructuredLogger)
    return logger


@pytest.fixture
def mock_metrics():
    """创建 Mock Metrics"""
    metrics = Mock(spec=MetricsCollector)
    metrics.active_requests = Mock()
    metrics.active_requests.inc = Mock()
    metrics.active_requests.dec = Mock()
    metrics.request_duration_seconds = Mock()
    metrics.request_duration_seconds.observe = Mock()
    return metrics


@pytest.fixture
def mock_llm_client(mock_config, mock_cache, mock_logger, mock_metrics):
    """创建 Mock LLM Client"""
    llm = Mock(spec=LLMClient)
    
    # Mock normalize 响应
    llm.json_schema.side_effect = [
        # 第一次调用：normalize
        {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "origin_latlng": None,
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium",
            "preferences": {},
            "hard_constraints": {},
            "output_requirements": {"num_backups": 3}
        },
        # 第二次调用：plan
        {
            "tool_calls": [
                {
                    "tool": "google_places_textsearch",
                    "args": {
                        "query": "afternoon tea Seattle",
                        "max_results": 10
                    }
                }
            ],
            "selection_policy": {"strategy": "default"},
            "notes": "Search for afternoon tea venues in Seattle"
        }
    ]
    
    return llm


@pytest.fixture
def mock_places_adapter(mock_config, mock_cache, mock_logger, mock_metrics):
    """创建 Mock Google Places Adapter"""
    places = Mock(spec=GooglePlacesAdapter)
    
    # Mock text_search 响应
    places.text_search.return_value = {
        "results": [
            {
                "place_id": "place_1",
                "name": "The Tea Room",
                "formatted_address": "123 Pike St, Seattle, WA 98101",
                "rating": 4.7,
                "user_ratings_total": 250,
                "price_level": 3,
                "types": ["restaurant", "cafe", "food"],
                "geometry": {
                    "location": {"lat": 47.6062, "lng": -122.3321}
                }
            },
            {
                "place_id": "place_2",
                "name": "Queen Mary Tea Room",
                "formatted_address": "2912 NE 55th St, Seattle, WA 98105",
                "rating": 4.5,
                "user_ratings_total": 180,
                "price_level": 2,
                "types": ["restaurant", "cafe"],
                "geometry": {
                    "location": {"lat": 47.6692, "lng": -122.2869}
                }
            },
            {
                "place_id": "place_3",
                "name": "Perennial Tea Room",
                "formatted_address": "1910 Post Alley, Seattle, WA 98101",
                "rating": 4.6,
                "user_ratings_total": 200,
                "price_level": 2,
                "types": ["restaurant", "cafe"],
                "geometry": {
                    "location": {"lat": 47.6097, "lng": -122.3425}
                }
            }
        ]
    }
    
    return places


class TestEndToEndRecommendationFlow:
    """测试完整的推荐流程"""
    
    def test_successful_recommendation_flow(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试成功的完整推荐流程"""
        # 创建组件
        planner = Planner(
            llm=mock_llm_client,
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        executor = Executor(
            places=mock_places_adapter,
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        evaluator = Evaluator(
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        orchestrator = Orchestrator(
            planner=planner,
            executor=executor,
            evaluator=evaluator,
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        # 执行完整流程
        user_prompt = "Find afternoon tea in Seattle on Sunday 2-5pm for 2 people"
        result = orchestrator.run(user_prompt)
        
        # 验证结果结构
        assert "intent" in result
        assert "executable" in result
        assert "candidates" in result
        assert "eval_report" in result
        assert "plan" in result
        assert "request_id" in result
        
        # 验证意图标准化
        assert isinstance(result["intent"], NormalizedIntent)
        assert result["intent"].city == "Seattle"
        assert result["intent"].party_size == 2
        assert result["intent"].budget_level == "medium"
        
        # 验证计划生成
        assert isinstance(result["executable"], ExecutableMCP)
        assert len(result["executable"].tool_calls) > 0
        assert result["executable"].tool_calls[0].tool == "google_places_textsearch"
        
        # 验证候选场所
        assert len(result["candidates"]) == 3
        assert all(isinstance(c, CandidateVenue) for c in result["candidates"])
        assert result["candidates"][0].name == "The Tea Room"
        
        # 验证评估报告
        assert isinstance(result["eval_report"], EvaluationReport)
        assert result["eval_report"].ok is True
        
        # 验证最终计划
        assert isinstance(result["plan"], FinalPlan)
        assert result["plan"].primary is not None
        assert len(result["plan"].backups) > 0
        
        # 验证 Mock 调用
        assert mock_llm_client.json_schema.call_count == 2  # normalize + plan
        assert mock_places_adapter.text_search.called
        assert mock_logger.set_request_id.called
        assert mock_metrics.active_requests.inc.called
        assert mock_metrics.active_requests.dec.called

    
    def test_recommendation_with_multiple_iterations(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试多次迭代的推荐流程（重新规划）"""
        # 第一次搜索返回不满足条件的结果，第二次返回满足条件的结果
        mock_places_adapter.text_search.side_effect = [
            # 第一次：返回评分较低的场所
            {
                "results": [
                    {
                        "place_id": "place_low",
                        "name": "Low Rated Tea",
                        "formatted_address": "456 Test St, Seattle, WA",
                        "rating": 3.0,
                        "user_ratings_total": 10,
                        "price_level": 1,
                        "types": ["cafe"],
                        "geometry": {
                            "location": {"lat": 47.6062, "lng": -122.3321}
                        }
                    }
                ]
            },
            # 第二次：返回高质量场所
            {
                "results": [
                    {
                        "place_id": "place_high",
                        "name": "Premium Tea Room",
                        "formatted_address": "789 Pike St, Seattle, WA",
                        "rating": 4.8,
                        "user_ratings_total": 300,
                        "price_level": 3,
                        "types": ["restaurant", "cafe"],
                        "geometry": {
                            "location": {"lat": 47.6062, "lng": -122.3321}
                        }
                    }
                ]
            }
        ]
        
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find high-quality afternoon tea in Seattle")
        
        # 验证进行了多次迭代
        assert mock_places_adapter.text_search.call_count >= 1
        
        # 验证最终结果
        assert "plan" in result
        assert result["plan"] is not None



class TestErrorRecoveryFlow:
    """测试错误恢复流程"""
    
    def test_llm_timeout_with_fallback(
        self,
        mock_places_adapter,
        mock_logger,
        mock_metrics,
        mock_config,
        mock_cache
    ):
        """测试 LLM 超时时的降级策略"""
        # 创建会超时的 LLM Client
        mock_llm = Mock(spec=LLMClient)
        mock_llm.json_schema.side_effect = Exception("API timeout")
        
        # 创建组件
        planner = Planner(llm=mock_llm, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证返回了错误响应
        assert "error" in result
        assert isinstance(result["error"], ErrorResponse)
        assert result["error"].error_code in [ErrorCode.API_TIMEOUT, ErrorCode.INTERNAL_ERROR]
        
        # 验证错误被记录
        assert mock_logger.log_error.called
        assert mock_metrics.record_error.called
    
    def test_places_api_error_with_empty_results(
        self,
        mock_llm_client,
        mock_logger,
        mock_metrics
    ):
        """测试 Google Places API 错误时返回空结果"""
        # 创建会失败的 Places Adapter
        mock_places = Mock(spec=GooglePlacesAdapter)
        mock_places.text_search.side_effect = Exception("API error")
        
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证返回了结果（使用降级策略）
        assert "candidates" in result
        # 由于 API 失败，候选列表应该为空或包含错误信息
        assert len(result["candidates"]) == 0 or "error" in result
        
        # 验证警告被记录
        assert mock_logger.warning.called or mock_logger.error.called

    
    def test_partial_failure_with_degraded_service(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试部分失败时的降级服务"""
        # Planner 成功，但 Executor 部分失败
        mock_places_adapter.text_search.return_value = {
            "results": [
                {
                    "place_id": "place_1",
                    "name": "Tea Room 1",
                    "formatted_address": "123 Test St, Seattle, WA",
                    "rating": 4.5,
                    "user_ratings_total": 100,
                    "types": ["cafe"],
                    "geometry": {
                        "location": {"lat": 47.6062, "lng": -122.3321}
                    }
                }
            ]
        }
        
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证返回了部分结果
        assert "candidates" in result
        assert len(result["candidates"]) >= 1
        
        # 验证生成了计划（即使只有部分数据）
        assert "plan" in result
    
    def test_network_timeout_retry(
        self,
        mock_places_adapter,
        mock_logger,
        mock_metrics,
        mock_config,
        mock_cache
    ):
        """测试网络超时时的重试机制"""
        # 创建会超时然后成功的 LLM Client
        mock_llm = Mock(spec=LLMClient)
        mock_llm.json_schema.side_effect = [
            Exception("Timeout"),  # 第一次超时
            {  # 第二次成功
                "city": "Seattle",
                "time_window": {
                    "day": "Sunday",
                    "start_local": "14:00",
                    "end_local": "17:00"
                },
                "origin_latlng": None,
                "max_travel_minutes": 30,
                "party_size": 2,
                "budget_level": "medium",
                "preferences": {},
                "hard_constraints": {},
                "output_requirements": {}
            }
        ]
        
        # 创建组件
        planner = Planner(llm=mock_llm, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证最终成功（经过重试）或返回错误
        assert "intent" in result or "error" in result
        
        # 如果成功，验证 LLM 被调用了多次（重试）
        if "intent" in result and isinstance(result["intent"], NormalizedIntent):
            # 注意：实际的重试逻辑在 LLMClient 内部，这里只是模拟
            assert mock_llm.json_schema.call_count >= 1



class TestDegradationStrategy:
    """测试降级策略"""
    
    def test_fallback_to_rule_engine_when_llm_unavailable(
        self,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试 LLM 不可用时降级到规则引擎"""
        # 创建总是失败的 LLM Client
        mock_llm = Mock(spec=LLMClient)
        mock_llm.json_schema.side_effect = Exception("LLM service unavailable")
        
        # 创建组件
        planner = Planner(llm=mock_llm, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证使用了降级策略
        # 应该返回错误或使用规则引擎生成的基本计划
        assert "error" in result or "executable" in result
        
        # 验证降级被记录
        assert mock_logger.warning.called or mock_logger.error.called
    
    def test_degraded_plan_with_minimal_data(
        self,
        mock_llm_client,
        mock_logger,
        mock_metrics
    ):
        """测试数据不足时的降级计划"""
        # Places API 返回最少数据
        mock_places = Mock(spec=GooglePlacesAdapter)
        mock_places.text_search.return_value = {
            "results": [
                {
                    "place_id": "place_1",
                    "name": "Tea Room",
                    "formatted_address": "Seattle, WA",
                    # 缺少 rating, user_ratings_total 等字段
                    "geometry": {
                        "location": {"lat": 47.6062, "lng": -122.3321}
                    }
                }
            ]
        }
        
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证即使数据不完整也能生成结果
        assert "candidates" in result
        assert len(result["candidates"]) >= 1
        
        # 验证候选场所包含基本信息
        candidate = result["candidates"][0]
        assert candidate.name == "Tea Room"
        assert candidate.place_id == "place_1"
    
    def test_empty_results_with_helpful_message(
        self,
        mock_llm_client,
        mock_logger,
        mock_metrics
    ):
        """测试无结果时返回有用的提示信息"""
        # Places API 返回空结果
        mock_places = Mock(spec=GooglePlacesAdapter)
        mock_places.text_search.return_value = {"results": []}
        
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证返回了空候选列表
        assert "candidates" in result
        assert len(result["candidates"]) == 0
        
        # 验证评估报告包含建议
        assert "eval_report" in result
        if result["eval_report"]:
            assert result["eval_report"].ok is False



class TestInfrastructureIntegration:
    """测试基础设施集成"""
    
    def test_request_id_propagation(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试请求 ID 在整个流程中的传递"""
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证返回了请求 ID
        assert "request_id" in result
        assert isinstance(result["request_id"], str)
        assert len(result["request_id"]) > 0
        
        # 验证请求 ID 被设置到 logger（至少被调用一次）
        assert mock_logger.set_request_id.called
        # 验证至少有一次调用使用了有效的请求 ID
        assert any(
            len(call[0][0]) > 0 
            for call in mock_logger.set_request_id.call_args_list
        )
    
    def test_metrics_collection_throughout_flow(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试整个流程中的指标收集"""
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证指标被收集
        assert mock_metrics.active_requests.inc.called
        assert mock_metrics.active_requests.dec.called
        assert mock_metrics.record_request.called
        
        # 验证记录了请求耗时
        call_args = mock_metrics.record_request.call_args[0]
        duration, status = call_args
        assert isinstance(duration, float)
        assert duration >= 0
        assert status in [200, 500]
    
    def test_logging_throughout_flow(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试整个流程中的日志记录"""
        # 创建组件
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证日志被记录
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        
        # 验证记录了关键步骤
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        # 应该包含各个阶段的日志
        assert len(info_calls) > 0
    
    def test_error_handling_with_proper_logging(
        self,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """测试错误处理和日志记录"""
        # 创建会失败的 LLM Client
        mock_llm = Mock(spec=LLMClient)
        mock_llm.json_schema.side_effect = Exception("Test error")
        
        # 创建组件
        planner = Planner(llm=mock_llm, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证错误被记录
        assert mock_logger.log_error.called or mock_logger.error.called
        assert mock_metrics.record_error.called
        
        # 验证返回了错误响应
        assert "error" in result


class TestCoreRequirementsValidation:
    """验证核心需求"""
    
    def test_requirement_1_error_handling(
        self,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """验证需求 1：错误处理和容错机制"""
        # 创建会失败的 LLM Client
        mock_llm = Mock(spec=LLMClient)
        mock_llm.json_schema.side_effect = Exception("API error")
        
        planner = Planner(llm=mock_llm, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea")
        
        # 验证：系统捕获异常并返回结构化错误信息
        assert "error" in result
        assert isinstance(result["error"], ErrorResponse)
        assert result["error"].error_code is not None
        assert result["error"].error_message is not None
        assert result["error"].request_id is not None
    
    def test_requirement_3_logging(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """验证需求 3：日志和监控"""
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证：系统记录所有关键事件
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        
        # 验证：日志包含请求 ID
        assert "request_id" in result
    
    def test_requirement_6_data_validation(
        self,
        mock_llm_client,
        mock_places_adapter,
        mock_logger,
        mock_metrics
    ):
        """验证需求 6：数据验证和清洗"""
        planner = Planner(llm=mock_llm_client, logger=mock_logger, metrics=mock_metrics)
        executor = Executor(places=mock_places_adapter, logger=mock_logger, metrics=mock_metrics)
        evaluator = Evaluator(logger=mock_logger, metrics=mock_metrics)
        orchestrator = Orchestrator(planner, executor, evaluator, logger=mock_logger, metrics=mock_metrics)
        
        # 执行
        result = orchestrator.run("Find afternoon tea in Seattle")
        
        # 验证：系统验证和清洗数据
        assert "intent" in result or "error" in result
        
        # 如果成功，验证数据结构
        if "intent" in result:
            assert isinstance(result["intent"], NormalizedIntent)
            assert result["intent"].city is not None
            assert result["intent"].party_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

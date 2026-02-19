"""单元测试：Orchestrator 模块

测试 Orchestrator 模块的核心功能：
- 完整的推荐流程
- 请求 ID 生成和传递
- 错误处理和降级策略
- 日志记录和指标收集
- 全局异常处理

验证需求：1.9
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from local_lifestyle_agent.orchestrator import Orchestrator, RunContext
from local_lifestyle_agent.schemas import (
    NormalizedIntent,
    ExecutableMCP,
    ToolCall,
    CandidateVenue,
    EvaluationReport,
    FinalPlan,
    PlanOption
)
from local_lifestyle_agent.infrastructure.error_handler import ErrorResponse, ErrorCode
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector


@pytest.fixture
def mock_planner():
    """创建 Mock Planner"""
    planner = Mock()
    return planner


@pytest.fixture
def mock_executor():
    """创建 Mock Executor"""
    executor = Mock()
    return executor


@pytest.fixture
def mock_evaluator():
    """创建 Mock Evaluator"""
    evaluator = Mock()
    return evaluator


@pytest.fixture
def mock_logger():
    """创建 Mock Logger"""
    logger = Mock(spec=StructuredLogger)
    return logger


@pytest.fixture
def mock_metrics():
    """创建 Mock Metrics"""
    metrics = Mock(spec=MetricsCollector)
    # 配置 active_requests 属性
    metrics.active_requests = Mock()
    metrics.active_requests.inc = Mock()
    metrics.active_requests.dec = Mock()
    # 配置其他方法
    metrics.record_request = Mock()
    metrics.record_error = Mock()
    return metrics


@pytest.fixture
def sample_intent():
    """创建示例 NormalizedIntent"""
    return NormalizedIntent(
        activity_type="afternoon_tea",
        city="Seattle",
        time_window={
            "day": "Sunday",
            "start_local": "14:00",
            "end_local": "17:00"
        },
        origin_latlng=None,
        max_travel_minutes=30,
        party_size=2,
        budget_level="medium",
        preferences={},
        hard_constraints={},
        output_requirements={"num_backups": 3}
    )


@pytest.fixture
def sample_executable():
    """创建示例 ExecutableMCP"""
    return ExecutableMCP(
        tool_calls=[
            ToolCall(
                tool="google_places_textsearch",
                args={"query": "afternoon tea Seattle", "max_results": 10}
            )
        ],
        selection_policy={"strategy": "default"},
        notes="Test plan"
    )


@pytest.fixture
def sample_candidates():
    """创建示例候选场所列表"""
    return [
        CandidateVenue(
            venue_id="venue1",
            place_id="place1",
            name="Tea House 1",
            address="123 Main St",
            rating=4.5,
            user_ratings_total=100,
            price_level=2,
            latlng="47.6062,-122.3321",
            category="cafe"
        ),
        CandidateVenue(
            venue_id="venue2",
            place_id="place2",
            name="Tea House 2",
            address="456 Oak Ave",
            rating=4.3,
            user_ratings_total=80,
            price_level=2,
            latlng="47.6062,-122.3321",
            category="cafe"
        )
    ]


@pytest.fixture
def sample_eval_report():
    """创建示例评估报告"""
    return EvaluationReport(
        ok=True,
        score_breakdown={
            "venue1": {"total": 0.8, "rating": 0.5, "popularity": 0.3},
            "venue2": {"total": 0.7, "rating": 0.4, "popularity": 0.3}
        }
    )


def test_orchestrator_initialization(mock_planner, mock_executor, mock_evaluator):
    """测试 Orchestrator 初始化"""
    orchestrator = Orchestrator(mock_planner, mock_executor, mock_evaluator)
    
    assert orchestrator.planner == mock_planner
    assert orchestrator.executor == mock_executor
    assert orchestrator.evaluator == mock_evaluator
    assert orchestrator.preference_signals == {}
    assert orchestrator.rejected_options == []


def test_orchestrator_with_infrastructure(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics
):
    """测试 Orchestrator 集成基础设施模块"""
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    assert orchestrator.logger == mock_logger
    assert orchestrator.metrics == mock_metrics
    assert orchestrator.error_handler is not None


def test_successful_orchestration(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_candidates,
    sample_eval_report
):
    """测试成功的完整推荐流程"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    # 创建排序后的候选场所
    ranked = [
        (sample_candidates[0], {"total": 0.8, "rating": 0.5, "popularity": 0.3}),
        (sample_candidates[1], {"total": 0.7, "rating": 0.4, "popularity": 0.3})
    ]
    mock_evaluator.evaluate.return_value = (sample_eval_report, ranked)
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea in Seattle")
    
    # 验证结果
    assert "intent" in result
    assert "executable" in result
    assert "candidates" in result
    assert "eval_report" in result
    assert "plan" in result
    assert "request_id" in result
    
    assert result["intent"] == sample_intent
    assert result["executable"] == sample_executable
    assert len(result["candidates"]) == 2
    assert result["eval_report"] == sample_eval_report
    assert isinstance(result["plan"], FinalPlan)
    
    # 验证 Mock 调用
    mock_planner.normalize.assert_called_once()
    mock_planner.plan.assert_called_once()
    mock_executor.execute.assert_called_once()
    mock_evaluator.evaluate.assert_called_once()
    
    # 验证日志记录
    assert mock_logger.set_request_id.called
    assert mock_logger.info.called
    
    # 验证指标收集
    assert mock_metrics.active_requests.inc.called
    assert mock_metrics.active_requests.dec.called
    assert mock_metrics.record_request.called


def test_orchestration_with_normalization_error(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics
):
    """测试标准化失败时的错误处理"""
    # 设置 Mock 返回错误响应
    error_response = ErrorResponse(
        error_code=ErrorCode.INVALID_INPUT,
        error_message="User prompt is too long",
        details={},
        request_id="test-request-id"
    )
    mock_planner.normalize.return_value = error_response
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea")
    
    # 验证结果包含错误
    assert "error" in result
    assert "request_id" in result
    assert result["error"] == error_response
    
    # 验证不会调用后续步骤
    mock_planner.plan.assert_not_called()
    mock_executor.execute.assert_not_called()
    mock_evaluator.evaluate.assert_not_called()
    
    # 验证错误记录
    assert mock_logger.error.called
    assert mock_metrics.record_error.called


def test_orchestration_with_plan_error_and_fallback(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_candidates,
    sample_eval_report
):
    """测试计划生成失败时的降级策略"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    
    # 第一次调用返回错误，触发降级策略
    error_response = ErrorResponse(
        error_code=ErrorCode.API_TIMEOUT,
        error_message="OpenAI API timeout",
        details={},
        request_id="test-request-id"
    )
    mock_planner.plan.return_value = error_response
    
    # Executor 返回候选场所
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    # Evaluator 返回评估结果
    ranked = [
        (sample_candidates[0], {"total": 0.8, "rating": 0.5, "popularity": 0.3}),
        (sample_candidates[1], {"total": 0.7, "rating": 0.4, "popularity": 0.3})
    ]
    mock_evaluator.evaluate.return_value = (sample_eval_report, ranked)
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea in Seattle")
    
    # 验证使用了降级策略
    assert "plan" in result
    assert result["plan"] is not None
    
    # 验证日志记录了降级策略
    assert mock_logger.warning.called
    assert mock_logger.info.called


def test_orchestration_with_executor_error(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_eval_report
):
    """测试执行失败时的降级策略"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    
    # Executor 返回错误
    error_response = ErrorResponse(
        error_code=ErrorCode.API_TIMEOUT,
        error_message="Google Places API timeout",
        details={},
        request_id="test-request-id"
    )
    mock_executor.execute.return_value = error_response
    
    # Evaluator 返回空结果
    empty_eval = EvaluationReport(
        ok=False,
        hard_violations=["no_candidates_pass_hard_constraints"],
        replan_suggestions=["broaden_queries"]
    )
    mock_evaluator.evaluate.return_value = (empty_eval, [])
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea in Seattle")
    
    # 验证使用了降级策略（空候选列表）
    assert "candidates" in result
    assert len(result["candidates"]) == 0
    
    # 验证日志记录了警告
    assert mock_logger.warning.called


def test_orchestration_with_evaluator_error(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_candidates
):
    """测试评估失败时的降级策略"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    # Evaluator 返回错误
    error_response = ErrorResponse(
        error_code=ErrorCode.INTERNAL_ERROR,
        error_message="Evaluation failed",
        details={},
        request_id="test-request-id"
    )
    mock_evaluator.evaluate.return_value = error_response
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea in Seattle")
    
    # 验证使用了降级策略（未排序的候选列表）
    assert "candidates" in result
    assert len(result["candidates"]) == 2
    
    # 验证日志记录了警告
    assert mock_logger.warning.called


def test_orchestration_with_global_exception(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics
):
    """测试全局异常处理"""
    # 设置 Mock 抛出异常
    mock_planner.normalize.side_effect = Exception("Unexpected error")
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea")
    
    # 验证返回错误响应
    assert "error" in result
    assert "request_id" in result
    assert isinstance(result["error"], ErrorResponse)
    
    # 验证错误记录
    assert mock_logger.log_error.called
    assert mock_metrics.record_error.called
    assert mock_metrics.record_request.called


def test_orchestration_multiple_iterations(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_candidates
):
    """测试多次迭代（重新规划）"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    # 第一次评估失败，第二次成功
    failed_eval = EvaluationReport(
        ok=False,
        hard_violations=["no_candidates_pass_hard_constraints"],
        replan_suggestions=["expand_radius_bias"]
    )
    success_eval = EvaluationReport(
        ok=True,
        score_breakdown={
            "venue1": {"total": 0.8, "rating": 0.5, "popularity": 0.3}
        }
    )
    
    ranked = [
        (sample_candidates[0], {"total": 0.8, "rating": 0.5, "popularity": 0.3})
    ]
    
    # 第一次调用返回失败，第二次返回成功
    mock_evaluator.evaluate.side_effect = [
        (failed_eval, []),
        (success_eval, ranked)
    ]
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea in Seattle")
    
    # 验证进行了多次迭代
    assert mock_planner.plan.call_count == 2
    assert mock_executor.execute.call_count == 2
    assert mock_evaluator.evaluate.call_count == 2
    
    # 验证最终成功
    assert "plan" in result
    assert result["plan"] is not None


def test_orchestration_max_iterations_reached(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_candidates
):
    """测试达到最大迭代次数"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    # 所有评估都失败
    failed_eval = EvaluationReport(
        ok=False,
        hard_violations=["no_candidates_pass_hard_constraints"],
        replan_suggestions=["expand_radius_bias"]
    )
    mock_evaluator.evaluate.return_value = (failed_eval, [])
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger,
        metrics=mock_metrics
    )
    
    # 运行推荐流程（最多 3 次迭代）
    ctx = RunContext(max_iterations=3)
    result = orchestrator.run("Find afternoon tea in Seattle", ctx)
    
    # 验证进行了 3 次迭代
    assert mock_planner.plan.call_count == 3
    assert mock_executor.execute.call_count == 3
    assert mock_evaluator.evaluate.call_count == 3
    
    # 验证没有生成最终计划
    assert result["plan"] is None
    assert result["eval_report"] == failed_eval
    
    # 验证日志记录了警告
    assert mock_logger.warning.called


def test_request_id_generation(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    sample_intent,
    sample_executable,
    sample_candidates,
    sample_eval_report
):
    """测试请求 ID 生成和传递"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    ranked = [
        (sample_candidates[0], {"total": 0.8, "rating": 0.5, "popularity": 0.3})
    ]
    mock_evaluator.evaluate.return_value = (sample_eval_report, ranked)
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea")
    
    # 验证返回了请求 ID
    assert "request_id" in result
    assert isinstance(result["request_id"], str)
    assert len(result["request_id"]) > 0
    
    # 验证请求 ID 被设置到 logger
    mock_logger.set_request_id.assert_called_once()
    call_args = mock_logger.set_request_id.call_args[0]
    assert call_args[0] == result["request_id"]


def test_metrics_collection(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_metrics,
    sample_intent,
    sample_executable,
    sample_candidates,
    sample_eval_report
):
    """测试指标收集"""
    # 设置 Mock 返回值
    mock_planner.normalize.return_value = sample_intent
    mock_planner.plan.return_value = sample_executable
    mock_executor.execute.return_value = {
        "tool_results": [],
        "candidates": sample_candidates
    }
    
    ranked = [
        (sample_candidates[0], {"total": 0.8, "rating": 0.5, "popularity": 0.3})
    ]
    mock_evaluator.evaluate.return_value = (sample_eval_report, ranked)
    
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        metrics=mock_metrics
    )
    
    # 运行推荐流程
    result = orchestrator.run("Find afternoon tea")
    
    # 验证指标收集
    # 活跃请求数应该增加和减少
    assert mock_metrics.active_requests.inc.called
    assert mock_metrics.active_requests.dec.called
    
    # 应该记录请求
    assert mock_metrics.record_request.called
    call_args = mock_metrics.record_request.call_args[0]
    duration, status = call_args
    assert isinstance(duration, float)
    assert duration >= 0
    assert status == 200


def test_fallback_plan_generation(
    mock_planner,
    mock_executor,
    mock_evaluator,
    mock_logger,
    sample_intent
):
    """测试降级计划生成"""
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator,
        logger=mock_logger
    )
    
    # 调用降级计划生成
    runtime_context = {"iteration": 1, "max_tool_calls": 6}
    fallback_plan = orchestrator._fallback_plan(sample_intent, runtime_context)
    
    # 验证生成了降级计划
    assert fallback_plan is not None
    assert isinstance(fallback_plan, ExecutableMCP)
    assert len(fallback_plan.tool_calls) > 0
    assert fallback_plan.tool_calls[0].tool == "google_places_textsearch"
    # activity_type 由 LLM 提取，fallback 时使用 intent 中的值
    assert sample_intent.activity_type in fallback_plan.tool_calls[0].args["query"]
    assert "Seattle" in fallback_plan.tool_calls[0].args["query"]
    assert fallback_plan.notes == "Generated by fallback rule engine (LLM unavailable)"
    
    # 验证日志记录
    assert mock_logger.info.called


def test_assemble_final_plan(
    mock_planner,
    mock_executor,
    mock_evaluator,
    sample_intent,
    sample_candidates
):
    """测试组装最终推荐计划"""
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator
    )
    
    # 创建排序后的候选场所
    ranked = [
        (sample_candidates[0], {
            "total": 0.8,
            "rating": 0.5,
            "popularity": 0.3,
            "price_fit": 0.7,
            "pref_bonus": 0.1
        }),
        (sample_candidates[1], {
            "total": 0.7,
            "rating": 0.4,
            "popularity": 0.3,
            "price_fit": 0.6,
            "pref_bonus": 0.0
        })
    ]
    
    # 组装最终计划
    plan = orchestrator._assemble(sample_intent, ranked, num_backups=1)
    
    # 验证计划结构
    assert isinstance(plan, FinalPlan)
    assert isinstance(plan.primary, PlanOption)
    assert plan.primary.name == "Tea House 1"
    assert len(plan.backups) == 1
    assert plan.backups[0].name == "Tea House 2"
    assert "arrive_at" in plan.schedule
    assert "leave_at" in plan.schedule
    assert len(plan.tips) > 0
    assert len(plan.assumptions) > 0


def test_apply_replan(
    mock_planner,
    mock_executor,
    mock_evaluator,
    sample_intent
):
    """测试应用重新规划建议"""
    # 创建 Orchestrator
    orchestrator = Orchestrator(
        mock_planner,
        mock_executor,
        mock_evaluator
    )
    
    # 记录原始值
    original_max_travel = sample_intent.max_travel_minutes
    
    # 应用重新规划建议
    suggestions = ["expand_radius_bias"]
    orchestrator._apply_replan(sample_intent, suggestions)
    
    # 验证 max_travel_minutes 增加了
    assert sample_intent.max_travel_minutes > original_max_travel
    assert sample_intent.max_travel_minutes == original_max_travel + 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

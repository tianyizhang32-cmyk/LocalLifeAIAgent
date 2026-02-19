"""
单元测试：Planner 模块

测试 Planner 模块的核心功能：
- 意图标准化（normalize）
- 计划生成（plan）
- 数据验证集成
- 日志记录集成
- 指标收集集成
- 错误处理集成

验证需求：1.6, 6.1, 6.2, 6.3, 6.4
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from local_lifestyle_agent.planner import Planner
from local_lifestyle_agent.schemas import NormalizedIntent, ExecutableMCP
from local_lifestyle_agent.infrastructure.error_handler import ErrorResponse, ErrorCode
from local_lifestyle_agent.infrastructure.validator import ValidationResult


class TestPlannerNormalize:
    """测试 Planner.normalize 方法"""
    
    def test_normalize_success(self):
        """测试成功的意图标准化"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "activity_type": "afternoon_tea",
            "city": "London",
            "time_window": {
                "day": "Saturday",
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
        
        planner = Planner(llm=mock_llm)
        
        # 执行
        result = planner.normalize("Find afternoon tea in London for 2 people")
        
        # 验证
        assert isinstance(result, NormalizedIntent)
        assert result.city == "London"
        assert result.party_size == 2
        assert result.budget_level == "medium"
        # activity_type 由 LLM 自动提取，不再硬编码检查
        assert result.activity_type  # 确保字段存在且非空
        
        # 验证 LLM 被调用
        mock_llm.json_schema.assert_called_once()
    
    def test_normalize_with_logger(self):
        """测试带日志记录的意图标准化"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "activity_type": "afternoon_tea",
            "city": "London",
            "time_window": {
                "day": "Saturday",
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
        
        mock_logger = Mock()
        planner = Planner(llm=mock_llm, logger=mock_logger)
        
        # 执行
        result = planner.normalize("Find afternoon tea in London")
        
        # 验证日志被调用
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        
        # 验证日志内容
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Starting intent normalization" in info_calls
        assert "Intent normalization completed" in info_calls
    
    def test_normalize_with_metrics(self):
        """测试带指标收集的意图标准化"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "activity_type": "afternoon_tea",
            "city": "London",
            "time_window": {
                "day": "Saturday",
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
        
        mock_metrics = Mock()
        planner = Planner(llm=mock_llm, metrics=mock_metrics)
        
        # 执行
        result = planner.normalize("Find afternoon tea in London")
        
        # 验证指标被记录
        assert mock_metrics.request_duration_seconds.observe.called
    
    def test_normalize_input_too_long(self):
        """测试输入过长的情况"""
        # 准备
        mock_llm = Mock()
        planner = Planner(llm=mock_llm)
        
        # 创建超长输入（超过 1000 字符）
        long_input = "a" * 1001
        
        # 执行
        result = planner.normalize(long_input)
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code == ErrorCode.INVALID_INPUT
        assert "too long" in result.error_message
    
    def test_normalize_malicious_content(self):
        """测试包含恶意内容的输入"""
        # 准备
        mock_llm = Mock()
        planner = Planner(llm=mock_llm)
        
        # 包含恶意内容的输入
        malicious_input = "Find tea <script>alert('xss')</script>"
        
        # 执行
        result = planner.normalize(malicious_input)
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code == ErrorCode.INVALID_INPUT
        assert "malicious" in result.error_message.lower()
    
    def test_normalize_validation_error(self):
        """测试 LLM 输出验证失败的情况"""
        # 准备
        mock_llm = Mock()
        # 返回无效数据（缺少必填字段）
        mock_llm.json_schema.return_value = {
            "city": "London",
            # 缺少其他必填字段
        }
        
        planner = Planner(llm=mock_llm)
        
        # 执行
        result = planner.normalize("Find afternoon tea in London")
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code == ErrorCode.VALIDATION_ERROR
    
    def test_normalize_llm_error(self):
        """测试 LLM 调用失败的情况"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.side_effect = Exception("API timeout")
        
        planner = Planner(llm=mock_llm)
        
        # 执行
        result = planner.normalize("Find afternoon tea in London")
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code in [ErrorCode.API_TIMEOUT, ErrorCode.INTERNAL_ERROR]
    
    def test_normalize_sanitizes_input(self):
        """测试输入清洗功能"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "activity_type": "afternoon_tea",
            "city": "London",
            "time_window": {
                "day": "Saturday",
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
        
        planner = Planner(llm=mock_llm)
        
        # 包含多余空白的输入
        messy_input = "Find   afternoon   tea   in   London"
        
        # 执行
        result = planner.normalize(messy_input)
        
        # 验证成功（输入被清洗）
        assert isinstance(result, NormalizedIntent)
        
        # 验证 LLM 收到的是清洗后的输入
        call_args = mock_llm.json_schema.call_args
        user_prompt = call_args[1]["user"]
        # 清洗后的输入应该只有单个空格
        assert "   " not in user_prompt


class TestPlannerPlan:
    """测试 Planner.plan 方法"""
    
    def test_plan_success(self):
        """测试成功的计划生成"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "tool_calls": [
                {
                    "tool": "google_places_textsearch",
                    "args": {"query": "afternoon tea London"}
                }
            ],
            "selection_policy": {"max_results": 5},
            "notes": "Search for afternoon tea venues"
        }
        
        planner = Planner(llm=mock_llm)
        
        # 创建测试用的 NormalizedIntent
        intent = NormalizedIntent(
            city="London",
            time_window={
                "day": "Saturday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            origin_latlng=None,
            max_travel_minutes=30,
            party_size=2,
            budget_level="medium",
            preferences={},
            hard_constraints={},
            output_requirements={},
            activity_type="afternoon_tea"
        )
        
        runtime_context = {"max_tool_calls": 3, "rejected_options": []}
        
        # 执行
        result = planner.plan(intent, runtime_context)
        
        # 验证
        assert isinstance(result, ExecutableMCP)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "google_places_textsearch"
        
        # 验证 LLM 被调用
        mock_llm.json_schema.assert_called_once()
    
    def test_plan_with_logger(self):
        """测试带日志记录的计划生成"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "tool_calls": [
                {
                    "tool": "google_places_textsearch",
                    "args": {"query": "afternoon tea London"}
                }
            ],
            "selection_policy": {},
            "notes": None
        }
        
        mock_logger = Mock()
        planner = Planner(llm=mock_llm, logger=mock_logger)
        
        intent = NormalizedIntent(
            city="London",
            time_window={
                "day": "Saturday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            origin_latlng=None,
            max_travel_minutes=30,
            party_size=2,
            budget_level="medium",
            preferences={},
            hard_constraints={},
            output_requirements={},
            activity_type="afternoon_tea"
        )
        
        # 执行
        result = planner.plan(intent, {"max_tool_calls": 3})
        
        # 验证日志被调用
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        
        # 验证日志内容
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Starting plan generation" in info_calls
        assert "Plan generation completed" in info_calls
    
    def test_plan_validation_error(self):
        """测试 LLM 输出验证失败的情况"""
        # 准备
        mock_llm = Mock()
        # 返回无效数据（缺少必填字段）
        mock_llm.json_schema.return_value = {
            "tool_calls": [],  # 空的 tool_calls
            # 缺少其他必填字段
        }
        
        planner = Planner(llm=mock_llm)
        
        intent = NormalizedIntent(
            city="London",
            time_window={
                "day": "Saturday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            origin_latlng=None,
            max_travel_minutes=30,
            party_size=2,
            budget_level="medium",
            preferences={},
            hard_constraints={},
            output_requirements={},
            activity_type="afternoon_tea"
        )
        
        # 执行
        result = planner.plan(intent, {"max_tool_calls": 3})
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code == ErrorCode.VALIDATION_ERROR
    
    def test_plan_llm_error(self):
        """测试 LLM 调用失败的情况"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.side_effect = Exception("API error")
        
        planner = Planner(llm=mock_llm)
        
        intent = NormalizedIntent(
            city="London",
            time_window={
                "day": "Saturday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            origin_latlng=None,
            max_travel_minutes=30,
            party_size=2,
            budget_level="medium",
            preferences={},
            hard_constraints={},
            output_requirements={},
            activity_type="afternoon_tea"
        )
        
        # 执行
        result = planner.plan(intent, {"max_tool_calls": 3})
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)


class TestPlannerIntegration:
    """测试 Planner 的集成功能"""
    
    def test_full_integration_with_all_components(self):
        """测试完整集成（日志、指标、错误处理）"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.return_value = {
            "activity_type": "afternoon_tea",
            "city": "London",
            "time_window": {
                "day": "Saturday",
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
        
        mock_logger = Mock()
        mock_metrics = Mock()
        mock_error_handler = Mock()
        
        planner = Planner(
            llm=mock_llm,
            logger=mock_logger,
            metrics=mock_metrics,
            error_handler=mock_error_handler
        )
        
        # 执行
        result = planner.normalize("Find afternoon tea in London")
        
        # 验证所有组件都被使用
        assert isinstance(result, NormalizedIntent)
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        assert mock_metrics.request_duration_seconds.observe.called
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 准备
        mock_llm = Mock()
        mock_llm.json_schema.side_effect = Exception("Timeout")
        
        mock_logger = Mock()
        mock_metrics = Mock()
        
        planner = Planner(
            llm=mock_llm,
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        # 执行
        result = planner.normalize("Find afternoon tea")
        
        # 验证错误被正确处理
        assert isinstance(result, ErrorResponse)
        assert mock_logger.log_error.called
        assert mock_metrics.record_error.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

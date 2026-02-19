"""
单元测试：Executor 模块

测试 Executor 模块的核心功能：
- 工具调用执行（execute）
- 参数验证
- 响应数据清洗
- 日志记录集成
- 指标收集集成
- 错误处理集成

验证需求：6.4, 6.5
"""

import pytest
from unittest.mock import Mock, MagicMock
from local_lifestyle_agent.executor import Executor
from local_lifestyle_agent.schemas import (
    ExecutableMCP, NormalizedIntent, ToolCall, ToolResult, CandidateVenue
)
from local_lifestyle_agent.infrastructure.error_handler import ErrorResponse, ErrorCode


class TestExecutorExecute:
    """测试 Executor.execute 方法"""
    
    def test_execute_textsearch_success(self):
        """测试成功的文本搜索执行"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {
            "results": [
                {
                    "place_id": "test_id_1",
                    "name": "Test Venue 1",
                    "formatted_address": "123 Test St",
                    "rating": 4.5,
                    "user_ratings_total": 100,
                    "price_level": 2,
                    "types": ["restaurant", "cafe"],
                    "geometry": {
                        "location": {"lat": 51.5074, "lng": -0.1278}
                    }
                }
            ]
        }
        
        executor = Executor(places=mock_places)
        
        # 创建测试数据
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea London", "max_results": 10}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert "tool_results" in result
        assert "candidates" in result
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].ok is True
        assert len(result["candidates"]) == 1
        assert result["candidates"][0].name == "Test Venue 1"
        
        # 验证 API 被调用
        mock_places.text_search.assert_called_once()
    
    def test_execute_details_success(self):
        """测试成功的场所详情执行"""
        # 准备
        mock_places = Mock()
        mock_places.details.return_value = {
            "result": {
                "place_id": "test_id_1",
                "name": "Test Venue 1",
                "formatted_address": "123 Test St",
                "rating": 4.8,
                "user_ratings_total": 200,
                "price_level": 3,
                "geometry": {
                    "location": {"lat": 51.5074, "lng": -0.1278}
                }
            }
        }
        
        executor = Executor(places=mock_places)
        
        # 创建测试数据
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_details",
                    args={"place_id": "test_id_1"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].ok is True
        
        # 验证 API 被调用
        mock_places.details.assert_called_once_with(place_id="test_id_1")
    
    def test_execute_with_logger(self):
        """测试带日志记录的执行"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {"results": []}
        
        mock_logger = Mock()
        executor = Executor(places=mock_places, logger=mock_logger)
        
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证日志被调用
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        assert mock_logger.debug.called
        
        # 验证日志内容
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Starting tool execution" in info_calls
        assert "Tool execution completed" in info_calls
    
    def test_execute_with_metrics(self):
        """测试带指标收集的执行"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {"results": []}
        
        mock_metrics = Mock()
        executor = Executor(places=mock_places, metrics=mock_metrics)
        
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证指标被记录
        assert mock_metrics.request_duration_seconds.observe.called
    
    def test_execute_validation_error_empty_tool_calls(self):
        """测试空工具调用列表的验证错误"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        # 创建无效的 ExecutableMCP（空 tool_calls）
        executable = ExecutableMCP(
            tool_calls=[],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert result.error_code == ErrorCode.VALIDATION_ERROR
        assert "tool_calls cannot be empty" in str(result.details)
    
    def test_execute_missing_query_parameter(self):
        """测试缺少 query 参数的情况"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        # 创建缺少 query 参数的工具调用
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={}  # 缺少 query
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].ok is False
        assert "missing_query" in result["tool_results"][0].error
    
    def test_execute_missing_place_id_parameter(self):
        """测试缺少 place_id 参数的情况"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        # 创建缺少 place_id 参数的工具调用
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_details",
                    args={}  # 缺少 place_id
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].ok is False
        assert "missing_place_id" in result["tool_results"][0].error
    
    def test_execute_multiple_tool_calls(self):
        """测试多个工具调用的情况"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {
            "results": [
                {
                    "place_id": "test_id_1",
                    "name": "Test Venue 1",
                    "formatted_address": "123 Test St",
                    "rating": 4.5,
                    "types": ["restaurant"],
                    "geometry": {
                        "location": {"lat": 51.5, "lng": -0.1}
                    }
                }
            ]
        }
        mock_places.details.return_value = {
            "result": {
                "place_id": "test_id_1",
                "name": "Test Venue 1",
                "formatted_address": "123 Test St",
                "rating": 4.8,
                "price_level": 3
            }
        }
        
        executor = Executor(places=mock_places)
        
        # 创建多个工具调用
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                ),
                ToolCall(
                    tool="google_places_details",
                    args={"place_id": "test_id_1"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert len(result["tool_results"]) == 2
        assert result["tool_results"][0].ok is True
        assert result["tool_results"][1].ok is True
        # 验证 details 调用更新了候选场所的信息
        assert len(result["candidates"]) == 1
        assert result["candidates"][0].rating == 4.8  # 被 details 更新
        assert result["candidates"][0].price_level == 3  # 被 details 更新
    
    def test_execute_api_error(self):
        """测试 API 调用失败的情况"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.side_effect = Exception("API timeout")
        
        executor = Executor(places=mock_places)
        
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证
        assert isinstance(result, dict)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].ok is False
        assert "Exception" in result["tool_results"][0].error


class TestExecutorCleanResponseData:
    """测试 Executor._clean_response_data 方法"""
    
    def test_clean_removes_none_values(self):
        """测试清洗移除 None 值"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "rating": None,
            "address": "123 Test St"
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证
        assert "name" in cleaned
        assert "rating" not in cleaned
        assert "address" in cleaned
    
    def test_clean_removes_empty_strings(self):
        """测试清洗移除空字符串"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "description": "",
            "address": "123 Test St"
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证
        assert "name" in cleaned
        assert "description" not in cleaned
        assert "address" in cleaned
    
    def test_clean_removes_empty_lists(self):
        """测试清洗移除空列表"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "types": [],
            "photos": ["photo1.jpg"]
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证
        assert "name" in cleaned
        assert "types" not in cleaned
        assert "photos" in cleaned
    
    def test_clean_removes_empty_dicts(self):
        """测试清洗移除空字典"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "metadata": {},
            "location": {"lat": 51.5, "lng": -0.1}
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证
        assert "name" in cleaned
        assert "metadata" not in cleaned
        assert "location" in cleaned
    
    def test_clean_nested_structures(self):
        """测试清洗嵌套结构"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "location": {
                "lat": 51.5,
                "lng": -0.1,
                "extra": None
            },
            "reviews": [
                {"text": "Great!", "rating": 5},
                {"text": "", "rating": 4}
            ]
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证
        assert "name" in cleaned
        assert "location" in cleaned
        assert "extra" not in cleaned["location"]
        assert len(cleaned["reviews"]) == 2
    
    def test_clean_idempotence(self):
        """测试清洗的幂等性：clean(clean(data)) == clean(data)"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "rating": None,
            "description": "",
            "types": [],
            "location": {
                "lat": 51.5,
                "lng": -0.1,
                "extra": None
            }
        }
        
        # 执行
        cleaned_once = executor._clean_response_data(data)
        cleaned_twice = executor._clean_response_data(cleaned_once)
        
        # 验证幂等性
        assert cleaned_once == cleaned_twice
    
    def test_clean_preserves_valid_data(self):
        """测试清洗保留有效数据"""
        # 准备
        mock_places = Mock()
        executor = Executor(places=mock_places)
        
        data = {
            "name": "Test Venue",
            "rating": 4.5,
            "price_level": 2,
            "types": ["restaurant", "cafe"],
            "location": {
                "lat": 51.5,
                "lng": -0.1
            }
        }
        
        # 执行
        cleaned = executor._clean_response_data(data)
        
        # 验证所有有效数据都被保留
        assert cleaned == data


class TestExecutorIntegration:
    """测试 Executor 的集成功能"""
    
    def test_full_integration_with_all_components(self):
        """测试完整集成（日志、指标、错误处理）"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {
            "results": [
                {
                    "place_id": "test_id",
                    "name": "Test Venue",
                    "formatted_address": "123 Test St",
                    "rating": 4.5,
                    "types": ["restaurant"],
                    "geometry": {
                        "location": {"lat": 51.5, "lng": -0.1}
                    }
                }
            ]
        }
        
        mock_logger = Mock()
        mock_metrics = Mock()
        mock_error_handler = Mock()
        
        executor = Executor(
            places=mock_places,
            logger=mock_logger,
            metrics=mock_metrics,
            error_handler=mock_error_handler
        )
        
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证所有组件都被使用
        assert isinstance(result, dict)
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        assert mock_metrics.request_duration_seconds.observe.called
    
    def test_data_cleaning_integration(self):
        """测试数据清洗集成"""
        # 准备
        mock_places = Mock()
        mock_places.text_search.return_value = {
            "results": [
                {
                    "place_id": "test_id",
                    "name": "Test Venue",
                    "formatted_address": "123 Test St",
                    "rating": 4.5,
                    "description": "",  # 应该被清洗掉
                    "extra_field": None,  # 应该被清洗掉
                    "types": ["restaurant"],
                    "geometry": {
                        "location": {"lat": 51.5, "lng": -0.1}
                    }
                }
            ]
        }
        
        executor = Executor(places=mock_places)
        
        executable = ExecutableMCP(
            tool_calls=[
                ToolCall(
                    tool="google_places_textsearch",
                    args={"query": "afternoon tea"}
                )
            ],
            selection_policy={},
            notes=None
        )
        
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
        result = executor.execute(executable, intent)
        
        # 验证数据被清洗
        assert isinstance(result, dict)
        assert len(result["candidates"]) == 1
        # 验证候选场所被正确创建（即使有无效字段）
        assert result["candidates"][0].name == "Test Venue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

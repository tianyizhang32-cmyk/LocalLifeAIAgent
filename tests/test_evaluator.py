"""
单元测试：Evaluator 模块

测试 Evaluator 模块的核心功能：
- 候选场所评估（evaluate）
- 数据验证集成
- 日志记录集成
- 指标收集集成
- 错误处理集成

验证需求：6.6
"""

import pytest
from unittest.mock import Mock
from local_lifestyle_agent.evaluator import Evaluator
from local_lifestyle_agent.schemas import (
    CandidateVenue,
    EvaluationReport,
    NormalizedIntent
)
from local_lifestyle_agent.infrastructure.error_handler import ErrorResponse


class TestEvaluatorEvaluate:
    """测试 Evaluator.evaluate 方法"""
    
    def test_evaluate_success(self):
        """测试成功的候选场所评估"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="The Ritz",
                address="150 Piccadilly, London",
                rating=4.8,
                user_ratings_total=1500,
                price_level=3,
                category="restaurant"
            ),
            CandidateVenue(
                venue_id="venue2",
                name="Claridge's",
                address="Brook Street, London",
                rating=4.7,
                user_ratings_total=1200,
                price_level=4,
                category="restaurant"
            ),
            CandidateVenue(
                venue_id="venue3",
                name="Local Cafe",
                address="High Street, London",
                rating=4.2,
                user_ratings_total=300,
                price_level=1,
                category="cafe"
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        assert isinstance(result, tuple)
        report, ranked = result
        
        assert isinstance(report, EvaluationReport)
        assert report.ok is True
        assert len(ranked) == 3
        
        # 验证排序（The Ritz 应该排第一）
        assert ranked[0][0].venue_id == "venue1"
        assert ranked[0][1]["total"] > 0
        
        # 验证评分组件
        scores = ranked[0][1]
        assert "total" in scores
        assert "rating" in scores
        assert "popularity" in scores
        assert "price_fit" in scores
        assert "pref_bonus" in scores
    
    def test_evaluate_with_rejected_ids(self):
        """测试过滤已拒绝的场所"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="The Ritz",
                address="150 Piccadilly, London",
                rating=4.8,
                user_ratings_total=1500,
                price_level=3
            ),
            CandidateVenue(
                venue_id="venue2",
                name="Claridge's",
                address="Brook Street, London",
                rating=4.7,
                user_ratings_total=1200,
                price_level=4
            )
        ]
        
        # venue1 被拒绝
        rejected_ids = ["venue1"]
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        assert report.ok is True
        assert len(ranked) == 1
        assert ranked[0][0].venue_id == "venue2"
    
    def test_evaluate_filters_low_rating(self):
        """测试过滤低评分场所"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="High Rating",
                address="Address 1",
                rating=4.5,
                user_ratings_total=1000,
                price_level=2
            ),
            CandidateVenue(
                venue_id="venue2",
                name="Low Rating",
                address="Address 2",
                rating=3.5,  # 低于 min_rating
                user_ratings_total=500,
                price_level=2
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        assert report.ok is True
        assert len(ranked) == 1
        assert ranked[0][0].venue_id == "venue1"
    
    def test_evaluate_no_candidates_pass(self):
        """测试没有候选场所通过评估的情况"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Low Rating",
                address="Address 1",
                rating=3.5,  # 低于 min_rating
                user_ratings_total=500,
                price_level=2
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        assert report.ok is False
        assert len(ranked) == 0
        assert "no_candidates_pass_hard_constraints" in report.hard_violations
        assert len(report.replan_suggestions) > 0
    
    def test_evaluate_preference_bonus(self):
        """测试偏好加分功能"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
            preferences={"quiet": True},  # 偏好安静
            hard_constraints={},
            output_requirements={},
            activity_type="afternoon_tea"
        )
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Tea House",
                address="Address 1",
                rating=4.5,
                user_ratings_total=1000,
                price_level=2,
                category="tea"  # 应该获得偏好加分
            ),
            CandidateVenue(
                venue_id="venue2",
                name="Restaurant",
                address="Address 2",
                rating=4.5,
                user_ratings_total=1000,
                price_level=2,
                category="restaurant"  # 不获得偏好加分
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        assert report.ok is True
        assert len(ranked) == 2
        
        # venue1 应该因为偏好加分排在前面
        assert ranked[0][0].venue_id == "venue1"
        assert ranked[0][1]["pref_bonus"] > 0
        assert ranked[1][1]["pref_bonus"] == 0
    
    def test_evaluate_with_logger(self):
        """测试带日志记录的评估"""
        # 准备
        mock_logger = Mock()
        evaluator = Evaluator(min_rating=4.0, logger=mock_logger)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="The Ritz",
                address="150 Piccadilly, London",
                rating=4.8,
                user_ratings_total=1500,
                price_level=3
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证日志被调用
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        
        # 验证日志内容
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Starting candidate evaluation" in info_calls
        assert "Evaluation completed successfully" in info_calls
    
    def test_evaluate_with_metrics(self):
        """测试带指标收集的评估"""
        # 准备
        mock_metrics = Mock()
        evaluator = Evaluator(min_rating=4.0, metrics=mock_metrics)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="The Ritz",
                address="150 Piccadilly, London",
                rating=4.8,
                user_ratings_total=1500,
                price_level=3
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证指标被记录
        assert mock_metrics.request_duration_seconds.observe.called
    
    def test_evaluate_with_invalid_candidate(self):
        """测试包含无效候选场所的情况"""
        # 准备
        mock_logger = Mock()
        evaluator = Evaluator(min_rating=4.0, logger=mock_logger)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Valid Venue",
                address="Address 1",
                rating=4.5,
                user_ratings_total=1000,
                price_level=2
            ),
            CandidateVenue(
                venue_id="",  # 无效：空 venue_id
                name="Invalid Venue",
                address="Address 2",
                rating=4.5,
                user_ratings_total=1000,
                price_level=2
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证：应该记录警告但继续处理
        assert mock_logger.warning.called
        
        # 验证结果
        report, ranked = result
        assert report.ok is True
        # 有效的候选场所应该被处理
        assert len(ranked) >= 1
    
    def test_evaluate_handles_missing_optional_fields(self):
        """测试处理缺少可选字段的候选场所"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Minimal Info",
                address="Address 1",
                rating=None,  # 缺少评分
                user_ratings_total=None,  # 缺少评论数
                price_level=None  # 缺少价格等级
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证：应该使用默认值处理
        report, ranked = result
        # 由于 rating 为 None，不会被 min_rating 过滤掉
        # 但评分会很低
        assert isinstance(report, EvaluationReport)
    
    def test_evaluate_error_handling(self):
        """测试错误处理"""
        # 准备
        mock_logger = Mock()
        mock_metrics = Mock()
        evaluator = Evaluator(
            min_rating=4.0,
            logger=mock_logger,
            metrics=mock_metrics
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
        
        # 传入无效的 candidates（不是列表）
        candidates = None
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证返回错误响应
        assert isinstance(result, ErrorResponse)
        assert mock_logger.log_error.called
        assert mock_metrics.record_error.called


class TestEvaluatorScoring:
    """测试评分算法"""
    
    def test_scoring_components(self):
        """测试评分组件计算"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Perfect Venue",
                address="Address 1",
                rating=5.0,  # 最高评分
                user_ratings_total=1200,  # 正好达到最大值
                price_level=2,  # 完美匹配预算
                category="restaurant"
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        scores = ranked[0][1]
        
        # 验证评分组件
        assert scores["rating"] == 1.0  # (5.0 - 4.0) / 1.0 = 1.0
        assert scores["popularity"] == 1.0  # 1200 / 1200 = 1.0
        assert scores["price_fit"] == 1.0  # 1.0 - abs(2 - 2) / 2.0 = 1.0
        assert scores["pref_bonus"] == 0.0  # 没有偏好
        
        # 验证总分
        expected_total = 0.45 * 1.0 + 0.30 * 1.0 + 0.15 * 1.0
        assert abs(scores["total"] - expected_total) < 0.01
    
    def test_sorting_by_total_score(self):
        """测试按总分排序"""
        # 准备
        evaluator = Evaluator(min_rating=4.0)
        
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="Medium Score",
                address="Address 1",
                rating=4.3,
                user_ratings_total=600,
                price_level=2
            ),
            CandidateVenue(
                venue_id="venue2",
                name="High Score",
                address="Address 2",
                rating=4.8,
                user_ratings_total=1500,
                price_level=2
            ),
            CandidateVenue(
                venue_id="venue3",
                name="Low Score",
                address="Address 3",
                rating=4.1,
                user_ratings_total=200,
                price_level=2
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证
        report, ranked = result
        
        # 验证排序：venue2 > venue1 > venue3
        assert ranked[0][0].venue_id == "venue2"
        assert ranked[1][0].venue_id == "venue1"
        assert ranked[2][0].venue_id == "venue3"
        
        # 验证分数递减
        assert ranked[0][1]["total"] > ranked[1][1]["total"]
        assert ranked[1][1]["total"] > ranked[2][1]["total"]


class TestEvaluatorIntegration:
    """测试 Evaluator 的集成功能"""
    
    def test_full_integration_with_all_components(self):
        """测试完整集成（日志、指标、错误处理）"""
        # 准备
        mock_logger = Mock()
        mock_metrics = Mock()
        mock_error_handler = Mock()
        
        evaluator = Evaluator(
            min_rating=4.0,
            logger=mock_logger,
            metrics=mock_metrics,
            error_handler=mock_error_handler
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
        
        candidates = [
            CandidateVenue(
                venue_id="venue1",
                name="The Ritz",
                address="150 Piccadilly, London",
                rating=4.8,
                user_ratings_total=1500,
                price_level=3
            )
        ]
        
        rejected_ids = []
        
        # 执行
        result = evaluator.evaluate(intent, candidates, rejected_ids)
        
        # 验证所有组件都被使用
        report, ranked = result
        assert isinstance(report, EvaluationReport)
        assert mock_logger.set_request_id.called
        assert mock_logger.info.called
        assert mock_metrics.request_duration_seconds.observe.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

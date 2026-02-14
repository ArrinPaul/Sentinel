"""
Unit tests for ScoringEngine
"""
import pytest
from app.services.scoring_engine import ScoringEngine


class TestScoringEngine:
    """Test suite for ScoringEngine class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ScoringEngine()
    
    def test_compute_final_score_basic(self):
        """Test basic score computation with known values"""
        result = self.engine.compute_final_score(
            liveness_score=0.8,
            deepfake_score=0.9,
            emotion_score=0.7
        )
        
        # Expected: 0.4 * 0.8 + 0.25 * 0.9 + 0.35 * 0.7 = 0.32 + 0.225 + 0.245 = 0.79
        expected_score = 0.79
        assert abs(result.final_score - expected_score) < 0.0001
        assert result.liveness_score == 0.8
        assert result.deepfake_score == 0.9
        assert result.emotion_score == 0.7
        assert result.passed is True
        assert result.timestamp > 0
    
    def test_threshold_boundary_pass(self):
        """Test that score exactly at 0.65 passes"""
        # Create scores that result in exactly 0.65
        # 0.4 * 0.65 + 0.25 * 0.65 + 0.35 * 0.65 = 0.65
        result = self.engine.compute_final_score(
            liveness_score=0.65,
            deepfake_score=0.65,
            emotion_score=0.65
        )
        
        assert abs(result.final_score - 0.65) < 0.0001
        assert result.passed is True
    
    def test_threshold_boundary_fail(self):
        """Test that score just below 0.65 fails"""
        # Create scores that result in below 0.65
        # 0.4 * 0.5 + 0.25 * 0.8 + 0.35 * 0.6 = 0.20 + 0.20 + 0.21 = 0.61
        result = self.engine.compute_final_score(
            liveness_score=0.5,
            deepfake_score=0.8,
            emotion_score=0.6
        )
        
        assert abs(result.final_score - 0.61) < 0.0001
        assert result.passed is False
    
    def test_perfect_scores(self):
        """Test with all perfect scores (1.0)"""
        result = self.engine.compute_final_score(
            liveness_score=1.0,
            deepfake_score=1.0,
            emotion_score=1.0
        )
        
        assert result.final_score == 1.0
        assert result.passed is True
    
    def test_zero_scores(self):
        """Test with all zero scores"""
        result = self.engine.compute_final_score(
            liveness_score=0.0,
            deepfake_score=0.0,
            emotion_score=0.0
        )
        
        assert result.final_score == 0.0
        assert result.passed is False
    
    def test_weighted_formula_liveness_dominant(self):
        """Test that liveness has the highest weight (0.4)"""
        # High liveness, low others
        result = self.engine.compute_final_score(
            liveness_score=1.0,
            deepfake_score=0.0,
            emotion_score=0.0
        )
        
        # Expected: 0.4 * 1.0 = 0.40
        assert abs(result.final_score - 0.40) < 0.0001
        assert result.passed is False  # Still below 0.65
    
    def test_weighted_formula_deepfake_and_emotion(self):
        """Test that deepfake and emotion have weights 0.25 and 0.35"""
        # Zero liveness, high deepfake and emotion
        result = self.engine.compute_final_score(
            liveness_score=0.0,
            deepfake_score=1.0,
            emotion_score=1.0
        )
        
        # Expected: 0.25 * 1.0 + 0.35 * 1.0 = 0.60
        assert abs(result.final_score - 0.60) < 0.0001
        assert result.passed is False
    
    def test_default_deepfake_score(self):
        """Test with default deepfake score of 1.0 (optional component)"""
        # As per user note: use 1.0 as default for deepfake
        result = self.engine.compute_final_score(
            liveness_score=0.8,
            deepfake_score=1.0,
            emotion_score=0.8
        )
        
        # Expected: 0.4 * 0.8 + 0.25 * 1.0 + 0.35 * 0.8 = 0.32 + 0.25 + 0.28 = 0.85
        expected_score = 0.85
        assert abs(result.final_score - expected_score) < 0.0001
        assert result.passed is True
    
    def test_emotion_as_liveness_proxy(self):
        """Test using liveness score as emotion proxy (optional component)"""
        # As per user note: use liveness score as proxy for emotion
        liveness = 0.75
        result = self.engine.compute_final_score(
            liveness_score=liveness,
            deepfake_score=1.0,
            emotion_score=liveness  # Using liveness as proxy
        )
        
        # Expected: 0.4 * 0.75 + 0.25 * 1.0 + 0.35 * 0.75 = 0.30 + 0.25 + 0.2625 = 0.8125
        expected_score = 0.8125
        assert abs(result.final_score - expected_score) < 0.0001
        assert result.passed is True
    
    def test_timestamp_is_set(self):
        """Test that timestamp is set in the result"""
        import time
        before = time.time()
        result = self.engine.compute_final_score(0.8, 0.9, 0.7)
        after = time.time()
        
        assert before <= result.timestamp <= after
    
    def test_weights_sum_to_one(self):
        """Verify that weights sum to 1.0"""
        total_weight = (
            self.engine.LIVENESS_WEIGHT +
            self.engine.DEEPFAKE_WEIGHT +
            self.engine.EMOTION_WEIGHT
        )
        assert abs(total_weight - 1.0) < 0.0001


# Property-Based Tests
from hypothesis import given, strategies as st


class TestScoringEngineProperties:
    """Property-based tests for ScoringEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ScoringEngine()
    
    @given(
        liveness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        deepfake=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        emotion=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_property_10_scoring_formula_correctness(self, liveness, deepfake, emotion):
        """
        Property 10: Scoring Formula Correctness
        
        **Validates: Requirements 7.1**
        
        For any set of component scores (liveness, deepfake, emotion), 
        the final score should equal exactly:
        0.4 × liveness_score + 0.3 × deepfake_score + 0.3 × emotion_score
        """
        result = self.engine.compute_final_score(liveness, deepfake, emotion)
        
        # Calculate expected score using the formula
        expected = 0.4 * liveness + 0.25 * deepfake + 0.35 * emotion
        
        # Verify the formula is applied correctly
        assert abs(result.final_score - expected) < 0.0001, (
            f"Final score {result.final_score} does not match expected {expected} "
            f"for inputs: liveness={liveness}, deepfake={deepfake}, emotion={emotion}"
        )
        
        # Verify component scores are preserved
        assert result.liveness_score == liveness
        assert result.deepfake_score == deepfake
        assert result.emotion_score == emotion
        
        # Verify timestamp is set
        assert result.timestamp > 0
    
    @given(
        liveness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        deepfake=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        emotion=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_property_11_threshold_based_decision(self, liveness, deepfake, emotion):
        """
        Property 11: Threshold-Based Verification Decision
        
        **Validates: Requirements 7.2, 7.3, 7.4**
        
        For any calculated final score, the verification should be marked as 
        successful if and only if the score is greater than or equal to 0.65.
        """
        result = self.engine.compute_final_score(liveness, deepfake, emotion)
        
        # Calculate expected score
        expected_score = 0.4 * liveness + 0.25 * deepfake + 0.35 * emotion
        
        # Verify threshold-based decision
        if expected_score >= 0.65:
            assert result.passed is True, (
                f"Verification should pass for score {expected_score:.4f} >= 0.65 "
                f"(liveness={liveness}, deepfake={deepfake}, emotion={emotion})"
            )
        else:
            assert result.passed is False, (
                f"Verification should fail for score {expected_score:.4f} < 0.65 "
                f"(liveness={liveness}, deepfake={deepfake}, emotion={emotion})"
            )
        
        # Verify the decision is consistent with the final score
        assert result.passed == (result.final_score >= self.engine.THRESHOLD)


class TestScoringEngineThresholdBoundary:
    """Unit tests for threshold boundary conditions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = ScoringEngine()
    
    def test_threshold_boundary_0_69_fails(self):
        """
        Test that score of exactly 0.64 fails verification
        
        **Validates: Requirements 7.3, 7.4**
        """
        # Create scores that result in below 0.65
        # 0.4 * 0.5 + 0.25 * 0.8 + 0.35 * 0.7 = 0.20 + 0.20 + 0.245 = 0.645
        result = self.engine.compute_final_score(
            liveness_score=0.5,
            deepfake_score=0.8,
            emotion_score=0.7
        )
        
        assert abs(result.final_score - 0.645) < 0.0001, (
            f"Expected score 0.645, got {result.final_score}"
        )
        assert result.passed is False, (
            "Score of 0.645 should fail (< 0.65 threshold)"
        )
    
    def test_threshold_boundary_0_70_passes(self):
        """
        Test that score of exactly 0.65 passes verification
        
        **Validates: Requirements 7.2, 7.3**
        """
        # Create scores that result in exactly 0.65
        # 0.4 * 0.65 + 0.25 * 0.65 + 0.35 * 0.65 = 0.65
        result = self.engine.compute_final_score(
            liveness_score=0.65,
            deepfake_score=0.65,
            emotion_score=0.65
        )
        
        assert abs(result.final_score - 0.65) < 0.0001, (
            f"Expected score 0.65, got {result.final_score}"
        )
        assert result.passed is True, (
            "Score of 0.65 should pass (>= 0.65 threshold)"
        )
    
    def test_threshold_boundary_just_below(self):
        """
        Test that score just below 0.65 fails
        
        **Validates: Requirements 7.4**
        """
        # Create scores that result in 0.649...
        # 0.4 * 0.649 + 0.25 * 0.65 + 0.35 * 0.65 = 0.2596 + 0.1625 + 0.2275 = 0.6496
        result = self.engine.compute_final_score(
            liveness_score=0.649,
            deepfake_score=0.65,
            emotion_score=0.65
        )
        
        assert result.final_score < 0.65, (
            f"Score {result.final_score} should be below 0.65"
        )
        assert result.passed is False, (
            f"Score {result.final_score} < 0.65 should fail"
        )
    
    def test_threshold_boundary_just_above(self):
        """
        Test that score just above 0.65 passes
        
        **Validates: Requirements 7.3**
        """
        # Create scores that result in 0.6504...
        # 0.4 * 0.651 + 0.25 * 0.65 + 0.35 * 0.65 = 0.2604 + 0.1625 + 0.2275 = 0.6504
        result = self.engine.compute_final_score(
            liveness_score=0.651,
            deepfake_score=0.65,
            emotion_score=0.65
        )
        
        assert result.final_score > 0.65, (
            f"Score {result.final_score} should be above 0.65"
        )
        assert result.passed is True, (
            f"Score {result.final_score} > 0.65 should pass"
        )

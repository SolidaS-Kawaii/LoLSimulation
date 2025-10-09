"""
Phase 3 Testing: Feature Engineering
Tests feature extraction components and integration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.features import (
    MetaAnalyzer, 
    SynergyCalculator, 
    MatchupCalculator,
    FeatureExtractor
)
from src.core.draft_engine import DraftEngine
import numpy as np


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_meta_analyzer():
    """Test meta analyzer functionality"""
    print_header("TEST 1: META ANALYZER")
    
    try:
        analyzer = MetaAnalyzer()
        
        # Test 1.1: Single champion meta
        print("\n📊 Test 1.1: Single champion meta")
        yasuo_meta = analyzer.get_champion_meta(157, "MIDDLE")
        assert 'pick_rate' in yasuo_meta, "Missing pick_rate"
        assert 'ban_rate' in yasuo_meta, "Missing ban_rate"
        assert 'win_rate' in yasuo_meta, "Missing win_rate"
        assert 0 <= yasuo_meta['pick_rate'] <= 1, "Pick rate out of range"
        assert 0 <= yasuo_meta['win_rate'] <= 1, "Win rate out of range"
        print(f"  ✓ Yasuo meta: Pick={yasuo_meta['pick_rate']:.3f}, "
              f"Ban={yasuo_meta['ban_rate']:.3f}, Win={yasuo_meta['win_rate']:.3f}")
        
        # Test 1.2: Team meta features
        print("\n📊 Test 1.2: Team meta features")
        team_ids = [157, 238, 61, 222, 412]
        team_roles = ["MIDDLE", "MIDDLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team_meta = analyzer.get_team_meta_features(team_ids, team_roles)
        assert 'avg_pick_rate' in team_meta, "Missing avg_pick_rate"
        assert 'max_win_rate' in team_meta, "Missing max_win_rate"
        print(f"  ✓ Team meta: Avg Pick={team_meta['avg_pick_rate']:.3f}, "
              f"Avg Win={team_meta['avg_win_rate']:.3f}")
        
        # Test 1.3: Banned champions meta
        print("\n📊 Test 1.3: Banned champions meta")
        banned = [157, 238, 61, 18, 11]
        ban_meta = analyzer.get_banned_champions_meta(banned)
        assert 'avg_ban_rate' in ban_meta, "Missing avg_ban_rate"
        print(f"  ✓ Ban meta: Avg Ban Rate={ban_meta['avg_ban_rate']:.3f}")
        
        print("\n✅ Meta Analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Meta Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synergy_calculator():
    """Test synergy calculator functionality"""
    print_header("TEST 2: SYNERGY CALCULATOR")
    
    try:
        calculator = SynergyCalculator()
        
        # Test 2.1: Pair synergy
        print("\n🤝 Test 2.1: Champion pair synergy")
        synergy = calculator.get_pair_synergy(157, "MIDDLE", 54, "TOP")
        assert 0 <= synergy <= 1, "Synergy score out of range"
        print(f"  ✓ Yasuo + Malphite synergy: {synergy:.3f}")
        
        # Test 2.2: Team synergy
        print("\n🤝 Test 2.2: Team synergy calculation")
        team_ids = [54, 64, 157, 222, 412]
        team_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team_syn = calculator.calculate_team_synergy(team_ids, team_roles)
        assert 'avg_synergy' in team_syn, "Missing avg_synergy"
        assert 'pairwise_synergies' in team_syn, "Missing pairwise_synergies"
        assert len(team_syn['pairwise_synergies']) == 10, "Expected 10 pairs"
        print(f"  ✓ Team synergy: Avg={team_syn['avg_synergy']:.3f}, "
              f"Min={team_syn['min_synergy']:.3f}, Max={team_syn['max_synergy']:.3f}")
        
        # Test 2.3: Role synergies
        print("\n🤝 Test 2.3: Role-specific synergies")
        role_syn = calculator.calculate_role_synergies(team_ids, team_roles)
        assert len(role_syn) == 5, "Expected 5 role synergies"
        print(f"  ✓ Role synergies calculated for {len(role_syn)} roles")
        
        # Test 2.4: Team comparison
        print("\n🤝 Test 2.4: Team synergy comparison")
        team2_ids = [86, 11, 238, 498, 53]
        team2_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        comparison = calculator.compare_team_synergies(
            team_ids, team_roles, team2_ids, team2_roles
        )
        assert 'synergy_diff' in comparison, "Missing synergy_diff"
        print(f"  ✓ Synergy difference: {comparison['synergy_diff']:+.3f}")
        
        print("\n✅ Synergy Calculator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Synergy Calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matchup_calculator():
    """Test matchup calculator functionality"""
    print_header("TEST 3: MATCHUP CALCULATOR")
    
    try:
        calculator = MatchupCalculator()
        
        # Test 3.1: Single matchup
        print("\n⚔️ Test 3.1: Champion matchup")
        matchup = calculator.get_matchup_score(157, "MIDDLE", 54, "TOP")
        assert 0 <= matchup <= 1, "Matchup score out of range"
        advantage = (matchup - 0.5) * 100
        print(f"  ✓ Yasuo vs Malphite: {matchup:.3f} ({advantage:+.1f}%)")
        
        # Test 3.2: Role matchups
        print("\n⚔️ Test 3.2: Role-by-role matchups")
        team1_ids = [86, 64, 157, 222, 412]
        team1_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team2_ids = [54, 11, 238, 498, 53]
        team2_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        
        role_matchups = calculator.calculate_role_matchups(
            team1_ids, team1_roles, team2_ids, team2_roles
        )
        assert len(role_matchups) == 5, "Expected 5 role matchups"
        print(f"  ✓ Role matchups calculated for {len(role_matchups)} roles")
        
        # Test 3.3: Team matchups
        print("\n⚔️ Test 3.3: Full team matchup analysis")
        team_matchups = calculator.calculate_team_matchups(
            team1_ids, team1_roles, team2_ids, team2_roles
        )
        assert 'overall_matchup_score' in team_matchups, "Missing overall score"
        assert 'favorable_matchups' in team_matchups, "Missing favorable count"
        print(f"  ✓ Overall matchup: {team_matchups['overall_matchup_score']:+.3f}")
        print(f"  ✓ Favorable: {team_matchups['favorable_matchups']}, "
              f"Unfavorable: {team_matchups['unfavorable_matchups']}")
        
        # Test 3.4: Counters
        print("\n⚔️ Test 3.4: Counter lookup")
        counters = calculator.get_best_counters(157, "MIDDLE", n=3)
        assert len(counters) <= 3, "Too many counters returned"
        print(f"  ✓ Found {len(counters)} counters to Yasuo")
        
        print("\n✅ Matchup Calculator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Matchup Calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extractor():
    """Test complete feature extraction"""
    print_header("TEST 4: FEATURE EXTRACTOR")
    
    try:
        extractor = FeatureExtractor()
        engine = DraftEngine(user_side='blue')
        
        # Test 4.1: Extract from initial state
        print("\n🔬 Test 4.1: Extract from initial state")
        features_initial = extractor.extract_features(engine.state)
        assert features_initial.shape == (129,), f"Expected shape (129,), got {features_initial.shape}"
        assert features_initial.dtype == np.float32, f"Expected float32, got {features_initial.dtype}"
        print(f"  ✓ Initial features: shape={features_initial.shape}, dtype={features_initial.dtype}")
        
        # Test 4.2: Extract after bans
        print("\n🔬 Test 4.2: Extract after bans")
        bans = [157, 238, 61, 18, 11, 201]
        for ban_id in bans:
            engine.execute_ban(ban_id)
        
        features_after_bans = extractor.extract_features(engine.state)
        assert features_after_bans.shape == (129,), "Shape mismatch after bans"
        assert not np.array_equal(features_initial, features_after_bans), "Features unchanged after bans"
        print(f"  ✓ Features changed after bans")
        
        # Test 4.3: Extract during picks
        print("\n🔬 Test 4.3: Extract during picks")
        picks = [
            (222, "Jinx", "BOTTOM"),
            (498, "Kai'Sa", "BOTTOM"),
            (412, "Thresh", "UTILITY"),
            (89, "Leona", "UTILITY"),
            (1, "Annie", "MIDDLE")
        ]
        
        for champ_id, name, role in picks:
            engine.execute_pick(champ_id, name, role)
        
        features_with_picks = extractor.extract_features(engine.state)
        assert features_with_picks.shape == (129,), "Shape mismatch with picks"
        print(f"  ✓ Features extracted with picks")
        
        # Test 4.4: Feature value ranges
        print("\n🔬 Test 4.4: Feature value ranges")
        assert not np.any(np.isnan(features_with_picks)), "NaN values detected"
        assert not np.any(np.isinf(features_with_picks)), "Inf values detected"
        print(f"  ✓ Min: {features_with_picks.min():.3f}, "
              f"Max: {features_with_picks.max():.3f}, "
              f"Mean: {features_with_picks.mean():.3f}")
        
        # Test 4.5: Feature names
        print("\n🔬 Test 4.5: Feature names")
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == 129, f"Expected 129 names, got {len(feature_names)}"
        print(f"  ✓ Generated {len(feature_names)} feature names")
        
        print("\n✅ Feature Extractor tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature Extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with Draft Engine"""
    print_header("TEST 5: INTEGRATION")
    
    try:
        print("\n🔄 Test 5.1: Complete draft simulation")
        engine = DraftEngine(user_side='blue')
        extractor = FeatureExtractor()
        
        # Simulate complete draft
        bans = [157, 238, 61, 18, 11, 201, 22, 64, 69, 110]
        picks = [
            (222, "Jinx", "BOTTOM"),
            (498, "Kai'Sa", "BOTTOM"),
            (412, "Thresh", "UTILITY"),
            (89, "Leona", "UTILITY"),
            (1, "Annie", "MIDDLE"),
            (8, "Vladimir", "MIDDLE"),
            (86, "Garen", "TOP"),
            (122, "Darius", "TOP"),
            (5, "Xin Zhao", "JUNGLE"),
            (77, "Udyr", "JUNGLE")
        ]
        
        # Execute bans
        for ban_id in bans:
            engine.execute_ban(ban_id)
        
        # Execute picks with feature extraction after each
        feature_history = []
        for champ_id, name, role in picks:
            engine.execute_pick(champ_id, name, role)
            features = extractor.extract_features(engine.state)
            feature_history.append(features)
        
        assert len(feature_history) == 10, f"Expected 10 feature vectors, got {len(feature_history)}"
        print(f"  ✓ Extracted features at {len(feature_history)} draft stages")
        
        # Verify feature evolution
        print("\n🔄 Test 5.2: Feature evolution")
        first_features = feature_history[0]
        last_features = feature_history[-1]
        difference = np.abs(last_features - first_features).sum()
        assert difference > 0, "Features did not change during draft"
        print(f"  ✓ Feature vector changed by {difference:.2f} total absolute difference")
        
        # Verify final state
        print("\n🔄 Test 5.3: Final state verification")
        assert engine.state.is_draft_complete(), "Draft not marked complete"
        final_features = extractor.extract_features(engine.state)
        assert final_features.shape == (129,), "Final features shape mismatch"
        print(f"  ✓ Draft complete with valid features")
        
        print("\n✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 tests"""
    print("\n" + "=" * 70)
    print("  PHASE 3: FEATURE ENGINEERING - TEST SUITE")
    print("=" * 70)
    
    results = {
        "Meta Analyzer": test_meta_analyzer(),
        "Synergy Calculator": test_synergy_calculator(),
        "Matchup Calculator": test_matchup_calculator(),
        "Feature Extractor": test_feature_extractor(),
        "Integration": test_integration()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed_count = sum(results.values())
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED! Phase 3 Complete! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
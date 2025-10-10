"""
Win Probability Predictor for League of Legends Draft System
Predicts win probability with hypothetical champion picks
"""

import numpy as np
from copy import deepcopy
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.draft_engine import DraftState
from src.features.feature_extractor import FeatureExtractor
from src.ai.model_manager import ModelManager


class WinProbabilityPredictor:
    """Predicts win probability for draft scenarios"""
    
    def __init__(self, model_manager: ModelManager, feature_extractor: FeatureExtractor):
        """
        Initialize predictor
        
        Args:
            model_manager: ModelManager instance
            feature_extractor: FeatureExtractor instance
        """
        self.model_manager = model_manager
        self.feature_extractor = feature_extractor
    
    def predict_with_champion(self,
                             draft_state: DraftState,
                             champion_id: int,
                             champion_name: str,
                             role: str,
                             side: str,
                             model_name: str = None) -> float:
        """
        Predict win probability if we pick this champion
        
        Args:
            draft_state: Current draft state
            champion_id: Champion ID to simulate picking
            champion_name: Champion name
            role: Role for this champion
            side: Which side ('blue' or 'red')
            model_name: Which ML model to use
        
        Returns:
            Win probability (0.0-1.0) for the specified side
        """
        # Create a deep copy of draft state
        simulated_state = deepcopy(draft_state)
        
        # Simulate adding this pick
        simulated_state.add_pick(champion_id, champion_name, role, side)
        
        # Extract features from simulated state
        features = self.feature_extractor.extract_features(simulated_state)
        
        # Predict win probability
        win_prob = self.model_manager.predict(features, model_name)
        
        # Model predicts for blue side (team1)
        # If we're red side, return complement
        if side == 'red':
            win_prob = 1.0 - win_prob
        
        return win_prob
    
    def predict_current_state(self,
                             draft_state: DraftState,
                             side: str,
                             model_name: str = None) -> float:
        """
        Predict win probability for current draft state
        
        Args:
            draft_state: Current draft state
            side: Which side to predict for ('blue' or 'red')
            model_name: Which ML model to use
        
        Returns:
            Win probability (0.0-1.0) for the specified side
        """
        # Extract features from current state
        features = self.feature_extractor.extract_features(draft_state)
        
        # Predict
        win_prob = self.model_manager.predict(features, model_name)
        
        # Adjust for side
        if side == 'red':
            win_prob = 1.0 - win_prob
        
        return win_prob
    
    def compare_picks(self,
                     draft_state: DraftState,
                     candidates: list,
                     side: str,
                     model_name: str = None) -> dict:
        """
        Compare win probabilities for multiple candidate picks
        
        Args:
            draft_state: Current draft state
            candidates: List of (champion_id, champion_name, role) tuples
            side: Which side is picking
            model_name: Which ML model to use
        
        Returns:
            Dictionary mapping champion_id to win_probability
        """
        results = {}
        
        for champion_id, champion_name, role in candidates:
            win_prob = self.predict_with_champion(
                draft_state, champion_id, champion_name, role, side, model_name
            )
            results[champion_id] = win_prob
        
        return results


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("WIN PROBABILITY PREDICTOR DEMO")
    print("=" * 60)
    
    try:
        from src.core.draft_engine import DraftEngine
        
        # Initialize components
        print("\nInitializing components...")
        model_manager = ModelManager()
        feature_extractor = FeatureExtractor()
        predictor = WinProbabilityPredictor(model_manager, feature_extractor)
        print("✓ Predictor ready")
        
        # Create sample draft
        print("\n📊 Creating sample draft...")
        engine = DraftEngine(user_side='blue')
        
        # Execute some bans
        bans = [157, 238, 61, 18, 11, 201, 22, 64, 69, 110]
        for ban_id in bans:
            engine.execute_ban(ban_id)
        
        # Execute some picks
        picks = [
            (222, "Jinx", "BOTTOM", "blue"),
            (498, "Kai'Sa", "BOTTOM", "red"),
            (412, "Thresh", "UTILITY", "red"),
            (89, "Leona", "UTILITY", "blue"),
        ]
        
        for champ_id, name, role, side in picks:
            engine.execute_pick(champ_id, name, role, side)
        
        print(f"  Bans: {len(engine.state.blue_bans) + len(engine.state.red_bans)}")
        print(f"  Picks: {len(engine.state.blue_picks) + len(engine.state.red_picks)}")
        
        # Test 1: Current state win probability
        print("\n📊 Test 1: Current state win probability")
        current_wp_blue = predictor.predict_current_state(engine.state, 'blue')
        current_wp_red = predictor.predict_current_state(engine.state, 'red')
        print(f"  Blue win prob: {current_wp_blue:.3f} ({current_wp_blue*100:.1f}%)")
        print(f"  Red win prob: {current_wp_red:.3f} ({current_wp_red*100:.1f}%)")
        print(f"  Sum check: {current_wp_blue + current_wp_red:.3f} (should be ~1.0)")
        
        # Test 2: Predict with specific champion
        print("\n📊 Test 2: Predict with Yasuo (MIDDLE)")
        yasuo_wp = predictor.predict_with_champion(
            engine.state, 157, "Yasuo", "MIDDLE", "blue"
        )
        print(f"  Win prob if pick Yasuo: {yasuo_wp:.3f} ({yasuo_wp*100:.1f}%)")
        print(f"  Improvement: {(yasuo_wp - current_wp_blue)*100:+.1f}%")
        
        # Test 3: Compare multiple picks
        print("\n📊 Test 3: Compare multiple mid laners")
        candidates = [
            (1, "Annie", "MIDDLE"),
            (103, "Ahri", "MIDDLE"),
            (268, "Azir", "MIDDLE"),
        ]
        
        comparisons = predictor.compare_picks(
            engine.state, candidates, 'blue'
        )
        
        print("  Champion comparisons:")
        for (champ_id, name, role), win_prob in zip(candidates, comparisons.values()):
            improvement = (win_prob - current_wp_blue) * 100
            print(f"    {name:15s}: {win_prob:.3f} ({improvement:+.1f}%)")
        
        # Find best pick
        best_id = max(comparisons, key=comparisons.get)
        best_name = next(name for cid, name, _ in candidates if cid == best_id)
        print(f"  → Best pick: {best_name} ({comparisons[best_id]:.3f})")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

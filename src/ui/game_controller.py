"""
Game Controller Module for Terminal UI
Manages game flow, AI opponent, and state management
"""

import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.draft_engine import DraftEngine
from src.core.champion_db import ChampionDatabase
from src.ai.recommender import ChampionRecommender
from src.ai.predictor import WinProbabilityPredictor
import config


class GameController:
    """
    Game controller for terminal UI
    
    Features:
    - Manage game modes (User vs AI, User vs User, AI vs AI)
    - Execute turns
    - AI opponent logic
    - State tracking
    - Draft history
    """
    
    def __init__(self, 
                 mode: str,
                 user_side: str,
                 model_name: str,
                 champion_db: ChampionDatabase,
                 recommender: ChampionRecommender,
                 predictor: WinProbabilityPredictor):
        """
        Initialize game controller
        
        Args:
            mode: Game mode ('user_vs_ai', 'user_vs_user', 'ai_vs_ai')
            user_side: User's side ('blue' or 'red')
            model_name: ML model to use
            champion_db: Champion database
            recommender: Champion recommender
            predictor: Win probability predictor
        """
        self.mode = mode
        self.user_side = user_side
        self.model_name = model_name
        
        # Components
        self.champion_db = champion_db
        self.recommender = recommender
        self.predictor = predictor
        
        # Game state
        self.engine = DraftEngine(user_side=user_side)
        self.draft_history: List[Dict] = []
        self.start_time = datetime.now()
    
    def is_user_turn(self) -> bool:
        """Check if it's user's turn"""
        if self.mode == 'ai_vs_ai':
            return False
        elif self.mode == 'user_vs_user':
            return True
        else:  # user_vs_ai
            return self.engine.is_user_turn()
    
    def execute_user_ban(self, champion_id: int, champion_name: str) -> bool:
        """
        Execute user's ban
        
        Args:
            champion_id: Champion ID to ban
            champion_name: Champion name
            
        Returns:
            True if successful
        """
        current_side = self.engine.get_current_side()
        
        if not current_side:
            return False
        
        success = self.engine.execute_ban(champion_id, current_side)
        
        if success:
            # Log to history
            self._log_action('ban', champion_name, None, current_side, is_ai=False)
        
        return success
    
    def execute_user_pick(self, champion_id: int, champion_name: str, role: str) -> bool:
        """
        Execute user's pick
        
        Args:
            champion_id: Champion ID
            champion_name: Champion name
            role: Role
            
        Returns:
            True if successful
        """
        current_side = self.engine.get_current_side()
        
        if not current_side:
            return False
        
        success = self.engine.execute_pick(champion_id, champion_name, role, current_side)
        
        if success:
            # Log to history
            self._log_action('pick', champion_name, role, current_side, is_ai=False)
        
        return success
    
    def execute_ai_turn(self) -> Optional[Dict]:
        """
        AI takes its turn (ban or pick)
        
        Returns:
            {
                'action': 'ban' | 'pick',
                'champion_id': int,
                'champion_name': str,
                'role': str (if pick),
                'score': float (if pick),
                'side': str
            }
            or None if draft complete
        """
        current_side = self.engine.get_current_side()
        
        if not current_side:
            return None
        
        turn_info = self.engine.get_turn_info()
        action = turn_info.get('action')
        
        if action == 'ban':
            return self._ai_ban(current_side)
        elif action == 'pick':
            return self._ai_pick(current_side)
        
        return None
    
    def _ai_ban(self, side: str) -> Dict:
        """
        AI executes a ban
        
        Strategy:
        1. Get top recommendations for opponent
        2. Ban the highest-scored champion that can be banned
        3. Fallback to banning high-meta champions
        """
        # Get opponent's side
        opponent_side = 'red' if side == 'blue' else 'blue'
        
        # Get recommendations for opponent
        try:
            recommendations = self.recommender.get_recommendations(
                self.engine.state,
                user_side=opponent_side,
                model_name=self.model_name
            )
            
            # Try to ban top recommendations
            for rec in recommendations[:5]:
                if self.engine.state.is_champion_available(rec.champion_id):
                    success = self.engine.execute_ban(rec.champion_id, side)
                    
                    if success:
                        self._log_action('ban', rec.champion_name, None, side, is_ai=True)
                        
                        return {
                            'action': 'ban',
                            'champion_id': rec.champion_id,
                            'champion_name': rec.champion_name,
                            'side': side,
                            'reason': f"Denying strong pick for {opponent_side}"
                        }
        
        except Exception as e:
            print(f"[AI] Error getting recommendations: {e}")
        
        # Fallback: Ban high-pick-rate champions
        return self._fallback_ban(side)
    
    def _fallback_ban(self, side: str) -> Dict:
        """Fallback ban strategy: ban high-meta champions"""
        all_champion_ids = self.champion_db.get_all_champion_ids()
        
        # Sort by pick rate (high to low)
        candidates = []
        for champ_id in all_champion_ids:
            if self.engine.state.is_champion_available(champ_id):
                champ_name = self.champion_db.get_name(champ_id)
                roles = self.champion_db.get_roles(champ_name)
                
                if roles:
                    max_pick_rate = max(r['pick_rate'] for r in roles)
                    candidates.append((champ_id, champ_name, max_pick_rate))
        
        # Sort by pick rate
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        if candidates:
            champion_id, champion_name, _ = candidates[0]
            success = self.engine.execute_ban(champion_id, side)
            
            if success:
                self._log_action('ban', champion_name, None, side, is_ai=True)
                
                return {
                    'action': 'ban',
                    'champion_id': champion_id,
                    'champion_name': champion_name,
                    'side': side,
                    'reason': 'High meta priority'
                }
        
        return None
    
    def _ai_pick(self, side: str) -> Dict:
        """
        AI executes a pick
        
        Strategy:
        1. Get recommendations from AI
        2. Pick the #1 recommendation
        """
        try:
            recommendations = self.recommender.get_recommendations(
                self.engine.state,
                user_side=side,
                model_name=self.model_name
            )
            
            if recommendations:
                # Pick #1 recommendation
                top_rec = recommendations[0]
                
                success = self.engine.execute_pick(
                    top_rec.champion_id,
                    top_rec.champion_name,
                    top_rec.role,
                    side
                )
                
                if success:
                    self._log_action('pick', top_rec.champion_name, top_rec.role, side, is_ai=True)
                    
                    return {
                        'action': 'pick',
                        'champion_id': top_rec.champion_id,
                        'champion_name': top_rec.champion_name,
                        'role': top_rec.role,
                        'score': top_rec.total_score,
                        'side': side
                    }
        
        except Exception as e:
            print(f"[AI] Error getting recommendations: {e}")
        
        # Fallback: pick random available champion
        return self._fallback_pick(side)
    
    def _fallback_pick(self, side: str) -> Optional[Dict]:
        """Fallback pick strategy"""
        vacant_roles = self.engine.state.get_vacant_roles(side)
        
        if not vacant_roles:
            return None
        
        target_role = vacant_roles[0]
        
        # Find any available champion for this role
        all_champion_ids = self.champion_db.get_all_champion_ids()
        
        for champ_id in all_champion_ids:
            if self.engine.state.is_champion_available(champ_id):
                champ_name = self.champion_db.get_name(champ_id)
                roles = self.champion_db.get_roles(champ_name)
                
                for role_data in roles:
                    if role_data['role'] == target_role:
                        success = self.engine.execute_pick(
                            champ_id,
                            champ_name,
                            target_role,
                            side
                        )
                        
                        if success:
                            self._log_action('pick', champ_name, target_role, side, is_ai=True)
                            
                            return {
                                'action': 'pick',
                                'champion_id': champ_id,
                                'champion_name': champ_name,
                                'role': target_role,
                                'score': 0.0,
                                'side': side
                            }
        
        return None
    
    def get_recommendations(self) -> List:
        """Get AI recommendations for current turn"""
        current_side = self.engine.get_current_side()
        
        if not current_side:
            return []
        
        # In user vs user mode, always show for current turn
        # In user vs ai mode, only show for user's side
        if self.mode == 'user_vs_ai' and current_side != self.user_side:
            return []
        
        try:
            recommendations = self.recommender.get_recommendations(
                self.engine.state,
                user_side=current_side,
                model_name=self.model_name
            )
            return recommendations
        
        except Exception as e:
            print(f"[Controller] Error getting recommendations: {e}")
            return []
    
    def is_complete(self) -> bool:
        """Check if draft is complete"""
        return self.engine.state.is_draft_complete()
    
    def get_winner_prediction(self) -> Dict:
        """
        Predict winner after draft complete
        
        Returns:
            {
                'blue_win_prob': float,
                'red_win_prob': float,
                'predicted_winner': 'blue' | 'red'
            }
        """
        try:
            blue_prob = self.predictor.predict_current_state(
                self.engine.state,
                side='blue',
                model_name=self.model_name
            )
            
            red_prob = 1.0 - blue_prob
            
            return {
                'blue_win_prob': blue_prob,
                'red_win_prob': red_prob,
                'predicted_winner': 'blue' if blue_prob > red_prob else 'red'
            }
        
        except Exception as e:
            print(f"[Controller] Error predicting winner: {e}")
            return {
                'blue_win_prob': 0.5,
                'red_win_prob': 0.5,
                'predicted_winner': 'unknown'
            }
    
    def _log_action(self, action: str, champion: str, role: Optional[str], 
                   side: str, is_ai: bool):
        """Log action to draft history"""
        self.draft_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'champion': champion,
            'role': role,
            'side': side,
            'is_ai': is_ai,
            'turn': len(self.draft_history) + 1
        })
    
    def export_draft(self, filename: Optional[str] = None) -> str:
        """
        Export draft history to file
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"draft_{timestamp}.json"
        
        # Ensure output directory exists
        output_dir = config.OUTPUTS_DIR / 'draft_history'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        # Prepare export data
        export_data = {
            'metadata': {
                'mode': self.mode,
                'user_side': self.user_side,
                'model': self.model_name,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'draft_state': self.engine.state.get_summary(),
            'history': self.draft_history
        }
        
        # Add winner prediction if complete
        if self.is_complete():
            export_data['prediction'] = self.get_winner_prediction()
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def get_summary(self) -> Dict:
        """Get game summary"""
        return {
            'mode': self.mode,
            'user_side': self.user_side,
            'model': self.model_name,
            'turns': len(self.draft_history),
            'bans': len(self.engine.state.blue_bans) + len(self.engine.state.red_bans),
            'picks': len(self.engine.state.blue_picks) + len(self.engine.state.red_picks),
            'complete': self.is_complete()
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test game controller"""
    
    print("="*70)
    print("GAME CONTROLLER - TESTING")
    print("="*70)
    
    from src.features.synergy_calculator import SynergyCalculator
    from src.features.matchup_calculator import MatchupCalculator
    from src.features.meta_analyzer import MetaAnalyzer
    from src.features.feature_extractor import FeatureExtractor
    from src.ai.model_manager import ModelManager
    
    # Initialize components
    print("\nInitializing components...")
    champion_db = ChampionDatabase()
    champion_db.load(verbose=False)
    
    model_manager = ModelManager()
    feature_extractor = FeatureExtractor()
    predictor = WinProbabilityPredictor(model_manager, feature_extractor)
    synergy_calc = SynergyCalculator()
    matchup_calc = MatchupCalculator()
    meta_analyzer = MetaAnalyzer()
    
    recommender = ChampionRecommender(
        predictor, synergy_calc, matchup_calc, meta_analyzer, champion_db
    )
    
    print("✓ All components initialized")
    
    # Test 1: User vs AI
    print("\n1. Testing User vs AI Mode:")
    print("-" * 70)
    
    controller = GameController(
        mode='user_vs_ai',
        user_side='blue',
        model_name='xgboost',
        champion_db=champion_db,
        recommender=recommender,
        predictor=predictor
    )
    
    print(f"Mode: {controller.mode}")
    print(f"User side: {controller.user_side}")
    print(f"Is user turn: {controller.is_user_turn()}")
    
    # Simulate a few turns
    print("\n2. Simulating Draft:")
    print("-" * 70)
    
    for turn in range(10):
        current_side = controller.engine.get_current_side()
        
        if not current_side:
            break
        
        is_user_turn = controller.is_user_turn()
        turn_info = controller.engine.get_turn_info()
        
        print(f"\nTurn {turn + 1}: {current_side.upper()} - {turn_info['action'].upper()}")
        print(f"  User turn: {is_user_turn}")
        
        if is_user_turn:
            # Simulate user action (execute ban/pick manually)
            print(f"  [User would take action here]")
            
            if turn_info['action'] == 'ban':
                # Use first available champion
                all_ids = champion_db.get_all_champion_ids()
                for champ_id in all_ids:
                    if controller.engine.state.is_champion_available(champ_id):
                        champ_name = champion_db.get_name(champ_id)
                        controller.execute_user_ban(champ_id, champ_name)
                        print(f"  User banned: {champ_name}")
                        break
        else:
            # AI turn
            ai_action = controller.execute_ai_turn()
            
            if ai_action:
                print(f"  AI {ai_action['action']}: {ai_action['champion_name']}", end='')
                if 'role' in ai_action:
                    print(f" ({ai_action['role']})")
                else:
                    print()
    
    # Test 3: Get recommendations
    print("\n3. Testing Recommendations:")
    print("-" * 70)
    
    recs = controller.get_recommendations()
    print(f"Got {len(recs)} recommendations")
    
    if recs:
        for rec in recs[:3]:
            print(f"  #{rec.rank}: {rec.champion_name} ({rec.role}) - {rec.total_score:.3f}")
    
    # Test 4: Export
    print("\n4. Testing Export:")
    print("-" * 70)
    
    filepath = controller.export_draft()
    print(f"✓ Draft exported to: {filepath}")
    
    # Test 5: Summary
    print("\n5. Game Summary:")
    print("-" * 70)
    
    summary = controller.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✓ Game controller testing complete")
    print("="*70)

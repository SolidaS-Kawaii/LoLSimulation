"""
Champion Recommender for League of Legends Draft System
Main recommendation engine that scores and ranks champions
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RECOMMENDATION_WEIGHTS, TOP_N_RECOMMENDATIONS
from src.core.draft_engine import DraftState
from src.core.champion_db import ChampionDatabase
from src.features.synergy_calculator import SynergyCalculator
from src.features.matchup_calculator import MatchupCalculator
from src.features.meta_analyzer import MetaAnalyzer
from src.ai.predictor import WinProbabilityPredictor


@dataclass
class Recommendation:
    """Champion recommendation with scores and structured data"""
    # Basic info
    champion_id: int
    champion_name: str
    role: str
    rank: int
    
    # Scores (0-1)
    total_score: float
    win_probability: float
    synergy_score: float
    counter_score: float
    meta_score: float
    
    # Structured data
    synergy_pairs: List[Tuple[str, float]]      # [(champ_name, score), ...]
    counter_matchups: List[Tuple[str, float]]   # [(enemy_name, win_rate), ...]
    meta_stats: Dict[str, float]                # {'pick_rate': ..., 'ban_rate': ..., 'win_rate': ...}


class ChampionRecommender:
    """Main recommendation engine"""
    
    def __init__(self,
                 predictor: WinProbabilityPredictor,
                 synergy_calculator: SynergyCalculator,
                 matchup_calculator: MatchupCalculator,
                 meta_analyzer: MetaAnalyzer,
                 champion_db: ChampionDatabase):
        """
        Initialize recommender
        
        Args:
            predictor: Win probability predictor
            synergy_calculator: Synergy calculator
            matchup_calculator: Matchup calculator
            meta_analyzer: Meta analyzer
            champion_db: Champion database
        """
        self.predictor = predictor
        self.synergy_calc = synergy_calculator
        self.matchup_calc = matchup_calculator
        self.meta_analyzer = meta_analyzer
        self.champion_db = champion_db
        
        # Weights from config
        self.weights = RECOMMENDATION_WEIGHTS
    
    def get_recommendations(self,
                           draft_state: DraftState,
                           user_side: str,
                           model_name: str = None) -> List[Recommendation]:
        """
        Get top 5 champion recommendations for all vacant roles
        
        Args:
            draft_state: Current draft state
            user_side: Which side user is playing ('blue' or 'red')
            model_name: Which ML model to use (default from config)
        
        Returns:
            List of top 5 Recommendation objects, sorted by total_score
        """
        try:
            # Get vacant roles
            vacant_roles = draft_state.get_vacant_roles(user_side)
            
            if not vacant_roles:
                print("  [DEBUG] No vacant roles")
                return []  # No recommendations if draft complete
            
            print(f"  [DEBUG] Vacant roles: {vacant_roles}")
            
            # Get all candidates
            candidates = self._get_candidates(draft_state, vacant_roles)
            
            if not candidates:
                print("  [DEBUG] No candidates found")
                return []  # No available champions
            
            print(f"  [DEBUG] Found {len(candidates)} candidates")
            
            # Score each candidate
            recommendations = []
            for i, (champion_id, champion_name, role) in enumerate(candidates[:20]):  # Limit to first 20 for speed
                try:
                    rec = self._score_candidate(
                        draft_state, champion_id, champion_name, role, user_side, model_name
                    )
                    recommendations.append(rec)
                    
                    if i < 3:  # Show first 3 for debugging
                        print(f"  [DEBUG] Scored {champion_name} ({role}): {rec.total_score:.3f}")
                        
                except Exception as e:
                    print(f"  [DEBUG] Failed to score {champion_name}: {e}")
                    continue
            
            if not recommendations:
                print("  [DEBUG] No recommendations after scoring")
                return []
            
            # Sort by total score (descending)
            recommendations.sort(key=lambda x: x.total_score, reverse=True)
            
            # Add ranks and return top 5
            for i, rec in enumerate(recommendations[:TOP_N_RECOMMENDATIONS]):
                rec.rank = i + 1
            
            print(f"  [DEBUG] Returning {len(recommendations[:TOP_N_RECOMMENDATIONS])} recommendations")
            return recommendations[:TOP_N_RECOMMENDATIONS]
            
        except Exception as e:
            print(f"  [ERROR] get_recommendations failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_candidates(self,
                       draft_state: DraftState,
                       vacant_roles: List[str]) -> List[Tuple[int, str, str]]:
        """
        Get all valid candidate champions for vacant roles
        
        Args:
            draft_state: Current draft state
            vacant_roles: List of vacant roles
        
        Returns:
            List of (champion_id, champion_name, role) tuples
        """
        candidates = []
        unavailable_ids = draft_state.get_unavailable_ids()
        
        # Get all champion IDs
        all_champion_ids = self.champion_db.get_all_champion_ids()
        
        for champion_id in all_champion_ids:
            # Skip if banned or picked
            if champion_id in unavailable_ids:
                continue
            
            # Get champion name
            champion_name = self.champion_db.get_name(champion_id)
            if not champion_name:
                continue
            
            # Get roles for this champion
            role_data_list = self.champion_db.get_roles(champion_name)

            MIN_PICK_RATE_THRESHOLD = 0.01  # 0.1% threshold

            for role_data in role_data_list:
                role = role_data['role']
                if role in vacant_roles:
                    # Filter: เฉพาะ pick_rate > threshold
                    if role_data['pick_rate'] >= MIN_PICK_RATE_THRESHOLD:
                        candidates.append((champion_id, champion_name, role))
        
        return candidates
        
        return candidates
    
    def _score_candidate(self,
                        draft_state: DraftState,
                        champion_id: int,
                        champion_name: str,
                        role: str,
                        user_side: str,
                        model_name: str) -> Recommendation:
        """
        Calculate all scores for a candidate champion
        
        Args:
            draft_state: Current draft state
            champion_id: Candidate champion ID
            champion_name: Candidate champion name
            role: Role for this pick
            user_side: User's side
            model_name: ML model to use
        
        Returns:
            Recommendation object with all scores
        """
        # 1. Win Probability
        win_prob = self.predictor.predict_with_champion(
            draft_state, champion_id, champion_name, role, user_side, model_name
        )
        
        # 2. Synergy Score
        synergy_data = self._calculate_synergy(
            draft_state, champion_id, role, user_side
        )
        
        # 3. Counter Score
        counter_data = self._calculate_counter(
            draft_state, champion_id, role, user_side
        )
        
        # 4. Meta Score
        meta_data = self._calculate_meta(champion_id, role)
        
        # 5. Total Score (weighted sum)
        total_score = (
            win_prob * self.weights['win_prob'] +
            synergy_data['score'] * self.weights['synergy'] +
            counter_data['score'] * self.weights['counter'] +
            meta_data['score'] * self.weights['meta']
        )
        
        # Create recommendation
        return Recommendation(
            champion_id=champion_id,
            champion_name=champion_name,
            role=role,
            rank=0,  # Will be set later
            total_score=total_score,
            win_probability=win_prob,
            synergy_score=synergy_data['score'],
            counter_score=counter_data['score'],
            meta_score=meta_data['score'],
            synergy_pairs=synergy_data['pairs'],
            counter_matchups=counter_data['matchups'],
            meta_stats=meta_data['stats']
        )
    
    def _calculate_synergy(self,
                          draft_state: DraftState,
                          champion_id: int,
                          role: str,
                          user_side: str) -> Dict:
        """Calculate synergy score and pairs"""
        # Get current team picks
        team_picks = draft_state.blue_picks if user_side == 'blue' else draft_state.red_picks
        
        if not team_picks:
            # No teammates yet
            return {
                'score': 0.5,  # Neutral
                'pairs': []
            }
        
        # Build team with candidate
        team_ids = [pick.champion_id for pick in team_picks] + [champion_id]
        team_roles = [pick.role for pick in team_picks] + [role]
        
        # Calculate synergies
        synergy_result = self.synergy_calc.calculate_team_synergy(team_ids, team_roles)
        
        # Get synergy with each teammate
        pairs = []
        for pick in team_picks:
            syn_score = self.synergy_calc.get_pair_synergy(
                champion_id, role, pick.champion_id, pick.role
            )
            pairs.append((pick.champion_name, syn_score))
        
        return {
            'score': synergy_result['avg_synergy'],
            'pairs': pairs
        }
    
    def _calculate_counter(self,
                          draft_state: DraftState,
                          champion_id: int,
                          role: str,
                          user_side: str) -> Dict:
        """Calculate counter score and matchups"""
        # Get enemy team picks
        enemy_picks = draft_state.red_picks if user_side == 'blue' else draft_state.blue_picks
        
        if not enemy_picks:
            # No enemies picked yet
            return {
                'score': 0.5,  # Neutral
                'matchups': []
            }
        
        # Get matchups against each enemy
        matchups = []
        total_wr = 0.0
        
        for enemy_pick in enemy_picks:
            win_rate = self.matchup_calc.get_matchup_score(
                champion_id, role, enemy_pick.champion_id, enemy_pick.role
            )
            matchups.append((enemy_pick.champion_name, win_rate))
            total_wr += win_rate
        
        # Average win rate as counter score
        avg_wr = total_wr / len(enemy_picks) if enemy_picks else 0.5
        
        return {
            'score': avg_wr,
            'matchups': matchups
        }
    
    def _calculate_meta(self, champion_id: int, role: str) -> Dict:
        """Calculate meta score and stats"""
        meta = self.meta_analyzer.get_champion_meta(champion_id, role)
        
        # Meta score = average of pick_rate and win_rate
        # (high pick rate + high win rate = strong in meta)
        meta_score = (meta['pick_rate'] + meta['win_rate']) / 2.0
        
        return {
            'score': meta_score,
            'stats': {
                'pick_rate': meta['pick_rate'],
                'ban_rate': meta['ban_rate'],
                'win_rate': meta['win_rate']
            }
        }


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("CHAMPION RECOMMENDER DEMO")
    print("=" * 60)
    
    try:
        from src.core.draft_engine import DraftEngine
        from src.features.feature_extractor import FeatureExtractor
        from src.ai.model_manager import ModelManager
        
        # Initialize all components
        print("\nInitializing components...")
        champion_db = ChampionDatabase()
        champion_db.load(verbose=False)  # Load database
        model_manager = ModelManager()
        feature_extractor = FeatureExtractor()
        predictor = WinProbabilityPredictor(model_manager, feature_extractor)
        synergy_calc = SynergyCalculator()
        matchup_calc = MatchupCalculator()
        meta_analyzer = MetaAnalyzer()
        
        recommender = ChampionRecommender(
            predictor, synergy_calc, matchup_calc, meta_analyzer, champion_db
        )
        print("✓ Recommender ready")
        
        # Create sample draft
        print("\n📊 Creating sample draft...")
        engine = DraftEngine(user_side='blue')
        
        # Execute bans
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
        
        print(f"  Current state: {len(engine.state.blue_picks)}/5 blue, "
              f"{len(engine.state.red_picks)}/5 red")
        print(f"  Vacant roles (blue): {engine.state.get_vacant_roles('blue')}")
        
        # Get recommendations
        print("\n🏆 Getting top 5 recommendations...")
        recommendations = recommender.get_recommendations(
            engine.state, user_side='blue', model_name='xgboost'  # Use XGBoost
        )
        
        print(f"\n{'='*60}")
        print("TOP 5 RECOMMENDATIONS")
        print('='*60)
        
        for rec in recommendations:
            print(f"\n#{rec.rank}: {rec.champion_name} ({rec.role})")
            print(f"  Total Score: {rec.total_score:.3f}")
            print(f"  ├─ Win Probability: {rec.win_probability:.3f} ({rec.win_probability*100:.1f}%)")
            print(f"  ├─ Synergy Score: {rec.synergy_score:.3f}")
            print(f"  ├─ Counter Score: {rec.counter_score:.3f}")
            print(f"  └─ Meta Score: {rec.meta_score:.3f}")
            
            if rec.synergy_pairs:
                print(f"  Synergies:")
                for champ, score in rec.synergy_pairs:
                    print(f"    • {champ}: {score:.3f}")
            
            if rec.counter_matchups:
                print(f"  Matchups:")
                for enemy, wr in rec.counter_matchups:
                    print(f"    • vs {enemy}: {wr:.3f} ({wr*100:.1f}%)")
            
            print(f"  Meta Stats:")
            print(f"    Pick Rate: {rec.meta_stats['pick_rate']:.3f}")
            print(f"    Ban Rate: {rec.meta_stats['ban_rate']:.3f}")
            print(f"    Win Rate: {rec.meta_stats['win_rate']:.3f}")
        
        print("\n✅ Demo complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
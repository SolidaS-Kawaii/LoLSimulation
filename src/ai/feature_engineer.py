"""
Feature Engineer
Converts draft state into ML model features

This is the CRITICAL component that must produce exactly the right features
matching the training data format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.draft_engine import DraftState, Pick
from src.core.champion_db import ChampionDatabase
from src.utils.data_utils import load_synergy_data, load_matchup_data, apply_bayesian_smoothing
import config


class FeatureEngineer:
    """
    Converts draft state into feature vector for ML model
    
    Must produce features matching training data format
    """
    
    def __init__(self, champion_db: ChampionDatabase, verbose: bool = True):
        """
        Initialize feature engineer
        
        Args:
            champion_db: Champion database instance
            verbose: Whether to print loading information
        """
        self.champion_db = champion_db
        
        # Load synergy and matchup data
        if verbose:
            print("📊 Loading synergy and matchup data...")
        
        self.synergy_df = load_synergy_data()
        self.matchup_df = load_matchup_data()
        
        # Create lookup dictionaries for fast access
        self._build_lookups()
        
        if verbose:
            print(f"✓ Feature Engineer ready")
            print(f"  Synergy pairs: {len(self.synergy_lookup):,}")
            print(f"  Matchup pairs: {len(self.matchup_lookup):,}")
    
    def _build_lookups(self):
        """Build fast lookup dictionaries for synergy and matchup data"""
        
        # Synergy lookup: (champ_A_id, role_A, champ_B_id, role_B) -> stats
        self.synergy_lookup = {}
        
        for _, row in self.synergy_df.iterrows():
            key = (int(row['Champion_ID_A']), str(row['Role_A']), 
                   int(row['Champion_ID_B']), str(row['Role_B']))
            
            # Apply Bayesian smoothing
            adjusted_wr = apply_bayesian_smoothing(
                int(row['Win_Count']), 
                int(row['Pick_Count'])
            )
            
            self.synergy_lookup[key] = {
                'win_rate': float(row['Win_Rate']),
                'adjusted_win_rate': adjusted_wr,
                'pick_count': int(row['Pick_Count'])
            }
        
        # Matchup lookup: same structure
        self.matchup_lookup = {}
        
        for _, row in self.matchup_df.iterrows():
            key = (int(row['Champion_ID_A']), str(row['Role_A']), 
                   int(row['Champion_ID_B']), str(row['Role_B']))
            
            adjusted_wr = apply_bayesian_smoothing(
                int(row['Win_Count']), 
                int(row['Pick_Count'])
            )
            
            self.matchup_lookup[key] = {
                'win_rate': float(row['Win_Rate']),
                'adjusted_win_rate': adjusted_wr,
                'pick_count': int(row['Pick_Count'])
            }
    
    def get_synergy_score(self, champ_A_id: int, role_A: str, 
                          champ_B_id: int, role_B: str) -> float:
        """
        Get synergy score between two champions
        
        Returns:
            Adjusted win rate (Bayesian smoothed) or 0.5 if not found
        """
        key = (champ_A_id, role_A, champ_B_id, role_B)
        
        if key in self.synergy_lookup:
            data = self.synergy_lookup[key]
            # Only use if sufficient games
            if data['pick_count'] >= config.MIN_GAMES_THRESHOLD:
                return data['adjusted_win_rate']
        
        # Default to neutral
        return 0.5
    
    def get_matchup_score(self, champ_A_id: int, role_A: str, 
                          champ_B_id: int, role_B: str) -> float:
        """
        Get matchup score (win rate of A vs B)
        
        Returns:
            Adjusted win rate or 0.5 if not found
        """
        key = (champ_A_id, role_A, champ_B_id, role_B)
        
        if key in self.matchup_lookup:
            data = self.matchup_lookup[key]
            if data['pick_count'] >= config.MIN_GAMES_THRESHOLD:
                return data['adjusted_win_rate']
        
        return 0.5
    
    def create_features(self, draft_state: DraftState, 
                       candidate_champion_id: Optional[int] = None,
                       candidate_side: Optional[str] = None,
                       candidate_role: Optional[str] = None) -> Dict[str, float]:
        """
        Create complete feature vector from draft state
        
        Args:
            draft_state: Current draft state
            candidate_champion_id: Optional champion to evaluate
            candidate_side: Which side to add candidate to
            candidate_role: Role for candidate champion
            
        Returns:
            Dictionary of features {feature_name: value}
        """
        features = {}
        
        # Make a copy of picks if evaluating a candidate
        blue_picks = draft_state.blue_picks.copy()
        red_picks = draft_state.red_picks.copy()
        
        if candidate_champion_id and candidate_side and candidate_role:
            candidate_name = self.champion_db.get_name(candidate_champion_id)
            if candidate_name:
                candidate_pick = Pick(candidate_champion_id, candidate_name, candidate_role)
                
                if candidate_side == 'blue':
                    blue_picks.append(candidate_pick)
                else:
                    red_picks.append(candidate_pick)
        
        # 1. BASIC FEATURES
        features.update(self._create_basic_features(
            blue_picks, red_picks,
            draft_state.blue_bans, draft_state.red_bans
        ))
        
        # 2. SYNERGY FEATURES
        features.update(self._create_synergy_features(blue_picks, red_picks))
        
        # 3. MATCHUP FEATURES
        features.update(self._create_matchup_features(blue_picks, red_picks))
        
        # 4. META FEATURES
        features.update(self._create_meta_features(blue_picks, red_picks))
        
        # 5. STRATEGIC FEATURES
        features.update(self._create_strategic_features(
            blue_picks, red_picks,
            draft_state.blue_bans, draft_state.red_bans
        ))
        
        return features
    
    def _create_basic_features(self, blue_picks: List[Pick], red_picks: List[Pick],
                              blue_bans: List[int], red_bans: List[int]) -> Dict:
        """Create basic pick/ban features (30 features)"""
        features = {}
        
        # Role encoding
        role_encoding = config.ROLE_ENCODING
        
        # Team 1 (Blue) picks and roles
        for i in range(5):
            pick_key = f'team1_pick{i+1}'
            role_key = f'team1_role{i+1}'
            
            if i < len(blue_picks):
                features[pick_key] = blue_picks[i].champion_id
                features[role_key] = role_encoding.get(blue_picks[i].role, 0)
            else:
                features[pick_key] = config.MISSING_VALUE_FILL
                features[role_key] = config.MISSING_VALUE_FILL
        
        # Team 1 (Blue) bans
        for i in range(5):
            ban_key = f'team1_ban{i+1}'
            if i < len(blue_bans):
                features[ban_key] = blue_bans[i]
            else:
                features[ban_key] = config.MISSING_VALUE_FILL
        
        # Team 2 (Red) picks and roles
        for i in range(5):
            pick_key = f'team2_pick{i+1}'
            role_key = f'team2_role{i+1}'
            
            if i < len(red_picks):
                features[pick_key] = red_picks[i].champion_id
                features[role_key] = role_encoding.get(red_picks[i].role, 0)
            else:
                features[pick_key] = config.MISSING_VALUE_FILL
                features[role_key] = config.MISSING_VALUE_FILL
        
        # Team 2 (Red) bans
        for i in range(5):
            ban_key = f'team2_ban{i+1}'
            if i < len(red_bans):
                features[ban_key] = red_bans[i]
            else:
                features[ban_key] = config.MISSING_VALUE_FILL
        
        return features
    
    def _create_synergy_features(self, blue_picks: List[Pick], 
                                red_picks: List[Pick]) -> Dict:
        """Create team synergy features (10 features)"""
        features = {}
        
        for team_num, picks in [(1, blue_picks), (2, red_picks)]:
            synergy_scores = []
            
            # Calculate pairwise synergies
            for i in range(len(picks)):
                for j in range(i + 1, len(picks)):
                    pick_A = picks[i]
                    pick_B = picks[j]
                    
                    synergy = self.get_synergy_score(
                        pick_A.champion_id, pick_A.role,
                        pick_B.champion_id, pick_B.role
                    )
                    synergy_scores.append(synergy)
            
            # Calculate statistics
            if synergy_scores:
                features[f'team{team_num}_avg_synergy'] = np.mean(synergy_scores)
                features[f'team{team_num}_max_synergy'] = np.max(synergy_scores)
                features[f'team{team_num}_min_synergy'] = np.min(synergy_scores)
                features[f'team{team_num}_synergy_variance'] = np.var(synergy_scores)
                features[f'team{team_num}_top_synergy_count'] = sum(1 for s in synergy_scores if s > 0.55)
            else:
                features[f'team{team_num}_avg_synergy'] = 0.5
                features[f'team{team_num}_max_synergy'] = 0.5
                features[f'team{team_num}_min_synergy'] = 0.5
                features[f'team{team_num}_synergy_variance'] = 0.0
                features[f'team{team_num}_top_synergy_count'] = 0
        
        return features
    
    def _create_matchup_features(self, blue_picks: List[Pick], 
                                red_picks: List[Pick]) -> Dict:
        """Create matchup features between teams (10 features)"""
        features = {}
        
        # Create role-indexed picks
        blue_by_role = {pick.role: pick for pick in blue_picks}
        red_by_role = {pick.role: pick for pick in red_picks}
        
        role_mapping = {
            'TOP': 'top_matchup',
            'JUNGLE': 'jungle_matchup',
            'MIDDLE': 'mid_matchup',
            'BOTTOM': 'bot_matchup',
            'UTILITY': 'support_matchup'
        }
        
        matchup_scores = []
        
        # Calculate role matchups
        for role, feature_name in role_mapping.items():
            if role in blue_by_role and role in red_by_role:
                blue_pick = blue_by_role[role]
                red_pick = red_by_role[role]
                
                matchup_score = self.get_matchup_score(
                    blue_pick.champion_id, blue_pick.role,
                    red_pick.champion_id, red_pick.role
                )
                
                features[feature_name] = matchup_score
                matchup_scores.append(matchup_score)
            else:
                features[feature_name] = 0.5
        
        # Team matchup aggregates
        if matchup_scores:
            avg_matchup = np.mean(matchup_scores)
            features['team1_avg_matchup'] = avg_matchup
            features['team1_favorable_matchups'] = sum(1 for s in matchup_scores if s > 0.52)
            features['team2_avg_matchup'] = 1.0 - avg_matchup
            features['team2_favorable_matchups'] = sum(1 for s in matchup_scores if s < 0.48)
        else:
            features['team1_avg_matchup'] = 0.5
            features['team1_favorable_matchups'] = 0
            features['team2_avg_matchup'] = 0.5
            features['team2_favorable_matchups'] = 0
        
        return features
    
    def _create_meta_features(self, blue_picks: List[Pick], 
                             red_picks: List[Pick]) -> Dict:
        """Create meta features (champion strength) (13 features)"""
        features = {}
        
        for team_num, picks in [(1, blue_picks), (2, red_picks)]:
            winrates = []
            pickrates = []
            banrates = []
            top_tier_count = 0
            
            for pick in picks:
                stats = self.champion_db.get_champion_stats(pick.champion_name, pick.role)
                
                if stats:
                    winrates.append(stats['win_rate'])
                    pickrates.append(stats['pick_rate'])
                    banrates.append(stats['ban_rate'])
                    
                    if stats['win_rate'] > 0.52:
                        top_tier_count += 1
            
            # Calculate averages
            features[f'team{team_num}_avg_winrate'] = np.mean(winrates) if winrates else 0.5
            features[f'team{team_num}_avg_pickrate'] = np.mean(pickrates) if pickrates else 0.0
            features[f'team{team_num}_avg_banrate'] = np.mean(banrates) if banrates else 0.0
            features[f'team{team_num}_top_tier_count'] = top_tier_count
            
            # Composite meta score
            meta_score = (
                features[f'team{team_num}_avg_winrate'] * 0.5 +
                features[f'team{team_num}_avg_pickrate'] * 100 * 0.3 +
                features[f'team{team_num}_avg_banrate'] * 100 * 0.2
            )
            features[f'team{team_num}_meta_score'] = meta_score
        
        # Meta advantage
        features['meta_advantage'] = features['team1_meta_score'] - features['team2_meta_score']
        
        return features
    
    def _create_strategic_features(self, blue_picks: List[Pick], red_picks: List[Pick],
                                  blue_bans: List[int], red_bans: List[int]) -> Dict:
        """Create strategic features (composition, diversity, etc.)"""
        features = {}
        
        # Basic composition features
        for team_num, picks in [(1, blue_picks), (2, red_picks)]:
            # Role diversity
            unique_roles = len(set(pick.role for pick in picks))
            features[f'team{team_num}_role_diversity'] = unique_roles
            
            # Champion count
            features[f'team{team_num}_champion_count'] = len(picks)
        
        # Strategic advantages
        features['synergy_advantage'] = (
            features.get('team1_avg_synergy', 0.5) - 
            features.get('team2_avg_synergy', 0.5)
        )
        
        features['matchup_advantage'] = features.get('team1_avg_matchup', 0.5) - 0.5
        
        # Overall strength differential
        team1_strength = (
            features.get('team1_meta_score', 25) * 0.4 +
            features.get('team1_avg_synergy', 0.5) * 40 +
            features.get('team1_avg_matchup', 0.5) * 20
        )
        
        team2_strength = (
            features.get('team2_meta_score', 25) * 0.4 +
            features.get('team2_avg_synergy', 0.5) * 40 +
            (1.0 - features.get('team1_avg_matchup', 0.5)) * 20
        )
        
        features['team1_overall_strength'] = team1_strength
        features['team2_overall_strength'] = team2_strength
        features['strength_differential'] = team1_strength - team2_strength
        
        return features
    
    def features_to_dataframe(self, features: Dict) -> pd.DataFrame:
        """Convert feature dictionary to DataFrame for model input"""
        return pd.DataFrame([features])
    
    def features_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        sorted_features = sorted(features.items())
        return np.array([v for k, v in sorted_features]).reshape(1, -1)
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        # Create dummy features to count
        dummy_state = DraftState()
        dummy_features = self.create_features(dummy_state)
        return len(dummy_features)


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test feature engineer"""
    
    print("="*70)
    print("FEATURE ENGINEER - TESTING")
    print("="*70)
    
    from src.core.draft_engine import DraftEngine
    
    # Initialize
    print("\n1. Initializing components...")
    print("-"*70)
    
    champion_db = ChampionDatabase()
    champion_db.load(verbose=False)
    
    feature_engineer = FeatureEngineer(champion_db, verbose=True)
    
    # Test 2: Create sample draft
    print("\n2. Creating sample draft...")
    print("-"*70)
    
    engine = DraftEngine(user_side='blue')
    
    # Add some bans
    ban_names = ["Yasuo", "Zed", "Orianna", "Ahri", "Syndra", 
                 "LeBlanc", "Katarina", "Fizz", "Talon", "Kassadin"]
    
    for name in ban_names:
        champ_id = champion_db.get_id(name)
        if champ_id:
            engine.execute_ban(champ_id)
    
    # Add some picks
    pick_names = [
        ("Jinx", "BOTTOM"),
        ("Kai'Sa", "BOTTOM"),
        ("Thresh", "UTILITY"),
    ]
    
    for name, role in pick_names:
        champ_id = champion_db.get_id(name)
        if champ_id:
            engine.execute_pick(champ_id, name, role)
    
    print(f"Draft state:")
    print(f"  Blue: {[p.champion_name for p in engine.state.blue_picks]}")
    print(f"  Red: {[p.champion_name for p in engine.state.red_picks]}")
    
    # Test 3: Generate features
    print("\n3. Generating features...")
    print("-"*70)
    
    features = feature_engineer.create_features(engine.state)
    
    print(f"Total features: {len(features)}")
    print(f"\nSample features:")
    for i, (key, value) in enumerate(list(features.items())[:15]):
        print(f"  {key:<30} {value}")
    
    # Test 4: Evaluate candidate
    print("\n4. Evaluating candidate champion...")
    print("-"*70)
    
    orianna_id = champion_db.get_id("Orianna")
    if orianna_id:
        # But Orianna is banned! Try different champion
        annie_id = champion_db.get_id("Annie")
        
        if annie_id:
            features_with_candidate = feature_engineer.create_features(
                engine.state,
                candidate_champion_id=annie_id,
                candidate_side='blue',
                candidate_role='MIDDLE'
            )
            
            print(f"Features with Annie added:")
            print(f"  team1_pick2: {features_with_candidate.get('team1_pick2', 'N/A')}")
            print(f"  team1_role2: {features_with_candidate.get('team1_role2', 'N/A')}")
            print(f"  team1_avg_synergy: {features_with_candidate.get('team1_avg_synergy', 0):.3f}")
    
    # Test 5: Feature types
    print("\n5. Testing feature types...")
    print("-"*70)
    
    feature_types = {
        'basic': [k for k in features.keys() if 'pick' in k or 'ban' in k or 'role' in k],
        'synergy': [k for k in features.keys() if 'synergy' in k],
        'matchup': [k for k in features.keys() if 'matchup' in k],
        'meta': [k for k in features.keys() if 'meta' in k or 'winrate' in k or 'pickrate' in k or 'banrate' in k],
        'strategic': [k for k in features.keys() if 'strength' in k or 'advantage' in k or 'diversity' in k]
    }
    
    for ftype, flist in feature_types.items():
        print(f"  {ftype.capitalize()}: {len(flist)} features")
    
    print("\n" + "="*70)
    print("✓ Feature engineer testing complete")
    print("="*70)

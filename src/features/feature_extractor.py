"""
Feature Extractor for League of Legends Draft System
Main orchestrator that combines all feature calculations
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.draft_engine import DraftState, Phase
from src.features.meta_analyzer import MetaAnalyzer
from src.features.synergy_calculator import SynergyCalculator
from src.features.matchup_calculator import MatchupCalculator


class FeatureExtractor:
    """
    Extracts 129-dimensional feature vectors from draft states
    
    Feature Groups:
    - Draft State (49): picks, bans, roles, phase indicators
    - Meta Features (15): pick/ban/win rates
    - Synergy Features (40): team synergies
    - Matchup Features (25): counter advantages
    """
    
    # Feature dimensions
    N_DRAFT_STATE = 49
    N_META = 15
    N_SYNERGY = 40
    N_MATCHUP = 25
    TOTAL_FEATURES = 129
    
    def __init__(self):
        """Initialize feature extractor with all calculators"""
        print("Initializing Feature Extractor...")
        
        self.meta_analyzer = MetaAnalyzer()
        self.synergy_calculator = SynergyCalculator()
        self.matchup_calculator = MatchupCalculator()
        
        print(f"✓ Feature Extractor ready ({self.TOTAL_FEATURES} features)")
    
    def extract_features(self, state: DraftState) -> np.ndarray:
        """
        Extract complete 129-dimensional feature vector from draft state
        
        Args:
            state: Current draft state
        
        Returns:
            NumPy array of shape (129,)
        """
        # Extract each feature group
        draft_features = self._extract_draft_state_features(state)
        meta_features = self._extract_meta_features(state)
        synergy_features = self._extract_synergy_features(state)
        matchup_features = self._extract_matchup_features(state)
        
        # Concatenate all features
        features = np.concatenate([
            draft_features,
            meta_features,
            synergy_features,
            matchup_features
        ])
        
        assert len(features) == self.TOTAL_FEATURES, \
            f"Expected {self.TOTAL_FEATURES} features, got {len(features)}"
        
        return features
    
    def _extract_draft_state_features(self, state: DraftState) -> np.ndarray:
        """
        Extract 49 draft state features
        
        Features:
        - Team 1 picks (5): champion IDs, 0 if not picked
        - Team 2 picks (5): champion IDs, 0 if not picked
        - Team 1 bans (5): champion IDs, 0 if not banned
        - Team 2 bans (5): champion IDs, 0 if not banned
        - Team 1 roles (5): encoded as 0-4 (TOP=0, JUNGLE=1, etc.)
        - Team 2 roles (5): encoded as 0-4
        - Phase indicators (9): one-hot encoding of current phase
        - Pick counts (4): blue_picks, red_picks, blue_bans, red_bans
        - Turn indicator (1): 0=blue, 1=red
        """
        features = []
        
        # Role encoding mapping
        role_map = {
            'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 
            'BOTTOM': 3, 'UTILITY': 4, '': -1
        }
        
        # Helper: Convert List[Pick] to dict {role: champion_id}
        def picks_to_dict(picks_list):
            return {pick.role: pick.champion_id for pick in picks_list}
        
        # Get picks as dictionaries
        blue_picks_dict = picks_to_dict(state.blue_picks)
        red_picks_dict = picks_to_dict(state.red_picks)
        
        # Team picks (10 features: 5 + 5)
        blue_picks = [blue_picks_dict.get(role, 0) 
                      for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']]
        red_picks = [red_picks_dict.get(role, 0) 
                     for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']]
        features.extend(blue_picks + red_picks)
        
        # Team bans (10 features: 5 + 5)
        blue_bans = state.blue_bans + [0] * (5 - len(state.blue_bans))
        red_bans = state.red_bans + [0] * (5 - len(state.red_bans))
        features.extend(blue_bans[:5] + red_bans[:5])
        
        # Roles encoded (10 features: 5 + 5)
        blue_role_ids = [role_map.get(blue_picks_dict.get(role, ''), -1) if role in blue_picks_dict else -1
                         for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']]
        red_role_ids = [role_map.get(red_picks_dict.get(role, ''), -1) if role in red_picks_dict else -1
                        for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']]
        features.extend(blue_role_ids + red_role_ids)
        
        # Phase indicators (9 features: one-hot)
        phase_vector = [0] * 9
        phase_map = {
            Phase.BAN_ROUND_1: 0,
            Phase.BAN_ROUND_2: 1,
            Phase.PICK: 2,
            Phase.COMPLETE: 8
        }
        if state.phase in phase_map:
            phase_vector[phase_map[state.phase]] = 1
        features.extend(phase_vector)
        
        # Pick/ban counts (4 features)
        features.extend([
            len(state.blue_picks),   # blue pick count
            len(state.red_picks),    # red pick count
            len(state.blue_bans),    # blue ban count
            len(state.red_bans)      # red ban count
        ])
        
        # Turn indicator (1 feature)
        features.append(1 if state.current_side == 'red' else 0)
        
        # Additional state features (5 features) - for flexibility
        features.extend([
            1.0 if state.phase == Phase.BAN_ROUND_1 else 0.0,
            1.0 if state.phase == Phase.BAN_ROUND_2 else 0.0,
            1.0 if state.phase == Phase.PICK else 0.0,
            1.0 if state.phase == Phase.COMPLETE else 0.0,
            len(state.blue_picks) + len(state.red_picks)  # Total picks so far
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_meta_features(self, state: DraftState) -> np.ndarray:
        """
        Extract 15 meta features
        
        Features:
        - Team 1 meta (7): avg/max pick_rate, ban_rate, win_rate, total_appearance
        - Team 2 meta (7): avg/max pick_rate, ban_rate, win_rate, total_appearance
        - Banned champions meta (1): avg_ban_rate
        """
        features = []
        
        # Helper: Convert List[Pick] to (ids, roles) tuples
        def picks_to_lists(picks_list):
            ids = [pick.champion_id for pick in picks_list]
            roles = [pick.role for pick in picks_list]
            # Pad to 5
            while len(ids) < 5:
                ids.append(0)
                roles.append('')
            return ids[:5], roles[:5]
        
        # Get team compositions
        blue_ids, blue_roles = picks_to_lists(state.blue_picks)
        red_ids, red_roles = picks_to_lists(state.red_picks)
        
        # Blue team meta features (7)
        blue_meta = self.meta_analyzer.get_team_meta_features(blue_ids, blue_roles)
        features.extend([
            blue_meta['avg_pick_rate'],
            blue_meta['avg_ban_rate'],
            blue_meta['avg_win_rate'],
            blue_meta['max_pick_rate'],
            blue_meta['max_ban_rate'],
            blue_meta['max_win_rate'],
            blue_meta['total_appearance']
        ])
        
        # Red team meta features (7)
        red_meta = self.meta_analyzer.get_team_meta_features(red_ids, red_roles)
        features.extend([
            red_meta['avg_pick_rate'],
            red_meta['avg_ban_rate'],
            red_meta['avg_win_rate'],
            red_meta['max_pick_rate'],
            red_meta['max_ban_rate'],
            red_meta['max_win_rate'],
            red_meta['total_appearance']
        ])
        
        # Banned champions meta (1)
        all_bans = state.blue_bans + state.red_bans
        ban_meta = self.meta_analyzer.get_banned_champions_meta(all_bans)
        features.append(ban_meta['avg_ban_rate'])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_synergy_features(self, state: DraftState) -> np.ndarray:
        """
        Extract 40 synergy features
        
        Features:
        - Team 1 pairwise synergies (10): all C(5,2) pair combinations
        - Team 2 pairwise synergies (10): all C(5,2) pair combinations
        - Team 1 role synergies (5): each role's avg synergy with team
        - Team 2 role synergies (5): each role's avg synergy with team
        - Team 1 aggregates (4): avg, min, max, std synergy
        - Team 2 aggregates (4): avg, min, max, std synergy
        - Comparison (2): synergy_diff, synergy_balance
        """
        features = []
        
        # Helper: Convert List[Pick] to (ids, roles) tuples
        def picks_to_lists(picks_list):
            ids = [pick.champion_id for pick in picks_list]
            roles = [pick.role for pick in picks_list]
            # Pad to 5
            while len(ids) < 5:
                ids.append(0)
                roles.append('')
            return ids[:5], roles[:5]
        
        # Get team compositions
        blue_ids, blue_roles = picks_to_lists(state.blue_picks)
        red_ids, red_roles = picks_to_lists(state.red_picks)
        
        # Blue team synergies
        blue_syn = self.synergy_calculator.calculate_team_synergy(blue_ids, blue_roles)
        pairwise1 = blue_syn['pairwise_synergies'] + [0.5] * (10 - len(blue_syn['pairwise_synergies']))
        features.extend(pairwise1[:10])
        
        # Red team synergies
        red_syn = self.synergy_calculator.calculate_team_synergy(red_ids, red_roles)
        pairwise2 = red_syn['pairwise_synergies'] + [0.5] * (10 - len(red_syn['pairwise_synergies']))
        features.extend(pairwise2[:10])
        
        # Role synergies
        blue_role_syn = self.synergy_calculator.calculate_role_synergies(blue_ids, blue_roles)
        features.extend([blue_role_syn[role] for role in 
                        ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']])
        
        red_role_syn = self.synergy_calculator.calculate_role_synergies(red_ids, red_roles)
        features.extend([red_role_syn[role] for role in 
                        ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']])
        
        # Team aggregates
        features.extend([
            blue_syn['avg_synergy'],
            blue_syn['min_synergy'],
            blue_syn['max_synergy'],
            blue_syn['std_synergy']
        ])
        
        features.extend([
            red_syn['avg_synergy'],
            red_syn['min_synergy'],
            red_syn['max_synergy'],
            red_syn['std_synergy']
        ])
        
        # Comparison metrics
        features.extend([
            blue_syn['avg_synergy'] - red_syn['avg_synergy'],  # diff
            abs(blue_syn['std_synergy'] - red_syn['std_synergy'])  # balance
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_matchup_features(self, state: DraftState) -> np.ndarray:
        """
        Extract 25 matchup features
        
        Features:
        - Role matchups (5): advantage per role (TOP, JG, MID, BOT, SUP)
        - Team 1 matchup stats (5): favorable, unfavorable, overall, max, min
        - Team 2 matchup stats (5): favorable, unfavorable, overall, max, min
        - Cross-team metrics (5): differential, counter_score, std, etc.
        - Additional features (5): to reach 25 total
        """
        features = []
        
        # Helper: Convert List[Pick] to (ids, roles) tuples
        def picks_to_lists(picks_list):
            ids = [pick.champion_id for pick in picks_list]
            roles = [pick.role for pick in picks_list]
            # Pad to 5
            while len(ids) < 5:
                ids.append(0)
                roles.append('')
            return ids[:5], roles[:5]
        
        # Get team compositions
        blue_ids, blue_roles = picks_to_lists(state.blue_picks)
        red_ids, red_roles = picks_to_lists(state.red_picks)
        
        # Role-by-role matchups (5)
        role_matchups = self.matchup_calculator.calculate_role_matchups(
            blue_ids, blue_roles, red_ids, red_roles
        )
        features.extend([role_matchups[role] for role in 
                        ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']])
        
        # Team matchup analysis (8)
        team_matchups = self.matchup_calculator.calculate_team_matchups(
            blue_ids, blue_roles, red_ids, red_roles
        )
        features.extend([
            team_matchups['overall_matchup_score'],
            team_matchups['favorable_matchups'],
            team_matchups['unfavorable_matchups'],
            team_matchups['neutral_matchups'],
            team_matchups['avg_matchup_advantage'],
            team_matchups['max_advantage'],
            team_matchups['min_advantage'],
            team_matchups['std_matchup']
        ])
        
        # Additional matchup metrics (12) - fill to reach 25
        # Normalized counts
        total_matchups = max(1, (team_matchups['favorable_matchups'] + 
                               team_matchups['unfavorable_matchups'] + 
                               team_matchups['neutral_matchups']))
        features.extend([
            team_matchups['favorable_matchups'] / total_matchups,
            team_matchups['unfavorable_matchups'] / total_matchups,
            team_matchups['neutral_matchups'] / total_matchups
        ])
        
        # Win probability estimate from matchups
        matchup_win_prob = 0.5 + team_matchups['overall_matchup_score']
        features.append(matchup_win_prob)
        
        # Role dominance (how many roles have advantage)
        role_advantages = sum(1 for score in role_matchups.values() if score > 0.05)
        role_disadvantages = sum(1 for score in role_matchups.values() if score < -0.05)
        features.extend([role_advantages, role_disadvantages])
        
        # Strongest and weakest roles
        role_scores = list(role_matchups.values())
        features.extend([
            max(role_scores) if role_scores else 0.0,
            min(role_scores) if role_scores else 0.0
        ])
        
        # Matchup consistency (inverse of std)
        consistency = 1.0 / (1.0 + team_matchups['std_matchup'])
        features.append(consistency)
        
        # Fill remaining to reach 25
        remaining = 25 - len(features)
        features.extend([0.0] * remaining)
        
        return np.array(features[:25], dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names for reference"""
        names = []
        
        # Draft state features (49)
        for team in ['team1', 'team2']:
            for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
                names.append(f'{team}_pick_{role}')
        
        for team in ['team1', 'team2']:
            for i in range(5):
                names.append(f'{team}_ban_{i}')
        
        for team in ['team1', 'team2']:
            for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
                names.append(f'{team}_role_{role}')
        
        for i in range(9):
            names.append(f'phase_indicator_{i}')
        
        names.extend(['team1_pick_count', 'team2_pick_count', 
                     'team1_ban_count', 'team2_ban_count', 'turn_indicator'])
        
        # Meta features (15)
        for team in ['team1', 'team2']:
            names.extend([f'{team}_avg_pick_rate', f'{team}_avg_ban_rate',
                         f'{team}_avg_win_rate', f'{team}_max_pick_rate',
                         f'{team}_max_ban_rate', f'{team}_max_win_rate',
                         f'{team}_total_appearance'])
        names.append('banned_avg_ban_rate')
        
        # Synergy features (40)
        for team in ['team1', 'team2']:
            for i in range(10):
                names.append(f'{team}_synergy_pair_{i}')
        
        for team in ['team1', 'team2']:
            for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
                names.append(f'{team}_role_synergy_{role}')
        
        for team in ['team1', 'team2']:
            names.extend([f'{team}_avg_synergy', f'{team}_min_synergy',
                         f'{team}_max_synergy', f'{team}_std_synergy'])
        
        names.extend(['synergy_diff', 'synergy_balance'])
        
        # Matchup features (25)
        for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
            names.append(f'role_matchup_{role}')
        
        names.extend(['overall_matchup', 'favorable_count', 'unfavorable_count',
                     'neutral_count', 'avg_advantage', 'max_advantage',
                     'min_advantage', 'std_matchup'])
        
        # Fill remaining
        remaining = 25 - (len(names) - 49 - 15 - 40)
        for i in range(remaining):
            names.append(f'matchup_extra_{i}')
        
        return names


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE EXTRACTOR DEMO")
    print("=" * 60)
    
    try:
        from src.core.draft_engine import DraftEngine
        
        # Initialize extractor
        extractor = FeatureExtractor()
        
        # Create a sample draft state
        print("\n🎮 Creating sample draft...")
        engine = DraftEngine(user_side='blue')
        
        # Simulate some bans and picks
        bans = [157, 238, 61, 18, 11, 201]
        for ban_id in bans:
            engine.execute_ban(ban_id)
        
        picks = [
            (222, "Jinx", "BOTTOM"),
            (498, "Kai'Sa", "BOTTOM"),
            (412, "Thresh", "UTILITY")
        ]
        for champ_id, name, role in picks:
            engine.execute_pick(champ_id, name, role)
        
        print(f"  Bans: {len(engine.state.team1_bans) + len(engine.state.team2_bans)}")
        print(f"  Picks: {engine.state.get_total_picks()}")
        
        # Extract features
        print("\n🔬 Extracting features...")
        features = extractor.extract_features(engine.state)
        
        print(f"  Feature vector shape: {features.shape}")
        print(f"  Feature vector dtype: {features.dtype}")
        print(f"  Min value: {features.min():.3f}")
        print(f"  Max value: {features.max():.3f}")
        print(f"  Mean value: {features.mean():.3f}")
        
        # Show sample features
        print("\n📊 Sample features (first 10):")
        feature_names = extractor.get_feature_names()
        for i in range(10):
            print(f"  {feature_names[i]}: {features[i]:.3f}")
        
        print("\n✅ Feature extraction successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
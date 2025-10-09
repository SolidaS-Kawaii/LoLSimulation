"""
Synergy Calculator for League of Legends Draft System
Calculates team synergy scores using Bayesian-weighted win rates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_DIR


class SynergyCalculator:
    """Calculates champion synergy scores using weighted win rates"""
    
    # Bayesian smoothing parameter (can be adjusted)
    DEFAULT_WEIGHT = 100  # Equivalent to 100 games of neutral data
    NEUTRAL_WIN_RATE = 0.5
    
    def __init__(self):
        """Initialize synergy calculator and load synergy data"""
        self.synergy_df = None
        self.synergy_lookup = {}  # {(champ_a, role_a, champ_b, role_b): score}
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load champion synergy data from CSV"""
        synergy_path = DATA_DIR / 'champion_synergy_data.csv'
        
        if not synergy_path.exists():
            raise FileNotFoundError(
                f"Synergy data file not found: {synergy_path}\n"
                "Please ensure champion_synergy_data.csv is in the data/ directory"
            )
        
        self.synergy_df = pd.read_csv(synergy_path)
        print(f"✓ Loaded {len(self.synergy_df)} synergy combinations")
    
    def _preprocess_data(self):
        """Preprocess synergy data with Bayesian smoothing"""
        # Strip whitespace
        self.synergy_df.columns = self.synergy_df.columns.str.strip()
        
        # Calculate Bayesian-smoothed synergy scores
        # Formula: (win_rate * pick_count + neutral_rate * weight) / (pick_count + weight)
        self.synergy_df['synergy_score'] = (
            (self.synergy_df['Win_Rate'] * self.synergy_df['Pick_Count'] + 
             self.NEUTRAL_WIN_RATE * self.DEFAULT_WEIGHT) /
            (self.synergy_df['Pick_Count'] + self.DEFAULT_WEIGHT)
        )
        
        # Create lookup dictionary (bidirectional)
        for _, row in self.synergy_df.iterrows():
            champ_a = int(row['Champion_ID_A'])
            role_a = row['Role_A'].strip()
            champ_b = int(row['Champion_ID_B'])
            role_b = row['Role_B'].strip()
            score = float(row['synergy_score'])
            
            # Store both directions
            self.synergy_lookup[(champ_a, role_a, champ_b, role_b)] = score
            self.synergy_lookup[(champ_b, role_b, champ_a, role_a)] = score
        
        print(f"✓ Preprocessed synergy data with Bayesian smoothing")
        print(f"  Weight parameter: {self.DEFAULT_WEIGHT}")
        print(f"  Unique pairs: {len(self.synergy_lookup) // 2}")
    
    def get_pair_synergy(self, 
                        champ_a: int, role_a: str,
                        champ_b: int, role_b: str) -> float:
        """
        Get synergy score for a champion pair
        
        Args:
            champ_a: First champion ID
            role_a: First champion role
            champ_b: Second champion ID
            role_b: Second champion role
        
        Returns:
            Synergy score (0.0-1.0), default 0.5 if not found
        """
        key = (champ_a, role_a.upper(), champ_b, role_b.upper())
        return self.synergy_lookup.get(key, self.NEUTRAL_WIN_RATE)
    
    def calculate_team_synergy(self, 
                               champion_ids: List[int],
                               roles: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive synergy features for a team
        
        Args:
            champion_ids: List of champion IDs (can contain 0 for unpicked)
            roles: List of corresponding roles
        
        Returns:
            Dictionary with synergy features:
            - pairwise_synergies: List of all pair synergies
            - avg_synergy: Average synergy score
            - min_synergy: Minimum synergy (weakest link)
            - max_synergy: Maximum synergy (best combo)
            - std_synergy: Standard deviation
            - synergy_count: Number of valid pairs
        """
        # Filter valid champions
        valid_pairs_data = [(cid, role) for cid, role in zip(champion_ids, roles) 
                            if cid > 0]
        
        if len(valid_pairs_data) < 2:
            # Not enough champions for synergy calculation
            return {
                'pairwise_synergies': [],
                'avg_synergy': self.NEUTRAL_WIN_RATE,
                'min_synergy': self.NEUTRAL_WIN_RATE,
                'max_synergy': self.NEUTRAL_WIN_RATE,
                'std_synergy': 0.0,
                'synergy_count': 0
            }
        
        # Calculate all pairwise synergies
        synergies = []
        for (champ_a, role_a), (champ_b, role_b) in combinations(valid_pairs_data, 2):
            synergy_score = self.get_pair_synergy(champ_a, role_a, champ_b, role_b)
            synergies.append(synergy_score)
        
        return {
            'pairwise_synergies': synergies,
            'avg_synergy': np.mean(synergies),
            'min_synergy': np.min(synergies),
            'max_synergy': np.max(synergies),
            'std_synergy': np.std(synergies),
            'synergy_count': len(synergies)
        }
    
    def calculate_role_synergies(self,
                                 champion_ids: List[int],
                                 roles: List[str]) -> Dict[str, float]:
        """
        Calculate synergy scores for each role with rest of team
        
        Args:
            champion_ids: List of champion IDs
            roles: List of corresponding roles
        
        Returns:
            Dictionary mapping role to average synergy with other champions
        """
        valid_pairs = [(cid, role) for cid, role in zip(champion_ids, roles) 
                       if cid > 0]
        
        if len(valid_pairs) < 2:
            return {role: self.NEUTRAL_WIN_RATE for role in 
                   ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']}
        
        role_synergies = {}
        
        for i, (champ_i, role_i) in enumerate(valid_pairs):
            synergies = []
            for j, (champ_j, role_j) in enumerate(valid_pairs):
                if i != j:
                    synergy = self.get_pair_synergy(champ_i, role_i, 
                                                   champ_j, role_j)
                    synergies.append(synergy)
            
            if synergies:
                role_synergies[role_i] = np.mean(synergies)
            else:
                role_synergies[role_i] = self.NEUTRAL_WIN_RATE
        
        # Fill missing roles with neutral
        for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
            if role not in role_synergies:
                role_synergies[role] = self.NEUTRAL_WIN_RATE
        
        return role_synergies
    
    def get_top_synergies(self, 
                         champion_id: int, 
                         role: str,
                         n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find champions with highest synergy with given champion
        
        Args:
            champion_id: Target champion ID
            role: Target champion role
            n: Number of top synergies to return
        
        Returns:
            List of (champion_id, role, synergy_score) tuples
        """
        synergies = []
        
        for (champ_a, role_a, champ_b, role_b), score in self.synergy_lookup.items():
            if champ_a == champion_id and role_a == role.upper():
                synergies.append((champ_b, role_b, score))
        
        # Sort by score descending and return top n
        synergies.sort(key=lambda x: x[2], reverse=True)
        return synergies[:n]
    
    def compare_team_synergies(self,
                              team1_ids: List[int], team1_roles: List[str],
                              team2_ids: List[int], team2_roles: List[str]) -> Dict[str, float]:
        """
        Compare synergy scores between two teams
        
        Returns:
            Dictionary with comparison metrics
        """
        team1_syn = self.calculate_team_synergy(team1_ids, team1_roles)
        team2_syn = self.calculate_team_synergy(team2_ids, team2_roles)
        
        return {
            'team1_avg': team1_syn['avg_synergy'],
            'team2_avg': team2_syn['avg_synergy'],
            'synergy_diff': team1_syn['avg_synergy'] - team2_syn['avg_synergy'],
            'team1_min': team1_syn['min_synergy'],
            'team2_min': team2_syn['min_synergy'],
            'team1_max': team1_syn['max_synergy'],
            'team2_max': team2_syn['max_synergy'],
        }


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("SYNERGY CALCULATOR DEMO")
    print("=" * 60)
    
    try:
        calculator = SynergyCalculator()
        
        # Test 1: Pair synergy
        print("\n🤝 Test 1: Champion pair synergy")
        # Yasuo + Malphite (famous combo)
        synergy = calculator.get_pair_synergy(157, "MIDDLE", 54, "TOP")
        print(f"  Yasuo (MID) + Malphite (TOP): {synergy:.3f}")
        
        # Test 2: Team synergy
        print("\n🤝 Test 2: Team synergy analysis")
        team_ids = [54, 64, 157, 222, 412]  # Malphite, Lee Sin, Yasuo, Jinx, Thresh
        team_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team_syn = calculator.calculate_team_synergy(team_ids, team_roles)
        print(f"  Team composition:")
        for cid, role in zip(team_ids, team_roles):
            print(f"    Champion {cid} - {role}")
        print(f"  Synergy metrics:")
        print(f"    Average: {team_syn['avg_synergy']:.3f}")
        print(f"    Min: {team_syn['min_synergy']:.3f}")
        print(f"    Max: {team_syn['max_synergy']:.3f}")
        print(f"    Std Dev: {team_syn['std_synergy']:.3f}")
        print(f"    Pairs analyzed: {team_syn['synergy_count']}")
        
        # Test 3: Role synergies
        print("\n🤝 Test 3: Role-specific synergies")
        role_syn = calculator.calculate_role_synergies(team_ids, team_roles)
        print(f"  Synergy by role:")
        for role, score in role_syn.items():
            print(f"    {role}: {score:.3f}")
        
        # Test 4: Top synergies for a champion
        print("\n🤝 Test 4: Best synergy partners")
        top_syn = calculator.get_top_synergies(157, "MIDDLE", n=5)
        print(f"  Top 5 synergies with Yasuo (MIDDLE):")
        for champ_id, role, score in top_syn:
            print(f"    Champion {champ_id} ({role}): {score:.3f}")
        
        # Test 5: Team comparison
        print("\n🤝 Test 5: Team synergy comparison")
        team2_ids = [86, 11, 238, 498, 53]  # Garen, Yi, Zed, Kai'Sa, Blitzcrank
        team2_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        comparison = calculator.compare_team_synergies(
            team_ids, team_roles,
            team2_ids, team2_roles
        )
        print(f"  Team 1 avg synergy: {comparison['team1_avg']:.3f}")
        print(f"  Team 2 avg synergy: {comparison['team2_avg']:.3f}")
        print(f"  Difference: {comparison['synergy_diff']:+.3f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

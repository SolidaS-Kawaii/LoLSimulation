"""
Matchup Calculator for League of Legends Draft System
Calculates champion counter/matchup advantages
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_DIR


class MatchupCalculator:
    """Calculates champion matchup and counter scores"""
    
    NEUTRAL_WIN_RATE = 0.5
    
    def __init__(self):
        """Initialize matchup calculator and load matchup data"""
        self.matchup_df = None
        self.matchup_lookup = {}  # {(champ_a, role_a, champ_b, role_b): win_rate}
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load champion matchup data from CSV"""
        matchup_path = DATA_DIR / 'champion_matchup_data.csv'
        
        if not matchup_path.exists():
            raise FileNotFoundError(
                f"Matchup data file not found: {matchup_path}\n"
                "Please ensure champion_matchup_data.csv is in the data/ directory"
            )
        
        self.matchup_df = pd.read_csv(matchup_path)
        print(f"✓ Loaded {len(self.matchup_df)} matchup combinations")
    
    def _preprocess_data(self):
        """Preprocess matchup data"""
        # Strip whitespace
        self.matchup_df.columns = self.matchup_df.columns.str.strip()
        
        # Create lookup dictionary
        # Note: A vs B means A's win rate against B
        for _, row in self.matchup_df.iterrows():
            champ_a = int(row['Champion_ID_A'])
            role_a = row['Role_A'].strip()
            champ_b = int(row['Champion_ID_B'])
            role_b = row['Role_B'].strip()
            win_rate = float(row['Win_Rate'])
            
            # Store A's win rate vs B
            self.matchup_lookup[(champ_a, role_a, champ_b, role_b)] = win_rate
            # B's win rate vs A is the complement
            self.matchup_lookup[(champ_b, role_b, champ_a, role_a)] = 1.0 - win_rate
        
        print(f"✓ Preprocessed matchup data")
        print(f"  Total matchup pairs: {len(self.matchup_lookup)}")
    
    def get_matchup_score(self,
                         champ_a: int, role_a: str,
                         champ_b: int, role_b: str) -> float:
        """
        Get win rate of champion A vs champion B
        
        Args:
            champ_a: First champion ID
            role_a: First champion role
            champ_b: Second champion ID (opponent)
            role_b: Second champion role
        
        Returns:
            Win rate (0.0-1.0), default 0.5 if not found
        """
        key = (champ_a, role_a.upper(), champ_b, role_b.upper())
        return self.matchup_lookup.get(key, self.NEUTRAL_WIN_RATE)
    
    def calculate_role_matchups(self,
                               team1_ids: List[int], team1_roles: List[str],
                               team2_ids: List[int], team2_roles: List[str]) -> Dict[str, float]:
        """
        Calculate matchup scores for each role (lane matchup)
        
        Args:
            team1_ids: Team 1 champion IDs
            team1_roles: Team 1 roles
            team2_ids: Team 2 champion IDs
            team2_roles: Team 2 roles
        
        Returns:
            Dictionary with role matchup scores (Team 1's perspective)
            Positive score = Team 1 advantage, Negative = Team 2 advantage
        """
        role_matchups = {}
        
        # Standard roles
        roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        
        for role in roles:
            # Find champions in this role
            team1_champ = None
            team2_champ = None
            
            for cid, r in zip(team1_ids, team1_roles):
                if r == role and cid > 0:
                    team1_champ = cid
                    break
            
            for cid, r in zip(team2_ids, team2_roles):
                if r == role and cid > 0:
                    team2_champ = cid
                    break
            
            # Calculate matchup if both champions exist
            if team1_champ and team2_champ:
                win_rate = self.get_matchup_score(team1_champ, role, 
                                                  team2_champ, role)
                # Convert to advantage score (-0.5 to +0.5)
                role_matchups[role] = win_rate - 0.5
            else:
                role_matchups[role] = 0.0  # Neutral if either missing
        
        return role_matchups
    
    def calculate_team_matchups(self,
                               team1_ids: List[int], team1_roles: List[str],
                               team2_ids: List[int], team2_roles: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive matchup features between two teams
        
        Args:
            team1_ids: Team 1 champion IDs
            team1_roles: Team 1 roles
            team2_ids: Team 2 champion IDs
            team2_roles: Team 2 roles
        
        Returns:
            Dictionary with matchup metrics
        """
        # Get valid champions
        team1_valid = [(cid, role) for cid, role in zip(team1_ids, team1_roles) 
                       if cid > 0]
        team2_valid = [(cid, role) for cid, role in zip(team2_ids, team2_roles) 
                       if cid > 0]
        
        if not team1_valid or not team2_valid:
            return {
                'overall_matchup_score': 0.0,
                'favorable_matchups': 0,
                'unfavorable_matchups': 0,
                'neutral_matchups': 0,
                'avg_matchup_advantage': 0.0,
                'max_advantage': 0.0,
                'min_advantage': 0.0,
                'std_matchup': 0.0
            }
        
        # Calculate all cross-team matchups
        matchup_scores = []
        favorable = 0
        unfavorable = 0
        neutral = 0
        
        for champ_a, role_a in team1_valid:
            for champ_b, role_b in team2_valid:
                win_rate = self.get_matchup_score(champ_a, role_a, champ_b, role_b)
                advantage = win_rate - 0.5
                matchup_scores.append(advantage)
                
                if advantage > 0.05:  # 55%+ win rate
                    favorable += 1
                elif advantage < -0.05:  # <45% win rate
                    unfavorable += 1
                else:
                    neutral += 1
        
        return {
            'overall_matchup_score': np.mean(matchup_scores),
            'favorable_matchups': favorable,
            'unfavorable_matchups': unfavorable,
            'neutral_matchups': neutral,
            'avg_matchup_advantage': np.mean(matchup_scores),
            'max_advantage': np.max(matchup_scores),
            'min_advantage': np.min(matchup_scores),
            'std_matchup': np.std(matchup_scores)
        }
    
    def get_counter_score(self,
                         team1_ids: List[int], team1_roles: List[str],
                         team2_ids: List[int], team2_roles: List[str]) -> float:
        """
        Calculate overall counter advantage for Team 1 vs Team 2
        
        Returns:
            Counter score (-0.5 to +0.5)
            Positive = Team 1 counters Team 2
            Negative = Team 2 counters Team 1
        """
        matchups = self.calculate_team_matchups(team1_ids, team1_roles,
                                               team2_ids, team2_roles)
        return matchups['overall_matchup_score']
    
    def get_best_counters(self,
                         champion_id: int,
                         role: str,
                         n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find champions that counter the given champion
        
        Args:
            champion_id: Target champion ID
            role: Target champion role
            n: Number of counters to return
        
        Returns:
            List of (champion_id, role, win_rate) tuples
        """
        counters = []
        
        # Find all matchups where opponent has high win rate vs target
        for (champ_a, role_a, champ_b, role_b), win_rate in self.matchup_lookup.items():
            if champ_b == champion_id and role_b == role.upper():
                # champ_a's win rate vs our target
                counters.append((champ_a, role_a, win_rate))
        
        # Sort by win rate descending (best counters)
        counters.sort(key=lambda x: x[2], reverse=True)
        return counters[:n]
    
    def get_favorable_matchups(self,
                              champion_id: int,
                              role: str,
                              n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find champions that the given champion counters
        
        Args:
            champion_id: Target champion ID
            role: Target champion role
            n: Number of favorable matchups to return
        
        Returns:
            List of (opponent_id, role, win_rate) tuples
        """
        favorable = []
        
        for (champ_a, role_a, champ_b, role_b), win_rate in self.matchup_lookup.items():
            if champ_a == champion_id and role_a == role.upper():
                favorable.append((champ_b, role_b, win_rate))
        
        # Sort by win rate descending
        favorable.sort(key=lambda x: x[2], reverse=True)
        return favorable[:n]


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("MATCHUP CALCULATOR DEMO")
    print("=" * 60)
    
    try:
        calculator = MatchupCalculator()
        
        # Test 1: Single matchup
        print("\n⚔️ Test 1: Champion matchup")
        # Yasuo vs Malphite
        matchup = calculator.get_matchup_score(157, "MIDDLE", 54, "TOP")
        print(f"  Yasuo (MID) vs Malphite (TOP): {matchup:.3f}")
        print(f"  Advantage: {(matchup - 0.5)*100:+.1f}%")
        
        # Test 2: Role matchups
        print("\n⚔️ Test 2: Role-by-role matchups")
        team1_ids = [86, 64, 157, 222, 412]  # Garen, Lee Sin, Yasuo, Jinx, Thresh
        team1_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team2_ids = [54, 11, 238, 498, 53]   # Malphite, Yi, Zed, Kai'Sa, Blitz
        team2_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        
        role_matchups = calculator.calculate_role_matchups(
            team1_ids, team1_roles, team2_ids, team2_roles
        )
        print(f"  Team 1 vs Team 2 (Team 1 perspective):")
        for role, score in role_matchups.items():
            advantage = "Advantage" if score > 0 else "Disadvantage" if score < 0 else "Neutral"
            print(f"    {role}: {score:+.3f} ({advantage})")
        
        # Test 3: Team matchup analysis
        print("\n⚔️ Test 3: Full team matchup analysis")
        team_matchups = calculator.calculate_team_matchups(
            team1_ids, team1_roles, team2_ids, team2_roles
        )
        print(f"  Overall matchup score: {team_matchups['overall_matchup_score']:+.3f}")
        print(f"  Favorable matchups: {team_matchups['favorable_matchups']}")
        print(f"  Unfavorable matchups: {team_matchups['unfavorable_matchups']}")
        print(f"  Neutral matchups: {team_matchups['neutral_matchups']}")
        
        # Test 4: Best counters
        print("\n⚔️ Test 4: Champions that counter Yasuo")
        counters = calculator.get_best_counters(157, "MIDDLE", n=5)
        print(f"  Top 5 counters to Yasuo (MIDDLE):")
        for champ_id, role, win_rate in counters:
            print(f"    Champion {champ_id} ({role}): {win_rate:.3f} ({(win_rate-0.5)*100:+.1f}%)")
        
        # Test 5: Favorable matchups
        print("\n⚔️ Test 5: Champions that Yasuo counters")
        favorable = calculator.get_favorable_matchups(157, "MIDDLE", n=5)
        print(f"  Top 5 favorable matchups for Yasuo (MIDDLE):")
        for champ_id, role, win_rate in favorable:
            print(f"    Champion {champ_id} ({role}): {win_rate:.3f} ({(win_rate-0.5)*100:+.1f}%)")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
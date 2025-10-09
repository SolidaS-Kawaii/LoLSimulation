"""
Meta Features Analyzer for League of Legends Draft System
Analyzes champion popularity and performance metrics from API data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_DIR


class MetaAnalyzer:
    """Analyzes champion meta features (pick rate, ban rate, win rate)"""
    
    def __init__(self):
        """Initialize meta analyzer and load champion stats"""
        self.stats_df = None
        self.champion_stats = {}  # {(champion_id, role): {stats}}
        self.global_stats = {}    # Global averages
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load champion stats from CSV"""
        stats_path = DATA_DIR / 'champion_stats_by_role_API.csv'
        
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Champion stats file not found: {stats_path}\n"
                "Please ensure champion_stats_by_role_API.csv is in the data/ directory"
            )
        
        self.stats_df = pd.read_csv(stats_path)
        print(f"✓ Loaded {len(self.stats_df)} champion-role combinations")
    
    def _preprocess_data(self):
        """Preprocess and normalize stats data"""
        # Strip whitespace from column names
        self.stats_df.columns = self.stats_df.columns.str.strip()
        
        # Normalize rates to 0-1 range
        # Pick_Rate, Ban_Rate are already 0-1 (percentages/100)
        # Win_Rate is 0-1 as well
        
        # Store normalized stats in dictionary
        for _, row in self.stats_df.iterrows():
            key = (int(row['Champion_ID']), row['Role'].strip())
            
            self.champion_stats[key] = {
                'pick_rate': float(row['Pick_Rate']),
                'ban_rate': float(row['Ban_Rate']),
                'win_rate': float(row['Win_Rate']),
                'pick_count': int(row['Pick_Count']),
                'ban_count': int(row['Ban_Count']),
                'appearance': float(row['Appearance'])
            }
        
        # Calculate global statistics
        self.global_stats = {
            'avg_pick_rate': self.stats_df['Pick_Rate'].mean(),
            'avg_ban_rate': self.stats_df['Ban_Rate'].mean(),
            'avg_win_rate': self.stats_df['Win_Rate'].mean(),
            'max_pick_rate': self.stats_df['Pick_Rate'].max(),
            'max_ban_rate': self.stats_df['Ban_Rate'].max(),
            'max_win_rate': self.stats_df['Win_Rate'].max(),
        }
        
        print(f"✓ Preprocessed meta data")
        print(f"  Global averages: Pick={self.global_stats['avg_pick_rate']:.3f}, "
              f"Ban={self.global_stats['avg_ban_rate']:.3f}, "
              f"Win={self.global_stats['avg_win_rate']:.3f}")
    
    def get_champion_meta(self, champion_id: int, role: str) -> Dict[str, float]:
        """
        Get normalized meta features for a champion-role combination
        
        Args:
            champion_id: Champion ID
            role: Role (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
        
        Returns:
            Dictionary with normalized meta features, or defaults if not found
        """
        key = (champion_id, role.upper())
        
        if key in self.champion_stats:
            return self.champion_stats[key].copy()
        
        # Return global averages if champion-role not found
        return {
            'pick_rate': self.global_stats['avg_pick_rate'],
            'ban_rate': self.global_stats['avg_ban_rate'],
            'win_rate': 0.5,  # Neutral win rate for unknown combinations
            'pick_count': 0,
            'ban_count': 0,
            'appearance': 0.0
        }
    
    def get_team_meta_features(self, 
                               champion_ids: list, 
                               roles: list) -> Dict[str, float]:
        """
        Calculate aggregated meta features for a team
        
        Args:
            champion_ids: List of champion IDs (can contain 0 for unpicked)
            roles: List of corresponding roles
        
        Returns:
            Dictionary with team meta features
        """
        # Filter out unpicked champions (ID = 0)
        valid_pairs = [(cid, role) for cid, role in zip(champion_ids, roles) 
                       if cid > 0]
        
        if not valid_pairs:
            # No champions picked yet
            return {
                'avg_pick_rate': 0.0,
                'avg_ban_rate': 0.0,
                'avg_win_rate': 0.5,
                'max_pick_rate': 0.0,
                'max_ban_rate': 0.0,
                'max_win_rate': 0.5,
                'total_appearance': 0.0
            }
        
        # Get stats for each champion
        stats_list = [self.get_champion_meta(cid, role) for cid, role in valid_pairs]
        
        # Calculate aggregates
        pick_rates = [s['pick_rate'] for s in stats_list]
        ban_rates = [s['ban_rate'] for s in stats_list]
        win_rates = [s['win_rate'] for s in stats_list]
        appearances = [s['appearance'] for s in stats_list]
        
        return {
            'avg_pick_rate': np.mean(pick_rates),
            'avg_ban_rate': np.mean(ban_rates),
            'avg_win_rate': np.mean(win_rates),
            'max_pick_rate': np.max(pick_rates),
            'max_ban_rate': np.max(ban_rates),
            'max_win_rate': np.max(win_rates),
            'total_appearance': np.sum(appearances)
        }
    
    def get_banned_champions_meta(self, banned_ids: list) -> Dict[str, float]:
        """
        Calculate meta features for banned champions
        
        Args:
            banned_ids: List of banned champion IDs
        
        Returns:
            Dictionary with banned champions meta features
        """
        valid_bans = [bid for bid in banned_ids if bid > 0]
        
        if not valid_bans:
            return {
                'avg_ban_rate': 0.0,
                'avg_pick_rate': 0.0,
                'avg_win_rate': 0.5
            }
        
        # Get best role for each banned champion (highest pick rate)
        ban_stats = []
        for champ_id in valid_bans:
            # Find all roles for this champion
            roles_stats = [(role, stats) for (cid, role), stats 
                          in self.champion_stats.items() if cid == champ_id]
            
            if roles_stats:
                # Pick role with highest pick rate
                best_role, best_stats = max(roles_stats, 
                                           key=lambda x: x[1]['pick_rate'])
                ban_stats.append(best_stats)
        
        if not ban_stats:
            return {
                'avg_ban_rate': 0.0,
                'avg_pick_rate': 0.0,
                'avg_win_rate': 0.5
            }
        
        return {
            'avg_ban_rate': np.mean([s['ban_rate'] for s in ban_stats]),
            'avg_pick_rate': np.mean([s['pick_rate'] for s in ban_stats]),
            'avg_win_rate': np.mean([s['win_rate'] for s in ban_stats])
        }
    
    def get_global_stats(self) -> Dict[str, float]:
        """Get global meta statistics"""
        return self.global_stats.copy()


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("META ANALYZER DEMO")
    print("=" * 60)
    
    try:
        analyzer = MetaAnalyzer()
        
        # Test 1: Single champion meta
        print("\n📊 Test 1: Get champion meta features")
        yasuo_meta = analyzer.get_champion_meta(157, "MIDDLE")
        print(f"  Yasuo (MIDDLE):")
        print(f"    Pick Rate: {yasuo_meta['pick_rate']:.3f}")
        print(f"    Ban Rate: {yasuo_meta['ban_rate']:.3f}")
        print(f"    Win Rate: {yasuo_meta['win_rate']:.3f}")
        
        # Test 2: Team meta features
        print("\n📊 Test 2: Team meta features")
        team_ids = [157, 238, 61, 222, 412]  # Yasuo, Zed, Orianna, Jinx, Thresh
        team_roles = ["MIDDLE", "MIDDLE", "MIDDLE", "BOTTOM", "UTILITY"]
        team_meta = analyzer.get_team_meta_features(team_ids, team_roles)
        print(f"  Team averages:")
        print(f"    Pick Rate: {team_meta['avg_pick_rate']:.3f}")
        print(f"    Ban Rate: {team_meta['avg_ban_rate']:.3f}")
        print(f"    Win Rate: {team_meta['avg_win_rate']:.3f}")
        
        # Test 3: Banned champions meta
        print("\n📊 Test 3: Banned champions meta")
        banned = [157, 238, 61, 18, 11]  # High priority bans
        ban_meta = analyzer.get_banned_champions_meta(banned)
        print(f"  Banned champions averages:")
        print(f"    Ban Rate: {ban_meta['avg_ban_rate']:.3f}")
        print(f"    Pick Rate: {ban_meta['avg_pick_rate']:.3f}")
        print(f"    Win Rate: {ban_meta['avg_win_rate']:.3f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Champion Database
Manages champion data, roles, and lookups
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.data_utils import load_champion_stats
from src.utils.text_utils import find_closest_champion
import config


class ChampionDatabase:
    """
    Manages champion data and provides efficient lookup functions
    
    Features:
    - ID ↔ Name mappings
    - Role information per champion (sorted by pick rate)
    - Champion statistics (win rate, pick rate, ban rate)
    - Fuzzy name matching for user input
    """
    
    def __init__(self):
        """Initialize empty champion database"""
        self.stats_df: Optional[pd.DataFrame] = None
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.champion_roles: Dict[str, List[Dict]] = {}  # name -> [{role, pick_rate, ...}, ...]
        self._loaded = False
    
    def load(self, verbose: bool = True):
        """
        Load champion data from CSV file
        
        Args:
            verbose: Whether to print loading information
            
        Raises:
            FileNotFoundError: If champion stats file not found
            ValueError: If data is invalid
        """
        if verbose:
            print("📚 Loading champion database...")
        
        try:
            # Load data
            self.stats_df = load_champion_stats()
            
            # Build ID mappings
            self._build_id_mappings()
            
            # Build role mappings
            self._build_role_mappings()
            
            self._loaded = True
            
            if verbose:
                print(f"✓ Loaded {len(self.name_to_id)} champions")
                print(f"✓ Total champion-role combinations: {len(self.stats_df)}")
                
                # Show sample multi-role champions
                multi_role = [name for name, roles in self.champion_roles.items() 
                             if len(roles) > 1]
                if multi_role and verbose:
                    print(f"✓ Multi-role champions: {len(multi_role)}")
                    sample = multi_role[:3]
                    for name in sample:
                        roles = [r['role'] for r in self.champion_roles[name]]
                        print(f"  • {name}: {', '.join(roles)}")
        
        except Exception as e:
            raise Exception(f"Failed to load champion database: {e}")
    
    def _build_id_mappings(self):
        """Build champion ID to name mappings"""
        for _, row in self.stats_df.iterrows():
            champ_id = int(row['Champion_ID'])
            champ_name = str(row['Champion_Name'])
            
            # Use first occurrence for ID mapping
            if champ_id not in self.id_to_name:
                self.id_to_name[champ_id] = champ_name
                self.name_to_id[champ_name] = champ_id
    
    def _build_role_mappings(self):
        """Build champion role mappings (sorted by pick rate)"""
        for champ_name in self.name_to_id.keys():
            champ_data = self.stats_df[self.stats_df['Champion_Name'] == champ_name]
            
            roles = []
            for _, row in champ_data.iterrows():
                # Only include roles with actual picks
                if row['Pick_Rate'] > 0 or row['Pick_Count'] > 0:
                    roles.append({
                        'role': row['Role'],
                        'pick_rate': float(row['Pick_Rate']),
                        'win_rate': float(row['Win_Rate']),
                        'pick_count': int(row['Pick_Count']),
                        'ban_rate': float(row['Ban_Rate'])
                    })
            
            # Sort by pick rate descending
            roles.sort(key=lambda x: x['pick_rate'], reverse=True)
            self.champion_roles[champ_name] = roles
    
    def is_loaded(self) -> bool:
        """Check if database is loaded"""
        return self._loaded
    
    def get_name(self, champion_id: int) -> Optional[str]:
        """
        Get champion name from ID
        
        Args:
            champion_id: Riot API champion ID
            
        Returns:
            Champion name or None if not found
        """
        return self.id_to_name.get(champion_id)
    
    def get_id(self, champion_name: str) -> Optional[int]:
        """
        Get champion ID from name
        
        Args:
            champion_name: Champion name (case-sensitive)
            
        Returns:
            Champion ID or None if not found
        """
        return self.name_to_id.get(champion_name)
    
    def get_roles(self, champion_name: str) -> List[Dict]:
        """
        Get all roles for a champion, sorted by pick rate
        
        Args:
            champion_name: Champion name
            
        Returns:
            List of role dictionaries: [{'role': 'MIDDLE', 'pick_rate': 0.05, ...}, ...]
            Empty list if champion not found
        """
        return self.champion_roles.get(champion_name, [])
    
    def get_best_role(self, champion_name: str, vacant_roles: List[str]) -> Optional[str]:
        """
        Get best role for champion given vacant roles
        
        Priority:
        1. First try to match vacant roles (by pick rate descending)
        2. If no vacant role matches, return most played role
        
        Args:
            champion_name: Champion name
            vacant_roles: List of roles not yet filled in team
            
        Returns:
            Best role to assign, or None if champion has no roles
            
        Examples:
            >>> db.get_best_role("Yasuo", ["TOP", "MIDDLE"])
            'MIDDLE'  # Because Yasuo mid has higher pick rate
            
            >>> db.get_best_role("Yasuo", ["BOTTOM", "UTILITY"])
            'MIDDLE'  # No vacant role matches, returns most played
        """
        roles = self.get_roles(champion_name)
        
        if not roles:
            return None
        
        # First try to match vacant roles (by pick rate)
        for role_data in roles:
            if role_data['role'] in vacant_roles:
                return role_data['role']
        
        # If no vacant role matches, return most played role
        return roles[0]['role']
    
    def get_champion_stats(self, champion_name: str, role: str) -> Optional[Dict]:
        """
        Get stats for specific champion-role combination
        
        Args:
            champion_name: Champion name
            role: Role (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
            
        Returns:
            Dictionary with stats or None if not found
            {
                'champion_id': int,
                'champion_name': str,
                'role': str,
                'pick_rate': float,
                'ban_rate': float,
                'win_rate': float,
                'pick_count': int
            }
        """
        data = self.stats_df[
            (self.stats_df['Champion_Name'] == champion_name) & 
            (self.stats_df['Role'] == role)
        ]
        
        if data.empty:
            return None
        
        row = data.iloc[0]
        return {
            'champion_id': int(row['Champion_ID']),
            'champion_name': str(row['Champion_Name']),
            'role': str(row['Role']),
            'pick_rate': float(row['Pick_Rate']),
            'ban_rate': float(row['Ban_Rate']),
            'win_rate': float(row['Win_Rate']),
            'pick_count': int(row['Pick_Count'])
        }
    
    def search_champion(self, query: str, threshold: float = 0.8) -> Optional[str]:
        """
        Search for champion with fuzzy matching
        
        Args:
            query: User input (can be partial or misspelled)
            threshold: Fuzzy match threshold (0-1)
            
        Returns:
            Matched champion name or None
            
        Examples:
            >>> db.search_champion("yas")
            'Yasuo'
            
            >>> db.search_champion("ori")
            'Orianna'
        """
        all_names = list(self.name_to_id.keys())
        return find_closest_champion(query, all_names, threshold)
    
    def get_all_champion_names(self) -> List[str]:
        """
        Get list of all champion names (sorted alphabetically)
        
        Returns:
            List of champion names
        """
        return sorted(self.name_to_id.keys())
    
    def get_all_champion_ids(self) -> List[int]:
        """
        Get list of all champion IDs (sorted)
        
        Returns:
            List of champion IDs
        """
        return sorted(self.id_to_name.keys())
    
    def get_champion_count(self) -> int:
        """Get total number of unique champions"""
        return len(self.name_to_id)
    
    def get_stats_summary(self) -> Dict:
        """
        Get summary statistics about the database
        
        Returns:
            Dictionary with summary info
        """
        if not self._loaded:
            return {'error': 'Database not loaded'}
        
        multi_role = [name for name, roles in self.champion_roles.items() 
                     if len(roles) > 1]
        
        return {
            'total_champions': self.get_champion_count(),
            'total_combinations': len(self.stats_df),
            'multi_role_champions': len(multi_role),
            'roles': config.ROLES
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test champion database"""
    
    print("="*70)
    print("CHAMPION DATABASE - TESTING")
    print("="*70)
    
    # Initialize and load
    db = ChampionDatabase()
    db.load(verbose=True)
    
    print("\n" + "-"*70)
    print("1. Testing ID ↔ Name Mappings:")
    print("-"*70)
    
    # Test some known champions
    test_ids = [157, 238, 61, 222]  # Yasuo, Zed, Orianna, Jinx
    
    for champ_id in test_ids:
        name = db.get_name(champ_id)
        if name:
            reverse_id = db.get_id(name)
            print(f"ID {champ_id} → '{name}' → ID {reverse_id} ✓")
        else:
            print(f"ID {champ_id} → Not found ✗")
    
    print("\n" + "-"*70)
    print("2. Testing Role Information:")
    print("-"*70)
    
    test_champions = ["Yasuo", "Orianna", "Lee Sin", "Jinx"]
    
    for champ_name in test_champions:
        roles = db.get_roles(champ_name)
        if roles:
            print(f"\n{champ_name}:")
            for i, role_data in enumerate(roles, 1):
                print(f"  {i}. {role_data['role']:<10} "
                      f"Pick Rate: {role_data['pick_rate']:.2%}, "
                      f"Win Rate: {role_data['win_rate']:.2%}")
        else:
            print(f"{champ_name}: No roles found ✗")
    
    print("\n" + "-"*70)
    print("3. Testing Best Role Assignment:")
    print("-"*70)
    
    test_cases = [
        ("Yasuo", ["TOP", "MIDDLE", "BOTTOM"]),
        ("Orianna", ["JUNGLE", "MIDDLE"]),
        ("Lee Sin", ["TOP", "JUNGLE"]),
        ("Jinx", ["BOTTOM", "UTILITY"])
    ]
    
    for champ_name, vacant_roles in test_cases:
        best_role = db.get_best_role(champ_name, vacant_roles)
        print(f"{champ_name} with vacant {vacant_roles}")
        print(f"  → Best role: {best_role}")
    
    print("\n" + "-"*70)
    print("4. Testing Champion Stats:")
    print("-"*70)
    
    stats_tests = [
        ("Yasuo", "MIDDLE"),
        ("Orianna", "MIDDLE"),
        ("Lee Sin", "JUNGLE")
    ]
    
    for champ_name, role in stats_tests:
        stats = db.get_champion_stats(champ_name, role)
        if stats:
            print(f"\n{champ_name} ({role}):")
            print(f"  Pick Rate: {stats['pick_rate']:.2%}")
            print(f"  Ban Rate:  {stats['ban_rate']:.2%}")
            print(f"  Win Rate:  {stats['win_rate']:.2%}")
            print(f"  Pick Count: {stats['pick_count']:,}")
        else:
            print(f"{champ_name} ({role}): Not found ✗")
    
    print("\n" + "-"*70)
    print("5. Testing Fuzzy Search:")
    print("-"*70)
    
    search_queries = ["yas", "ori", "lee", "tf", "kisa"]
    
    for query in search_queries:
        result = db.search_champion(query, threshold=0.7)
        if result:
            print(f"'{query}' → {result} ✓")
        else:
            print(f"'{query}' → Not found ✗")
    
    print("\n" + "-"*70)
    print("6. Database Summary:")
    print("-"*70)
    
    summary = db.get_stats_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*70)
    print("✓ Champion database testing complete")
    print("="*70)

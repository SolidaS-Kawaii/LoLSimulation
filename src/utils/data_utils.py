"""
Data Loading Utilities
Functions for loading and processing CSV data
"""

import pandas as pd
from typing import Optional
import config


def load_champion_stats() -> pd.DataFrame:
    """
    Load champion statistics from CSV
    
    Returns:
        DataFrame with columns: Champion_ID, Champion_Name, Role, 
                                Pick_Rate, Ban_Rate, Win_Rate, etc.
    
    Raises:
        FileNotFoundError: If champion stats file not found
    """
    try:
        df = pd.read_csv(config.CHAMPION_STATS_FILE)
        # Clean whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Ensure required columns exist
        required_cols = ['Champion_ID', 'Champion_Name', 'Role', 'Pick_Rate', 
                        'Ban_Rate', 'Win_Rate', 'Pick_Count']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Champion stats file not found: {config.CHAMPION_STATS_FILE}\n"
            f"Please ensure the data file exists in the data/ directory"
        )


def load_synergy_data() -> pd.DataFrame:
    """
    Load champion synergy data from CSV
    
    Returns:
        DataFrame with columns: Champion_ID_A, Champion_Name_A, Role_A,
                                Champion_ID_B, Champion_Name_B, Role_B,
                                Pick_Count, Win_Count, Win_Rate
    
    Raises:
        FileNotFoundError: If synergy data file not found
    """
    try:
        df = pd.read_csv(config.SYNERGY_DATA_FILE)
        df.columns = df.columns.str.strip()
        
        required_cols = ['Champion_ID_A', 'Champion_Name_A', 'Role_A',
                        'Champion_ID_B', 'Champion_Name_B', 'Role_B',
                        'Pick_Count', 'Win_Count', 'Win_Rate']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Synergy data file not found: {config.SYNERGY_DATA_FILE}\n"
            f"Please ensure the data file exists in the data/ directory"
        )


def load_matchup_data() -> pd.DataFrame:
    """
    Load champion matchup/counter data from CSV
    
    Returns:
        DataFrame with same structure as synergy data
    
    Raises:
        FileNotFoundError: If matchup data file not found
    """
    try:
        df = pd.read_csv(config.MATCHUP_DATA_FILE)
        df.columns = df.columns.str.strip()
        
        required_cols = ['Champion_ID_A', 'Champion_Name_A', 'Role_A',
                        'Champion_ID_B', 'Champion_Name_B', 'Role_B',
                        'Pick_Count', 'Win_Count', 'Win_Rate']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Matchup data file not found: {config.MATCHUP_DATA_FILE}\n"
            f"Please ensure the data file exists in the data/ directory"
        )


def apply_bayesian_smoothing(wins: int, games: int, 
                            prior_mean: float = config.BAYESIAN_PRIOR_MEAN,
                            prior_strength: int = config.BAYESIAN_PRIOR_STRENGTH) -> float:
    """
    Apply Bayesian smoothing to win rate to reduce impact of small samples
    
    Formula: (wins + prior_mean * prior_strength) / (games + prior_strength)
    
    This helps stabilize win rates from small sample sizes:
    - 1 win in 1 game (100%) becomes ~52%
    - 5 wins in 5 games (100%) becomes ~60%
    - Converges to actual win rate as sample size increases
    
    Args:
        wins: Number of wins
        games: Number of games played
        prior_mean: Prior belief about win rate (default 0.5 = 50%)
        prior_strength: Strength of prior (equivalent games, default 20)
        
    Returns:
        Smoothed win rate (0.0 to 1.0)
    
    Examples:
        >>> apply_bayesian_smoothing(1, 1)  # 1/1 = 100% raw
        0.524  # Smoothed to ~52%
        
        >>> apply_bayesian_smoothing(5, 5)  # 5/5 = 100% raw
        0.600  # Smoothed to 60%
        
        >>> apply_bayesian_smoothing(50, 100)  # 50/100 = 50% raw
        0.500  # Stays at 50% (matches prior)
    """
    if games == 0:
        return prior_mean
    
    smoothed = (wins + prior_mean * prior_strength) / (games + prior_strength)
    return smoothed


def get_data_info() -> dict:
    """
    Get information about loaded data files
    
    Returns:
        Dictionary with data statistics
    """
    info = {}
    
    try:
        champion_df = load_champion_stats()
        info['champion_stats'] = {
            'rows': len(champion_df),
            'unique_champions': champion_df['Champion_ID'].nunique(),
            'file': config.CHAMPION_STATS_FILE
        }
    except Exception as e:
        info['champion_stats'] = {'error': str(e)}
    
    try:
        synergy_df = load_synergy_data()
        info['synergy_data'] = {
            'rows': len(synergy_df),
            'file': config.SYNERGY_DATA_FILE
        }
    except Exception as e:
        info['synergy_data'] = {'error': str(e)}
    
    try:
        matchup_df = load_matchup_data()
        info['matchup_data'] = {
            'rows': len(matchup_df),
            'file': config.MATCHUP_DATA_FILE
        }
    except Exception as e:
        info['matchup_data'] = {'error': str(e)}
    
    return info


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test data loading functions"""
    
    print("="*70)
    print("DATA UTILS - TESTING")
    print("="*70)
    
    # Test Bayesian smoothing
    print("\n1. Testing Bayesian Smoothing:")
    print("-" * 50)
    
    test_cases = [
        (1, 1, "1 win in 1 game (100% raw)"),
        (5, 5, "5 wins in 5 games (100% raw)"),
        (10, 20, "10 wins in 20 games (50% raw)"),
        (30, 50, "30 wins in 50 games (60% raw)"),
        (100, 200, "100 wins in 200 games (50% raw)")
    ]
    
    for wins, games, desc in test_cases:
        raw_wr = wins / games if games > 0 else 0
        smoothed = apply_bayesian_smoothing(wins, games)
        print(f"{desc}")
        print(f"  Raw WR: {raw_wr:.1%} → Smoothed: {smoothed:.1%}")
    
    # Test data loading
    print("\n2. Testing Data Loading:")
    print("-" * 50)
    
    info = get_data_info()
    
    for data_type, data_info in info.items():
        print(f"\n{data_type}:")
        if 'error' in data_info:
            print(f"  ❌ Error: {data_info['error']}")
        else:
            for key, value in data_info.items():
                print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✓ Data utils testing complete")
    print("="*70)

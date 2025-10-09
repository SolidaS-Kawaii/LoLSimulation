"""
Configuration file for LoL Draft Simulator
All constants and settings in one place
"""

from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Data files
CHAMPION_STATS_FILE = DATA_DIR / 'champion_stats_by_role_API.csv'
SYNERGY_DATA_FILE = DATA_DIR / 'champion_synergy_data.csv'
MATCHUP_DATA_FILE = DATA_DIR / 'champion_matchup_data.csv'

# ==================== DRAFT RULES ====================
# Ban Phase: 3-3-2-2 format (Pro play)
# Round 1: B-R-B-R-B-R (3 bans each)
# Round 2: B-R-B-R (2 bans each)
BAN_PHASE_CONFIG = {
    'round1': {'blue': 3, 'red': 3},
    'round2': {'blue': 2, 'red': 2}
}

# Pick Phase: 1-2-2-2-2-1 format
PICK_PHASE_SEQUENCE = [
    'blue',           # Pick 1
    'red', 'red',     # Pick 1, 2
    'blue', 'blue',   # Pick 2, 3
    'red', 'red',     # Pick 3, 4
    'blue', 'blue',   # Pick 4, 5
    'red'             # Pick 5
]

# Roles
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

# Role encoding for features
ROLE_ENCODING = {
    'TOP': 1,
    'JUNGLE': 2,
    'MIDDLE': 3,
    'BOTTOM': 4,
    'UTILITY': 5
}

# ==================== DATA PROCESSING ====================
# Minimum games threshold for synergy/matchup data
MIN_GAMES_THRESHOLD = 20

# Bayesian smoothing parameters
BAYESIAN_PRIOR_MEAN = 0.5      # 50% baseline win rate
BAYESIAN_PRIOR_STRENGTH = 20   # Equivalent to 20 games

# ==================== RECOMMENDATION ====================
# Weighted scoring for recommendations
RECOMMENDATION_WEIGHTS = {
    'win_prob': 0.40,    # 40% weight on win probability
    'synergy': 0.30,     # 30% weight on team synergy
    'counter': 0.20,     # 20% weight on counter matchups
    'meta': 0.10         # 10% weight on meta strength
}

TOP_N_RECOMMENDATIONS = 5

# ==================== UI ====================
# Fuzzy matching threshold for champion names
FUZZY_MATCH_THRESHOLD = 0.8

# Display settings
USE_COLOR = True
MAX_DISPLAY_WIDTH = 70

# ==================== MODEL ====================
# Available models
AVAILABLE_MODELS = {
    'lightgbm': {
        'filename': 'trained_model_lightgbm.pkl',
        'name': 'LightGBM',
        'accuracy': 0.8310
    },
    'xgboost': {
        'filename': 'trained_model_xgboost.pkl',
        'name': 'XGBoost',
        'accuracy': 0.8295
    },
    'random_forest': {
        'filename': 'trained_model_random_forest.pkl',
        'name': 'Random Forest',
        'accuracy': 0.8245
    }
}

DEFAULT_MODEL = 'lightgbm'

# Missing value fill for incomplete drafts
MISSING_VALUE_FILL = 0

# ==================== DEBUGGING ====================
DEBUG = False
VERBOSE = False
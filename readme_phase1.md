# 🎮 LoL Draft Simulator - AI-Powered Recommendation System

## 📋 Phase 1: Core Data Management ✅

### Project Overview
AI-powered champion recommendation system for League of Legends draft phase using machine learning models.

---

## 📁 Project Structure

```
lol_draft_simulator/
│
├── README.md                        # This file
├── config.py                        # ✅ Configuration constants
├── requirements.txt                 # ✅ Python dependencies
├── test_phase1.py                   # ✅ Phase 1 testing script
│
├── data/                           # 📊 Data files (you need to add these)
│   ├── champion_stats_by_role_API.csv
│   ├── champion_synergy_data.csv
│   └── champion_matchup_data.csv
│
├── models/                         # 🤖 Trained ML models (you need to add these)
│   ├── trained_model_lightgbm.pkl
│   ├── trained_model_xgboost.pkl
│   └── trained_model_random_forest.pkl
│
├── outputs/                        # 💾 Output directory (auto-created)
│   ├── draft_history/
│   └── reports/
│
└── src/                           # 📦 Source code
    ├── __init__.py
    │
    ├── utils/                     # ✅ Utility functions
    │   ├── __init__.py
    │   ├── data_utils.py          # ✅ Data loading and Bayesian smoothing
    │   └── text_utils.py          # ✅ Fuzzy matching and text processing
    │
    ├── core/                      # ✅ Core components (Phase 1 COMPLETE)
    │   ├── __init__.py
    │   ├── champion_db.py         # ✅ Champion database management
    │   ├── draft_engine.py        # 🔲 Coming in Phase 2
    │   └── model_manager.py       # 🔲 Coming in Phase 3
    │
    ├── ai/                        # 🔲 AI components (Phase 3-4)
    │   ├── __init__.py
    │   ├── feature_engineer.py    # 🔲 Coming in Phase 3
    │   ├── recommender.py         # 🔲 Coming in Phase 4
    │   └── predictor.py           # 🔲 Coming in Phase 4
    │
    └── ui/                        # 🔲 UI components (Phase 5)
        ├── __init__.py
        ├── display.py             # 🔲 Coming in Phase 5
        └── input_handler.py       # 🔲 Coming in Phase 5
```

---

## 🚀 Quick Start - Phase 1

### 1. Setup Environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Data Files

Place these CSV files in the `data/` directory:
- `champion_stats_by_role_API.csv`
- `champion_synergy_data.csv`
- `champion_matchup_data.csv`

### 3. Run Tests

```bash
# Test Phase 1 components
python test_phase1.py
```

Expected output:
```
✅ PASS Data Loading
✅ PASS Bayesian Smoothing
✅ PASS Text Utilities
✅ PASS Champion Database
✅ PASS Integration

Total: 5/5 tests passed

🎉 ALL TESTS PASSED! Phase 1 complete! 🚀
✓ Ready to proceed to Phase 2: Draft Engine
```

---

## 📚 Phase 1 Components

### ✅ Completed Components

#### 1. **Configuration (`config.py`)**
- All constants and settings
- Draft rules (ban/pick sequences)
- Model configurations
- Paths and file locations

#### 2. **Data Utilities (`src/utils/data_utils.py`)**
- `load_champion_stats()` - Load champion statistics
- `load_synergy_data()` - Load synergy data
- `load_matchup_data()` - Load matchup data
- `apply_bayesian_smoothing()` - Win rate smoothing for small samples

#### 3. **Text Utilities (`src/utils/text_utils.py`)**
- `fuzzy_match()` - Fuzzy string matching
- `find_closest_champion()` - Find best match for user input
- `format_suggestions()` - Generate suggestions for typos
- `clean_champion_name()` - Normalize champion names

#### 4. **Champion Database (`src/core/champion_db.py`)**
- Champion ID ↔ Name mappings
- Role information (sorted by pick rate)
- Champion statistics lookup
- Fuzzy search functionality
- Best role assignment logic

---

## 🔧 Key Features

### Bayesian Smoothing
Handles extreme win rates from small sample sizes:
```python
# 1 win in 1 game (100% raw) → ~52% smoothed
# 5 wins in 5 games (100% raw) → ~60% smoothed
# Converges to actual rate with more games
```

### Fuzzy Matching
Intelligent champion name matching:
```python
"yas" - "Yasuo"
"ori" - "Orianna"
"lee" - "Lee Sin"
"tf"  - "Twisted Fate"
```

### Role Priority
Assigns best role based on:
1. Vacant roles in team (by pick rate)
2. Most played role if no vacant match

---

## 🧪 Testing Individual Components

### Test Data Loading
```bash
python -c "from src.utils.data_utils import *; print(get_data_info())"
```

### Test Champion Database
```bash
python src/core/champion_db.py
```

### Test Text Utils
```bash
python src/utils/text_utils.py
```

---

## 📊 Data Requirements

### Champion Stats CSV Format
Required columns:
- `Champion_ID` (int): Riot API champion ID
- `Champion_Name` (str): Champion name
- `Role` (str): TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
- `Pick_Rate` (float): Pick rate (0-1)
- `Ban_Rate` (float): Ban rate (0-1)
- `Win_Rate` (float): Win rate (0-1)
- `Pick_Count` (int): Number of picks

### Synergy Data CSV Format
Required columns:
- `Champion_ID_A`, `Champion_Name_A`, `Role_A`
- `Champion_ID_B`, `Champion_Name_B`, `Role_B`
- `Pick_Count` (int): Times played together
- `Win_Count` (int): Wins together
- `Win_Rate` (float): Win rate (0-1)

### Matchup Data CSV Format
Same structure as synergy data (represents A vs B matchups)

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Ban phase format
BAN_PHASE_CONFIG = {
    'round1': {'blue': 3, 'red': 3},  # 3-3
    'round2': {'blue': 2, 'red': 2}   # 2-2
}

# Bayesian smoothing
BAYESIAN_PRIOR_MEAN = 0.5      # 50% baseline
BAYESIAN_PRIOR_STRENGTH = 20   # Strength (games)

# Recommendation weights
RECOMMENDATION_WEIGHTS = {
    'win_prob': 0.40,
    'synergy': 0.30,
    'counter': 0.20,
    'meta': 0.10
}
```

---

## 🐛 Troubleshooting

### "File not found" error
Make sure CSV files are in the `data/` directory with correct names.

### Import errors
Ensure you're running from project root directory:
```bash
python test_phase1.py  # ✓ Correct
cd src && python ../test_phase1.py  # ✗ Wrong
```

### "Module not found" error
Install requirements:
```bash
pip install -r requirements.txt
```

---

## 📈 Next Steps

### Phase 2: Draft Engine (Coming Next)
- Draft state management
- Ban/Pick turn progression
- Draft validation rules
- Game flow logic

### Phase 3: Feature Engineering
- Draft state → Feature vector conversion
- 129 features for ML model
- Synergy/matchup calculations

### Phase 4: AI Recommendation
- Win probability prediction
- Champion scoring system
- Recommendation ranking

### Phase 5: Terminal UI
- Interactive draft interface
- Real-time recommendations
- Final analysis display

---

## 📝 Development Notes

### Code Style
- Use type hints for function parameters
- Include docstrings for all functions
- Follow PEP 8 naming conventions

### Testing
- Test each component individually
- Run integration tests after changes
- Use `test_phase1.py` before proceeding to Phase 2

---

## 🎯 Phase 1 Checklist

- [x] Project structure created
- [x] Configuration file (`config.py`)
- [x] Requirements file (`requirements.txt`)
- [x] Data utilities (`data_utils.py`)
- [x] Text utilities (`text_utils.py`)
- [x] Champion database (`champion_db.py`)
- [x] Testing script (`test_phase1.py`)
- [x] Documentation (this README)

**Status: ✅ Phase 1 Complete!**

---

## 📞 Support

If you encounter issues:
1. Check file paths in `config.py`
2. Verify CSV files are present
3. Run `test_phase1.py` for diagnostics
4. Check Python version (requires 3.8+)

---

## 📄 License

Academic Project - For Educational Purposes

---

**Ready to proceed to Phase 2!** 🚀

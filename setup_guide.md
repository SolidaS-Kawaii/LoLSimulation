# 🚀 Complete Setup Guide

Step-by-step instructions to set up the LoL Draft Simulator project.

---

## 📋 Prerequisites

- Python 3.8 or higher
- Git (optional, for version control)
- 2GB+ free disk space (for data and models)

---

## 🛠️ Setup Steps

### **Step 1: Create Project Structure**

Create the following folder structure:

```
lol_draft_simulator/
├── .gitignore              # ← Create this
├── config.py               # ← Create this
├── requirements.txt        # ← Create this
├── test_phase1.py         # ← Create this
├── README.md              # ← Create this
├── SETUP.md               # ← This file
│
├── data/                  # ← Create folder
│   ├── README.md          # ← Create this
│   └── .gitkeep           # ← Create empty file
│
├── models/                # ← Create folder
│   ├── README.md          # ← Create this
│   └── .gitkeep           # ← Create empty file
│
├── outputs/               # ← Create folder
│   ├── draft_history/     # ← Create folder
│   │   └── .gitkeep       # ← Create empty file
│   └── reports/           # ← Create folder
│       └── .gitkeep       # ← Create empty file
│
└── src/                   # ← Create folder
    ├── __init__.py        # ← Create EMPTY file
    │
    ├── utils/             # ← Create folder
    │   ├── __init__.py    # ← Create EMPTY file
    │   ├── data_utils.py  # ← Create this
    │   └── text_utils.py  # ← Create this
    │
    ├── core/              # ← Create folder
    │   ├── __init__.py    # ← Create EMPTY file
    │   └── champion_db.py # ← Create this
    │
    ├── ai/                # ← Create folder (empty for now)
    │   └── __init__.py    # ← Create EMPTY file
    │
    └── ui/                # ← Create folder (empty for now)
        └── __init__.py    # ← Create EMPTY file
```

**Quick Command (Linux/Mac):**
```bash
# Create all directories
mkdir -p lol_draft_simulator/{data,models,outputs/{draft_history,reports},src/{utils,core,ai,ui}}

# Create .gitkeep files
touch lol_draft_simulator/data/.gitkeep
touch lol_draft_simulator/models/.gitkeep
touch lol_draft_simulator/outputs/draft_history/.gitkeep
touch lol_draft_simulator/outputs/reports/.gitkeep

# Create __init__.py files
touch lol_draft_simulator/src/__init__.py
touch lol_draft_simulator/src/{utils,core,ai,ui}/__init__.py
```

**Windows PowerShell:**
```powershell
# Create directories
New-Item -ItemType Directory -Force -Path lol_draft_simulator\data, lol_draft_simulator\models, lol_draft_simulator\outputs\draft_history, lol_draft_simulator\outputs\reports, lol_draft_simulator\src\utils, lol_draft_simulator\src\core, lol_draft_simulator\src\ai, lol_draft_simulator\src\ui

# Create empty files
New-Item -ItemType File -Force -Path lol_draft_simulator\data\.gitkeep, lol_draft_simulator\models\.gitkeep, lol_draft_simulator\outputs\draft_history\.gitkeep, lol_draft_simulator\outputs\reports\.gitkeep, lol_draft_simulator\src\__init__.py, lol_draft_simulator\src\utils\__init__.py, lol_draft_simulator\src\core\__init__.py, lol_draft_simulator\src\ai\__init__.py, lol_draft_simulator\src\ui\__init__.py
```

---

### **Step 2: Copy Code Files**

Copy all the code files from the artifacts:

1. **Root files:**
   - `.gitignore`
   - `config.py`
   - `requirements.txt`
   - `test_phase1.py`
   - `README.md`

2. **Utility files:**
   - `src/utils/data_utils.py`
   - `src/utils/text_utils.py`

3. **Core files:**
   - `src/core/champion_db.py`

4. **Documentation:**
   - `data/README.md`
   - `models/README.md`

---

### **Step 3: Install Dependencies**

```bash
# Navigate to project directory
cd lol_draft_simulator

# (Optional) Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Expected packages:
```
pandas, numpy, scikit-learn
xgboost, lightgbm, joblib
colorama
```

---

### **Step 4: Add Data Files**

1. Place your CSV files in the `data/` folder:
   - `champion_stats_by_role_API.csv`
   - `champion_synergy_data.csv`
   - `champion_matchup_data.csv`

2. Verify files exist:
   ```bash
   ls data/*.csv  # Linux/Mac
   dir data\*.csv  # Windows
   ```

---

### **Step 5: Add Model Files (Phase 3+)**

1. Place trained model files in `models/` folder:
   - `trained_model_lightgbm.pkl`
   - `trained_model_xgboost.pkl`
   - `trained_model_random_forest.pkl`

2. These are needed for Phase 3+ (not required for Phase 1)

---

### **Step 6: Run Tests**

```bash
# Run Phase 1 tests
python test_phase1.py
```

**Expected Output:**
```
🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮
LoL DRAFT SIMULATOR - PHASE 1 TESTING
🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮

Testing Core Data Management components...
...

Results:
  Data Loading             ✅ PASS
  Bayesian Smoothing       ✅ PASS
  Text Utilities           ✅ PASS
  Champion Database        ✅ PASS
  Integration              ✅ PASS

Total: 5/5 tests passed

🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
ALL TESTS PASSED! Phase 1 complete! 🚀
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉

✓ Ready to proceed to Phase 2: Draft Engine
```

---

### **Step 7: Initialize Git (Optional)**

```bash
# Initialize git repository
git init

# Add files
git add .

# First commit
git commit -m "Initial commit - Phase 1 complete"
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
# Make sure you're in the project root
pwd  # Should show .../lol_draft_simulator

# Check __init__.py files exist
ls src/__init__.py
ls src/utils/__init__.py
ls src/core/__init__.py
```

### Problem: "FileNotFoundError: champion_stats_by_role_API.csv"
**Solution:**
```bash
# Check if data files exist
ls data/champion_stats_by_role_API.csv

# If not, copy your CSV files to data/ folder
```

### Problem: "ImportError: cannot import name 'load_champion_stats'"
**Solution:**
```bash
# Verify files are in correct locations
ls src/utils/data_utils.py
ls src/core/champion_db.py

# Check file content is correct (not empty)
```

### Problem: Tests fail with "assertion error"
**Solution:**
```bash
# Run individual component tests
python src/core/champion_db.py
python src/utils/text_utils.py

# Check config.py paths are correct
```

---

## ✅ Verification Checklist

Before proceeding to Phase 2, ensure:

- [ ] All folders created
- [ ] All `__init__.py` files created (can be empty)
- [ ] All `.gitkeep` files created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files (CSV) in `data/` folder
- [ ] `test_phase1.py` passes all tests (5/5)
- [ ] (Optional) Git repository initialized

---

## 📚 Next Steps

After Phase 1 setup is complete:

1. **Phase 2:** Draft Engine (draft state, turn progression)
2. **Phase 3:** Feature Engineering (129 features)
3. **Phase 4:** AI Recommendation (ML predictions)
4. **Phase 5:** Terminal UI (interactive interface)

---

## 📞 Need Help?

If you encounter issues:

1. Check file paths in `config.py`
2. Verify CSV column names match expected format
3. Run `python test_phase1.py -v` for verbose output
4. Check Python version: `python --version` (need 3.8+)

---

**Setup complete! Ready for Phase 2! 🚀**

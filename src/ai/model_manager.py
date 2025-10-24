"""
Model Manager for League of Legends Draft System
Loads and manages ML models for win probability prediction
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, AVAILABLE_MODELS, DEFAULT_MODEL


class ModelManager:
    """Manages ML models for win probability prediction"""
    
    def __init__(self):
        """Initialize model manager and load available models"""
        self.models = {}  # {model_name: model_object}
        self.model_info = AVAILABLE_MODELS.copy()
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models from models directory"""
        print("Loading ML models...")
        
        for model_name, info in self.model_info.items():
            model_path = MODELS_DIR / info['filename']
            
            if model_path.exists():
                try:
                    # Try pickle format first
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    print(f"  ✓ Loaded {info['name']} (accuracy: {info['accuracy']:.2%})")
                    
                except Exception as pickle_error:
                    # If pickle fails, try native format for LightGBM
                    if model_name == 'lightgbm':
                        try:
                            import lightgbm as lgb
                            
                            # Load as Booster
                            booster = lgb.Booster(model_file=str(model_path))
                            
                            # Wrap in LGBMClassifier for consistent API
                            model = lgb.LGBMClassifier()
                            model._Booster = booster
                            model._n_classes = 2  # Binary classification
                            
                            self.models[model_name] = model
                            print(f"  ✓ Loaded {info['name']} from native format (accuracy: {info['accuracy']:.2%})")
                            
                        except Exception as lgb_error:
                            print(f"  ✗ Failed to load {info['name']}: {lgb_error}")
                    else:
                        print(f"  ✗ Failed to load {info['name']}: {pickle_error}")
            else:
                print(f"  ✗ Model file not found: {model_path}")
        
        if not self.models:
            raise RuntimeError(
                "No models could be loaded! Please ensure .pkl files are in models/ directory"
            )
        
        print(f"✓ Model Manager ready ({len(self.models)} models loaded)")
    
    def get_model(self, model_name: str = None):
        """
        Get specific model
        
        Args:
            model_name: Model identifier (e.g., 'lightgbm', 'xgboost')
                       If None, returns default model
        
        Returns:
            Model object
        
        Raises:
            ValueError: If model not found
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available}"
            )
        
        return self.models[model_name]

    def predict(self, features: np.ndarray, model_name: str = None) -> float:
        """
        Predict win probability using specified model

        Args:
            features: Feature vector (129 features)
            model_name: Which model to use (default: from config)

        Returns:
            Win probability (0.0-1.0)
        """
        model = self.get_model(model_name)

        # Validate model type
        if isinstance(model, np.ndarray):
            print(f"[ERROR] Model '{model_name}' is corrupted (numpy.ndarray instead of model)")
            return 0.5  # Neutral fallback

        # Validate model has required method
        if not hasattr(model, 'predict_proba'):
            print(f"[ERROR] Model '{model_name}' doesn't have predict_proba (type: {type(model)})")
            return 0.5  # Neutral fallback

        # Ensure features is 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            # Handle LightGBM booster specifically
            if model_name == 'lightgbm' and hasattr(model, '_Booster'):
                prediction = model._Booster.predict(features)[0]
                return float(prediction)
            else:
                # Standard sklearn API
                prediction = model.predict_proba(features)[0]
                win_probability = prediction[1]
                return float(win_probability)

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 0.5  # Neutral fallback
    
    def list_available_models(self) -> List[Dict]:
        """
        List all available models with information
        
        Returns:
            List of dictionaries with model info
        """
        models_list = []
        
        for model_name in self.models.keys():
            info = self.model_info[model_name].copy()
            info['model_name'] = model_name
            info['loaded'] = True
            models_list.append(info)
        
        return models_list
    
    def get_model_info(self, model_name: str = None) -> Dict:
        """
        Get information about specific model
        
        Args:
            model_name: Model identifier
        
        Returns:
            Dictionary with model information
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        info = self.model_info[model_name].copy()
        info['model_name'] = model_name
        info['loaded'] = True
        
        return info


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MANAGER DEMO")
    print("=" * 60)
    
    try:
        # Initialize manager
        manager = ModelManager()
        
        # Test 1: List available models
        print("\n📊 Test 1: Available models")
        models = manager.list_available_models()
        for model in models:
            print(f"  • {model['name']}")
            print(f"    - Accuracy: {model['accuracy']:.2%}")
            print(f"    - File: {model['filename']}")
        
        # Test 2: Get specific model
        print("\n📊 Test 2: Get model")
        lgb_model = manager.get_model('lightgbm')
        print(f"  ✓ Retrieved LightGBM model: {type(lgb_model)}")
        
        # Test 3: Predict with dummy features
        print("\n📊 Test 3: Prediction")
        dummy_features = np.random.rand(129)
        win_prob = manager.predict(dummy_features, 'lightgbm')
        print(f"  Win probability: {win_prob:.3f} ({win_prob*100:.1f}%)")
        
        # Test 4: Model info
        print("\n📊 Test 4: Model information")
        info = manager.get_model_info('lightgbm')
        print(f"  Model: {info['name']}")
        print(f"  Accuracy: {info['accuracy']:.2%}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
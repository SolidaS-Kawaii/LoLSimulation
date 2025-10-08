"""
Model Manager
Handles loading and managing ML models
"""

import joblib
import os
from typing import Dict, Optional, Any
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config


class ModelManager:
    """
    Manages ML models (loading, switching, predictions)
    
    Features:
    - Load models from .pkl files
    - Switch between different models
    - Get model metadata
    - Make predictions
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.models: Dict[str, Any] = {}
        self.current_model_key: str = config.DEFAULT_MODEL
        self.current_model: Optional[Any] = None
        self._loaded = False
    
    def load_model(self, model_key: str, verbose: bool = True) -> bool:
        """
        Load a specific model from disk
        
        Args:
            model_key: Model identifier (e.g., 'lightgbm', 'xgboost')
            verbose: Whether to print loading information
            
        Returns:
            True if successful, False otherwise
        """
        if model_key not in config.AVAILABLE_MODELS:
            if verbose:
                print(f"❌ Error: Unknown model '{model_key}'")
                print(f"Available models: {list(config.AVAILABLE_MODELS.keys())}")
            return False
        
        model_info = config.AVAILABLE_MODELS[model_key]
        model_path = os.path.join(config.MODELS_DIR, model_info['filename'])
        
        if not os.path.exists(model_path):
            if verbose:
                print(f"❌ Error: Model file not found: {model_path}")
                print(f"Please ensure the model file exists in the models/ directory")
            return False
        
        try:
            if verbose:
                print(f"📦 Loading model: {model_info['name']}...")
            
            model = joblib.load(model_path)
            self.models[model_key] = model
            
            if verbose:
                print(f"✓ Model loaded successfully")
                print(f"  Accuracy: {model_info['accuracy']:.2%}")
            
            return True
        
        except Exception as e:
            if verbose:
                print(f"❌ Error loading model: {e}")
            return False
    
    def load_all_models(self, verbose: bool = True) -> int:
        """
        Load all available models
        
        Args:
            verbose: Whether to print loading information
            
        Returns:
            Number of models successfully loaded
        """
        if verbose:
            print("\n📦 Loading all available models...")
        
        loaded_count = 0
        for model_key in config.AVAILABLE_MODELS.keys():
            if self.load_model(model_key, verbose=False):
                loaded_count += 1
                if verbose:
                    model_info = config.AVAILABLE_MODELS[model_key]
                    print(f"  ✓ {model_info['name']:<20} ({model_info['accuracy']:.2%})")
        
        if verbose:
            print(f"\n✓ Loaded {loaded_count}/{len(config.AVAILABLE_MODELS)} models")
        
        return loaded_count
    
    def set_active_model(self, model_key: str, verbose: bool = True) -> bool:
        """
        Set the active model for predictions
        
        Args:
            model_key: Model identifier
            verbose: Whether to print information
            
        Returns:
            True if successful
        """
        # Load model if not already loaded
        if model_key not in self.models:
            if not self.load_model(model_key, verbose=verbose):
                return False
        
        self.current_model_key = model_key
        self.current_model = self.models[model_key]
        self._loaded = True
        
        if verbose:
            model_info = config.AVAILABLE_MODELS[model_key]
            print(f"✓ Active model: {model_info['name']}")
        
        return True
    
    def get_active_model(self) -> Optional[Any]:
        """
        Get the currently active model
        
        Returns:
            Model object or None if no model loaded
        """
        if self.current_model is None and not self._loaded:
            # Try to load default model
            self.set_active_model(config.DEFAULT_MODEL, verbose=False)
        
        return self.current_model
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """
        Get metadata about a specific model
        
        Args:
            model_key: Model identifier
            
        Returns:
            Dictionary with model information or None
        """
        return config.AVAILABLE_MODELS.get(model_key)
    
    def get_all_model_info(self) -> Dict:
        """
        Get metadata about all available models
        
        Returns:
            Dictionary of all model information
        """
        return config.AVAILABLE_MODELS.copy()
    
    def is_loaded(self, model_key: Optional[str] = None) -> bool:
        """
        Check if a model is loaded
        
        Args:
            model_key: Model identifier (None for current model)
            
        Returns:
            True if loaded, False otherwise
        """
        if model_key is None:
            return self._loaded and self.current_model is not None
        
        return model_key in self.models
    
    def predict(self, features, use_proba: bool = True) -> float:
        """
        Make prediction with current model
        
        Args:
            features: Feature vector (array-like or DataFrame)
            use_proba: Whether to return probability (True) or class (False)
            
        Returns:
            Prediction value
            
        Raises:
            RuntimeError: If no model is loaded
        """
        model = self.get_active_model()
        
        if model is None:
            raise RuntimeError("No model loaded. Call set_active_model() first.")
        
        try:
            if use_proba and hasattr(model, 'predict_proba'):
                # Return probability of class 1 (team1 wins)
                proba = model.predict_proba(features)
                return float(proba[0][1])
            else:
                # Return class prediction
                prediction = model.predict(features)
                return float(prediction[0])
        
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Optional[Dict]:
        """
        Get feature importance from current model (if available)
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            Dictionary with feature importance or None
        """
        model = self.get_active_model()
        
        if model is None:
            return None
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            return None
        
        try:
            importances = model.feature_importances_
            
            # Create dict with feature names (if available)
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                importance_dict = dict(zip(feature_names, importances))
            else:
                # Use indices as names
                importance_dict = {f"feature_{i}": imp 
                                 for i, imp in enumerate(importances)}
            
            # Sort by importance
            sorted_items = sorted(importance_dict.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            
            if top_n:
                sorted_items = sorted_items[:top_n]
            
            return dict(sorted_items)
        
        except Exception:
            return None
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of model manager state
        
        Returns:
            Dictionary with summary information
        """
        return {
            'loaded_models': list(self.models.keys()),
            'current_model': self.current_model_key if self._loaded else None,
            'available_models': list(config.AVAILABLE_MODELS.keys()),
            'models_loaded': len(self.models),
            'is_ready': self._loaded
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test model manager"""
    
    print("="*70)
    print("MODEL MANAGER - TESTING")
    print("="*70)
    
    # Test 1: Initialize
    print("\n1. Testing Initialization:")
    print("-"*70)
    
    manager = ModelManager()
    print(f"Default model: {config.DEFAULT_MODEL}")
    print(f"Available models: {list(config.AVAILABLE_MODELS.keys())}")
    
    # Test 2: Model info
    print("\n2. Testing Model Info:")
    print("-"*70)
    
    for key, info in manager.get_all_model_info().items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  File: {info['filename']}")
        print(f"  Accuracy: {info['accuracy']:.2%}")
    
    # Test 3: Load model
    print("\n3. Testing Model Loading:")
    print("-"*70)
    
    # Try to load default model
    print(f"\nAttempting to load default model ({config.DEFAULT_MODEL})...")
    success = manager.set_active_model(config.DEFAULT_MODEL)
    
    if success:
        print("✓ Model loaded successfully!")
        
        # Test 4: Check loaded state
        print("\n4. Testing Loaded State:")
        print("-"*70)
        
        summary = manager.get_model_summary()
        print(f"Loaded models: {summary['loaded_models']}")
        print(f"Current model: {summary['current_model']}")
        print(f"Is ready: {summary['is_ready']}")
        
        # Test 5: Feature importance
        print("\n5. Testing Feature Importance:")
        print("-"*70)
        
        importance = manager.get_feature_importance(top_n=10)
        if importance:
            print("Top 10 features:")
            for i, (feature, imp) in enumerate(importance.items(), 1):
                print(f"  {i}. {feature:<30} {imp:.4f}")
        else:
            print("Feature importance not available for this model")
    
    else:
        print("\n⚠️  Model loading failed.")
        print("This is expected if model files haven't been placed in models/ yet.")
        print("\nNote: Models are required for Phase 4+, not Phase 3.")
    
    # Test 6: Load all models
    print("\n6. Testing Load All Models:")
    print("-"*70)
    
    manager2 = ModelManager()
    loaded_count = manager2.load_all_models(verbose=True)
    
    if loaded_count == 0:
        print("\n⚠️  No models loaded (expected if model files not present)")
    
    print("\n" + "="*70)
    print("✓ Model manager testing complete")
    print("="*70)
    
    # Summary
    print("\n📝 Summary:")
    print(f"  Model files location: {config.MODELS_DIR}")
    print(f"  Expected files:")
    for key, info in config.AVAILABLE_MODELS.items():
        print(f"    - {info['filename']}")
    
    print("\n💡 Note:")
    print("  Model files (.pkl) should be placed in models/ directory")
    print("  They are required for Phase 4 (AI Recommendation)")
    print("  Phase 3 (Feature Engineering) can proceed without them")

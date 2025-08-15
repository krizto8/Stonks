#!/usr/bin/env python3
"""
LSTM Model Training for Chart Pattern Recognition

This script trains an LSTM neural network to classify stock chart patterns.
It uses the dataset created by dataset_builder.py and implements the specified
architecture with proper training procedures.

Features:
- LSTM architecture with dropout layers
- Early stopping and model checkpointing
- Training/validation curves plotting
- Model evaluation and metrics
- Categorical crossentropy loss for multi-class classification

Usage:
    python train_model.py --epochs 50 --batch_size 32 --save_model
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config

class LSTMPatternClassifier:
    """LSTM model for chart pattern classification."""
    
    def __init__(self, sequence_length, n_features, n_classes=4):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        self.logger = self._setup_logger()
        
        # Initialize scalers for live prediction compatibility
        self.scaler_X = MinMaxScaler()
        self.scaler_fitted = False
        
        # Pattern class mapping
        self.pattern_classes = {
            0: 'Uptrend',
            1: 'Downtrend', 
            2: 'Head-and-Shoulders',
            3: 'Double Bottom'
        }
        
        # Feature columns (will be set during training)
        self.feature_columns = None
    
    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def prepare_data(self, X, y=None, fit_scaler=False):
        """Prepare and scale data for training or prediction."""
        # Reshape X for scaling (samples * sequence_length, features)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            self.logger.info("Fitting scaler on training data")
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            self.scaler_fitted = True
        else:
            if not self.scaler_fitted:
                raise ValueError("Scaler has not been fitted yet. Call with fit_scaler=True first.")
            X_scaled = self.scaler_X.transform(X_reshaped)
        
        # Reshape back to original shape
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled if y is None else (X_scaled, y)
    
    def build_model(self):
        """Build the LSTM model architecture as specified."""
        self.logger.info(f"🏗️ Building LSTM model...")
        self.logger.info(f"   Input shape: ({self.sequence_length}, {self.n_features})")
        self.logger.info(f"   Output classes: {self.n_classes}")
        
        model = Sequential([
            # First LSTM layer with 64 units, return_sequences=True
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            
            # Second LSTM layer with 64 units
            LSTM(64),
            Dropout(0.2),
            
            # Dense output layer with softmax activation
            Dense(self.n_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Print model summary
        self.logger.info("✓ Model architecture:")
        model.summary()
        
        return model
    
    def prepare_callbacks(self, model_save_path):
        """Prepare training callbacks."""
        callbacks = [
            # Early stopping if validation loss doesn't improve for 5 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save best model weights
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_save_path=None):
        """Train the LSTM model."""
        self.logger.info(f"🚀 Starting training...")
        self.logger.info(f"   Training samples: {len(X_train)}")
        self.logger.info(f"   Validation samples: {len(X_val)}")
        self.logger.info(f"   Batch size: {batch_size}")
        self.logger.info(f"   Max epochs: {epochs}")
        
        # Prepare callbacks
        if model_save_path is None:
            model_save_path = f"models/lstm_pattern_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        callbacks = self.prepare_callbacks(model_save_path)
        
        # Prepare and scale data
        X_train_scaled = self.prepare_data(X_train, fit_scaler=True)
        X_val_scaled = self.prepare_data(X_val, fit_scaler=False)
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.n_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.n_classes)
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("✓ Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        self.logger.info("📊 Evaluating model on test data...")
        
        # Scale test data
        X_test_scaled = self.prepare_data(X_test, fit_scaler=False)
        
        # Convert labels to categorical
        y_test_cat = to_categorical(y_test, num_classes=self.n_classes)
        
        # Get test metrics
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        class_names = [self.pattern_classes[i] for i in range(self.n_classes)]
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        self.logger.info(f"✓ Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"✓ Test Loss: {test_loss:.4f}")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss/accuracy curves."""
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', color='#2E86AB')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='#A23B72')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss', color='#F18F01')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='#C73E1D')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📈 Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        
        class_names = [self.pattern_classes[i] for i in range(self.n_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Pattern', fontsize=12)
        plt.ylabel('True Pattern', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def predict_pattern(self, sequence):
        """Predict pattern for a single sequence."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure sequence has correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        # Scale the input sequence
        sequence_scaled = self.prepare_data(sequence, fit_scaler=False)
        
        # Get prediction probabilities
        pred_proba = self.model.predict(sequence_scaled, verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        
        result = {
            'predicted_class': int(pred_class),
            'predicted_pattern': self.pattern_classes[pred_class],
            'probabilities': {
                self.pattern_classes[i]: float(pred_proba[i]) 
                for i in range(self.n_classes)
            },
            'confidence': float(np.max(pred_proba))
        }
        
        return result
    
    def save_model(self, model_name, base_dir="models"):
        """Save the trained model with scalers and metadata for live prediction compatibility."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(base_dir, f"{model_name}.h5")
        self.model.save(model_path)
        
        # Save scaler
        if self.scaler_fitted:
            scaler_path = os.path.join(base_dir, f"{model_name}_scaler_X.pkl")
            joblib.dump(self.scaler_X, scaler_path)
            self.logger.info(f"💾 Scaler saved to {scaler_path}")
        
        # Save metadata for live prediction compatibility
        metadata = {
            'model_type': 'classification',
            'prediction_type': 'classification',
            'n_classes': self.n_classes,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'pattern_classes': self.pattern_classes,
            'feature_columns': self.feature_columns,
            'scaler_fitted': self.scaler_fitted,
            'created_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(base_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"💾 Model saved to {model_path}")
        self.logger.info(f"💾 Metadata saved to {metadata_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path if self.scaler_fitted else None,
            'metadata_path': metadata_path
        }
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        self.logger.info(f"📂 Model loaded from {filepath}")

def load_dataset(data_dir='data'):
    """Load the processed dataset."""
    print(f"📂 Loading dataset from {data_dir}/...")
    
    # Load arrays
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load metadata
    with open(os.path.join(data_dir, 'dataset_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Dataset loaded successfully:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Features: {metadata['n_features']}")
    print(f"   Sequence length: {metadata['sequence_length']}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), metadata

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for chart pattern recognition')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the processed dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for the saved model')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for saved models and plots')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting LSTM model training...")
        
        # 1. Load dataset
        print("\n📂 Step 1: Loading dataset...")
        (X_train, X_val, X_test), (y_train, y_val, y_test), metadata = load_dataset(args.data_dir)
        
        # 2. Create model
        print("\n🏗️ Step 2: Building LSTM model...")
        classifier = LSTMPatternClassifier(
            sequence_length=metadata['sequence_length'],
            n_features=metadata['n_features'],
            n_classes=len(metadata['pattern_classes'])
        )
        
        # Set feature columns for live prediction compatibility
        classifier.feature_columns = metadata.get('feature_columns', [f"feature_{i}" for i in range(metadata['n_features'])])
        
        model = classifier.build_model()
        
        # 3. Train model
        print("\n🚀 Step 3: Training model...")
        
        # Prepare model save path
        if args.model_name:
            model_save_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = os.path.join(args.output_dir, f"lstm_pattern_classifier_{timestamp}.h5")
        
        history = classifier.train(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=model_save_path
        )
        
        # 4. Evaluate model
        print("\n📊 Step 4: Evaluating model...")
        results = classifier.evaluate(X_test, y_test)
        
        # Print classification report
        print("\n📋 Classification Report:")
        print(results['classification_report'])
        
        # 5. Plot training curves
        print("\n📈 Step 5: Plotting training curves...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        curves_path = os.path.join(args.output_dir, 'training_curves.png')
        classifier.plot_training_history(save_path=curves_path)
        
        # 6. Plot confusion matrix
        print("\n📊 Step 6: Plotting confusion matrix...")
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        classifier.plot_confusion_matrix(np.array(results['confusion_matrix']), save_path=cm_path)
        
        # 7. Save results
        print("\n💾 Step 7: Saving results...")
        
        # Save detailed results
        results_with_metadata = {
            **results,
            'model_metadata': {
                'sequence_length': metadata['sequence_length'],
                'n_features': metadata['n_features'],
                'n_classes': len(metadata['pattern_classes']),
                'pattern_classes': metadata['pattern_classes'],
                'epochs_trained': len(history.history['loss']),
                'batch_size': args.batch_size,
                'model_architecture': 'LSTM-64-Dropout-LSTM-64-Dropout-Dense'
            },
            'training_history': {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            }
        }
        
        results_path = os.path.join(args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # 8. Save model if requested
        if args.save_model:
            model_name = args.model_name or f"lstm_pattern_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n💾 Step 8: Saving model as '{model_name}' with scalers and metadata...")
            
            # Save with enhanced method for live prediction compatibility
            save_paths = classifier.save_model(model_name, base_dir=args.output_dir)
            
            print("✓ Saved files:")
            print(f"   Model: {save_paths['model_path']}")
            if save_paths['scaler_path']:
                print(f"   Scaler: {save_paths['scaler_path']}")
            print(f"   Metadata: {save_paths['metadata_path']}")
        
        # Summary
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📋 Training Summary:")
        print(f"   Final validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"   Test accuracy: {results['test_accuracy']:.4f}")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        print(f"   Best model saved to: {model_save_path}")
        
        print(f"\n📁 Outputs saved to {args.output_dir}/:")
        print(f"   - training_curves.png")
        print(f"   - confusion_matrix.png") 
        print(f"   - training_results.json")
        if args.save_model:
            model_name = args.model_name or f"lstm_pattern_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"   - {model_name}.h5 (model)")
            print(f"   - {model_name}_scaler_X.pkl (scaler)")
            print(f"   - {model_name}_metadata.json (metadata)")
        
        print("\nNext steps:")
        print("1. Use predict.py to make predictions on new data")
        print("2. Analyze the confusion matrix to understand model performance")
        print("3. Consider hyperparameter tuning if needed")
        
        # Example prediction
        print(f"\n🔮 Example prediction on first test sample:")
        example_result = classifier.predict_pattern(X_test[0])
        print(f"   Predicted: {example_result['predicted_pattern']}")
        print(f"   Confidence: {example_result['confidence']:.3f}")
        print(f"   True label: {classifier.pattern_classes[y_test[0]]}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

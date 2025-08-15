import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib
import logging
import os
import sys
from typing import Tuple, Optional, Dict, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class LSTMModel:
    """
    LSTM model for stock price prediction and trading signal generation.
    """
    
    def __init__(self, prediction_type: str = "regression"):
        """
        Initialize LSTM model.
        
        Args:
            prediction_type: "regression" for price prediction, "classification" for signals
        """
        self.prediction_type = prediction_type
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.logger = self._setup_logger()
        self.feature_columns = None
        self.history = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        try:
            self.logger.info(f"Building LSTM model with input shape: {input_shape}")
            
            model = keras.Sequential()
            
            # First LSTM layer
            model.add(layers.LSTM(
                units=Config.LSTM_UNITS[0],
                return_sequences=True if len(Config.LSTM_UNITS) > 1 else False,
                input_shape=input_shape,
                dropout=Config.DROPOUT_RATE,
                recurrent_dropout=Config.DROPOUT_RATE
            ))
            
            # Additional LSTM layers
            for i, units in enumerate(Config.LSTM_UNITS[1:], 1):
                return_sequences = i < len(Config.LSTM_UNITS) - 1
                model.add(layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.DROPOUT_RATE
                ))
            
            # Dense layers
            model.add(layers.Dense(25, activation='relu'))
            model.add(layers.Dropout(Config.DROPOUT_RATE))
            
            # Output layer
            if self.prediction_type == "classification":
                model.add(layers.Dense(3, activation='softmax'))  # Buy, Sell, Hold
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.add(layers.Dense(1))  # Price prediction
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                    loss='mse',
                    metrics=['mae']
                )
            
            self.model = model
            self.logger.info("Model built successfully")
            
            # Print model summary
            model.summary()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training.
        
        Args:
            X: Feature data
            y: Target data
            fit_scalers: Whether to fit the scalers (True for training, False for prediction)
            
        Returns:
            Scaled X and y data
        """
        try:
            # Reshape X for scaling
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            
            if fit_scalers:
                self.logger.info("Fitting scalers on training data")
                X_scaled = self.scaler_X.fit_transform(X_reshaped)
                
                if self.prediction_type == "regression" and self.scaler_y is not None:
                    y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                else:
                    y_scaled = y  # No scaling for classification
            else:
                X_scaled = self.scaler_X.transform(X_reshaped)
                
                if self.prediction_type == "regression" and self.scaler_y is not None:
                    y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
                else:
                    y_scaled = y
            
            # Reshape back to original format
            X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            return X_scaled, y_scaled
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def create_labels(self, prices: np.ndarray) -> np.ndarray:
        """
        Create classification labels from price data.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of labels (0=Sell, 1=Hold, 2=Buy)
        """
        try:
            price_changes = np.diff(prices) / prices[:-1]
            labels = np.ones(len(price_changes))  # Default to Hold (1)
            
            # Buy signal (2) for price increases above threshold
            labels[price_changes > Config.BUY_THRESHOLD] = 2
            
            # Sell signal (0) for price decreases below threshold
            labels[price_changes < Config.SELL_THRESHOLD] = 0
            
            return labels.astype(int)
            
        except Exception as e:
            self.logger.error(f"Error creating labels: {str(e)}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = None, batch_size: int = None, 
             validation_split: float = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            epochs: Number of training epochs (default: Config.EPOCHS)
            batch_size: Batch size for training (default: Config.BATCH_SIZE)
            validation_split: Validation split ratio (default: Config.VALIDATION_SPLIT)
            
        Returns:
            Training history
        """
        try:
            self.logger.info("Starting model training")
            
            # Prepare data
            X_scaled, y_scaled = self.prepare_data(X, y, fit_scalers=True)
            
            # Store feature columns for later use
            self.feature_columns = list(range(X.shape[2]))
            
            # Build model if not already built
            if self.model is None:
                self.build_model((X.shape[1], X.shape[2]))
            
            # Prepare validation data if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_scaled, y_val_scaled = self.prepare_data(X_val, y_val, fit_scalers=False)
                validation_data = (X_val_scaled, y_val_scaled)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=Config.PATIENCE,
                    restore_best_weights=True,
                    monitor='val_loss' if validation_data else 'loss'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    monitor='val_loss' if validation_data else 'loss'
                )
            ]
            
            # Use provided parameters or fall back to config defaults
            epochs = epochs or Config.EPOCHS
            batch_size = batch_size or Config.BATCH_SIZE
            validation_split = validation_split or Config.VALIDATION_SPLIT
            
            # Train model
            self.history = self.model.fit(
                X_scaled, y_scaled,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                validation_split=validation_split if validation_data is None else 0,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Model training completed")
            return self.history.history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Prepare data
            X_scaled, _ = self.prepare_data(X, np.zeros(X.shape[0]), fit_scalers=False)
            
            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Inverse scale for regression (only if y_scaler exists)
            if self.prediction_type == "regression" and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(X)
            
            if self.prediction_type == "regression":
                mse = mean_squared_error(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mse)
                
                # Calculate directional accuracy
                actual_direction = np.sign(np.diff(y))
                pred_direction = np.sign(np.diff(predictions))
                directional_accuracy = np.mean(actual_direction == pred_direction)
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy
                }
                
            else:  # classification
                pred_classes = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(y, pred_classes)
                
                metrics = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y, pred_classes, output_dict=True)
                }
            
            self.logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_name: str) -> None:
        """
        Save the trained model and scalers.
        
        Args:
            model_name: Name for the saved model
        """
        try:
            os.makedirs(Config.MODELS_DIR, exist_ok=True)
            
            # Save model
            model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.h5")
            self.model.save(model_path)
            
            # Save scalers
            scaler_X_path = os.path.join(Config.MODELS_DIR, f"{model_name}_scaler_X.pkl")
            joblib.dump(self.scaler_X, scaler_X_path)
            
            # Only save Y scaler for regression models
            if self.prediction_type == "regression" and self.scaler_y is not None:
                scaler_y_path = os.path.join(Config.MODELS_DIR, f"{model_name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_y_path)
            
            # Save metadata
            metadata = {
                'prediction_type': self.prediction_type,
                'feature_columns': self.feature_columns,
                'sequence_length': Config.SEQUENCE_LENGTH,
                'features': Config.FEATURES
            }
            
            metadata_path = os.path.join(Config.MODELS_DIR, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_name: str) -> None:
        """
        Load a saved model and scalers.
        
        Args:
            model_name: Name of the saved model
        """
        try:
            # Load model with custom objects for compatibility
            model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.h5")
            
            # Try loading with different compatibility modes
            try:
                self.model = keras.models.load_model(model_path)
            except (TypeError, ValueError) as e:
                if "batch_shape" in str(e):
                    # Handle batch_shape compatibility issue
                    self.logger.warning("Attempting to load model with compatibility mode...")
                    import tensorflow as tf
                    custom_objects = {}
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        self.model = keras.models.load_model(model_path, compile=False)
                        # Recompile the model
                        self.model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy' if self.prediction_type == 'classification' else 'mse',
                            metrics=['accuracy'] if self.prediction_type == 'classification' else ['mae']
                        )
                else:
                    raise e
            
            # Load scalers
            scaler_X_path = os.path.join(Config.MODELS_DIR, f"{model_name}_scaler_X.pkl")
            scaler_y_path = os.path.join(Config.MODELS_DIR, f"{model_name}_scaler_y.pkl")
            
            # Load X scaler (required)
            self.scaler_X = joblib.load(scaler_X_path)
            
            # Load Y scaler (optional for classification models)
            if os.path.exists(scaler_y_path):
                self.scaler_y = joblib.load(scaler_y_path)
            else:
                self.logger.info("No Y scaler found - assuming classification model")
                self.scaler_y = None
            
            # Load metadata (optional)
            metadata_path = os.path.join(Config.MODELS_DIR, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.prediction_type = metadata.get('prediction_type', 'classification')
                self.feature_columns = metadata.get('feature_columns', [])
            else:
                self.logger.info("No metadata found - using default settings for classification model")
                # Default to classification if no Y scaler and no metadata
                if self.scaler_y is None:
                    self.prediction_type = 'classification'
                else:
                    self.prediction_type = 'regression'
                self.feature_columns = []
            
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(f"Prediction type: {self.prediction_type}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise


class TradingSignalGenerator:
    """
    Generates trading signals based on model predictions.
    """
    
    def __init__(self, model: LSTMModel):
        self.model = model
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_signal(self, current_data: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        Generate trading signal based on current data.
        
        Args:
            current_data: Recent data for prediction
            current_price: Current stock price
            
        Returns:
            Dictionary containing signal information
        """
        try:
            # Make prediction
            prediction = self.model.predict(current_data.reshape(1, *current_data.shape))
            
            if self.model.prediction_type == "regression":
                predicted_price = prediction[0]
                price_change = (predicted_price - current_price) / current_price
                
                # Generate signal based on thresholds
                if price_change > Config.BUY_THRESHOLD:
                    signal = "BUY"
                    confidence = min(abs(price_change) / Config.BUY_THRESHOLD, 1.0)
                elif price_change < Config.SELL_THRESHOLD:
                    signal = "SELL"
                    confidence = min(abs(price_change) / abs(Config.SELL_THRESHOLD), 1.0)
                else:
                    signal = "HOLD"
                    confidence = 1.0 - min(abs(price_change) / Config.BUY_THRESHOLD, 1.0)
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'expected_change': price_change,
                    'timestamp': pd.Timestamp.now()
                }
                
            else:  # classification - Pattern Recognition Model
                probabilities = prediction[0]
                signal_idx = np.argmax(probabilities)
                confidence = probabilities[signal_idx]
                
                # Map pattern classes to trading signals
                # 0: Uptrend -> BUY, 1: Downtrend -> SELL, 2: Head-and-Shoulders -> SELL, 3: Double Bottom -> BUY
                if signal_idx == 0 or signal_idx == 3:  # Uptrend or Double Bottom
                    signal = "BUY"
                elif signal_idx == 1 or signal_idx == 2:  # Downtrend or Head-and-Shoulders
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'pattern_predicted': Config.PATTERN_CLASSES.get(signal_idx, 'Unknown'),
                    'pattern_probabilities': {
                        'uptrend': probabilities[0],
                        'downtrend': probabilities[1], 
                        'head_shoulders': probabilities[2],
                        'double_bottom': probabilities[3]
                    },
                    'current_price': current_price,
                    'timestamp': pd.Timestamp.now()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            raise
    
    def is_signal_valid(self, signal_data: Dict[str, Any]) -> bool:
        """
        Check if signal meets confidence threshold.
        
        Args:
            signal_data: Signal data from generate_signal
            
        Returns:
            True if signal is valid
        """
        return signal_data['confidence'] >= Config.CONFIDENCE_THRESHOLD

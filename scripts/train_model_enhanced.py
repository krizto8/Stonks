#!/usr/bin/env python3
"""
Enhanced LSTM Model Training Script for Pattern Recognition

This script focuses on pattern recognition in stock trading with advanced feature engineering.
It can either download fresh data or use existing preprocessed datasets for training.

Usage:
    # Train with fresh data:
    python train_model_enhanced.py --ticker AAPL --period 2y --save_data --epochs 50 --batch_size 32
    
    # Train with existing .npy dataset files:
    python train_model_enhanced.py --use_existing_data --data_dir ../data --epochs 100 --batch_size 64
    
    # Train with custom parameters:
    python train_model_enhanced.py --use_existing_data --sequence_length 60 --test_size 0.2 --epochs 75
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_utils import DataFetcher
from utils.model_utils import LSTMModel

class PatternRecognitionPreprocessor:
    """Enhanced preprocessing focused on pattern recognition."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.scaler = RobustScaler()  # More robust to outliers than MinMaxScaler
        
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
    
    def calculate_candlestick_patterns(self, data):
        """Calculate candlestick pattern indicators."""
        df = data.copy()
        
        # Body and shadow ratios
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios for pattern recognition
        df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-8)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-8)
        
        # Candlestick patterns using pandas_ta
        df['doji'] = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = ta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = ta.cdl_engulfing(df['open'], df['high'], df['low'], df['close'])
        
        # Simple pattern indicators
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        
        return df
    
    def calculate_price_patterns(self, data):
        """Calculate price pattern indicators."""
        df = data.copy()
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        df['gap_size'] = df['open'] - df['close'].shift(1)
        
        # Price velocity and acceleration
        df['price_velocity'] = df['close'].diff()
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Rolling patterns
        for window in [5, 10, 20]:
            df[f'high_{window}d'] = df['high'].rolling(window).max()
            df[f'low_{window}d'] = df['low'].rolling(window).min()
            df[f'range_{window}d'] = df[f'high_{window}d'] - df[f'low_{window}d']
            df[f'price_rank_{window}d'] = df['close'].rolling(window).rank() / window
        
        return df
    
    def calculate_volume_patterns(self, data):
        """Calculate volume pattern indicators."""
        df = data.copy()
        
        # Volume analysis
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        
        # Volume price trend
        df['vpt'] = ta.vpt(df['close'], df['volume'])
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Volume patterns
        df['volume_spike'] = (df['volume'] > df['volume_ma_20'] * 2).astype(int)
        df['low_volume'] = (df['volume'] < df['volume_ma_20'] * 0.5).astype(int)
        
        # Price-Volume relationship
        df['pv_trend'] = ((df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1))).astype(int)
        
        return df
    
    def calculate_momentum_patterns(self, data):
        """Calculate momentum and trend patterns."""
        df = data.copy()
        
        # Multiple timeframe RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)
        
        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.roc(df['close'], length=period)
        
        # Commodity Channel Index
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        # Money Flow Index
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend strength
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])
        
        return df
    
    def calculate_volatility_patterns(self, data):
        """Calculate volatility indicators."""
        df = data.copy()
        
        # True Range and ATR
        df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        
        # Volatility ratios
        df['volatility_ratio'] = df['atr'] / df['close']
        
        # Bollinger Band indicators
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc['KCUe_20_2']
        df['kc_middle'] = kc['KCBe_20_2']
        df['kc_lower'] = kc['KCLe_20_2']
        
        return df
    
    def calculate_support_resistance_levels(self, data):
        """Calculate support and resistance levels."""
        df = data.copy()
        
        # Pivot points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Distance from pivot levels
        df['dist_from_pivot'] = (df['close'] - df['pivot']) / df['close']
        df['dist_from_r1'] = (df['close'] - df['r1']) / df['close']
        df['dist_from_s1'] = (df['close'] - df['s1']) / df['close']
        
        # Rolling support/resistance
        for window in [20, 50]:
            df[f'resistance_{window}'] = df['high'].rolling(window).max()
            df[f'support_{window}'] = df['low'].rolling(window).min()
            df[f'sr_ratio_{window}'] = (df['close'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'] + 1e-8)
        
        return df
    
    def create_pattern_features(self, data):
        """Main function to create all pattern recognition features."""
        self.logger.info("Starting pattern recognition feature engineering...")
        
        df = data.copy()
        
        # Apply all pattern calculations
        df = self.calculate_candlestick_patterns(df)
        df = self.calculate_price_patterns(df)
        df = self.calculate_volume_patterns(df)
        df = self.calculate_momentum_patterns(df)
        df = self.calculate_volatility_patterns(df)
        df = self.calculate_support_resistance_levels(df)
        
        # Create interaction features
        df['rsi_bb_interaction'] = df['rsi_14'] * df['bb_position']
        df['volume_momentum'] = df['volume_ratio'] * df['roc_10']
        df['volatility_momentum'] = df['volatility_ratio'] * df['rsi_14']
        
        # Create target patterns (for classification)
        df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
        df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
        df['future_return_10'] = df['close'].shift(-10) / df['close'] - 1
        
        # Pattern-based labels
        df['bullish_pattern'] = (
            (df['future_return_5'] > 0.02) & 
            (df['rsi_14'] < 70) & 
            (df['bb_position'] < 0.8)
        ).astype(int)
        
        df['bearish_pattern'] = (
            (df['future_return_5'] < -0.02) & 
            (df['rsi_14'] > 30) & 
            (df['bb_position'] > 0.2)
        ).astype(int)
        
        self.logger.info(f"Created {len(df.columns)} features for pattern recognition")
        return df
    
    def prepare_sequences(self, data, sequence_length=60, target_col='close'):
        """Create sequences for LSTM training with pattern focus."""
        
        # Select relevant features for pattern recognition
        feature_cols = [col for col in data.columns if not col.startswith('future_') 
                       and col not in ['bullish_pattern', 'bearish_pattern']]
        
        # Remove any remaining NaN values
        clean_data = data[feature_cols + ['future_return_5']].dropna()
        
        self.logger.info(f"Using {len(feature_cols)} features for sequences")
        self.logger.info(f"Clean data shape: {clean_data.shape}")
        
        # Scale features
        feature_data = clean_data[feature_cols]
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(clean_data['future_return_5'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y, feature_cols

def save_datasets(raw_data, processed_data, ticker, period):
    """Save datasets as CSV files."""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Save raw data
    raw_file = os.path.join(Config.DATA_DIR, f"{ticker}_{period}_raw_data.csv")
    raw_data.to_csv(raw_file)
    print(f"✓ Raw data saved to: {raw_file}")
    
    # Save processed data
    processed_file = os.path.join(Config.DATA_DIR, f"{ticker}_{period}_processed_data.csv")
    processed_data.to_csv(processed_file)
    print(f"✓ Processed data saved to: {processed_file}")
    
    # Save feature summary
    summary_file = os.path.join(Config.DATA_DIR, f"{ticker}_{period}_feature_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("PATTERN RECOGNITION FEATURES\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Features: {len(processed_data.columns)}\n\n")
        
        categories = {
            'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
            'Candlestick Patterns': [col for col in processed_data.columns if any(x in col for x in ['doji', 'hammer', 'engulfing', 'body_', 'shadow'])],
            'Price Patterns': [col for col in processed_data.columns if any(x in col for x in ['price_position', 'gap_', 'velocity', 'acceleration', 'rank'])],
            'Volume Patterns': [col for col in processed_data.columns if any(x in col for x in ['volume_', 'vpt', 'obv', 'pv_trend'])],
            'Momentum': [col for col in processed_data.columns if any(x in col for x in ['rsi_', 'williams_r', 'roc_', 'cci', 'mfi', 'adx'])],
            'Volatility': [col for col in processed_data.columns if any(x in col for x in ['atr', 'volatility', 'bb_', 'kc_'])],
            'Support/Resistance': [col for col in processed_data.columns if any(x in col for x in ['pivot', 'r1', 'r2', 's1', 's2', 'resistance', 'support', 'sr_ratio'])],
        }
        
        for category, features in categories.items():
            if features:
                f.write(f"{category}:\n")
                for feature in features:
                    if feature in processed_data.columns:
                        f.write(f"  - {feature}\n")
                f.write("\n")
    
    print(f"✓ Feature summary saved to: {summary_file}")

def load_existing_dataset(data_dir='data'):
    """Load existing .npy dataset files."""
    print(f"📂 Loading existing dataset from {data_dir}/...")
    
    try:
        # Load arrays
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Combine train and validation for enhanced training
        X_train_combined = np.concatenate([X_train, X_val], axis=0)
        y_train_combined = np.concatenate([y_train, y_val], axis=0)
        
        # Load metadata
        import json
        with open(os.path.join(data_dir, 'dataset_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Dataset loaded successfully:")
        print(f"   Train: {X_train_combined.shape}")
        print(f"   Test:  {X_test.shape}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Sequence length: {metadata['sequence_length']}")
        
        return X_train_combined, X_test, y_train_combined, y_test, metadata
        
    except Exception as e:
        print(f"❌ Error loading existing dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced LSTM training for pattern recognition')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='2y', help='Historical data period')
    parser.add_argument('--save_data', action='store_true', help='Save datasets as CSV files')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length for LSTM')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--use_existing_data', action='store_true', help='Use existing .npy dataset files from data/')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing .npy dataset files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED PATTERN RECOGNITION TRAINING")
    print("="*50)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.period}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Test Size: {args.test_size}")
    if args.use_existing_data:
        print(f"Data Source: Existing dataset from {args.data_dir}")
    else:
        print(f"Data Source: Fresh data from market")
    
    try:
        if args.use_existing_data:
            # Load existing dataset
            print("\n📂 Loading existing dataset...")
            X_train, X_test, y_train, y_test, metadata = load_existing_dataset(args.data_dir)
            
            # Use metadata values
            sequence_length = metadata['sequence_length']
            n_features = metadata['n_features']
            feature_names = list(range(n_features))  # Generic feature names
            
        else:
            # 1. Fetch data
            print("\n📊 Fetching market data...")
            data_fetcher = DataFetcher(ticker=args.ticker, interval='1d')  # Daily data for patterns
            raw_data = data_fetcher.fetch_historical_data(period=args.period)
            print(f"✓ Fetched {len(raw_data)} data points")
            
            # 2. Enhanced feature engineering
            print("\n🔍 Creating pattern recognition features...")
            preprocessor = PatternRecognitionPreprocessor()
            processed_data = preprocessor.create_pattern_features(raw_data)
            print(f"✓ Created {len(processed_data.columns)} features")
            
            # 3. Save datasets if requested
            if args.save_data:
                print("\n💾 Saving datasets...")
                save_datasets(raw_data, processed_data, args.ticker, args.period)
            
            # 4. Prepare sequences for LSTM
            print("\n🔄 Preparing sequences for LSTM...")
            X, y, feature_names = preprocessor.prepare_sequences(
                processed_data, 
                sequence_length=args.sequence_length
            )
            
            # 5. Split data
            print("\n📊 Splitting data...")
            train_size = int(len(X) * (1 - args.test_size))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            sequence_length = args.sequence_length
            n_features = X.shape[2]
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        print(f"✓ Features: {n_features}")
        
        # 6. Train model
        print("\n🤖 Training LSTM model...")
        model = LSTMModel(prediction_type='classification')  # Use classification for pattern recognition
        
        # Update model configuration for pattern recognition
        model.sequence_length = sequence_length
        model.n_features = n_features
        
        history = model.train(
            X_train, y_train, 
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # 7. Evaluate model
        print("\n📈 Evaluating model...")
        test_metrics = model.evaluate(X_test, y_test)
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 8. Save model
        if args.use_existing_data:
            model_name = "lstm_pattern_classifier_enhanced"
        else:
            model_name = f"{args.ticker}_{args.period}_pattern_model"
        
        print(f"\n💾 Saving model as: {model_name}")
        model.save_model(model_name)
        
        # 9. Save training results
        results = {
            'model_name': model_name,
            'ticker': args.ticker if not args.use_existing_data else 'multi_ticker',
            'period': args.period if not args.use_existing_data else 'existing_data',
            'sequence_length': sequence_length,
            'n_features': n_features,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'test_size': args.test_size,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_metrics': test_metrics,
            'feature_names': feature_names,
            'used_existing_data': args.use_existing_data
        }
        
        import json
        results_path = os.path.join(Config.MODELS_DIR, f"{model_name}_results.json")
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {results_path}")
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Check saved CSV files for data analysis")
        print("2. Review feature importance and patterns")
        print("3. Use trained model for predictions")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

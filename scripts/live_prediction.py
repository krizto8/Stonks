#!/usr/bin/env python3
"""
Real-time Stock Prediction and Signal Generation Script

This script continuously fetches live data, makes predictions, generates trading signals,
and sends alerts. It's designed to run as a background service.

Usage:
    python live_prediction.py --model_name AAPL_1h_regression_model --ticker AAPL
"""

import argparse
import logging
import os
import sys
import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import deque
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_utils import DataFetcher, DataPreprocessor
from utils.model_utils import LSTMModel, TradingSignalGenerator
from utils.alert_utils import AlertManager

class LivePredictor:
    """
    Handles live prediction and signal generation.
    """
    
    def __init__(self, model_name: str, ticker: str, interval: str = None):
        self.model_name = model_name
        self.ticker = ticker
        self.interval = interval or Config.DATA_INTERVAL
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher(ticker=self.ticker, interval=self.interval)
        self.data_preprocessor = DataPreprocessor()
        self.model = LSTMModel()
        self.signal_generator = None
        self.alert_manager = AlertManager()
        
        # Load model
        self._load_model()
        
        # Initialize data buffer
        self.data_buffer = deque(maxlen=Config.SEQUENCE_LENGTH * 2)
        self.predictions_history = deque(maxlen=100)
        self.signals_history = deque(maxlen=50)
        
        # Performance tracking
        self.last_prediction_time = None
        self.prediction_count = 0
        self.error_count = 0
        
        self.logger.info(f"LivePredictor initialized for {ticker} using model {model_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        logger = logging.getLogger(f"LivePredictor_{self.ticker}")
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                os.path.join(Config.LOGS_DIR, f'live_prediction_{self.ticker}.log')
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model.load_model(self.model_name)
            self.signal_generator = TradingSignalGenerator(self.model)
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _fetch_initial_data(self):
        """Fetch initial data to populate the buffer."""
        try:
            self.logger.info("Fetching initial data...")
            
            # Fetch recent data (more than sequence length)
            days_needed = max(7, Config.SEQUENCE_LENGTH // 100)  # Estimate based on interval
            raw_data = self.data_fetcher.fetch_live_data(days=days_needed)
            
            # Process data
            processed_data = self.data_preprocessor.calculate_technical_indicators(raw_data)
            feature_data = self.data_preprocessor.prepare_features(processed_data)
            
            # Populate buffer with recent data
            for idx in range(len(feature_data)):
                self.data_buffer.append({
                    'timestamp': feature_data.index[idx],
                    'data': feature_data.iloc[idx].values,
                    'close_price': processed_data['close'].iloc[idx]
                })
            
            self.logger.info(f"Initialized data buffer with {len(self.data_buffer)} records")
            
        except Exception as e:
            self.logger.error(f"Error fetching initial data: {str(e)}")
            raise
    
    def _fetch_latest_data(self):
        """Fetch the latest data point."""
        try:
            # Fetch recent data
            raw_data = self.data_fetcher.fetch_live_data(days=1)
            
            if raw_data.empty:
                self.logger.warning("No new data available")
                return None
            
            # Get the latest data point
            latest_raw = raw_data.iloc[-1:]
            
            # Process data
            processed_data = self.data_preprocessor.calculate_technical_indicators(raw_data)
            feature_data = self.data_preprocessor.prepare_features(processed_data)
            
            # Get the latest processed data point
            latest_processed = feature_data.iloc[-1:]
            latest_close = processed_data['close'].iloc[-1]
            
            # Check if this is truly new data
            if self.data_buffer and self.data_buffer[-1]['timestamp'] >= latest_processed.index[0]:
                self.logger.debug("No new data since last fetch")
                return None
            
            new_data_point = {
                'timestamp': latest_processed.index[0],
                'data': latest_processed.iloc[0].values,
                'close_price': latest_close
            }
            
            return new_data_point
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            return None
    
    def _make_prediction(self):
        """Make prediction and generate trading signal."""
        try:
            if len(self.data_buffer) < Config.SEQUENCE_LENGTH:
                self.logger.warning(f"Insufficient data for prediction. Need {Config.SEQUENCE_LENGTH}, have {len(self.data_buffer)}")
                return None
            
            # Prepare sequence data
            sequence_data = np.array([point['data'] for point in list(self.data_buffer)[-Config.SEQUENCE_LENGTH:]])
            current_price = self.data_buffer[-1]['close_price']
            
            # Generate signal
            signal_data = self.signal_generator.generate_signal(sequence_data, current_price)
            
            # Add metadata
            signal_data['model_name'] = self.model_name
            signal_data['ticker'] = self.ticker
            signal_data['data_timestamp'] = self.data_buffer[-1]['timestamp']
            
            # Store prediction
            self.predictions_history.append(signal_data)
            
            # Update counters
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            self.logger.info(f"Prediction: {signal_data['signal']} (confidence: {signal_data['confidence']:.2%}) for {self.ticker} at ${current_price:.2f}")
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            self.error_count += 1
            return None
    
    def _process_signal(self, signal_data):
        """Process trading signal and send alerts if necessary."""
        try:
            # Check if signal is valid (meets confidence threshold)
            if not self.signal_generator.is_signal_valid(signal_data):
                self.logger.debug(f"Signal confidence too low: {signal_data['confidence']:.2%}")
                return
            
            # Store signal
            self.signals_history.append(signal_data)
            
            # Send alert
            alert_sent = self.alert_manager.send_signal_alert(signal_data, self.ticker)
            
            if alert_sent:
                self.logger.info(f"Alert sent for {signal_data['signal']} signal")
            
            # Save signal to file
            self._save_signal_to_file(signal_data)
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
    
    def _save_signal_to_file(self, signal_data):
        """Save signal data to file for dashboard."""
        try:
            signals_file = os.path.join(Config.DATA_DIR, f"live_signals_{self.ticker}.json")
            
            # Load existing signals
            if os.path.exists(signals_file):
                with open(signals_file, 'r') as f:
                    signals = json.load(f)
            else:
                signals = []
            
            # Add new signal
            signals.append(signal_data)
            
            # Keep only recent signals (last 100)
            signals = signals[-100:]
            
            # Save updated signals
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving signal to file: {str(e)}")
    
    def _save_status(self):
        """Save current status for monitoring."""
        try:
            status = {
                'ticker': self.ticker,
                'model_name': self.model_name,
                'last_update': datetime.now(),
                'last_prediction_time': self.last_prediction_time,
                'prediction_count': self.prediction_count,
                'error_count': self.error_count,
                'buffer_size': len(self.data_buffer),
                'recent_signals': list(self.signals_history)[-5:],  # Last 5 signals
                'current_price': self.data_buffer[-1]['close_price'] if self.data_buffer else None
            }
            
            status_file = os.path.join(Config.DATA_DIR, f"live_status_{self.ticker}.json")
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving status: {str(e)}")
    
    def run_prediction_cycle(self):
        """Run one prediction cycle."""
        try:
            self.logger.debug("Starting prediction cycle")
            
            # Fetch latest data
            new_data = self._fetch_latest_data()
            
            if new_data is not None:
                # Add to buffer
                self.data_buffer.append(new_data)
                self.logger.debug(f"Added new data point: {new_data['timestamp']}")
                
                # Make prediction
                signal_data = self._make_prediction()
                
                if signal_data is not None:
                    # Process signal
                    self._process_signal(signal_data)
            
            # Save status
            self._save_status()
            
        except Exception as e:
            self.logger.error(f"Error in prediction cycle: {str(e)}")
            self.error_count += 1
    
    def start_live_prediction(self, update_interval_minutes: int = 1):
        """Start the live prediction loop."""
        try:
            self.logger.info(f"Starting live prediction for {self.ticker}")
            self.logger.info(f"Update interval: {update_interval_minutes} minutes")
            
            # Fetch initial data
            self._fetch_initial_data()
            
            # Schedule prediction cycles
            schedule.every(update_interval_minutes).minutes.do(self.run_prediction_cycle)
            
            # Send startup notification
            startup_message = f"""
Live prediction started for {self.ticker}

Model: {self.model_name}
Update Interval: {update_interval_minutes} minutes
Prediction Type: {self.model.prediction_type}
"""
            self.alert_manager.alert_system.send_system_alert(startup_message, "INFO")
            
            # Run initial prediction
            self.run_prediction_cycle()
            
            # Main loop
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Stopping live prediction (KeyboardInterrupt)")
        except Exception as e:
            self.logger.error(f"Error in live prediction loop: {str(e)}")
            
            # Send error notification
            error_message = f"Live prediction stopped due to error: {str(e)}"
            self.alert_manager.alert_system.send_system_alert(error_message, "ERROR")
            raise
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return {
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.prediction_count, 1),
            'last_prediction_time': self.last_prediction_time,
            'uptime': datetime.now() - (self.last_prediction_time or datetime.now()),
            'buffer_size': len(self.data_buffer),
            'signals_generated': len(self.signals_history)
        }


def main():
    parser = argparse.ArgumentParser(description='Live stock prediction and signal generation')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the trained model to use')
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--interval', type=str, default=Config.DATA_INTERVAL,
                       help='Data interval (1m, 5m, 1h, 1d)')
    parser.add_argument('--update_interval', type=int, default=1,
                       help='Update interval in minutes')
    
    args = parser.parse_args()
    
    try:
        # Create live predictor
        predictor = LivePredictor(
            model_name=args.model_name,
            ticker=args.ticker,
            interval=args.interval
        )
        
        # Start live prediction
        predictor.start_live_prediction(update_interval_minutes=args.update_interval)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/bin/bash
"""
Management script for the Stock Trading System

This script provides utilities for managing the trading system.
"""

import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import subprocess

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_utils import DataFetcher, DataPreprocessor
from utils.model_utils import LSTMModel

def list_models():
    """List available trained models."""
    print("Available Models:")
    print("-" * 50)
    
    models_dir = Config.MODELS_DIR
    if not os.path.exists(models_dir):
        print("No models directory found")
        return
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5'):
            model_name = file.replace('.h5', '')
            
            # Try to load metadata
            metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                models.append({
                    'name': model_name,
                    'type': metadata.get('prediction_type', 'unknown'),
                    'features': len(metadata.get('features', [])),
                    'sequence_length': metadata.get('sequence_length', 'unknown')
                })
            else:
                models.append({
                    'name': model_name,
                    'type': 'unknown',
                    'features': 'unknown',
                    'sequence_length': 'unknown'
                })
    
    if models:
        df = pd.DataFrame(models)
        print(df.to_string(index=False))
    else:
        print("No models found")

def show_model_info(model_name):
    """Show detailed information about a model."""
    print(f"Model Information: {model_name}")
    print("-" * 50)
    
    # Load metadata
    metadata_file = os.path.join(Config.MODELS_DIR, f"{model_name}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Prediction Type: {metadata.get('prediction_type', 'unknown')}")
        print(f"Sequence Length: {metadata.get('sequence_length', 'unknown')}")
        print(f"Features: {len(metadata.get('features', []))}")
        print(f"Feature List: {', '.join(metadata.get('features', []))}")
    else:
        print("No metadata found for this model")
    
    # Load results if available
    results_file = os.path.join(Config.MODELS_DIR, f"{model_name}_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\nTraining Results:")
        print(f"Ticker: {results.get('ticker', 'unknown')}")
        print(f"Interval: {results.get('interval', 'unknown')}")
        print(f"Training Samples: {results.get('training_samples', 'unknown')}")
        print(f"Test Samples: {results.get('test_samples', 'unknown')}")
        
        test_metrics = results.get('test_metrics', {})
        if test_metrics:
            print(f"\nTest Metrics:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

def check_system_status():
    """Check the status of running systems."""
    print("System Status Check")
    print("-" * 50)
    
    # Check for status files
    data_dir = Config.DATA_DIR
    if not os.path.exists(data_dir):
        print("No data directory found")
        return
    
    status_files = [f for f in os.listdir(data_dir) if f.startswith('live_status_')]
    
    if not status_files:
        print("No live prediction systems found")
        return
    
    for status_file in status_files:
        ticker = status_file.replace('live_status_', '').replace('.json', '')
        
        try:
            with open(os.path.join(data_dir, status_file), 'r') as f:
                status = json.load(f)
            
            print(f"\nTicker: {ticker.upper()}")
            print(f"Model: {status.get('model_name', 'unknown')}")
            print(f"Last Update: {status.get('last_update', 'unknown')}")
            print(f"Predictions: {status.get('prediction_count', 0)}")
            print(f"Errors: {status.get('error_count', 0)}")
            print(f"Current Price: ${status.get('current_price', 0):.2f}")
            
            # Check if system is running (last update within 10 minutes)
            last_update = status.get('last_update')
            if last_update:
                try:
                    last_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    time_diff = datetime.now() - last_time.replace(tzinfo=None)
                    if time_diff < timedelta(minutes=10):
                        print("Status: 🟢 RUNNING")
                    else:
                        print("Status: 🔴 STOPPED")
                except:
                    print("Status: ❓ UNKNOWN")
            else:
                print("Status: ❓ UNKNOWN")
                
        except Exception as e:
            print(f"Error reading status for {ticker}: {e}")

def show_recent_signals(ticker, limit=10):
    """Show recent trading signals for a ticker."""
    print(f"Recent Signals for {ticker.upper()}")
    print("-" * 50)
    
    signals_file = os.path.join(Config.DATA_DIR, f"live_signals_{ticker}.json")
    
    if not os.path.exists(signals_file):
        print("No signals file found")
        return
    
    try:
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        
        if not signals:
            print("No signals found")
            return
        
        # Get recent signals
        recent_signals = signals[-limit:]
        
        for signal in recent_signals:
            timestamp = signal.get('timestamp', 'unknown')
            signal_type = signal.get('signal', 'unknown')
            confidence = signal.get('confidence', 0)
            price = signal.get('current_price', 0)
            
            emoji = "🟢" if signal_type == 'BUY' else "🔴" if signal_type == 'SELL' else "🟡"
            print(f"{emoji} {timestamp} - {signal_type} - ${price:.2f} ({confidence:.1%})")
            
    except Exception as e:
        print(f"Error reading signals: {e}")

def cleanup_old_files(days=30):
    """Clean up old log and data files."""
    print(f"Cleaning up files older than {days} days...")
    
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0
    
    # Clean logs
    logs_dir = Config.LOGS_DIR
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, file)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file}")
    
    print(f"Cleanup completed. Deleted {deleted_count} files.")

def backup_models():
    """Create a backup of all models."""
    print("Creating models backup...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"models_backup_{timestamp}"
    
    try:
        import shutil
        shutil.copytree(Config.MODELS_DIR, backup_dir)
        print(f"Backup created: {backup_dir}")
    except Exception as e:
        print(f"Backup failed: {e}")

def run_dashboard():
    """Start the dashboard."""
    print("Starting dashboard...")
    try:
        subprocess.run(["streamlit", "run", "dashboard/dashboard_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stock Trading System Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models command
    subparsers.add_parser('list-models', help='List available trained models')
    
    # Model info command
    model_info_parser = subparsers.add_parser('model-info', help='Show model information')
    model_info_parser.add_argument('model_name', help='Name of the model')
    
    # System status command
    subparsers.add_parser('status', help='Check system status')
    
    # Recent signals command
    signals_parser = subparsers.add_parser('signals', help='Show recent trading signals')
    signals_parser.add_argument('ticker', help='Stock ticker symbol')
    signals_parser.add_argument('--limit', type=int, default=10, help='Number of recent signals to show')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old files')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Files older than this many days will be deleted')
    
    # Backup command
    subparsers.add_parser('backup', help='Backup models')
    
    # Dashboard command
    subparsers.add_parser('dashboard', help='Start the dashboard')
    
    args = parser.parse_args()
    
    if args.command == 'list-models':
        list_models()
    elif args.command == 'model-info':
        show_model_info(args.model_name)
    elif args.command == 'status':
        check_system_status()
    elif args.command == 'signals':
        show_recent_signals(args.ticker, args.limit)
    elif args.command == 'cleanup':
        cleanup_old_files(args.days)
    elif args.command == 'backup':
        backup_models()
    elif args.command == 'dashboard':
        run_dashboard()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

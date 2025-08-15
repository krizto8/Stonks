#!/usr/bin/env python3
"""
Quick Start Script for Stock Trading System

This script provides a quick way to get started with the trading system.
It sets up the environment, trains a sample model, and starts the dashboard.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              🚀 STOCK TRADING SYSTEM QUICK START 🚀              ║
    ║                                                                  ║
    ║     Live Stock Price Pattern Recognition & Trading Signals       ║
    ║                    Powered by LSTM Neural Networks               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_command(command, description, show_output=False):
    """Run a command with error handling."""
    print(f"\n🔄 {description}...")
    try:
        if show_output:
            result = subprocess.run(command, shell=True, check=True)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are met."""
    print("\n🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if pip is available
    try:
        subprocess.run(["pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
    except:
        print("❌ pip is not available")
        return False
    
    return True

def setup_environment():
    """Set up the environment."""
    print("\n🛠️  Setting up environment...")
    
    # Create directories
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Set up environment file
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
    
    return True

def train_sample_model():
    """Train a sample model."""
    print("\n🧠 Training sample LSTM model...")
    print("This will train a model for AAPL stock with 1-hour intervals")
    print("Training may take several minutes depending on your hardware...")
    
    command = "python scripts/train_model.py --ticker AAPL --interval 1h --prediction_type regression"
    
    if run_command(command, "Training AAPL model", show_output=True):
        print("✅ Sample model trained successfully!")
        print("📁 Model saved as: AAPL_1h_regression_model")
        return True
    else:
        print("❌ Model training failed. You can try training manually later.")
        return False

def start_services():
    """Start the services."""
    print("\n🚀 Starting services...")
    
    # Ask user what they want to start
    print("\nWhat would you like to start?")
    print("1. Dashboard only")
    print("2. Live prediction + Dashboard")
    print("3. Nothing (manual start)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🎨 Starting dashboard...")
        print("Dashboard will open at: http://localhost:8501")
        time.sleep(2)
        subprocess.run(["streamlit", "run", "dashboard/dashboard_app.py"])
    
    elif choice == "2":
        print("\n📊 Starting live prediction service...")
        print("This will start monitoring AAPL and generating trading signals")
        
        # Start live prediction in background
        import threading
        
        def start_prediction():
            command = "python scripts/live_prediction.py --model_name AAPL_1h_regression_model --ticker AAPL"
            subprocess.run(command, shell=True)
        
        prediction_thread = threading.Thread(target=start_prediction)
        prediction_thread.daemon = True
        prediction_thread.start()
        
        print("✅ Live prediction started in background")
        time.sleep(3)
        
        print("\n🎨 Starting dashboard...")
        print("Dashboard will open at: http://localhost:8501")
        time.sleep(2)
        subprocess.run(["streamlit", "run", "dashboard/dashboard_app.py"])
    
    else:
        print("\n✅ Setup complete! You can start services manually:")
        print("\nTo start live prediction:")
        print("python scripts/live_prediction.py --model_name AAPL_1h_regression_model --ticker AAPL")
        print("\nTo start dashboard:")
        print("streamlit run dashboard/dashboard_app.py")

def main():
    """Main function."""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the Stonks directory")
        print("Make sure you have the complete project files")
        return
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please install Python 3.8+ and pip, then try again")
        return
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed!")
        return
    
    # Ask if user wants to train a model
    print("\n🤖 Would you like to train a sample model now?")
    print("This will download AAPL data and train an LSTM model (recommended for first-time users)")
    
    train_choice = input("Train sample model? (y/n): ").strip().lower()
    
    model_trained = False
    if train_choice in ['y', 'yes']:
        model_trained = train_sample_model()
    
    # Configuration reminder
    print("\n⚙️  Configuration:")
    print("• Edit .env file to add your API keys for alerts")
    print("• Telegram: Get bot token from @BotFather")
    print("• Email: Use app passwords for Gmail")
    
    # Start services
    if model_trained:
        start_services()
    else:
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Train a model: python scripts/train_model.py --ticker AAPL")
        print("2. Start dashboard: streamlit run dashboard/dashboard_app.py")
        print("3. Read README.md for detailed instructions")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user. You can run this script again anytime!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the README.md for manual setup instructions")

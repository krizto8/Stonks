# 📈 Stock Trading System with LSTM

Real-time stock pattern recognition using LSTM neural networks with an interactive dashboard for live trading signals.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start dashboard
streamlit run dashboard/dashboard_app.py 

Navigate to `http://localhost:8503`, enter any ticker (AAPL, GOOGL, etc.), and click **Start Prediction**.

## ✨ Features

- **Real-time Pattern Recognition**: LSTM model detects 4 chart patterns (Uptrend, Downtrend, Head-Shoulders, Double Bottom)
- **Interactive Dashboard**: Live charts, signals, and performance metrics
- **One-Click Predictions**: Auto-start live predictions for any ticker
- **High Accuracy**: 93-96% confidence on trading signals
- **Balanced Training**: SMOTE augmentation + Focal Loss for unbiased predictions

## 📁 Structure

```
Stonks/
├── dashboard/
│   └── dashboard_app_v2.py      # Main dashboard
├── scripts/
│   ├── live_prediction.py       # Real-time predictions
│   └── train_model.py           # Model training
├── utils/
│   ├── data_utils.py            # Data processing
│   └── model_utils.py           # LSTM model
├── models/
│   └── lstm_improved_*.h5       # Trained models
├── data/                        # Live signals & status
└── config/config.py             # Configuration
```

## 🤖 Model Architecture

```
LSTM (64 units) → LSTM (64 units) → Dense (25) → Output (4 classes)
- Input: 60 timesteps × 13 features
- Training: Focal Loss (γ=2.0) for class balance
- Dataset: 68,552 samples (17,138 per class, 0% duplicates)
```

## 💻 Usage

### Start Dashboard
```bash
streamlit run dashboard/dashboard_app.py 
```

### Manual Live Prediction
```bash
python scripts/live_prediction.py \
  --model_name lstm_improved_20251015_022146_focal \
  --ticker AAPL
```

### Train New Model
```bash
python train_model.py --epochs 50
```

## 📊 Signal Output

```json
{
  "signal": "BUY",
  "confidence": 0.9599,
  "pattern_predicted": "Double Bottom",
  "current_price": 232.85,
  "timestamp": "2025-10-16 01:15:22"
}
```

## 🛠️ Technical Details

**13 Technical Indicators:**
- OHLCV (Open, High, Low, Close, Volume)
- Returns, Volume Ratio, Price Ratios
- SMA 5/20, RSI, Volatility

**Key Improvements:**
- Fixed severe class bias (85% → 25% per class)
- SMOTE augmentation removes duplicate data
- Focal Loss handles remaining imbalance
- Production-ready error handling





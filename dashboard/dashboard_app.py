#!/usr/bin/env python3
"""
Live Stock Trading Dashboard

Interactive Streamlit dashboard for monitoring live predictions and trading signals.

Usage:
    streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os
import sys
import subprocess
import threading
import time
from datetime import datetime, timedelta
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_utils import DataFetcher, DataPreprocessor
from utils.model_utils import LSTMModel

# Page configuration
st.set_page_config(
    page_title="Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Dashboard:
    """Main dashboard class."""
    
    def __init__(self):
        self.data_fetcher = None
        self.data_preprocessor = DataPreprocessor()
        
    def start_live_prediction(self, ticker):
        """Start live prediction for a ticker in the background."""
        try:
            # Check if already running
            if self.is_live_prediction_running(ticker):
                return True
                
            # Command to start live prediction
            cmd = [
                sys.executable, 
                "scripts/live_prediction.py",
                "--model_name", "lstm_pattern_classifier", 
                "--ticker", ticker
            ]
            
            # Start in background
            subprocess.Popen(
                cmd, 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Give it a moment to start
            time.sleep(2)
            return True
            
        except Exception as e:
            st.error(f"Failed to start live prediction for {ticker}: {str(e)}")
            return False
    
    def is_live_prediction_running(self, ticker):
        """Check if live prediction is already running for a ticker."""
        status_file = os.path.join(Config.DATA_DIR, f"live_status_{ticker}.json")
        if not os.path.exists(status_file):
            return False
            
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            # Check if the last update was recent (within last 5 minutes)
            last_update = status.get('last_update')
            if last_update:
                last_update_time = pd.to_datetime(last_update)
                time_diff = (pd.Timestamp.now() - last_update_time).total_seconds()
                return time_diff < 300  # 5 minutes
                
        except Exception:
            return False
            
        return False
        
    def load_live_signals(self, ticker):
        """Load live signals from file."""
        try:
            signals_file = os.path.join(Config.DATA_DIR, f"live_signals_{ticker}.json")
            if os.path.exists(signals_file):
                with open(signals_file, 'r') as f:
                    signals = json.load(f)
                return pd.DataFrame(signals)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading signals: {str(e)}")
            return pd.DataFrame()
    
    def load_live_status(self, ticker):
        """Load live status from file."""
        try:
            status_file = os.path.join(Config.DATA_DIR, f"live_status_{ticker}.json")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                return status
            return {}
        except Exception as e:
            st.error(f"Error loading status: {str(e)}")
            return {}
    
    def load_historical_data(self, ticker, interval, period):
        """Load historical data for charts."""
        try:
            self.data_fetcher = DataFetcher(ticker=ticker, interval=interval)
            raw_data = self.data_fetcher.fetch_historical_data(period=period)
            processed_data = self.data_preprocessor.calculate_technical_indicators(raw_data)
            return processed_data
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()
    
    def create_candlestick_chart(self, data, signals_df):
        """Create candlestick chart with signals overlay."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Signals', 'Volume', 'Technical Indicators'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if all(col in data.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add trading signals
        if not signals_df.empty:
            for _, signal in signals_df.iterrows():
                try:
                    timestamp = pd.to_datetime(signal['timestamp'])
                    signal_type = signal['signal']
                    price = signal['current_price']
                    
                    color = 'green' if signal_type == 'BUY' else 'red' if signal_type == 'SELL' else 'yellow'
                    symbol = 'triangle-up' if signal_type == 'BUY' else 'triangle-down' if signal_type == 'SELL' else 'circle'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[timestamp],
                            y=[price],
                            mode='markers',
                            marker=dict(
                                symbol=symbol,
                                size=12,
                                color=color,
                                line=dict(color='black', width=1)
                            ),
                            name=f'{signal_type} Signal',
                            showlegend=False,
                            hovertemplate=f"<b>{signal_type}</b><br>" +
                                        f"Price: ${price:.2f}<br>" +
                                        f"Confidence: {signal['confidence']:.1%}<br>" +
                                        f"Time: {timestamp}<extra></extra>"
                        ),
                        row=1, col=1
                    )
                except Exception as e:
                    continue
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Technical indicators
        if 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title="Stock Price Analysis with Trading Signals",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        return fig
    
    def create_signals_table(self, signals_df):
        """Create signals summary table."""
        if signals_df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp descending
        signals_df = signals_df.sort_values('timestamp', ascending=False)
        
        # Format for display
        display_df = signals_df[['timestamp', 'signal', 'confidence', 'current_price']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert confidence to float and format as percentage
        display_df['confidence'] = pd.to_numeric(display_df['confidence'], errors='coerce')
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        
        display_df.columns = ['Timestamp', 'Signal', 'Confidence', 'Price']
        
        return display_df
    
    def run(self):
        """Run the dashboard."""
        st.title("📈 Live Stock Trading Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Configuration")
        
        # Stock selection
        ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
        
        # Time settings
        interval = st.sidebar.selectbox(
            "Data Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d"],
            index=2
        )
        
        period = st.sidebar.selectbox(
            "Historical Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=2
        )
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
        
        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            st.rerun()
        
        # Load data
        with st.spinner("Loading data..."):
            # Load historical data
            historical_data = self.load_historical_data(ticker, interval, period)
            
            # Load live signals
            signals_df = self.load_live_signals(ticker)
            
            # Load live status
            status = self.load_live_status(ticker)
            
            # Check if live prediction is running for this ticker
            if not status:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.warning(f"⚠️ No live prediction running for {ticker}")
                    
                with col2:
                    if st.button(f"🚀 Start Live Prediction for {ticker}", type="primary"):
                        with st.spinner(f"Starting live prediction for {ticker}..."):
                            if self.start_live_prediction(ticker):
                                st.success(f"✅ Live prediction started for {ticker}!")
                                st.info("Waiting for first prediction... (may take 1-2 minutes)")
                                time.sleep(3)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to start live prediction for {ticker}")
                
                # Show available tickers
                available_tickers = []
                try:
                    for file in os.listdir(Config.DATA_DIR):
                        if file.startswith("live_status_") and file.endswith(".json"):
                            ticker_name = file.replace("live_status_", "").replace(".json", "")
                            if self.is_live_prediction_running(ticker_name):
                                available_tickers.append(ticker_name)
                except:
                    pass
                
                if available_tickers:
                    st.info(f"💡 Currently running predictions: {', '.join(available_tickers)}")
                else:
                    st.info("💡 No live predictions currently running")
        
        # Main content
        if not historical_data.empty:
            # Current price and status
            current_price = historical_data['close'].iloc[-1]
            price_change = historical_data['close'].iloc[-1] - historical_data['close'].iloc[-2]
            price_change_pct = (price_change / historical_data['close'].iloc[-2]) * 100
            
            # Status indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=f"{ticker} Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                if status:
                    last_update = status.get('last_update', 'Unknown')
                    if isinstance(last_update, str):
                        try:
                            last_update = pd.to_datetime(last_update).strftime('%H:%M:%S')
                        except:
                            pass
                    st.metric("Last Update", last_update)
                else:
                    st.metric("Last Update", "No Data")
            
            with col3:
                if status:
                    prediction_count = status.get('prediction_count', 0)
                    error_count = status.get('error_count', 0)
                    error_rate = error_count / max(prediction_count, 1) * 100
                    st.metric("Error Rate", f"{error_rate:.1f}%")
                else:
                    st.metric("Error Rate", "N/A")
            
            with col4:
                if not signals_df.empty:
                    latest_signal = signals_df.iloc[-1]
                    signal_color = "🟢" if latest_signal['signal'] == 'BUY' else "🔴" if latest_signal['signal'] == 'SELL' else "🟡"
                    st.metric("Latest Signal", f"{signal_color} {latest_signal['signal']}")
                else:
                    st.metric("Latest Signal", "No Signals")
            
            # Main chart
            st.subheader("Price Chart & Trading Signals")
            chart = self.create_candlestick_chart(historical_data, signals_df)
            st.plotly_chart(chart, use_container_width=True)
            
            # Two columns for additional info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recent Trading Signals")
                signals_table = self.create_signals_table(signals_df.tail(10))
                if not signals_table.empty:
                    st.dataframe(signals_table, use_container_width=True)
                else:
                    st.info("No trading signals available")
            
            with col2:
                st.subheader("System Status")
                if status:
                    status_data = {
                        'Metric': ['Model Name', 'Prediction Count', 'Buffer Size', 'Last Prediction'],
                        'Value': [
                            str(status.get('model_name', 'N/A')),
                            str(status.get('prediction_count', 0)),
                            str(status.get('buffer_size', 0)),
                            str(status.get('last_prediction_time', 'N/A'))
                        ]
                    }
                    st.dataframe(pd.DataFrame(status_data), use_container_width=True)
                else:
                    st.info("No system status available")
            
            # Performance metrics
            if not signals_df.empty:
                st.subheader("Signal Performance")
                
                # Signal distribution
                signal_counts = signals_df['signal'].value_counts()
                fig_pie = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="Signal Distribution",
                    color_discrete_map={
                        'BUY': 'green',
                        'SELL': 'red',
                        'HOLD': 'yellow'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Confidence distribution
                fig_hist = px.histogram(
                    signals_df,
                    x='confidence',
                    nbins=20,
                    title="Signal Confidence Distribution",
                    labels={'confidence': 'Confidence Level', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.error("Failed to load historical data. Please check your configuration.")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

# Cryptocurrency Price Prediction System 📈

A complete, end-to-end time-series forecasting system that predicts cryptocurrency prices (specifically **Bitcoin - BTC-USD**) using deep learning.

## About the Project
This system analyzes historical market datasets alongside technical trading volume and momentum indicators to forecast future price trajectories. The predictive inference is powered by a PyTorch-based **Long Short-Term Memory (LSTM)** neural network natively integrated into a minimalist, interactive web interface.

## Core Features
1. **Data Collection**: Dynamically fetches the latest live and historical market data using the Yahoo Finance API (`yfinance`).
2. **Feature Engineering**: Computes advanced technical metrics such as **RSI, MACD, and Bollinger Bands** using the `ta` library to inject trading momentum and volatility context into the dataset.
3. **Deep Learning Engine**: Implements a robust 2-layer recurrent LSTM architecture optimized for sequential time-series extraction. It utilizes a `MinMaxScaler` across a 60-day sliding window.
4. **Interactive UI**: Contains a fully responsive Python Flask web dashboard, aesthetically styled with **Tailwind CSS** and leveraging **Chart.js** to visualize recent market trends alongside live, next-day neural network inferences.

## Technology Stack
- **Backend Model engine**: Python, PyTorch (LSTM API), scikit-learn
- **Data Engineering**: Pandas, NumPy, yfinance, ta
- **Web App Integration**: Flask
- **Frontend Dashboard**: HTML5, Tailwind CSS, Chart.js

## Getting Started

### Prerequisites
Make sure you have Python 3 installed. It's recommended to deploy this locally within a virtual environment.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rmuruganantham2004/crytocurrency.git
   cd crytocurrency
   ```
2. Activate a local virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application Structure
If you want to re-train the model weights from the ground up:
```bash
python src/train.py
```
*(This extracts the historical sequences, executes the EDA/Preprocessing, runs the Epochs through the LSTM, evaluates the MSE/MAE performance, and stores the `.pt` and `.pkl` artifacts internally).*

To load the server and access the interactive predictive dashboard:
```bash
python app.py
```
Then, open your preferred web browser and navigate to `http://localhost:5000/`.

# 📈 Stock Market Prediction Web App

A Flask-based web application that predicts stock market trends using ARIMA and LSTM models. It integrates Twitter sentiment analysis and leverages financial data from the Alpha Vantage API to enhance predictions.

---

## 🚀 Features

- 📊 **Time Series Prediction**: Uses ARIMA and LSTM models to forecast future stock prices.
- 🧠 **Sentiment Analysis**: Analyzes Twitter data using NLP to gauge market sentiment.
- 📈 **Alpha Vantage Integration**: Fetches real-time stock data via the Alpha Vantage API.
- 💬 **Tweepy & TextBlob**: Pulls tweets and evaluates sentiment polarity to influence predictions.
- 📉 **Visualization**: Displays historical vs predicted stock prices using Matplotlib.
- 🌐 **Web Interface**: Built with Flask for seamless interaction and result viewing.

---

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS (Jinja2 templating via Flask)
- **Backend**: Python (Flask)
- **Data & ML**:
  - ARIMA (statsmodels)
  - LSTM (TensorFlow/Keras)
  - Linear Regression (scikit-learn)
- **APIs & Libraries**:
  - Alpha Vantage
  - yFinance
  - Tweepy + TextBlob
  - NLTK + Regex
- **Visualization**: Matplotlib

---

## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/stock-market-prediction-web-app.git
   cd stock-market-prediction-web-app
2. **Add your API keys**

Create a constants.py file and include:
```
CONSUMER_KEY = "your_key"
CONSUMER_SECRET = "your_secret"
ACCESS_TOKEN = "your_token"
ACCESS_TOKEN_SECRET = "your_token_secret"
BEARER_TOKEN = "your_bearer_token"
NUM_OF_TWEETS = 100  # or any number
```
3. **Run the app**
```
python main.py``
```

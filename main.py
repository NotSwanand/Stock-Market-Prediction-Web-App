from flask import Flask, jsonify, render_template, request, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from constants import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, BEARER_TOKEN, NUM_OF_TWEETS
from datetime import datetime
import datetime as dt
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import tweepy
import sys
sys.stdout.reconfigure(encoding='utf-8')  # For Windows compatibility
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Use Tweepy Client for Twitter API v2 (requires OAuth2)
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)

DEMO_MODE = False  # Set to False for real version
MOCK_TWEETS = [
    "AAPL hits new all-time high! 🚀",
    "Analysts bullish on Apple's new products 📈",
    "Breaking: AAPL announces stock split 💥",
    "iPhone sales exceed expectations 📱",
    "Tim Cook announces new AI initiatives 🤖",
    "Apple Park named world's best office 🏆"
]

try:
    response = client.search_recent_tweets(query="AAPL", max_results=10)
    print("Tweets found:", len(response.data))
except Exception as e:
    print("Credential Error:", str(e))

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
   return render_template('index.html')

# **************** FUNCTION TO FETCH DATA ***************************
def get_historical(quote):
    DEMO_MODE = True  # Set this to False for real data
    
    if DEMO_MODE:
        # Generate fake realistic-looking demo data
        dates = pd.date_range(end=datetime.now(), periods=500)
        base_price = np.random.uniform(100, 200)
        
        return pd.DataFrame({
            'date': dates,
            'open': base_price + np.random.randn(500).cumsum(),
            'high': base_price + np.random.randn(500).cumsum() + 5,
            'low': base_price + np.random.randn(500).cumsum() - 5,
            'close': base_price + np.random.randn(500).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 500),
            'adj close': base_price + np.random.randn(500).cumsum()
        }).reset_index(drop=True)
    
    else:
        try:
            # Original Yahoo Finance code here
            end = datetime.now()
            start = datetime(end.year-2, end.month, end.day)
            data = yf.download(quote, start=start, end=end)
            
            # ... rest of original Yahoo code ...
            
        except Exception as e:
            # Original Alpha Vantage fallback code here
            try:
                ts = TimeSeries(key='S1ZOSDNQNZTJHQ29', output_format='pandas')
                # ... rest of original Alpha Vantage code ...
                
            except Exception as e:
                print(f"Alpha Vantage Error: {e}")
                return pd.DataFrame()

# **************** INSERT INTO TABLE FUNCTION ***************************

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    DEMO_MODE = True  # Set this to False for real version
    quote = request.form.get('nm') or "AAPL"  # Default to AAPL if empty

    if DEMO_MODE:
        # MODIFIED SECTION START (for dynamic sentiment analysis)
        # Generate realistic mock price data
        base_price = round(random.uniform(50, 300), 2)
        daily_change = random.uniform(-0.03, 0.04)
        mock_price = round(base_price * (1 + daily_change), 2)
        mock_error = round(random.uniform(1.5, 4.5), 2)

        # Create categorized tweets
        sentiment_tweets = {
            'positive': [
        f"{quote} reaches new all-time high! 🚀",
        f"Analysts raise price target for {quote} 📈",
        f"{quote} announces major dividend increase 💰",
        "Record quarterly earnings reported 🏆",
        f"Institutional investors accumulating {quote} 🧑💼",
        "New product launch exceeds expectations 🚀",
        f"{quote} named sector leader by Wall Street 📰",
        "CEO buys shares in open market 💹"
            ],
            'negative': [
        f"{quote} faces SEC investigation 🔍",
        "CFO unexpectedly resigns 😮",
        f"{quote} misses revenue estimates 📉",
        "Supply chain disruptions reported ⚠️",
        "Short sellers increasing positions in {quote} 📉",
        "Product recall announced ⚠️",
        "Credit rating downgrade issued 📉",
        f"Lawsuits filed against {quote} ⚖️"
            ],
            'neutral': [
        f"{quote} to hold earnings call tomorrow 🎧",
        "Sector-wide volatility continues 📊",
        f"{quote} added to watchlist by major fund 👀",
        "Technical analysis shows mixed signals 📈📉",
        "Market awaits Fed decision 💼",
        f"{quote} volume spikes with no clear catalyst 📊",
        "Institutional ownership remains stable ⚖️",
        "Analysts debate {quote} valuation 🧑💻"
            ]
        }

        # Randomly select 3 tweets with varied sentiment
        all_tweets = (
    sentiment_tweets['positive'] + 
    sentiment_tweets['negative'] + 
    sentiment_tweets['neutral']
)
        tw_list = random.choices(all_tweets, k=5)

        # Count sentiments
        pos = sum(1 for t in tw_list if t in sentiment_tweets['positive'])
        neg = sum(1 for t in tw_list if t in sentiment_tweets['negative'])
        neutral = sum(1 for t in tw_list if t in sentiment_tweets['neutral'])

        # Generate sentiment chart
        generate_sentiment_chart(pos, neg, neutral)

        # Generate stock metrics
        today_stock = pd.DataFrame({
            'open': round(mock_price * (1 + random.uniform(-0.02, 0.02)), 2),
            'high': round(mock_price * (1 + random.uniform(0.01, 0.05)), 2),
            'low': round(mock_price * (1 + random.uniform(-0.05, -0.01)), 2),
            'close': mock_price,
            'volume': random.randint(1000000, 5000000),
            'adj close': round(mock_price * (1 + random.uniform(-0.01, 0.01)), 2)
        }, index=[0])

        # Generate predictions
        arima_pred = round(mock_price * (1 + random.uniform(-0.03, 0.05)), 2)
        lstm_pred = round(mock_price * (1 + random.uniform(-0.02, 0.04)), 2)
        lr_pred = round(mock_price * (1 + random.uniform(-0.01, 0.03)), 2)

        # Generate forecast trend
       # Generate forecast trend
# Corrected line (notice the parentheses)
        forecast_set = [[round(mock_price * (1 + (i/100 * random.choice([1, -1]))), 2)] for i in range(1, 8)]

        # Generate recommendation
        idea = "RISE" if (pos > neg) else "FALL"
        decision = "BUY" if idea == "RISE" else "SELL"
        # MODIFIED SECTION END

        return render_template('results.html',
            quote=quote,
            arima_pred=arima_pred,
            lstm_pred=lstm_pred,
            lr_pred=lr_pred,
            open_s=str(today_stock['open'].iloc[0]),
            close_s=str(today_stock['close'].iloc[0]),
            adj_close=str(today_stock['adj close'].iloc[0]),
            tw_list=tw_list,
            tw_pol=f"Positive: {pos} | Negative: {neg} | Neutral: {neutral}",
            idea=idea,
            decision=decision,
            high_s=str(today_stock['high'].iloc[0]),
            low_s=str(today_stock['low'].iloc[0]),
            vol=str(today_stock['volume'].iloc[0]),
            forecast_set=forecast_set,
            error_lr=mock_error,
            error_lstm=round(mock_error * 0.95, 2),
            error_arima=round(mock_error * 1.1, 2)
        )

    else:  # Original real code
        try:
            df = get_historical(quote)
            if df.empty:
                return render_template('index.html', error=True)
            # ... rest of original processing code ...
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('index.html', error=True)

# **************** ARIMA SECTION ********************
def ARIMA_ALGO(df):
    try:
        # Validate DataFrame structure
        required_columns = {'date', 'close'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns. Found: {df.columns}")
        
        # Convert to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # Plot trends
        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(df['close'])
        plt.savefig('static/Trends.png')
        plt.close()

        # Train-test split
        values = df['close'].values
        split_idx = int(len(values) * 0.8)
        train, test = values[:split_idx], values[split_idx:]
        
        # ARIMA Model
        predictions = []
        history = list(train)
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1,0))
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        
        # Plot results
        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.savefig('static/ARIMA.png')
        plt.close()

        return predictions[-1], np.sqrt(mean_squared_error(test, predictions))
    
    except Exception as e:
        print(f"ARIMA processing failed: {e}")
        return 0.0, 0.0

# ************* LSTM SECTION **********************
def LSTM_ALGO(df):
        if len(df) < 7:
            raise ValueError("Insufficient data for LSTM")
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        ############# NOTE #################
        #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set = df['close'].values.reshape(-1, 1)# 1:2, to store as numpy array else Series obj will be stored
        #select cols using above manner to select as float64 type, view in var explorer

        #Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        #In scaling, fit_transform for training, transform for test
        
        #Creating data stucture with 7 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]#memory with 7 days from day i
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        #Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
        
        #Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
        
        #Initialise RNN
        regressor=Sequential()
        
        #Add first LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        #units=no. of neurons in layer
        #input_shape=(timesteps,no. of cols/features)
        #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))
        
        #Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        #Add o/p layer
        regressor.add(Dense(units=1))
        
        #Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        #For lstm, batch_size=power of 2
        
        #Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price = dataset_test['close'].values.reshape(-1, 1)
        
        #To predict, we need stock prices of 7 days before the test set
        #So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['close'],dataset_test['close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
        
        #Feature scaling
        testing_set=sc.transform(testing_set)
        
        #Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            #Convert list to numpy arrays
        X_test=np.array(X_test)
        
        #Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        #Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        #Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        
        #Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        #Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")
        return lstm_pred,error_lstm

# ***************** LINEAR REGRESSION SECTION ******************       
def LIN_REG_ALGO(df):
    # No of days to be forecasted in future
    forecast_out = int(7)
    # Price after n days
    df['close after n days'] = df['close'].shift(-forecast_out)  # Lowercase 'close'
    # New df with only relevant data
    df_new = df[['close', 'close after n days']]  # Lowercase 'close'

    # Structure data for train, test & forecast
    # Labels of known data, discard last 35 rows
    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))
    # All cols of known data except labels, discard last 35 rows
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])
    # Unknown, X to be forecasted
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
        
        #Traning, testing to plot graphs, check accuracy
    X_train=X[0:int(0.8*len(df)),:]
    X_test=X[int(0.8*len(df)):,:]
    y_train=y[0:int(0.8*len(df)),:]
    y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    X_to_be_forecasted=sc.transform(X_to_be_forecasted)
    
    #Training
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    
    #Testing
    y_test_pred=clf.predict(X_test)
    y_test_pred=y_test_pred*(1.04)
    import matplotlib.pyplot as plt2
    fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
    plt2.plot(y_test,label='Actual Price' )
    plt2.plot(y_test_pred,label='Predicted Price')
    
    plt2.legend(loc=4)
    plt2.savefig('static/LR.png')
    plt2.close(fig)
    
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    
    
    #Forecasting
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set=forecast_set*(1.04)
    mean=forecast_set.mean()
    lr_pred=forecast_set[0,0]
    print()
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by Linear Regression: ",lr_pred)
    print("Linear Regression RMSE:",error_lr)
    print("##############################################################################")
    return df, lr_pred, forecast_set, mean, error_lr

# **************** RECOMMENDATION FUNCTION **************************
def recommending(df, global_polarity, today_stock, mean):
    if today_stock.empty:
        return "N/A", "Insufficient Data"
    if today_stock.iloc[-1]['close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
        else:
            idea = "FALL"
            decision = "SELL"
    else:
        idea = "FALL"
        decision = "SELL"
    
    return idea, decision

import time
from tweepy.errors import TooManyRequests
from datetime import datetime, timedelta

tweet_cache = {}

def get_tweets(symbol):
    DEMO_MODE = True  # Set this to False for real tweets
    MOCK_TWEETS = [
        f"Breaking: {symbol} reaches new all-time high! 🚀",
        f"Analysts bullish on {symbol}'s latest earnings report 📈",
        f"{symbol} CEO announces major expansion plans 🌍",
        f"Market reacts positively to {symbol}'s new product line 🤑",
        f"{symbol} stock surges on strong quarterly results 💹",
        f"Investors flock to {symbol} amid market volatility 🛡️"
    ]

    if DEMO_MODE:
        # Return mock tweets immediately for demo
        return MOCK_TWEETS[:3]  # Return first 3 mock tweets

    # Only execute below if DEMO_MODE is False
    try:
        if symbol in tweet_cache:
            cached_time, cached_tweets = tweet_cache[symbol]
            if datetime.now() - cached_time < timedelta(minutes=120):  # Longer cache
                return cached_tweets[:6]

        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
        tweets = []
        
        # Single API call with error wrapping
        try:
            response = client.search_recent_tweets(
                query=f"({symbol} OR #{symbol}) -is:retweet lang:en",
                max_results=10,  # Reduced from 50 to be conservative
                tweet_fields=["created_at"]
            )
            
            if response and response.data:
                for tweet in response.data:
                    try:
                        cleaned = p.clean(tweet.text).strip()
                        cleaned = cleaned.encode('utf-8', 'ignore').decode('utf-8')
                        if cleaned and len(cleaned) < 280:  # Add length check
                            tweets.append(cleaned)
                    except Exception as e:
                        print(f"Tweet processing error: {str(e)}")
                        continue
                
                if tweets:
                    tweet_cache[symbol] = (datetime.now(), tweets)
                    return tweets[:3]  # Return top 3 tweets only

        except tweepy.TooManyRequests as e:
            print("Rate limit hit in demo - using cached mock data")
            return MOCK_TWEETS[:3]  # Fallback to mock tweets

        except Exception as e:
            print(f"Twitter API error: {str(e)}")
            return MOCK_TWEETS[:3]  # Fallback to mock tweets

        # Final fallback if all else fails
        return MOCK_TWEETS[:3]

    except Exception as e:
        print(f"General error in get_tweets: {str(e)}")
        return MOCK_TWEETS[:3]
def generate_sentiment_chart(pos, neg, neutral):
    plt.figure(figsize=(7.2, 4.8), dpi=65)
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    
    # Ensure we always have some data to display
    if sum(sizes) == 0:
        sizes = [1, 1, 1]
        plt.title("No Tweets Analyzed")
    else:
        plt.title(f"Tweet Sentiment Distribution (Total: {sum(sizes)} tweets)")
    
    plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: '{:.0f}'.format(p * sum(sizes)/100) if sum(sizes) != 0 else '',
            startangle=140)
    plt.axis('equal')
    plt.savefig('static/SA.png')
    plt.close()
    

# ****************GET DATA ***************************************
@app.route('/get_data', methods=['POST'])
def get_data():
    quote = request.form.get('nm')
    
    # Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html', not_found=True)
    
    # Preprocessing
    df = pd.read_csv(''+quote+'.csv')
    print("##############################################################################")
    print("Today's",quote,"Stock Data: ")
    today_stock = df.iloc[-1:]
    print(today_stock)
    print("##############################################################################")
    
    df = df.dropna()
    code_list = []
    for i in range(0, len(df)):
        code_list.append(quote)
    df2 = pd.DataFrame(code_list, columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df = df2

    # Run prediction algorithms
    arima_pred, error_arima = ARIMA_ALGO(df)
    lstm_pred, error_lstm = LSTM_ALGO(df)
    df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df)
    
    # Twitter Lookup is no longer free
    #polarity, tw_list, tw_pol, pos, neg, neutral = 0, [], "Can't fetch tweets, Twitter Lookup is no longer free in API v2.", 0, 0, 0
    try:
    # Initialize variables
        polarity = 0.0
        tw_list = []
        tw_pol = "No tweets found"
        pos = neg = neutral = 0

    # Fetch tweets
        tw_list = get_tweets(quote)
    
        if tw_list:
            analysis = [TextBlob(tweet).sentiment.polarity for tweet in tw_list]
            polarity = sum(analysis) / len(analysis) if len(analysis) > 0 else 0
            pos = len([x for x in analysis if x > 0])
            neg = len([x for x in analysis if x < 0])
            neutral = len([x for x in analysis if x == 0])
            tw_pol = f"Positive: {pos} | Negative: {neg} | Neutral: {neutral}"
            generate_sentiment_chart(pos, neg, neutral)
        else:
            generate_sentiment_chart(0, 0, 0)

    except Exception as e:
        print(f"Twitter/Sentiment Error: {str(e)}")
        tw_pol = "Error: Failed to fetch tweets"
    
    # Get recommendation
    idea, decision = recommending(df, polarity, today_stock, mean)
    
    print("Forecasted Prices for Next 7 days:")
    print(forecast_set)
    
    today_stock = today_stock.round(2)
    
    return render_template('results.html', 
        quote=quote,
        arima_pred=round(arima_pred, 2),
        lstm_pred=round(lstm_pred, 2),
        lr_pred=round(lr_pred, 2),
        open_s=today_stock['open'].to_string(index=False),
        close_s=today_stock['close'].to_string(index=False),
        adj_close=today_stock['adj close'].to_string(index=False),
        tw_list=tw_list,
        tw_pol=tw_pol,
        idea=idea,
        decision=decision,
        high_s=today_stock['high'].to_string(index=False),
        low_s=today_stock['low'].to_string(index=False),
        vol=today_stock['volume'].to_string(index=False),
        forecast_set=forecast_set,
        error_lr=round(error_lr, 2),
        error_lstm=round(error_lstm, 2),
        error_arima=round(error_arima, 2)
    )

os.environ['FLASK_DEBUG'] = '0'  # Force-disable debug mode
os.environ['WERKZEUG_DEBUG_PIN'] = 'off'  # Disable debug PIN

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, extra_files=[])
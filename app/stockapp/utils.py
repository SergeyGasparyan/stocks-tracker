import yfinance as yf
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from sklearn import preprocessing, model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def fetch_stock_data():
    data = yf.download(
        tickers=['AAPL', 'AMZN', 'GOOG', 'META', 'NVDA', 'MSFT'],
        group_by='ticker',
        threads=True, # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )
    data.reset_index(level=0, inplace=True)
    
    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['GOOG']['Adj Close'], name="GOOGL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['MSFT']['Adj Close'], name="MSFT")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    df1 = yf.download(tickers='AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers='AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers='GOOG', period='1d', interval='1d')
    df4 = yf.download(tickers='META', period='1d', interval='1d')
    df5 = yf.download(tickers='NVDA', period='1d', interval='1d')
    df6 = yf.download(tickers='MSFT', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "GOOG")
    df4.insert(0, "Ticker", "META")
    df5.insert(0, "Ticker", "NVDA")
    df6.insert(0, "Ticker", "MSFT")

    stock_df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    stock_df.reset_index(level=0, inplace=True)
    stock_df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    stock_df = stock_df.astype(convert_dict)
    stock_df.drop('Date', axis=1, inplace=True)

    return stock_df, plot_div_left


def ml_pipeline(df_ml, forecast_out):
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]

    # Reshape X into 3D
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_forecast = X_forecast.reshape((X_forecast.shape[0], 1, X_forecast.shape[1]))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Applying LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    confidence = model.evaluate(X_test, y_test, verbose=0)

    # Predicting for 'n' days stock data
    forecast_prediction = model.predict(X_forecast)
    forecast = forecast_prediction.ravel().tolist()

    return forecast, confidence

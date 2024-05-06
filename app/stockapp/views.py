from django.shortcuts import render
from  django.contrib import messages
from django.http import JsonResponse
from .utils import fetch_stock_data, ml_pipeline
from .ticker_names import Valid_Ticker
import json
import datetime as dt
import csv
import pandas as pd
import yfinance as yf
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter


# Create your views here.
def search_ticker(request):
    if request.method == 'POST':
        try:
            search_str = json.loads(request.body).get('searchText')
            data = []
            count = 0
            with open('data/new_tickers.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if search_str.lower() in row['Name'].lower():
                        data.append({'Symbol': row['Symbol'], 'Name': row['Name']})
                        count += 1
                    if count == 20:  # limit the number of rows to append to 20
                        break
            return JsonResponse(list(data), safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)})


def index(request):
    stock_df, plot_div_left = fetch_stock_data()
    json_records = stock_df.reset_index().to_json(orient='records')
    recent_stocks = json.loads(json_records)

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })


def search(request):
    return render(request, 'search.html', {})


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers=ticker_value, period='1d', interval='1m')
        print(f"Downloaded {ticker_value} ticker successfully")
    except:
        messages.error(request, 'We’re sorry for inconvinience, the API Server seems to be under maintainance.')
        return render(request, 'search.html', {})

    try:
        number_of_days = int(number_of_days)
    except:
        messages.error(request, 'Please enter valid number of days!')
        return render(request, 'search.html', {})
 
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')

    try:
        df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1h')
    except:
        df_ml = yf.download(tickers='AAPL', period='3mo', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)

    forecast, confidence = ml_pipeline(df_ml, forecast_out)
    
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')


    ticker = pd.read_csv('/home/serggasparyann/Projects/stocks-tracker/ml/data/tickers.csv')
    to_search = ticker_value
    ticker.columns = [
        'Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
        'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry'
    ]
    
    for i in range(ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break
    
    context = {
        'plot_div': plot_div, 
        'confidence': confidence,
        'forecast': forecast,
        'ticker_value': ticker_value,
        'number_of_days': number_of_days,
        'plot_div_pred': plot_div_pred,
        'Symbol': Symbol,
        'Name': Name,
        'Market_Cap': Market_Cap,
        'Country': Country,
        'IPO_Year': IPO_Year,
        'Sector': Sector,
        'Industry': Industry,
    }

    return render(request, "result.html", context)

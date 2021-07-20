import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from flask import Flask
from flask_ngrok import run_with_ngrok
import json

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
btc_df = yf.download('BTC-USD',start_date, today)

# Preprocessing the data
btc_df.reset_index(inplace=True)

df = btc_df[["Date", "Open"]]
new_names = {
    "Date": "ds", 
    "Open": "y",
}
df.rename(columns=new_names, inplace=True)

# Training the model using fbprophet
m = Prophet(
    seasonality_mode="multiplicative" 
)
m.fit(df)

# Using the model to predict the future values for periods=365 days
future = m.make_future_dataframe(periods = 365)
forecast=m.predict(future)

# Hosting the model on a Flask server as a REST API
app=Flask(__name__)
run_with_ngrok(app)

@app.route('/predict-btcusd')
def predict():
  next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
  return json.dumps({
      "prediction":'Bitcoin is projected to change by '+str(forecast[forecast['ds'] == next_day]['yhat'].item()-df['y'].iloc[-1])+' today, hitting an opening value of '+str(forecast[forecast['ds'] == next_day]['yhat'].item())+' tomorrow.'
  })
app.run()
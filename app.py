import onnxruntime as ort
import pickle
import joblib
import numpy as np
import json
import os
import requests
import pandas as pd
from flask import Flask,render_template,jsonify,request
import plotly.graph_objects as go
from livereload import Server
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
import plotly.io as pio
import ChatBot as cb

load_dotenv()
api=os.environ['API_KEY']
head={
    "x-cg-demo-api-key":api
}
params={
    "vs_currency":"usd",
    "days":1
}
lines=requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",params=params,headers=head).json()

def preprocess(lines):
    cols=["open_time","open","high","low","close"]
    data=pd.DataFrame(lines,columns=cols)
    data['open_time']=pd.to_datetime(data['open_time'],unit='ms')
    data['rsi']=RSIIndicator(close=data["close"],window=14,fillna=0).rsi()
    macd=MACD(close=data["close"])
    data["macd"]=macd.macd()
    data['macd_signal']=macd.macd_signal()
    data['ema_20']=EMAIndicator(close=data['close'],window=14,fillna=0).ema_indicator()
    data['atr']=AverageTrueRange(high=data['high'],low=data['low'],close=data['close'],window=14).average_true_range()
    data['weekday']=data['open_time'].dt.day_of_week
    data['day_of_year']=data['open_time'].dt.day_of_year
    data['hour']=data['open_time'].dt.hour
    data['month_end']=data['open_time'].dt.is_month_end.astype(np.float32)
    data['month_start']=data['open_time'].dt.is_month_start.astype(np.float32)
    data['quarter_start']=data['open_time'].dt.is_quarter_start.astype(np.float32)
    data["quarter_end"]=data['open_time'].dt.is_quarter_end.astype(np.float32)
    data["year"]=data['open_time'].dt.year
    data['month']=data['open_time'].dt.month
    data['is_recent']=data['open_time'].dt.year>2023
    data['is_recent']=data['is_recent'].astype(np.float64)
    data=data[data["open_time"].dt.minute==0].reset_index(drop=True)
    return data
data=preprocess(lines)

session=ort.InferenceSession("models/lstm_model.onnx")

with open("models/hourly_stat.pkl","rb") as f:
    hourly_model=pickle.load(f)

with open("models/daily_stat.pkl","rb") as f:
    daily_model=pickle.load(f)

input_scaler=joblib.load("models/input_scaler.pkl")

output_scaler=joblib.load("models/ouput_scaler.pkl")

with open("Data/news.json","r") as f:
    news=json.load(f)

with open("Data/portfolio.json","r") as f:
    portfolio=json.load(f)

def predict_onnx(data):
    ourdata=data.drop(columns=['open_time'])
    impfactor=np.linspace(0,1,15).reshape(15,1)
    scaled=input_scaler.transform(ourdata)
    X=np.append(scaled,impfactor,axis=1).reshape(1,15,20)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: X.astype(np.float32)})
    return pd.DataFrame({
        "ds":data[-15:]['open_time']+pd.Timedelta(hours=1),
        "x":output_scaler.inverse_transform(output[0][0].reshape(1,15)).reshape(15)
    })
def precict_ml(data):
    prediction=hourly_model.predict(data.rename(columns={"open_time":"ds","close":"previous_close"}).fillna(0))
    return pd.DataFrame({
        "ds":data['open_time']+pd.Timedelta(hours=1),
        "y":prediction['yhat']
    })
def plot_chart():
    temp=pd.DataFrame()
    batchsize=15
    for i in range(0,len(data),batchsize):
        input_data=data[len(data)-i-batchsize:len(data)-i].fillna(0)
        if len(input_data)<batchsize:
            prediction=predict_onnx(data[:15].fillna(0))[:-(batchsize-(len(data)-i))]
            temp=pd.concat([prediction,temp])
            break
        prediction=predict_onnx(input_data)
        temp=pd.concat([prediction,temp])
    predicted=temp
    mlprediction=precict_ml(data)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(
         x=data["open_time"],
         open=data["open"],
         high=data['high'],
         low=data['low'],
         close=data['close'],
         name="liveprice"
    ))
    fig.add_trace(go.Scatter(
         x=predicted['ds'],
         y=predicted['x'],
         name="lstm_prediction"
    ))
    fig.add_trace(go.Scatter(
        x=mlprediction['ds'],
        y=mlprediction['y'],
        name="MlPrediction"
    )
    )
    fig.update_layout(
        xaxis=dict(
            range=[data['open_time'].iloc[-15],data['open_time'].iloc[-1]],
            rangeslider=dict(visible=False),
            fixedrange=False
        ),
        paper_bgcolor='rgba(113, 78, 161, 0.5)',
        plot_bgcolor='rgba(113, 78, 161, 0.5)',
        font_color="white",
        dragmode="pan"
    )
    return fig

def Predict20Days():
    today=pd.to_datetime("today").normalize()
    timerange=pd.date_range(start=today,end=today+pd.Timedelta(days=20),freq="d")
    input_data=pd.DataFrame(timerange,columns=['ds'])
    pred=daily_model.predict(input_data)
    return {
        "ds":timerange,
        "y":pred['yhat']
    }

def make_20_days_chart():
    df=Predict20Days()
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y']
    ))
    fig.update_layout(
        paper_bgcolor='rgba(113, 78, 161, 0.5)',
        plot_bgcolor='rgba(113, 78, 161, 0.5)',
        font_color="white",
        dragmode="pan"
    )
    return fig

app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.route("/")
def front():
    figure=plot_chart()
    next20days=make_20_days_chart()
    config={
        "displayModeBar":False
    }
    plot=pio.to_html(figure,full_html=False,config=config)
    plot2=pio.to_html(next20days,full_html=False,config=config)
    return render_template("index.html",chart=plot,newsset=news,pred20=plot2)

@app.route("/update_chart")
def give_update():
    global lines
    global data
    newlines=requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",params=params,headers=head).json()
    lines.append(newlines)
    data=preprocess(lines)
    data=data.drop_duplicates()
    predonnx=predict_onnx(data.iloc[-15:].fillna(0))
    predml=precict_ml(data.iloc[[-1]])
    return jsonify({
        "x":[str(data['open_time'].iloc[-1]),str(predonnx['ds']),str(predml['ds'])],
        "open":float(data['open'].iloc[-1]),
        "high":float(data["high"].iloc[-1]),
        "low":float(data['low'].iloc[-1]),
        "close":float(data['close'].iloc[-1]),
        "y":[float(predonnx['x']),float(predml['y'])]
    })

@app.route("/get_response",methods=['POST'])
def chat():
    res=request.get_json()
    query=res['text']
    inputinvoke={
        "userinput":query,
        "movewhere":"",
        "aimessages":[],
        "query":"",
        "finalanswer":""
    }
    response=cb.graph.invoke(inputinvoke)
    return jsonify(response=response['finalanswer'])
if __name__=="__main__":
    server=Server(app.wsgi_app)
    server.watch("templates/*.html")
    server.serve(open_url_delay=True,port=8000)



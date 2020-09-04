#Import
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

## Definitions

# Visualisation

def plotly_candlestick(data, instrumentName)
    fig = go.Figure(data=[go.Candlestick(x=data['datetime'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'])])
    fig.update_layout(
        title= {
            'text': instrumentName,
        'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="#7f7f7f"
        )
        )
    fig.show()
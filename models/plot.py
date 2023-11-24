import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go


def linear_approx(x, start, end):
    
    # Calculate the length of the x array
    length = len(x)
    
    # Calculate the slope based on the start and end values
    slope = (end - start) / length
    
    # Create an empty array to store the predicted values
    prediction = np.zeros(length)
    
    # Loop through the array and fill it with the linear equation
    for i in range(length):
        prediction[i] = start + slope * i
    
    # Return the prediction array
    return prediction


# Assuming df is your DataFrame and it has been defined earlier
df = pd.read_excel('C:/Users/aruizr/OneDrive/9. Valor Ganado/data/processed/df_wbs_pr.xlsx')

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Tablero Valor Ganado'),
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=df['Fecha'].min(),
        max_date_allowed=df['Fecha'].max(),
        start_date=df['Fecha'].min(),
        end_date=df['Fecha'].max()
    ),
    dcc.Dropdown(
        id='wbs-dropdown',
        options=[{'label': i, 'value': i} for i in df['WBS'].unique()],
        value='3616_'
    ),
    dcc.Graph(id='time-series-graph')
])

# Define the callback
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)

def update_graph(start_date, end_date, wbs_value):
    mask = (df['Fecha'] >= start_date) & (df['Fecha'] <= end_date) & (df['WBS'] == wbs_value)
    data = df.loc[mask]
    
    # Filter the 'PV' series based on the selected 'WBS' value
    pv_data = df.loc[df['WBS'] == wbs_value]
    
    # Calculate the 'EACt' series from 'end_date' to the last date
    eact_data = df.loc[(df['Fecha'] >= end_date) & (df['WBS'] == wbs_value)].copy()
    eact_data['EACt'] = eact_data['AcAcum'].expanding().mean() * len(eact_data)
    
    # Adjust the 'EACt' series to end at the 'EAC' value of the selected end date
    adjustment_factor = eact_data.loc[eact_data.index[-1], 'EAC'] / eact_data.loc[eact_data.index[-1], 'EACt']
    eact_data['EACt'] *= adjustment_factor


    # Get the 'AcAcum' value at the end-date from the original DataFrame
    start_value = df.loc[(df['Fecha'] == end_date) & (df['WBS'] == wbs_value), 'AcAcum'].values[0]
    # Use the function to predict the 'EACt' values
    x_values = np.array(range(len(data), len(data) + len(eact_data)))
    # Get the 'EAC' value at the end-date from the original DataFrame
    end_value = df.loc[(df['Fecha'] == end_date) & (df['WBS'] == wbs_value), 'EAC'].values[0]
    eact_data['EACt'] = linear_approx(x_values, start_value, end_value)

       
    traces = [
        go.Scatter(
            x=data['Fecha'], 
            y=data['AcAcum'], 
            mode='lines', 
            name='AcAcum'
        ),
        go.Scatter(
            x=data['Fecha'], 
            y=data['EV'], 
            mode='lines', 
            name='EV'
        ),
        go.Scatter(
            x=pv_data['Fecha'], 
            y=pv_data['PV'], 
            mode='lines', 
            name='PV'
        ),
        go.Scatter(
            x=eact_data['Fecha'], 
            y=eact_data['EACt'], 
            mode='lines+markers', 
            line=dict(dash='dot'), 
            name='EACt'
        )
    ]
    
    return {'data': traces, 'layout': go.Layout(title='AcAcum, PV, EV, EACt over time')}


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_serve_dev_bundles=False)

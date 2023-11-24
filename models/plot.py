import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

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

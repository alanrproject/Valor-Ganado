import dash
from dash import dcc, html, dash_table
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import locale

# Define la función de aproximación lineal
def linear_approx(x, start, end, dfP, M):
    # Calculate the length of the x array
    #length = len(x)
    length = len(dfP)
    # Calculate the slope based on the start and end values
    #slope = (end - start) / length
    #if len(dfP)>0:
        #CanDia=(dfP.iloc[len(dfP)-1,8]-dfP.iloc[0,8]).days
        #CanDia=CanDia.astype(int)
        #if CanDia!=0:
            #slope = (end - start) / CanDia
        #else:
            #slope = 0.0
     ## convirtiendo a entero
    # Create an empty array to store the predicted values
    prediction = np.zeros(length)
    # Loop through the array and fill it with the linear equation
   
    #PorcTot=0.0
    for i in range(length):
        #CanDia=(dfP.iloc[i,8]-dfP.iloc[0,8]).days
            #prediction[i]=start+slope*CanDia
        
        if i==0:
            prediction[i] =dfP.iloc[0,M]
            ValAnt=prediction[i]
            Porc=0.0
        else:
            Porc=(dfP.iloc[i,13]-dfP.iloc[i-1,13])/(dfP.iloc[len(dfP)-1,13]-dfP.iloc[0,13])
            prediction[i] = ValAnt+(dfP.iloc[len(dfP)-1,M]-dfP.iloc[0,M])*Porc
            ValAnt=prediction[i]
        #PorcTot=PorcTot+Porc
    # Return the prediction array 
    return prediction

# Lee el dataframe base
df = pd.read_excel('models/df_wbs_pr.xlsx')
#df = pd.read_excel('/Users/ramonalzate/Downloads/9. Valor Ganado/data/processed/df_wbs_pr.xlsx')
app = dash.Dash(__name__)
server = app.server
n=0
max_char_length = 9
filtered_wbs_options = [{'label': i, 'value': i} for i in df['WBS'].unique() if len(i) <= max_char_length]

# Define the layout
app.layout = html.Div([
    html.H1(children='EARNED VALUE DASHBOARD',
            style={'textAlign': 'center','color': 'black','fontSize': 28,'fontFamily': 'Arial'}
            ),
    html.Div([
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=df['Fecha'].min(),
        max_date_allowed=df['Fecha'].max(),
        start_date=df['Fecha'].min(),
        end_date=df['Fecha'].max(),
        style={'font-family': 'Arial', 'font-size': '12px','width': '50%','margin': 'auto','height': '40px','borderRadius': '5px','border': '1px solid #ccc','padding': '4px'}
    ),
    dcc.Dropdown(
        id='wbs-dropdown',
        options=filtered_wbs_options,
        value='3616_',
        style={'font-family': 'Arial', 'font-size': '12px','width': '50%','margin': 'auto'}
    ),
    ], style={'width': '50%', 'margin': 'auto', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

    dash_table.DataTable(
        id='table',
        columns=[
            {'name': 'WBS', 'id': 'WBS'},
            {'name': 'Comienzo Plan', 'id': 'Comienzo'},
            {'name': 'Fin Plan', 'id': 'Fin'},
            {'name': 'BAC', 'id': 'LB Costo COP'},
            {'name': 'EAC', 'id': 'EAC'},
            {'name': 'Fin Real', 'id': 'FechaFin'},
        ],
        style_table={'height': '100px', 'overflowY': 'auto'},
        style_cell={
            'fontFamily': 'Arial',
            'fontSize': '12px',
            'textAlign': 'center',},
        style_header={'fontWeight': 'bold'},
    ),

    html.Div([
        dcc.Graph(id='time-series-graph', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='time-series-graph-avance', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'flex'}),
    
    html.Div([
        dcc.Graph(id='cpi-gauge', style={'width': '20%', 'display': 'inline-block','margin': '1'}),
        dcc.Graph(id='spi-gauge', style={'width': '20%', 'display': 'inline-block'}),
        dcc.Graph(id='acum-bar-chart', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'})
])

@app.callback(
    Output('table', 'data'),
    [Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)
def update_table(end_date, wbs_value):
    # Filtra los datos para la última fecha seleccionada y el WBS específico
    mask = (df['Fecha'] == end_date) & (df['WBS'] == wbs_value)
    filtered_data = df.loc[mask, ['WBS', 'Comienzo', 'Fin', 'EAC', 'FechaFin', 'LB Costo COP']]
    
    # Formatea las columnas de fecha
    filtered_data['Comienzo'] = pd.to_datetime(filtered_data['Comienzo']).dt.strftime('%Y-%m-%d')
    filtered_data['Fin'] = pd.to_datetime(filtered_data['Fin']).dt.strftime('%Y-%m-%d')
    filtered_data['FechaFin'] = pd.to_datetime(filtered_data['FechaFin']).dt.strftime('%Y-%m-%d')
    filtered_data['EAC'] = filtered_data['EAC'].apply(lambda x: '{:,.0f}'.format(x))
    filtered_data['LB Costo COP'] = filtered_data['LB Costo COP'].apply(lambda x: '{:,.0f}'.format(x))

    # Convierte los datos a formato de diccionario para la tabla
    table_data = filtered_data.to_dict('records')
    
    return table_data


# Define el callback para el gráfico de barras acumulativas
@app.callback(
    Output('acum-bar-chart', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)
def update_acum_bar_chart(start_date, end_date, wbs_value):
    # Filtra los datos por WBS y rango de fechas
    mask_wbs = df['WBS'].str.startswith(wbs_value)
    mask_date = (df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)
    data_filtered = df.loc[mask_wbs & mask_date]

    # Aplica el filtro adicional según la longitud y prefijo del Dropdown
    if wbs_value == '3616_':
        data_filtered = data_filtered[data_filtered['WBS'].str.len() == 9]
    elif wbs_value == '3616_C000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]
    elif wbs_value == '3616_P000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]
    elif wbs_value == '3616_G000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]
    elif wbs_value == '3616_S000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]
    elif wbs_value == '3616_M000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]
    elif wbs_value == '3616_E000':
        data_filtered = data_filtered[(data_filtered['AcAcum'] != 0)]

    # Limita la longitud máxima de caracteres en el eje X a 12
    data_filtered['WBS'] = data_filtered['WBS'].apply(lambda x: x[:12])

    # Crea el gráfico de barras
    fig = px.bar(data_filtered, x='WBS', y='AcAcum', color=data_filtered['WBS'].str[:9], title=f'AC POR WBS',
                 labels={'AcAcum': '', 'WBS': 'WBS'},)
    fig.update_traces(showlegend=False)
    fig.update_layout(
        plot_bgcolor='white',  # Fondo del área del gráfico
        paper_bgcolor='white',  # Fondo del papel o lienzo del gráfico
        title=dict(text='AC POR WBS', x=0.5,)
    )
    
    return fig


# Define la callback para el gráfico de medidor (gauge)
@app.callback(
    Output('cpi-gauge', 'figure'),
    [Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)
def update_cpi_gauge(end_date, wbs_value):
    # Filtra los datos por WBS y fecha
    mask = (df['WBS'] == wbs_value) & (df['Fecha'] == end_date)
    data_filtered = df.loc[mask]

    # Extrae el valor del CPI del DataFrame
    if not data_filtered.empty and 'CPI' in data_filtered.columns:
        cpi_value = data_filtered['CPI'].values[0]
    else:
        cpi_value = 1  # Puedes ajustar este valor por defecto si no hay datos

    # Crea el gráfico de medidor (gauge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cpi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CPI"},
        gauge={
            'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "gray",'thickness': 0.1,},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.9], 'color': 'red'},
                {'range': [0.9, 1.2], 'color': 'yellow'},
                {'range': [1.2, 2], 'color': 'green'},
            ],
        },
    ))

    fig.update_layout(title='')

    return fig

# Define la callback para el gráfico de medidor (gauge) del SPI
@app.callback(
    Output('spi-gauge', 'figure'),
    [Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)
def update_spi_gauge(end_date, wbs_value):
    mask = (df['WBS'] == wbs_value) & (df['Fecha'] == end_date)
    data_filtered = df.loc[mask]

    if not data_filtered.empty and 'SPI' in data_filtered.columns:
        spi_value = data_filtered['SPI'].values[0]
    else:
        spi_value = 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "SPI"},
        gauge={
            'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "gray", 'thickness': 0.1},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.9], 'color': 'red'},
                {'range': [0.9, 1.2], 'color': 'yellow'},
                {'range': [1.2, 2], 'color': 'green'},
            ],
        },
    ))

    fig.update_layout(title='')

    return fig

# Define the callback
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)

def update_graph(start_date, end_date, wbs_value):
    mask = (df['Fecha'] >= start_date) & (df['Fecha'] <= end_date) & (df['WBS'] == wbs_value)
    #mask1=(df['Fecha'] >end_date) & (df['Fecha'] <= end_date) & (df['WBS'] == wbs_value)
    data = df.loc[mask]
    N_end_date= data.loc[data.index[(len(data.index)-1)],'FechaFin']
    #data1=df.loc[mask1]

    
    

    # Filter the 'PV' series based on the selected 'WBS' value
    pv_data = df.loc[df['WBS'] == wbs_value]
    
    # Calculate the 'EACt' series from 'end_date' to the last date
    eact_data = df.loc[(df['Fecha'] >= end_date) & (df['Fecha'] <= N_end_date) & (df['WBS'] == wbs_value)].copy()
    if len(eact_data)>0:

    #Se ingresa un registro a eact_data con los datos del punto final donde debe terminar la proyeccion
        eact_data.loc[len(eact_data)]={}
        eact_data.iloc[len(eact_data)-1,8]=N_end_date
        eact_data.iloc[len(eact_data)-1,24]=N_end_date
        eact_data.iloc[len(eact_data)-1,12]=data.loc[data.index[(len(data.index)-1)],'EAC']
        eact_data.iloc[len(eact_data)-1,13]=data.loc[data.index[(len(data.index)-1)],'LB Costo COP']
        eact_data.iloc[len(eact_data)-1,14]=data.loc[data.index[(len(data.index)-1)],'LB Costo COP']
        eact_data.iloc[len(eact_data)-1,19]=data.loc[data.index[(len(data.index)-1)],'EAC']


    vectors_to_proyect = {'AcAcum':'EAC', 'EV':'LB Costo COP'}
    for key,value in vectors_to_proyect.items():
        if key =='AcAcum':
            M=12
            end_value = data.loc[data.index[(len(data.index)-1)],'EAC']
        else:
            M=14
            end_value = data.loc[data.index[(len(data.index)-1)],'LB Costo COP']
        start_value = df.loc[(df['Fecha'] == end_date) & (df['WBS'] == wbs_value), key].values[0]
        x_values = np.array(range(len(data), len(data) + len(eact_data)))
        print(x_values)
        eact_data[f'{value}t'] = linear_approx(x_values, start_value, end_value,eact_data,M)
    
    #_______________________
    # Grafica tabla
    last_date_data = data.loc[data['Fecha'] == end_date]
    table_data = last_date_data[['AC', 'EV', 'PV', 'EAC']].to_dict('records')

    traces = [
        go.Scatter(
            x=data['Fecha'], 
            y=data['AcAcum'], 
            mode='lines', 
            name='AC'
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
            name='EAC'
        ),
        go.Scatter(
            x=eact_data['Fecha'], 
            y=eact_data['LB Costo COPt'], 
            mode='lines+markers', 
            line=dict(dash='dot'), 
            name='VAC'
        )
    ]

    table = go.Table(
        header=dict(values=[ 'AC', 'EV', 'PV']),
        cells=dict(values=[
        '{:,.0f}'.format(last_date_data['AcAcum'].iloc[0] / 1000000),
        '{:,.0f}'.format(last_date_data['EV'].iloc[0] / 1000000),
        '{:,.0f}'.format(last_date_data['PV'].iloc[0] / 1000000)
    ]),
        visible=True,
        domain=dict(x=[0.1, 0.5], y=[0, 1]),
        columnwidth=[0.8, 0.8, 0.8]
    )
    layout = go.Layout(
        title='EARNED VALUE',
        showlegend=True,
        updatemenus=[
            # ... (si tienes botones de actualización)
        ],
        annotations=[dict(text='Valores', showarrow=False, x=0.5, y=1.15, xref='paper', yref='paper')],
        yaxis=dict(title='Valor',tickformat=',.0f'),
    )

    return {'data': traces+ [table], 'layout': go.Layout(title='EARNED VALUE')}

@app.callback(
    Output('time-series-graph-avance', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('wbs-dropdown', 'value')]
)
def update_graph_avance(start_date, end_date, wbs_value):
     # Filtra los datos por WBS
    mask_wbs = df['WBS'] == wbs_value
    data_wbs = df.loc[mask_wbs]
    # Obtén el último valor de AvancePlan
    last_avance_plan = data_wbs['AvancePlan'].iloc[-1]
    # Filtra los datos por rango de fechas
    mask_date = (df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)
    data_filtered = df.loc[mask_date & mask_wbs]

    # Gráfica para AvancePlan y AvanceReal
    traces_avance = [
        go.Scatter(
            x=data_wbs['Fecha'], 
            y=data_wbs['AvancePlan'], 
            mode='lines', 
            name='PLAN'
        ),
        go.Scatter(
            x=data_filtered['Fecha'], 
            y=data_filtered['AvanceReal'], 
            mode='lines', 
            name='REAL'
        )
    ]

    layout_avance = go.Layout(
        title='CURVA S',
        xaxis=dict(title=''),
        yaxis=dict(title='',tickformat='.0%',hoverformat='.0%'),
        showlegend=True,
        annotations=[
        {
            'x': end_date,  # Última fecha en el DataFrame
            'y': data_wbs['AvancePlan'].loc[data_wbs['Fecha'] == end_date].values[0],  # Valor correspondiente de AvancePlan en la última fecha
            'xref': 'x',
            'yref': 'y',
            'text': f'Plan: {data_wbs["AvancePlan"].loc[data_wbs["Fecha"] == end_date].values[0]:.1%}',
            'showarrow': True,
            'arrowhead': 4,
            'ax': 0,
            'ay': -20,
            'font': dict(color='blue', size=12)
        },
        {
            'x': end_date,  # Última fecha en el DataFrame
            'y': data_wbs['AvanceReal'].loc[data_wbs['Fecha'] == end_date].values[0],  # Valor correspondiente de AvanceReal en la última fecha
            'xref': 'x',
            'yref': 'y',
            'text': f'Real: {data_wbs["AvanceReal"].loc[data_wbs["Fecha"] == end_date].values[0]:.1%}',
            'showarrow': True,
            'arrowhead': 4,
            'ax': 0,
            'ay': 20,
            'font': dict(color='orange', size=12)
        },
    ]
)
    return {'data': traces_avance, 'layout': layout_avance}


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_serve_dev_bundles=False)

import pandas as pd
import numpy as np
import json



def ac_cleaner(folder, sheet):
     df = pd.read_excel(folder, sheet_name=sheet, skiprows=6)
     total = df['Valor'].sum()
     
     ## Borrar espacios en blanco de títulos de columna y datos
     df.columns = df.columns.str.replace(' ', '')
     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

     ## Organizar por campo fecha
     df['Fecha'] = pd.to_datetime(df['Fecha'])
     df = df.sort_values(by='Fecha')
     df.reset_index(inplace=True)
     df.drop('index', axis=1, inplace=True)


     ## reemplazar guiones
     df['Proyecto'] = df['Proyecto'].str.replace('-','_',case=False)

         
     with open('C:/Users/aruizr/OneDrive/9. Valor Ganado/data/raw/reglas.json', 'r', encoding='utf-8') as f:
          dict_replace = json.load(f)

     # Itera sobre el DataFrame
     for i, row in df.iterrows():
          key = str(row['Proyecto']) + '.' + str(row['ACTIVIDAD'])
          key = 1
          if key in dict_replace:
               df.at[i, 'WBS'] = dict_replace[key]
     
     df_grouped = df.groupby('Proveedor').agg({'Proveedor':'size','ACTIVIDAD':'first', 'Proyecto':'first','WBS':'first', 'Valor':'sum'})




     return total, df, df_grouped
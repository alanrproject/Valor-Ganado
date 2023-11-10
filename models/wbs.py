import pandas as pd
import numpy as np

def get_wbs(folder):
     
     # Importar dataframe del archivo de excel
     df = pd.read_excel(folder)
     
     # Filtrar por WBS hasta sub-actividades
     mask = df['WBS'].str.count('_') <= 2
     df = df.loc[mask]
     
     # Convertir columnas con datos de avance en filas del mismo tipo de dato
     df = df.melt(id_vars=df.columns[:6], var_name='Fecha', value_name='Avance')

     # Limpieza
     df['plan_or_real'] = df['Fecha'].str.startswith('Unnamed').map({True:None, False:'real'})
     df['plan_or_real'] = df['plan_or_real'].fillna('plan')
     mask = ~df['Fecha'].astype(str).str.startswith('Unnamed')
     df = df.loc[mask]
     df['Avance'] = df['Avance'].fillna(0)
     df['LB Costo COP'] = df['LB Costo COP'].fillna(0)

     return df, mask

import pandas as pd
import numpy as np

def ac_cleaner(folder,file_ac, sheet):

     ## Leer y limpiar base de datos del GP
     df = pd.read_excel(folder+file_ac, sheet_name=sheet, skiprows=6)
     total = df['Valor'].sum()
     df.columns = df.columns.str.replace(' ', '')
     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

     ## Organizar por campo fecha
     df['Fecha'] = pd.to_datetime(df['Fecha'])
     df = df.sort_values(by='Fecha')
     df.reset_index(inplace=True)
     df.drop('index', axis=1, inplace=True)

     ## reemplazar guiones
     df['Proyecto'] = df['Proyecto'].str.replace('-','_',case=False)
     df['Actividad'] = df['Proyecto'].str.split('_').str[1]
     df['Proyecto'] = df['Proyecto'].str.split('_').str[0]

     ## Leer y limpiar archivo de asignación de wbs    
     wbs_df = pd.read_excel(folder+'wbs_correct.xlsx', sheet_name='proveedor')
     wbs_df = wbs_df[['Proveedor','Proyecto','WBS_Corregida']]
     wbs_df['WBS_Corregida'] = wbs_df['WBS_Corregida'].astype(str).str.replace(' ', '')
     wbs_df['WBS_Corregida'] = wbs_df['WBS_Corregida'].str.replace(';', '_')
     wbs_df['WBS'] = wbs_df['WBS_Corregida'].str.split('_')

     ## Crear columnas de asignación wbs
     max_len = max(wbs_df['WBS'].apply(lambda x: len(x)))
     for i in range(max_len):
          wbs_df['WBS_'+ str(i)] = wbs_df.apply(lambda row: str(row['Proyecto']) + '_' + row['WBS'][i] if i < len(row['WBS']) else None, axis=1)
     
     ## Filtrar dataframe para dejar solo los campos necesarios
     wbs_df['Actividad'] = wbs_df['Proyecto'].str.split('_').str[1]
     wbs_df['Proyecto'] = wbs_df['Proyecto'].str.split('_').str[0]
     wbs_df = wbs_df[['Proveedor','Proyecto','Actividad','WBS','WBS_0','WBS_1','WBS_2']]

     ## merge de los dos dataframes
     df = df.merge(wbs_df, on=['Proveedor', 'Actividad', 'Proyecto'], how='left')
     df['WBS'] = df['WBS_y']
     df.drop(columns=['WBS_x','WBS_y'], inplace=True)

     ### dividir el valor en los casos en los que se deba asignar a más de una wbs

     # Create an empty list to store the new rows
     new_rows = []

     # Iterate over the rows in the original dataframe
     for index, row in df.iterrows():
     # Check if there's data in 'WBS_2'
          if pd.notna(row['WBS_2']):
               # Split the 'Valor' value by 3 and create two additional rows
               for i, wbs in enumerate(['WBS_0', 'WBS_1', 'WBS_2']):
                    row_copy = row.copy()
                    row_copy['Valor'] /= 3
                    row_copy['WBS_0'] = row[wbs]
                    new_rows.append(row_copy)
               # Delete the original row from df
               df.drop(index, inplace=True)
          elif pd.notna(row['WBS_1']):
               # Split the 'Valor' value by 2 and create one additional row
               for i, wbs in enumerate(['WBS_0', 'WBS_1']):
                    row_copy = row.copy()
                    row_copy['Valor'] /= 2
                    row_copy['WBS_0'] = row[wbs]
                    new_rows.append(row_copy)
               # Delete the original row from df
               df.drop(index, inplace=True)
          else:
               # If there's only data in 'WBS_0', keep the row as is
               new_rows.append(row)
               df.drop(index, inplace=True)

     # Create a new DataFrame from the list of new rows
     df_new = pd.DataFrame(new_rows, columns=df.columns)

     # Concatenate the original dataframe with the new rows
     df = pd.concat([df, df_new], ignore_index=True)
     df = df.iloc[:,:-3]
     df['wbs'] = df['WBS_0'].str.split('_').str[2]

     total_1 = df['Valor'].sum()
     
     return total, df
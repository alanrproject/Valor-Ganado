import pandas as pd
import numpy as np
from ac_cleaner import ac_cleaner
from wbs import get_wbs

folder = 'C:/Users/aruizr/OneDrive/9. Valor Ganado/data/raw/'
file_wbs = '3616 La Union.xlsx'
file_ac = 'AC.xlsx'
sheet = 'rpt_PROY_CON_ComprasGP'

def main():
     total_ac, df_ac = ac_cleaner(folder,file_ac, sheet)
     df_ac.to_excel('C:/Users/aruizr/OneDrive/9. Valor Ganado/data/processed/df_ac.xlsx', sheet_name='ac')
     df, mask = get_wbs(folder+file_wbs)
     df.to_excel('C:/Users/aruizr/OneDrive/9. Valor Ganado/data/processed/df_wbs.xlsx', sheet_name='wbs')
     print(total_ac)

if __name__ == "__main__":
     main()
     
     
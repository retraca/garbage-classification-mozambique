from data import Data
import pandas as pd


all_picture_data_path = 'main.csv'

with open(all_picture_data_path, 'r') as f:
    data = pd.read_csv(f)
    
for idx, row in data.iterrows():
    a= row.loc_dirty
    print(a)
    if a<=2:
        aux= 1
    elif a<=4 and a>2:
        aux= 2
    elif a<=6 and a>4:
        aux=3
    elif a<=8 and a>6:
        aux=4
    elif a<=10 and a>8:
        aux=5
    data.loc[idx, 'scaled_loc_dirty']=aux
    
data.to_csv('1000nvlr_crashed2200_scale_of_5.csv', index=False)
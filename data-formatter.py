import csv
import numpy as np
import pandas as pd

df = pd.read_csv('data/test_data_3.csv',
                 header=0, sep='\t', index_col=0)
df = df[['Eng_speed_PCM[rpm]', 'Engine_speed_TCM[rpm]', 'Eng_throttle_electronic_control_actual_PCM[deg]',
         'Eng_air_fuel_ratio_commanded_bank1_PCM[]', 'Eng_air_fuel_ratio_commanded_bank2_PCM[]']]

df['Eng_air_fuel_ratio_commanded_bank1_PCM[]'] = df['Eng_air_fuel_ratio_commanded_bank1_PCM[]']/14.7

#df = df.iloc[::2, :]  # downsampling

inputs = []
for i in range(2, 5800):  # shorter sample for testing
    inputs.append([df.iloc[i-1]['Eng_speed_PCM[rpm]'], df.iloc[i-2]['Eng_speed_PCM[rpm]'], df.iloc[i-1]['Eng_throttle_electronic_control_actual_PCM[deg]'], df.iloc[i-2]
                   ['Eng_throttle_electronic_control_actual_PCM[deg]'], df.iloc[i-1]['Eng_air_fuel_ratio_commanded_bank1_PCM[]'], df.iloc[i-2]['Eng_air_fuel_ratio_commanded_bank1_PCM[]']])

print(df.shape)

file = open('data/X_AR_3.csv', 'w+', newline='')
writer = csv.writer(file)
with file:
    writer.writerows(inputs)

labels = df['Eng_air_fuel_ratio_commanded_bank1_PCM[]']
labels = labels.head(5800)
labels.to_csv('data/y_AR_3.csv', header=False, index=False, sep=',')
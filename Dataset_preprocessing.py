import pandas as pd

# Načítanie datasetu
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)

# Zistenie hodnôt pre každú premennú
values = {column: df[column].unique() for column in df.columns}

# Zobrazenie hodnôt
for col, val in values.items():
    print(f"{col}: {val}\n")

# Mapovanie hodnôt pre textové premenné
mapping_dict = {
    'yes': 1,
    'no': 0,
    'Female': 1,
    'Male': 0,
    'Sometimes': 2,
    'Frequently': 3,
    'Always': 4,
    'Public_Transportation': 0,
    'Walking': 1,
    'Automobile': 2,
    'Motorbike': 3,
    'Bike': 4,
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6,
}

# Použitie mapovania na všetky stĺpce, kde je to potrebné
df_numeric = df.map(lambda x: mapping_dict[x] if x in mapping_dict else x)

# Uloženie nového datasetu do súboru
output_file_path = 'obesity_dataset.csv'
df_numeric.to_csv(output_file_path, index=False)

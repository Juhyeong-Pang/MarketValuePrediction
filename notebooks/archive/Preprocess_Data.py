#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import requests
import json
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler

import re


# In[5]:


squad_rank_map = {
    # [21-22, 22-23, 23-24, 24-25]
    'Ajaccio': [22, 18, 22, 22], 'Alavés': [20, 22, 10, 12], 'Almería': [22, 17, 19, 22], 'Angers': [14, 20, 22, 18],
    'Arminia': [17, 22, 22, 22], 'Arsenal': [5, 2, 2, 2], 'Aston Villa': [14, 7, 4, 5], 'Atalanta': [8, 5, 4, 3],
    'Athletic Club': [8, 8, 5, 6], 'Atlético Madrid': [3, 3, 4, 3], 'Augsburg': [14, 15, 11, 13], 'Auxerre': [22, 17, 22, 15],
    'Barcelona': [2, 1, 2, 1], 'Bayern Munich': [1, 1, 3, 1], 'Bochum': [13, 14, 16, 17], 'Bologna': [13, 9, 5, 8],
    'Bordeaux': [20, 22, 22, 22],'Bournemouth': [22, 15, 12, 11],'Brentford': [13, 9, 16, 14],'Brest': [11, 14, 3, 9],
    'Brighton': [9, 6, 11, 10],'Burnley': [18, 22, 19, 22],'Cagliari': [18, 22, 16, 16],'Celta Vigo': [11, 13, 14, 11],
    'Chelsea': [3, 12, 6, 4],'Clermont Foot': [17, 8, 18, 22],'Cremonese': [22, 19, 22, 22],'Crystal Palace': [12, 11, 10, 12],
    'Cádiz': [17, 14, 18, 22],'Darmstadt 98': [22, 22, 18, 22],'Dortmund': [2, 2, 5, 4],'Eintracht Frankfurt': [11, 7, 6, 6],
    'Elche': [13, 20, 22, 22],'Empoli': [14, 14, 17, 14],'Espanyol': [14, 19, 22, 17],'Everton': [16, 17, 15, 15],
    'Fiorentina': [7, 8, 8, 7], 'Freiburg': [6, 5, 10, 8], 'Frosinone': [22, 22, 18, 22],
    'Fulham': [22, 10, 13, 12], 'Genoa': [19, 22, 11, 13], 'Getafe': [15, 15, 12, 14], 'Girona': [22, 10, 3, 7],
    'Gladbach': [10, 10, 14, 11], 'Granada': [18, 22, 20, 22], 'Greuther Fürth': [18, 22, 22, 22], 'Heidenheim': [22, 22, 8, 12],
    'Hellas Verona': [9, 18, 13, 15], 'Hertha BSC': [16, 18, 22, 22], 'Hoffenheim': [9, 12, 7, 10], 'Inter': [2, 3, 1, 2],
    'Juventus': [4, 7, 3, 3], 'Köln': [7, 11, 17, 22], 'Las Palmas': [22, 22, 16, 17], 'Lazio': [5, 2, 7, 6],
    'Le Havre': [22, 22, 15, 16], 'Lecce': [22, 16, 14, 15], 'Leeds United': [17, 19, 22, 22], 'Leicester City': [8, 18, 22, 16],
    'Lens': [7, 2, 7, 8], 'Levante': [19, 22, 22, 22], 'Leverkusen': [3, 6, 1, 3], 'Lille': [10, 5, 4, 5],
    'Liverpool': [2, 5, 3, 1], 'Lorient': [16, 10, 19, 22], 'Luton Town': [22, 22, 18, 22], 'Lyon': [8, 7, 6, 6],
    'Mainz 05': [8, 9, 13, 12], 'Mallorca': [16, 9, 15, 13], 'Manchester City': [1, 1, 1, 2], 'Manchester Utd': [6, 3, 8, 7],
    'Marseille': [2, 3, 8, 5], 'Metz': [19, 22, 18, 22], 'Milan': [1, 4, 2, 4], 'Monaco': [3, 6, 2, 2],
    'Montpellier': [13, 12, 12, 14], 'Monza': [22, 11, 12, 15], 'Nantes': [9, 16, 14, 16], 'Napoli': [3, 1, 10, 4],
    'Newcastle United': [11, 4, 7, 6], 'Nice': [5, 9, 5, 6], 'Norwich City': [20, 22, 22, 22], 'Nottingham Forest': [22, 16, 17, 14],
    'Osasuna': [10, 7, 11, 10], 'Paris Saint-Germain': [1, 1, 1, 1], 'RB Leipzig': [4, 3, 4, 4], 'Rayo Vallecano': [12, 11, 17, 15],
    'Real Betis': [5, 6, 7, 7], 'Real Madrid': [1, 2, 1, 2], 'Real Sociedad': [6, 4, 6, 8], 'Reims': [12, 11, 9, 10],
    'Rennes': [4, 4, 10, 9], 'Roma': [6, 6, 6, 7], 'Saint-Étienne': [18, 22, 22, 16], 'Salernitana': [17, 15, 20, 22],
    'Sampdoria': [15, 20, 22, 22], 'Sassuolo': [11, 13, 19, 22], 'Schalke 04': [22, 17, 22, 22], 'Sevilla': [4, 12, 14, 12],
    'Sheffield United': [22, 22, 20, 22], 'Southampton': [15, 20, 22, 19], 'Spezia': [16, 18, 22, 22],'Strasbourg': [6, 15, 13, 13],
    'Stuttgart': [15, 16, 2, 7],'Torino': [10, 10, 9, 11], 'Tottenham Hotspur': [4, 8, 5, 5],'Toulouse': [22, 13, 11, 12],
    'Troyes': [15, 19, 22, 22],'Udinese': [12, 12, 15, 13], 'Union Berlin': [5, 4, 15, 11],'Valencia': [9, 14, 9, 13],
    'Valladolid': [22, 18, 22, 19],'Venezia': [20, 22, 22, 20], 'Villarreal': [7, 5, 8, 6],'Watford': [19, 22, 22, 22],
    'Werder Bremen': [22, 13, 9, 10],'West Ham United': [7, 14, 9, 11], 'Wolfsburg': [12, 8, 12, 12],'Wolves': [10, 13, 14, 15],
}

def get_squad_rank_map(year, full_map=squad_rank_map):
    index = year - 2021
    if (index < 0 or index >= len(full_map)):
        print("No Data For That Year")
        return

    temp_dict = {}
    for squad in full_map:
        temp_dict[squad] = full_map[squad][index]

    return temp_dict


# In[8]:


def apply_squad_rank(df, year):
    df = df.copy()
    rank_map_for_year = get_squad_rank_map(year)

    if rank_map_for_year is None:
        return df

    df['Squad_Rank'] = df['Unnamed: 4_level_0_Squad'].map(rank_map_for_year).apply(lambda x: 21 - x if pd.notnull(x) else 0)
    df = df.drop(columns=['Unnamed: 4_level_0_Squad'])

    return df

def clean_dataframe(df):
    df = df.dropna(subset = ["Value (€)"])
    df = df.drop(columns = ["Name_Clean", 
                            "Unnamed: 0_y", 
                            "Name_y", 
                            "Unnamed: 0_x", 
                            "Unnamed: 0_level_0_Rk", 
                            "Unnamed: 24_level_0_Matches",
                            "Unnamed: 0"
                           ])
    df = df.rename(columns={
        'Unnamed: 3_level_0_Pos': 'Position', 
        'Unnamed: 5_level_0_Age': 'Age',
        'Unnamed: 6_level_0_Born': 'Born',
        'Unnamed: 2_level_0_Nation': 'Nationality',
        'Playing Time_MP': 'Match Played',
        'Name_x': 'Name',
        'Playing Time_Starts': 'Match Started',
        'Playing Time_Min': 'Minutes Played',
        'Playing Time_90s': 'Minutes Played / 90',
        'Performance_Gls': 'Goals',
        'Performance_Ast': 'Assists',
        'Performance_G+A': 'Goals + Assists',
        'Performance_G-PK': 'Non-Penality Goals',
        'Performance_PK': 'Penalty Kick Goals',
        'Performance_PKatt': 'Penalty Kick Attempted',
        'Performance_CrdY': 'Yellow Cards',
        'Performance_CrdR': 'Red Cards',
        'Per 90 Minutes_Gls': 'Goals Per 90 Minutes',
        'Per 90 Minutes_Ast': 'Assists Per 90 Minutes',
        'Per 90 Minutes_G+A': 'G+A Per 90 Minutes',
        'Per 90 Minutes_G-PK': 'Non-Penality Goals Per 90 Minutes',
        'Per 90 Minutes_G+A-PK': 'Non-Penalty Goals + Assists/90',
        'Value (€)': 'Value',
        # '': '',
        # '': '',
        # '': '',
        # '': '',
        # '': '',
        # '': '',
    })

    intger_column_list = ["Age", "Born", "Match Played", "Match Started", "Minutes Played", "Goals", "Assists", "Goals + Assists", 
                  "Non-Penality Goals", "Penalty Kick Goals", "Penalty Kick Attempted", "Yellow Cards", "Red Cards"]

    float_column_list = ["Minutes Played / 90", "Goals Per 90 Minutes", "Assists Per 90 Minutes", 
                  "G+A Per 90 Minutes", "Non-Penality Goals Per 90 Minutes", "Non-Penalty Goals + Assists/90"]

    numeric_column_list = []
    numeric_column_list.extend(intger_column_list)
    numeric_column_list.extend(float_column_list)

    for column in numeric_column_list:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

def preprocess_dataframe(df):
    df = df.copy()

    df_rest = df.drop(columns=["Value"])

    scaler = StandardScaler()
    numeric_cols = df_rest.select_dtypes(include=['float64', 'int64']).columns
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    position_dummies = df['Position'].str.get_dummies(sep=',')
    df = pd.concat([df, position_dummies], axis=1)   
    df = df.set_index('Name')
    df = df.drop(columns = ["Nationality", "Born", "Position"])

    df = df[[c for c in df.columns if c != "Value"] + ["Value"]]
    return df, scaler

def end_to_end_process_data(df, year):
    df = apply_squad_rank(df, year)
    df = clean_dataframe(df)
    df, _ = preprocess_dataframe(df)
    return df

def end_to_end_load_data(filename):
    df = pd.read_csv(filename)
    year = int("".join(re.findall(r'\d+', filename)))

    df = end_to_end_process_data(df, year)

    return df


# In[10]:


folder_path = "Datasets"

test_2024 = end_to_end_load_data(os.path.join(folder_path, "test_2024.csv"))
train_2023 = end_to_end_load_data(os.path.join(folder_path, "train_2023.csv"))
train_2022 = end_to_end_load_data(os.path.join(folder_path, "train_2022.csv"))
train_2021 = end_to_end_load_data(os.path.join(folder_path, "train_2021.csv"))


# In[10]:


df = pd.concat([train_2023,train_2022,train_2021], axis=0)


# In[11]:


df.head(20)


# In[12]:


from sklearn.model_selection import train_test_split

X, y = df.drop(columns = ["Value"]), df["Value"]
X_train, _, y_train, _ = train_test_split(X, y, random_state=42)

X_test, y_test = test_2024.drop(columns = ["Value"]), test_2024["Value"]


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# model = LinearRegression()
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = model.score(X_test, y_test) 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared (R2) Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

### LinearRegression:

R-squared (R2) Score: 0.43060017234650616
Mean Absolute Error (MAE): 9686289.349116301
Mean Squared Error (MSE): 254330624533508.6
Root Mean Squared Error (RMSE): 15947746.691414075
# In[14]:


correlation_matrix = df.corr()

plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1)
plt.show()


# In[15]:


comp = pd.DataFrame({
    "true": y_test,
    "pred": y_pred
})
comp["error"] = comp["true"] - comp["pred"] # -x : 과대평가, +x : 과소평가

comp['true'] = comp['true'].apply(lambda x: f"{x:.6e}")


# In[16]:


comp.sort_values(by="error")


# In[17]:


test_2024.loc["Mika Biereth"]


# In[18]:


round(pd.DataFrame(comp["error"]).describe(), 2)


# In[19]:


round(pd.DataFrame(y_test).describe(), 2)


# In[20]:


round(pd.DataFrame(y_pred).describe(), 2)


# In[21]:


round(df.describe(), 2)


# In[22]:


test_2024_raw = pd.read_csv("test_2024.csv")


# In[23]:


test_2024_raw[test_2024_raw["Name_x"] == "Mika Biereth"]


# In[24]:


from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

models = {
    # "Baseline (Mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression": linear_model.Ridge(alpha=1.0),
    "Lasso Regression": linear_model.Lasso(alpha=0.001),
    "ElasticNet": linear_model.ElasticNet(alpha=0.001, l1_ratio=0.5),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.1)
}

models_evaluation = {}
for model_name in models:
    model = models[model_name]
    model_info = {}

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_info["y_pred"] = y_pred

    r2 = model.score(X_test, y_test) 
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    model_info["metrics"] = [r2, mae, mse, rmse]

    models_evaluation[model_name] = model_info


# In[25]:


def print_models_evalutation(models_evaluation):

    for model in models_evaluation:
        metrics = models_evaluation[model]["metrics"]
        y_pred = models_evaluation[model]["y_pred"]

        print(f"{model} : ")


        r2, mae, mse, rmse = metrics[0], metrics[1], metrics[2], metrics[3]

        plt.figure(figsize=(10, 5))
        sns.histplot(y_test, kde=True, color='skyblue', label='Actual (y_test)', alpha=0.5)
        sns.histplot(y_pred, kde=True, color='red', label=f'Predicted ({model})', alpha=0.5)
        plt.title(f'Comparison of Actual vs Predicted Distribution ({model})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()

        print(f"    - R2: {r2}")
        print(f"    - MAE: {mae}")
        print(f"    - MSE: {mse}")
        print(f"    - RMSE: {rmse}")
        print()


# In[26]:


print_models_evalutation(models_evaluation)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





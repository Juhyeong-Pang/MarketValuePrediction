import os
import re
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder


FOLDER_PATH = os.path.join("..", "data")
SCALER_DICT_PATH = os.path.join(FOLDER_PATH, "scalers.joblib")

SQUAD_RANK_MAP = {
    # [21-22, 22-23, 23-24, 24-25]
    "Ajaccio": [22, 18, 22, 22],
    "Alavés": [20, 22, 10, 12],
    "Almería": [22, 17, 19, 22],
    "Angers": [14, 20, 22, 18],
    "Arminia": [17, 22, 22, 22],
    "Arsenal": [5, 2, 2, 2],
    "Aston Villa": [14, 7, 4, 5],
    "Atalanta": [8, 5, 4, 3],
    "Athletic Club": [8, 8, 5, 6],
    "Atlético Madrid": [3, 3, 4, 3],
    "Augsburg": [14, 15, 11, 13],
    "Auxerre": [22, 17, 22, 15],
    "Barcelona": [2, 1, 2, 1],
    "Bayern Munich": [1, 1, 3, 1],
    "Bochum": [13, 14, 16, 17],
    "Bologna": [13, 9, 5, 8],
    "Bordeaux": [20, 22, 22, 22],
    "Bournemouth": [22, 15, 12, 11],
    "Brentford": [13, 9, 16, 14],
    "Brest": [11, 14, 3, 9],
    "Brighton": [9, 6, 11, 10],
    "Burnley": [18, 22, 19, 22],
    "Cagliari": [18, 22, 16, 16],
    "Celta Vigo": [11, 13, 14, 11],
    "Chelsea": [3, 12, 6, 4],
    "Clermont Foot": [17, 8, 18, 22],
    "Cremonese": [22, 19, 22, 22],
    "Crystal Palace": [12, 11, 10, 12],
    "Cádiz": [17, 14, 18, 22],
    "Darmstadt 98": [22, 22, 18, 22],
    "Dortmund": [2, 2, 5, 4],
    "Eintracht Frankfurt": [11, 7, 6, 6],
    "Elche": [13, 20, 22, 22],
    "Empoli": [14, 14, 17, 14],
    "Espanyol": [14, 19, 22, 17],
    "Everton": [16, 17, 15, 15],
    "Fiorentina": [7, 8, 8, 7],
    "Freiburg": [6, 5, 10, 8],
    "Frosinone": [22, 22, 18, 22],
    "Fulham": [22, 10, 13, 12],
    "Genoa": [19, 22, 11, 13],
    "Getafe": [15, 15, 12, 14],
    "Girona": [22, 10, 3, 7],
    "Gladbach": [10, 10, 14, 11],
    "Granada": [18, 22, 20, 22],
    "Greuther Fürth": [18, 22, 22, 22],
    "Heidenheim": [22, 22, 8, 12],
    "Hellas Verona": [9, 18, 13, 15],
    "Hertha BSC": [16, 18, 22, 22],
    "Hoffenheim": [9, 12, 7, 10],
    "Inter": [2, 3, 1, 2],
    "Juventus": [4, 7, 3, 3],
    "Köln": [7, 11, 17, 22],
    "Las Palmas": [22, 22, 16, 17],
    "Lazio": [5, 2, 7, 6],
    "Le Havre": [22, 22, 15, 16],
    "Lecce": [22, 16, 14, 15],
    "Leeds United": [17, 19, 22, 22],
    "Leicester City": [8, 18, 22, 16],
    "Lens": [7, 2, 7, 8],
    "Levante": [19, 22, 22, 22],
    "Leverkusen": [3, 6, 1, 3],
    "Lille": [10, 5, 4, 5],
    "Liverpool": [2, 5, 3, 1],
    "Lorient": [16, 10, 19, 22],
    "Luton Town": [22, 22, 18, 22],
    "Lyon": [8, 7, 6, 6],
    "Mainz 05": [8, 9, 13, 12],
    "Mallorca": [16, 9, 15, 13],
    "Manchester City": [1, 1, 1, 2],
    "Manchester Utd": [6, 3, 8, 7],
    "Marseille": [2, 3, 8, 5],
    "Metz": [19, 22, 18, 22],
    "Milan": [1, 4, 2, 4],
    "Monaco": [3, 6, 2, 2],
    "Montpellier": [13, 12, 12, 14],
    "Monza": [22, 11, 12, 15],
    "Nantes": [9, 16, 14, 16],
    "Napoli": [3, 1, 10, 4],
    "Newcastle United": [11, 4, 7, 6],
    "Nice": [5, 9, 5, 6],
    "Norwich City": [20, 22, 22, 22],
    "Nottingham Forest": [22, 16, 17, 14],
    "Osasuna": [10, 7, 11, 10],
    "Paris Saint-Germain": [1, 1, 1, 1],
    "RB Leipzig": [4, 3, 4, 4],
    "Rayo Vallecano": [12, 11, 17, 15],
    "Real Betis": [5, 6, 7, 7],
    "Real Madrid": [1, 2, 1, 2],
    "Real Sociedad": [6, 4, 6, 8],
    "Reims": [12, 11, 9, 10],
    "Rennes": [4, 4, 10, 9],
    "Roma": [6, 6, 6, 7],
    "Saint-Étienne": [18, 22, 22, 16],
    "Salernitana": [17, 15, 20, 22],
    "Sampdoria": [15, 20, 22, 22],
    "Sassuolo": [11, 13, 19, 22],
    "Schalke 04": [22, 17, 22, 22],
    "Sevilla": [4, 12, 14, 12],
    "Sheffield United": [22, 22, 20, 22],
    "Southampton": [15, 20, 22, 19],
    "Spezia": [16, 18, 22, 22],
    "Strasbourg": [6, 15, 13, 13],
    "Stuttgart": [15, 16, 2, 7],
    "Torino": [10, 10, 9, 11],
    "Tottenham Hotspur": [4, 8, 5, 5],
    "Toulouse": [22, 13, 11, 12],
    "Troyes": [15, 19, 22, 22],
    "Udinese": [12, 12, 15, 13],
    "Union Berlin": [5, 4, 15, 11],
    "Valencia": [9, 14, 9, 13],
    "Valladolid": [22, 18, 22, 19],
    "Venezia": [20, 22, 22, 20],
    "Villarreal": [7, 5, 8, 6],
    "Watford": [19, 22, 22, 22],
    "Werder Bremen": [22, 13, 9, 10],
    "West Ham United": [7, 14, 9, 11],
    "Wolfsburg": [12, 8, 12, 12],
    "Wolves": [10, 13, 14, 15],
}


def get_squad_rank_map(year, full_map=SQUAD_RANK_MAP):
    index = year - 2021
    if index < 0 or index >= len(full_map):
        print("No Data For That Year")
        return

    temp_dict = {}
    for squad in full_map:
        temp_dict[squad] = full_map[squad][index]

    return temp_dict


def apply_squad_rank(df, year):
    df = df.copy()
    rank_map_for_year = get_squad_rank_map(year)

    if rank_map_for_year is None:
        return df

    df["Squad_Rank"] = (
        df["Unnamed: 4_level_0_Squad"]
        .map(rank_map_for_year)
        .apply(lambda x: 21 - x if pd.notnull(x) else 0)
    )
    df = df.drop(columns=["Unnamed: 4_level_0_Squad"])

    return df


def clean_dataframe(df):
    df = df.dropna(subset=["Value (€)"])
    df = df.drop(
        columns=[
            "Name_Clean",
            "Unnamed: 0_y",
            "Name_y",
            "Unnamed: 0_x",
            "Unnamed: 0_level_0_Rk",
            "Unnamed: 24_level_0_Matches",
            "Unnamed: 0",
        ]
    )
    df = df.rename(
        columns={
            "Unnamed: 3_level_0_Pos": "Position",
            "Unnamed: 5_level_0_Age": "Age",
            "Unnamed: 6_level_0_Born": "Born",
            "Unnamed: 2_level_0_Nation": "Nationality",
            "Playing Time_MP": "Match Played",
            "Name_x": "Name",
            "Playing Time_Starts": "Match Started",
            "Playing Time_Min": "Minutes Played",
            "Playing Time_90s": "Minutes Played / 90",
            "Performance_Gls": "Goals",
            "Performance_Ast": "Assists",
            "Performance_G+A": "Goals + Assists",
            "Performance_G-PK": "Non-Penality Goals",
            "Performance_PK": "Penalty Kick Goals",
            "Performance_PKatt": "Penalty Kick Attempted",
            "Performance_CrdY": "Yellow Cards",
            "Performance_CrdR": "Red Cards",
            "Per 90 Minutes_Gls": "Goals Per 90 Minutes",
            "Per 90 Minutes_Ast": "Assists Per 90 Minutes",
            "Per 90 Minutes_G+A": "G+A Per 90 Minutes",
            "Per 90 Minutes_G-PK": "Non-Penality Goals Per 90 Minutes",
            "Per 90 Minutes_G+A-PK": "Non-Penalty Goals + Assists/90",
            "Value (€)": "Value",
            # '': '',
            # '': '',
            # '': '',
            # '': '',
            # '': '',
            # '': '',
        }
    )

    intger_column_list = [
        "Age",
        "Born",
        "Match Played",
        "Match Started",
        "Minutes Played",
        "Goals",
        "Assists",
        "Goals + Assists",
        "Non-Penality Goals",
        "Penalty Kick Goals",
        "Penalty Kick Attempted",
        "Yellow Cards",
        "Red Cards",
    ]

    float_column_list = [
        "Minutes Played / 90",
        "Goals Per 90 Minutes",
        "Assists Per 90 Minutes",
        "G+A Per 90 Minutes",
        "Non-Penality Goals Per 90 Minutes",
        "Non-Penalty Goals + Assists/90",
    ]

    numeric_column_list = []
    numeric_column_list.extend(intger_column_list)
    numeric_column_list.extend(float_column_list)

    for column in numeric_column_list:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def preprocess_dataframe(df):
    df = df.copy()

    df_rest = df.drop(columns=["Value"])

    scaler = StandardScaler()
    numeric_cols = df_rest.select_dtypes(include=["float64", "int64"]).columns
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = divide_position(df)
    df = df.set_index("Name")
    df = df.drop(columns=["Nationality", "Born", "Position"])

    df = df[[c for c in df.columns if c != "Value"] + ["Value"]]
    return df, scaler


def divide_position(df):
    position_dummies = df["Position"].str.get_dummies(sep=",")
    return pd.concat([df, position_dummies], axis=1)


def load_raw_data(filepath, year, folder_path="../data"):
    return apply_squad_rank(
        divide_position(
            clean_dataframe(pd.read_csv(os.path.join(folder_path, filepath)))
        ).drop(columns=["Position", "Name", "Nationality", "Born"]),
        year,
    )


def end_to_end_process_data(df, year):
    df = apply_squad_rank(df, year)
    df = clean_dataframe(df)
    df, _ = preprocess_dataframe(df)
    return df


def end_to_end_load_data(filename):
    df = pd.read_csv(filename)
    year = int("".join(re.findall(r"\d+", filename)))

    df = end_to_end_process_data(df, year)

    return df

def feature_engineering(input_df: pd.DataFrame, scale=True, scalers: dict[Any] = None) -> tuple[pd.DataFrame, dict[Any]]:
    """
    1. Filters FW
    2. Drop Unused Columns
    3. Bin "Match Played"
    4. Binarize "Penalty Kick Goals"
    5. Log Transform "Goals", "Assists", "Non-Penalty Goals", "Goals Per 90 Minutes", "Assists Per 90 Minutes", "Non-Penalty Goals Per 90 Minutes", "Non-Penalty Goals + Assists/90"
    6. Min-Max Scale "Match Started", "Minutes Played"

    Parameters:
        input_df : dataframe loaded with train.csv or test.csv
        scale : Default to True. If True, returns dictionary containing the scaler with key 'mm_scaler'
        scalers : Default to None. Uses the provided scaler to when scaling. 

    Returns:
        1. feature engineered dataframe
        2. Dictionary containing scaler used
    """
    # Filter only FW & MF
    df = input_df[(input_df["FW"] == 1) | (input_df["MF"] == 1)].copy().reset_index(drop=True)
    df = df.drop(columns=["DF", "GK"])

    # Drop columns
    df = df.drop(columns=["Goals + Assists", "Penalty Kick Attempted", "Yellow Cards", "Red Cards", "G+A Per 90 Minutes", "Minutes Played / 90"])

    # Bin "Match Played"
    df["Experience Level"] = pd.cut(
        df["Match Played"], 
        bins=[0, 10, 20, 30, np.inf], 
        labels=["Level 1", "Level 2", "Level 3", "Level 4"],
        include_lowest=True,
    )
    df = df.drop(columns=["Match Played"])

    # Binarize "Penalty Kick Goals"
    df["Penalty Kicker"] = pd.cut(
        df["Penalty Kick Goals"], 
        bins=[0, 1, np.inf], 
        labels=["Not Penalty taker", "Penalty taker"],
        include_lowest=True,
        right = False,
    )
    df = df.drop(columns=["Penalty Kick Goals"])

    encoder = OrdinalEncoder(dtype=int)
    df[["Experience Level", "Penalty Kicker"]] = encoder.fit_transform(df[["Experience Level", "Penalty Kicker"]])

    # Log Transform (f(x) = ln(x + 1)) "Goals", "Assists", "Non-Penalty Goals", "Goals Per 90 Minutes", "Assists Per 90 Minutes", "Non-Penalty Goals Per 90 Minutes", "Non-Penalty Goals + Assists/90"
    target_cols = ["Goals Per 90 Minutes", "Assists Per 90 Minutes", "Non-Penality Goals Per 90 Minutes", "Non-Penalty Goals + Assists/90", "Goals", "Assists", "Non-Penality Goals"]
    df[target_cols] = np.log1p(df[target_cols])

    # Min-Max Scale "Match Started", "Minutes Played"
    if scale:
        if scalers and ('mm_scaler' in scalers):
            df[["Match Started", "Minutes Played"]] = scalers['mm_scaler'].transform(df[["Match Started", "Minutes Played"]])
            return df, scalers
        else:
            scalers = {}

            mm_scaler = MinMaxScaler()
            df[["Match Started", "Minutes Played"]] = mm_scaler.fit_transform(df[["Match Started", "Minutes Played"]])

            scalers['mm_scaler'] = mm_scaler

            return df, scalers

    return df, scalers

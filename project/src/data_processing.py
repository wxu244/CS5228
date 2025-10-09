import os
from glob import glob

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pathlib import Path


def read_files(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(base_dir, "data")

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    aux_dir = os.path.join(data_dir, "aux")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    auxiliary = {}
    for csv_path in glob(os.path.join(aux_dir, "*.csv")):
        file_stem = os.path.splitext(os.path.basename(csv_path))[0]
        auxiliary[file_stem] = pd.read_csv(csv_path)

    return train_df, test_df, auxiliary


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    split_data = df['MONTH'].str.split('-')
    df['YEAR'] = split_data.str[0].astype(int)
    df["MONTH_NUM"] = split_data.str[1].astype(int)
    df.drop(columns=['MONTH'], inplace=True)
    return df


def process_flat_type(df: pd.DataFrame) -> pd.DataFrame:
    df['FLAT_TYPE'] = df['FLAT_TYPE'].str.upper()
    df['FLAT_TYPE'] = df['FLAT_TYPE'].str.replace('-', '_', regex=False)
    df['FLAT_TYPE'] = df['FLAT_TYPE'].str.replace(' ', '_', regex=False)
    df['FLAT_TYPE_ORIGINAL'] = df['FLAT_TYPE']
    # 也可以用label encoding
    df = pd.get_dummies(df, columns=['FLAT_TYPE'], dtype=int)
    return df


def calculate_floor(df: pd.DataFrame) -> pd.DataFrame:
    regex_pattern = r'(\d+)\s*to\s*(\d+)'
    extracted_df = df['FLOOR_RANGE'].str.extract(regex_pattern)
    lower_values = extracted_df[0].astype(int)
    upper_values = extracted_df[1].astype(int)
    df['FLOOR'] = ((lower_values + upper_values) / 2).astype(int)
    return df


def engineer_flat_model_group(df: pd.DataFrame) -> pd.DataFrame:
    # df['FLAT_MODEL_GROUPED'] = df['FLAT_MODEL'].map(FLAT_MODEL_MAPPING).fillna('Other')

    # 按均值分类的
    # if df.get('RESALE_PRICE') is not None:
    #     group_stats = df.groupby('FLAT_MODEL_GROUPED').agg(
    #         count=('RESALE_PRICE', 'size'),
    #         mean_price=('RESALE_PRICE', 'mean'),
    #         median_price=('RESALE_PRICE', 'median'),
    #         mean_area=('FLOOR_AREA_SQM', 'mean')
    #     ).sort_values('mean_price')
    #     print(group_stats)

    le = LabelEncoder()
    df['FLAT_MODEL_ENCODED'] = le.fit_transform(df['FLAT_MODEL'])
    # df['FLAT_MODEL_ENCODED'] = le.fit_transform(df['FLAT_MODEL_GROUPED'])
    return df


def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    df['AGE'] = df['YEAR'] - df['LEASE_COMMENCE_DATA']
    return df


def clean_and_normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df['TOWN'] = df['TOWN'].str.lower()
    df['STREET'] = df['STREET'].str.lower()
    return df


def train_test_process(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_date_features(df)
    df = clean_and_normalize_text(df)
    df = process_flat_type(df)
    df = calculate_floor(df)
    df = engineer_flat_model_group(df)
    df = calculate_age(df)

    df = df.drop(columns=['ECO_CATEGORY'])
    return df

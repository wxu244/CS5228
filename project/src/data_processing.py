import os
from glob import glob

import numpy as np
import pandas as pd
from distributed.utils_test import throws
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
    label_encoder = LabelEncoder()
    df['FLAT_TYPE'] = label_encoder.fit_transform(df['FLAT_TYPE'])
    return df


def calculate_floor(df: pd.DataFrame) -> pd.DataFrame:
    regex_pattern = r'(\d+)\s*to\s*(\d+)'
    extracted_df = df['FLOOR_RANGE'].str.extract(regex_pattern)
    lower_values = extracted_df[0].astype(int)
    upper_values = extracted_df[1].astype(int)
    df['FLOOR'] = ((lower_values + upper_values) / 2).astype(int)
    return df


def engineer_flat_model_group(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # df['FLAT_MODEL_GROUPED'] = df['FLAT_MODEL'].map(FLAT_MODEL_MAPPING).fillna('Other')

    # 1:label encode
    # le = LabelEncoder()
    # df['FLAT_MODEL_ENCODED'] = le.fit_transform(df['FLAT_MODEL'])

    # 2:target encode
    # transfrom the resale_price using log if it exists
    if 'RESALE_PRICE' in train_df.columns:
        train_df['LOG_RESALE_PRICE'] = np.log1p(train_df['RESALE_PRICE'])

    target_column = 'LOG_RESALE_PRICE' if 'LOG_RESALE_PRICE' in train_df.columns else 'RESALE_PRICE'
    target_mean = train_df.groupby('FLAT_MODEL')[target_column].transform('mean')

    # save the result
    train_df['FLAT_MODEL_ENCODED'] = target_mean

    # save the encoding_map
    target_encoding_map = train_df.groupby('FLAT_MODEL')['FLAT_MODEL_ENCODED'].first().to_dict()

    # apply the data into test set
    test_df['FLAT_MODEL_ENCODED'] = test_df['FLAT_MODEL'].map(target_encoding_map)

    # deal with unknown value
    unknown_value = train_df['FLAT_MODEL_ENCODED'].mean()  # 使用训练集的平均值作为未知类别的默认值
    test_df['FLAT_MODEL_ENCODED'].fillna(unknown_value, inplace=True)

    # delete the column of log_resale_price
    if 'LOG_RESALE_PRICE' in train_df.columns:
        train_df.drop(columns=['LOG_RESALE_PRICE'], inplace=True)


    return train_df, test_df

def engineer_flat_model_group_3(df: pd.DataFrame) -> pd.DataFrame:
    structure_map = {
        '2-room': 'flat',
        'standard': 'flat',
        'simplified': 'flat',
        'new generation': 'flat',
        'model a': 'flat',
        'model a2': 'flat',
        'improved': 'flat',
        'apartment': 'flat',
        'type s1': 'flat',
        'type s2': 'flat',
        'dbss': 'flat',
        'premium apartment': 'flat',

        'maisonette': 'maisonette',
        'model a maisonette': 'maisonette',
        'improved maisonette': 'maisonette',
        'premium maisonette': 'maisonette',
        'premium apartment loft': 'maisonette',

        'multi generation': 'special',
        '3gen': 'special',
        'adjoined flat': 'special',
        'terrace': 'special'
    }

    quality_map = {
        '2-room': 'basic',
        'standard': 'basic',
        'simplified': 'basic',
        'new generation': 'basic',

        'model a': 'improved',
        'model a2': 'improved',
        'improved': 'improved',
        'apartment': 'improved',
        'type s1': 'improved',
        'type s2': 'improved',
        'maisonette': 'improved',
        'model a maisonette': 'improved',
        'improved maisonette': 'improved',
        'multi generation': 'improved',

        'premium apartment': 'premium',
        'premium apartment loft': 'premium',
        'premium maisonette': 'premium',
        'dbss': 'premium',
        '3gen': 'premium',
        'adjoined flat': 'premium',
        'terrace': 'premium'
    }

    # label encoding with setted config
    le_structure = LabelEncoder()
    le_quality = LabelEncoder()

    # process the data set
    df['structure_type'] = df['FLAT_MODEL'].str.lower().map(structure_map)
    df['quality_level'] = df['FLAT_MODEL'].str.lower().map(quality_map)
    df['structure_type_encoded'] = le_structure.fit_transform(df['structure_type'])
    df['quality_level_encoded'] = le_quality.fit_transform(df['quality_level'])


    return df


def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    df['AGE'] = df['YEAR'] - df['LEASE_COMMENCE_DATA']
    return df


def clean_and_normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df['TOWN'] = df['TOWN'].str.lower()
    df['STREET'] = df['STREET'].str.lower()
    return df


def train_test_process(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = extract_date_features(train_df)
    test_df = extract_date_features(test_df)

    train_df = clean_and_normalize_text(train_df)
    test_df = clean_and_normalize_text(test_df)

    train_df = process_flat_type(train_df)
    test_df = process_flat_type(test_df)

    train_df = calculate_floor(train_df)
    test_df = calculate_floor(test_df)

    train_df, test_df = engineer_flat_model_group(train_df, test_df)

    # train_df = engineer_flat_model_group_3(train_df)
    # test_df = engineer_flat_model_group_3(test_df)

    train_df = calculate_age(train_df)
    test_df = calculate_age(test_df)

    train_df = train_df.drop(columns=['ECO_CATEGORY'], errors='ignore')
    test_df = test_df.drop(columns=['ECO_CATEGORY'], errors='ignore')

    return train_df, test_df

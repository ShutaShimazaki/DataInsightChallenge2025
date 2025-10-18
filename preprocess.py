import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# 列の種類定義（柔軟に修正可能）
# ------------------------------
COLUMNS = {
    "categorical": ['業界', '上場種別', '取引形態', '特徴'],
    "text": ['企業概要','組織図','今後のDX展望'],
    "survey": ['アンケート１','アンケート２','アンケート３','アンケート４','アンケート５','アンケート６','アンケート７','アンケート８','アンケート９','アンケート１０','アンケート１１'],
    "numeric": ['従業員数','事業所数','工場数','店舗数','資本金','総資産','流動資産',
                '固定資産','負債','短期借入金','長期借入金','純資産','自己資本',
                '売上','営業利益','経常利益','当期純利益','営業CF','減価償却費',
                '運転資本変動','投資CF','有形固定資産変動','無形固定資産変動(ソフトウェア関連)']
}

# ---------------------------------------
# 欠損値処理
# ---------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS["numeric"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in COLUMNS["categorical"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in COLUMNS["text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in COLUMNS["survey"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------
# 指定カテゴリ列のみ Label Encoding
# ---------------------------------------
def encode_categories(df: pd.DataFrame, categorical_cols=None, encoders=None):
    if encoders is None:
        encoders = {}
    if categorical_cols is None:
        categorical_cols = COLUMNS["categorical"]

    for col in categorical_cols:
        if col in df.columns:
            le = encoders.get(col, LabelEncoder())
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


# ---------------------------------------
# TF-IDF特徴量
# ---------------------------------------
def add_tfidf_features(df: pd.DataFrame, tfidf_cols=None, max_features=100, vectorizers=None):
    if vectorizers is None:
        vectorizers = {}
    if tfidf_cols is None:
        tfidf_cols = COLUMNS["text"]

    for col in tfidf_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            tfidf = vectorizers.get(col, TfidfVectorizer(max_features=max_features))
            tfidf_matrix = tfidf.fit_transform(df[col])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
            df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
            vectorizers[col] = tfidf

    return df, vectorizers


# ---------------------------------------
# 数値特徴量追加
# ---------------------------------------
def add_numeric_features(df: pd.DataFrame):
    if "売上" in df.columns and "従業員数" in df.columns:
        df["売上_per_employee"] = df["売上"] / (df["従業員数"] + 1)
    if "営業利益" in df.columns and "売上" in df.columns:
        df["営業利益率"] = df["営業利益"] / (df["売上"] + 1)
    return df


# ---------------------------------------
# DX意識スコア作成
# ---------------------------------------
def add_dx_awareness_score(df: pd.DataFrame):
    score = (
        df["アンケート１"] + df["アンケート２"] + df["アンケート３"]
        + (6 - df["アンケート４"])  # 反転
        + df["アンケート５"]
        + df["アンケート６"].apply(lambda x: 1 if x == 1 else 0)
        + df["アンケート７"] + df["アンケート８"]
        + df["アンケート９"] + df["アンケート１０"] + df["アンケート１１"]
    )
    df["DX意識スコア"] = score
    return df


# ---------------------------------------
# 前処理まとめ関数
# ---------------------------------------
def preprocess_data(df: pd.DataFrame,
                    categorical_cols=None,
                    tfidf_cols=None,
                    encoders=None,
                    vectorizers=None):
    df = handle_missing_values(df)
    df, encoders = encode_categories(df, categorical_cols, encoders)
    df, vectorizers = add_tfidf_features(df, tfidf_cols, vectorizers=vectorizers)
    df = add_numeric_features(df)
    df = add_dx_awareness_score(df)

    return df, encoders, vectorizers

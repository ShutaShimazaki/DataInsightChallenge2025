import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer

# ------------------------------
# 列の種類定義
# ------------------------------
COLUMNS = {
    "categorical": ['業界', '上場種別', '取引形態'],
    "text": ['企業概要', '今後のDX展望'],
    "survey": ['アンケート１','アンケート２','アンケート３','アンケート４','アンケート５',
               'アンケート６','アンケート７','アンケート８','アンケート９','アンケート１０','アンケート１１'],
    "numeric": ['従業員数','事業所数','工場数','店舗数','資本金','総資産','流動資産','固定資産','負債',
                '短期借入金','長期借入金','純資産','自己資本','売上','営業利益','経常利益','当期純利益',
                '営業CF','減価償却費','運転資本変動','投資CF','有形固定資産変動','無形固定資産変動(ソフトウェア関連)']
}

# ---------------------------------------
# 欠損値処理
# ---------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS["numeric"] + COLUMNS["survey"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in COLUMNS["categorical"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in COLUMNS["text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df

# ---------------------------------------
# カテゴリ変数エンコード
# ---------------------------------------
def encode_categories(df: pd.DataFrame, encoders=None):
    if encoders is None:
        encoders = {}

    # --- 業界 (One-Hot) ---
    if "業界" in df.columns:
        ohe = encoders.get("業界", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        df_ohe = pd.DataFrame(ohe.fit_transform(df[["業界"]]),
                              columns=[f"業界_{cat}" for cat in ohe.categories_[0]],
                              index=df.index)
        df = pd.concat([df.drop(columns=["業界"]), df_ohe], axis=1)
        encoders["業界"] = ohe

    # --- 上場種別 (Ordinal) ---
    if "上場種別" in df.columns:
        oe = encoders.get("上場種別", OrdinalEncoder(categories=[["PR", "ST", "GR"]]))
        df["上場種別"] = oe.fit_transform(df[["上場種別"]])
        encoders["上場種別"] = oe

    # --- 取引形態 (MultiLabelBinarizer) ---
    if "取引形態" in df.columns:
        mlb = encoders.get("取引形態", MultiLabelBinarizer())
        transformed = mlb.fit_transform(df["取引形態"].str.split(", "))
        df_mlb = pd.DataFrame(transformed, columns=[f"取引形態_{cls}" for cls in mlb.classes_], index=df.index)
        df = pd.concat([df.drop(columns=["取引形態"]), df_mlb], axis=1)
        encoders["取引形態"] = mlb

    return df, encoders

# ---------------------------------------
# TF-IDF特徴量
# ---------------------------------------
tokenizer = Tokenizer()

def tokenize_ja(text):
    return " ".join(token.surface for token in tokenizer.tokenize(text))

def add_tfidf_features(df: pd.DataFrame, vectorizers=None):
    if vectorizers is None:
        vectorizers = {}

    for col in COLUMNS["text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

            if col in vectorizers:
                tfidf = vectorizers[col]
                tfidf_matrix = tfidf.transform(df[col])
            else:
                tfidf = TfidfVectorizer(max_features=10, tokenizer=tokenize_ja, ngram_range=(1,2), min_df=2)
                tfidf_matrix = tfidf.fit_transform(df[col])
                vectorizers[col] = tfidf

            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                                    index=df.index)
            df = pd.concat([df, tfidf_df], axis=1)

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
# DX意識スコア
# ---------------------------------------
def add_dx_awareness_score(df: pd.DataFrame):
    df["DX意識スコア"] = (
        df["アンケート１"] + df["アンケート２"] + df["アンケート３"]
        + (6 - df["アンケート４"])
        + df["アンケート５"]
        + df["アンケート６"].apply(lambda x: 1 if x == 1 else 0)
        + df["アンケート７"] + df["アンケート８"]
        + df["アンケート９"] + df["アンケート１０"] + df["アンケート１１"]
    )
    return df

# ---------------------------------------
# 前処理
# ---------------------------------------
def preprocess_train(df: pd.DataFrame):
    """train専用前処理"""
    # ========= 不要データの削除 =========
    #　不要な取引形態
    if "取引形態" in df.columns:
        remove_patterns = ["BtoB, BtoC, CtoC", "CtoC"]
        df = df[~df["取引形態"].isin(remove_patterns)].reset_index(drop=True)

    #　不要な業界
    if "業界" in df.columns:
        remove_industries = ["専門サービス", "その他"]
        df = df[~df["業界"].isin(remove_industries)].reset_index(drop=True)

    df = handle_missing_values(df)
    df, encoders = encode_categories(df)
    df, vectorizers = add_tfidf_features(df)
    df = add_numeric_features(df)
    df = add_dx_awareness_score(df)
    return df, encoders, vectorizers

def preprocess_test(df: pd.DataFrame, encoders, vectorizers):
    """test専用前処理（trainで学習したencoder/vectorizerを使用）"""
    df = handle_missing_values(df)
    df, _ = encode_categories(df, encoders=encoders)
    df, _ = add_tfidf_features(df, vectorizers=vectorizers)
    df = add_numeric_features(df)
    df = add_dx_awareness_score(df)
    return df

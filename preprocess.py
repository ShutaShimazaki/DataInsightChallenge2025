import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer

# ------------------------------
# 列の種類定義（柔軟に修正可能）
# ------------------------------
COLUMNS = {
    "categorical": ['業界', '上場種別', '取引形態'],
    "text": ['企業概要','今後のDX展望'],
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

    # --- 業界 (名義尺度 → One-Hot) ---
    if "業界" in categorical_cols and "業界" in df.columns:
        ohe = encoders.get("業界", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        transformed = ohe.fit_transform(df[["業界"]])
        ohe_cols = [f"業界_{cat}" for cat in ohe.categories_[0]]
        df_ohe = pd.DataFrame(transformed, columns=ohe_cols, index=df.index)
        df = pd.concat([df.drop(columns=["業界"]), df_ohe], axis=1)
        encoders["業界"] = ohe

    # --- 上場種別 (順序尺度 → Ordinal) ---
    if "上場種別" in categorical_cols and "上場種別" in df.columns:
        # PR < ST < GR の順序を仮定
        order = [["PR", "ST", "GR"]]
        oe = encoders.get("上場種別", OrdinalEncoder(categories=order))
        df["上場種別"] = oe.fit_transform(df[["上場種別"]])
        encoders["上場種別"] = oe

    # --- 取引形態 (名義尺度、複数ラベル → MultiLabelBinarizer) ---
    if "取引形態" in categorical_cols and "取引形態" in df.columns:
        mlb = encoders.get("取引形態", MultiLabelBinarizer())
        transformed = mlb.fit_transform(df["取引形態"].fillna("").str.split(", "))
        mlb_cols = [f"取引形態_{cls}" for cls in mlb.classes_]
        df_mlb = pd.DataFrame(transformed, columns=mlb_cols, index=df.index)
        df = pd.concat([df.drop(columns=["取引形態"]), df_mlb], axis=1)
        encoders["取引形態"] = mlb

    return df, encoders


# ---------------------------------------
# TF-IDF特徴量
# ---------------------------------------
# Janomeトークナイザー（日本語用）
tokenizer = Tokenizer()

def tokenize_ja(text):
    """日本語テキストを分かち書きする関数"""
    return " ".join(token.surface for token in tokenizer.tokenize(text))

def add_tfidf_features(df: pd.DataFrame, tfidf_cols=None, max_features=20, vectorizers=None):
    """
    日本語対応TF-IDF特徴量生成
    """
    if vectorizers is None:
        vectorizers = {}
    if tfidf_cols is None:
        tfidf_cols = COLUMNS["text"]  # あなたのコードに合わせて使用

    for col in tfidf_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

            # 既存のベクトライザがあれば再利用、なければ初期化
            if col in vectorizers:
                tfidf = vectorizers[col]
                tfidf_matrix = tfidf.transform(df[col])
            else:
                tfidf = TfidfVectorizer(
                    max_features=max_features,
                    tokenizer=tokenize_ja,  # ✅日本語対応
                    ngram_range=(1, 2),     # ✅精度改善（ユニグラム＋バイグラム）
                    min_df=2                # ✅ノイズ抑制
                )
                tfidf_matrix = tfidf.fit_transform(df[col])
                vectorizers[col] = tfidf

            # データフレームに結合
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
            df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

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
# 前処理
# ---------------------------------------
def preprocess_data(df: pd.DataFrame,
                    categorical_cols=None,
                    tfidf_cols=None,
                    encoders=None,
                    vectorizers=None):
    
    # ========= 不要データの削除 =========
    # 取引形態が "BtoB, BtoC, CtoC" または "CtoC" の行を削除
    if "取引形態" in df.columns:
        remove_patterns = ["BtoB, BtoC, CtoC", "CtoC"]
        df = df[~df["取引形態"].isin(remove_patterns)].reset_index(drop=True)

    # ========= 欠損値=========
    df = handle_missing_values(df) 

    # ========= カテゴリ変数の数値化 =========
    df, encoders = encode_categories(df, categorical_cols, encoders)

    # ========= テキストデータの数値化（tfidf） =========
    df, vectorizers = add_tfidf_features(df, tfidf_cols, vectorizers=vectorizers)

    # ========= 新規特徴量の設計 =========
    df = add_numeric_features(df)

    # ========= 新規特徴量の設計(DX意識スコア) =========
    df = add_dx_awareness_score(df)

    return df, encoders, vectorizers

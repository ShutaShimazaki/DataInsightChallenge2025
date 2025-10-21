from preprocess import add_tfidf_features, COLUMNS
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import re
import scipy.stats as stats


def normalize_katakana_words(words):
    normalized = []
    temp = ""
    for w in words:
        if re.match(r'[\u30A0-\u30FF]+', w):  # カタカナ
            temp += w
        else:
            if temp:
                normalized.append(temp)
                temp = ""
            normalized.append(w)
    if temp:
        normalized.append(temp)
    return normalized

tokenizer = Tokenizer()

def tokenize_ja(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
        if part_of_speech in ["名詞", "動詞", "形容詞"]:
            word = token.surface.replace(" ", "").replace("\u3000", "")  # 半角・全角スペース除去
            if word:  # 空文字除外
                words.append(word)

    words = normalize_katakana_words(words)
    return " ".join(words)

def add_tfidf_features(df: pd.DataFrame, vectorizers=None):
    """TF-IDF特徴量追加"""
    if vectorizers is None:
        vectorizers = {}

    for col in COLUMNS["text"]:
        if col in df.columns:
            # 先に tokenize_ja で単語列に変換
            df[col] = df[col].fillna("").astype(str).apply(tokenize_ja)

            if col in vectorizers:
                tfidf = vectorizers[col]
                tfidf_matrix = tfidf.transform(df[col])
            else:
                tfidf = TfidfVectorizer(max_features=400, tokenizer=None, ngram_range=(1,2), min_df=10)
                tfidf_matrix = tfidf.fit_transform(df[col])
                vectorizers[col] = tfidf

            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                index=df.index
            )
            df = pd.concat([df, tfidf_df], axis=1)

    return df, vectorizers

def select_useful_tfidf_features(df: pd.DataFrame, vectorizers: dict, target_col="購入フラグ", top_k=30, output_csv="tfidf_feature_ranking.csv"):
    # TF-IDF列を抽出
    tfidf_cols = [c for c in df.columns if "tfidf" in c]
    X = df[tfidf_cols]
    y = df[target_col]

    # Mutual Information (MI)による特徴スコア計算
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # 結果をDataFrameにまとめる
    mi_df = pd.DataFrame({
        "feature": tfidf_cols,
        "mi_score": mi_scores
    }).sort_values(by="mi_score", ascending=False)

    # 列名に対応する単語を取得
    feature_to_word = {}
    for col_name in COLUMNS["text"]:
        if col_name in vectorizers:
            words = vectorizers[col_name].get_feature_names_out()
            for i, w in enumerate(words):
                feature_to_word[f"{col_name}_tfidf_{i}"] = w

    # 元単語を追加
    mi_df["word"] = mi_df["feature"].map(feature_to_word)

    # 上位 top_k の特徴量のみ抽出
    top_features_df = mi_df.head(top_k)
    top_features = top_features_df["feature"].tolist()

    # χ²検定（不均衡考慮：出現0または片側のみのときp=1）
    chi2_p_list = []
    purchase_rate_list = []  # 購入クラスでの出現率
    for f in top_features:
        binarized = (X[f] > 0).astype(int)
        contingency_table = pd.crosstab(binarized, y)
        # 出現0や片側のみのときは無効とする
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            p = 1.0
        else:
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
        chi2_p_list.append(p)

        # 購入クラスでの出現率
        purchase_rate = df[df[f] > 0][target_col].mean() if (df[f] > 0).sum() > 0 else 0
        purchase_rate_list.append(purchase_rate)

    top_features_df["chi2_p"] = chi2_p_list
    top_features_df["purchase_rate"] = purchase_rate_list

    # CSV出力
    top_features_df.to_csv(output_csv, index=False)

    # dfには top_k の特徴量のみ残す
    df = df.drop(columns=[c for c in tfidf_cols if c not in top_features])

    return df, top_features_df

# ---------------------------
# 最終特徴量選択
# ---------------------------
def select_final_tfidf_features(top_features_df, df, target_col="購入フラグ",
                                chi2_p_threshold=0.05, top_n_per_col=15, output_csv="final_selected_features.csv"):
    final_features = []

    # 全体購入率
    purchase_rate_threshold = df[target_col].mean()

    for col_name in COLUMNS["text"]:
        col_df = top_features_df[top_features_df["feature"].str.startswith(col_name)].copy()
        col_df = col_df[
            (col_df["chi2_p"] < chi2_p_threshold) &
            (col_df["purchase_rate"] > purchase_rate_threshold)
        ]
        col_df = col_df.sort_values(by="mi_score", ascending=False).head(top_n_per_col)
        final_features.append(col_df)

    final_df = pd.concat(final_features).reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)
    print(f"最終選択特徴量 CSV 出力: {output_csv}")
    return final_df


base_folder = "./"
train = pd.read_csv(base_folder + "train.csv")
top_k = 30
vectorizers = {}
final_features_all = []

for col in COLUMNS["text"]:  # 例: ["企業概要", "今後のDX展望"]
    if col not in train.columns:
        continue

    print(f"=== {col} の処理 ===")

    # TF-IDF追加
    df_temp, vectorizers = add_tfidf_features(train[[col, "購入フラグ"]].copy(), vectorizers)

    # MI計算・上位top_k抽出
    df_temp, top_features_df = select_useful_tfidf_features(
        df_temp,
        vectorizers=vectorizers,
        target_col="購入フラグ",
        top_k=top_k,
        output_csv=f"{base_folder}tfidf_feature_ranking_{col}.csv"
    )

    # 最終選択
    final_features_df = select_final_tfidf_features(
        top_features_df,
        df=train,
        target_col="購入フラグ",
        chi2_p_threshold=0.05,
        top_n_per_col=10,
        output_csv=f"{base_folder}final_selected_features_{col}.csv"
    )

    final_features_all.append(final_features_df)
    print(final_features_all)

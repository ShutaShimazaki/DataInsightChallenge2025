import pandas as pd
from preprocess import preprocess_data
import pickle

if __name__ == "__main__":
    base_folder = "./"
    train = pd.read_csv(base_folder + "train.csv")
    test = pd.read_csv(base_folder + "test.csv")

    # 各列の欠損値の数を確認
    missing_counts = train.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print("欠損値のある列と件数:")
    print(missing_counts)

    # 前処理
    train, encoders, vectorizers = preprocess_data(train)
    test, _, _ = preprocess_data(test, encoders=encoders, vectorizers=vectorizers)
    # print(train.info())

    # 各列の欠損値の数を確認
    missing_counts = train.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print("欠損値のある列と件数:")
    print(missing_counts)

    # 保存
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("vectorizers.pkl", "wb") as f:
        pickle.dump(vectorizers, f)

    print("前処理完了")
    pd.set_option('display.max_columns', 300)
    
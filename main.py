from preprocess import preprocess_data
from train_model import train_model
import pandas as pd
import pickle

if __name__ == "__main__":
    base_folder = "./"
    train = pd.read_csv(base_folder + "train.csv")
    test = pd.read_csv(base_folder + "test.csv")
    pd.set_option('display.max_columns', 300)

     # ========= 前処理 =========
    train, encoders, vectorizers = preprocess_data(train)
    test, _, _ = preprocess_data(test, encoders=encoders, vectorizers=vectorizers)

    # IDや名前などモデリングに不要な列は除去（※購入フラグは目的変数なので学習時に扱う）
    drop_cols = ["企業ID", "企業名", "組織図", '企業概要' ,'今後のDX展望','アンケート１','アンケート２','アンケート３','アンケート４','アンケート５','アンケート６','アンケート７','アンケート８','アンケート９','アンケート１０','アンケート１１']
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    print(train.info())
    print(test.info())


    # # # 各列の欠損値の数を確認
    # # missing_counts = train.isna().sum()
    # # missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    # # print("欠損値のある列と件数:")
    # # print(missing_counts)

    # # ========= 学習 =========
    # train_model(train)

    # # ========= 予測・提出ファイル作成 =========
    # # 学習済みモデルを読み込む
    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)

    # # testデータで予測
    # test_preds = model.predict(test)

    # # 提出ファイルの作成
    # sample_submit = pd.read_csv(base_folder + "sample_submit.csv", header=None)
    # sample_submit.iloc[:, 1] = test_preds
    # sample_submit.to_csv(base_folder + "submission.csv", index=False, header=None)
    # print("提出ファイルを作成しました: submission.csv")
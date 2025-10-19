from preprocess import preprocess_train, preprocess_test, COLUMNS
import pandas as pd
import pickle

if __name__ == "__main__":
    base_folder = "./"
    train = pd.read_csv(base_folder + "train.csv")
    test = pd.read_csv(base_folder + "test.csv")
    pd.set_option('display.max_columns', 300)

    # ====================
    # 前処理
    # ====================
    print("=== Train 前処理 ===")
    train, encoders, vectorizers = preprocess_train(train)

    print("=== Test 前処理 ===")
    test = preprocess_test(test, encoders, vectorizers)

    # ====================
    # 不要列の削除
    # ====================
    drop_cols = ["企業ID", "企業名", "組織図"] + COLUMNS["text"] + COLUMNS["survey"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    print("=== Train info ===")
    print(train.info())
    print(train.shape)
    print("=== Test info ===")
    print(test.info())
    print(train.shape)

    # ====================
    # 学習・予測部分（コメントアウト）
    # ====================
    from train_model import train_model
    train_model(train)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        test_preds = model.predict(test)
        sample_submit = pd.read_csv(base_folder + "sample_submit.csv", header=None)
        sample_submit.iloc[:, 1] = test_preds
        sample_submit.to_csv(base_folder + "submission.csv", index=False, header=None)

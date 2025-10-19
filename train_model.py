import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

def train_model(train_df):
    target_col = "購入フラグ"
    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])

    model = LGBMClassifier(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    print("\n===== 学習開始（5-fold Cross Validation）=====")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        f1_scores.append(f1)
        print(f"[Fold {fold}] F1 Score: {f1:.4f}")

    print(f"\n平均F1スコア: {np.mean(f1_scores):.4f}")

    # 全データで再学習して保存
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("モデル保存完了: model.pkl")
    

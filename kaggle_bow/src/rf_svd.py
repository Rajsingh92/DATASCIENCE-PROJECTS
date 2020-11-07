import pandas as pd
from sklearn import ensemble
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

def run_training(fold):
    df = pd.read_csv("../input/train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    x_train = tfv.transform(df_train.review.values)
    x_valid = tfv.transform(df_valid.review.values)

    svd  = decomposition.TruncatedSVD(n_components=120)
    svd.fit(x_train)

    x_train_svd = svd.transform(x_train)
    x_valid_svd = svd.transform(x_valid)

    y_train = df_train.sentiment.values
    y_valid = df_valid.sentiment.values

    clf = ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1)
    clf.fit(x_train_svd,y_train)
    preds = clf.predict_proba(x_valid_svd)[:,1]

    auc = metrics.roc_auc_score(y_valid,preds)
    print(f"fold = {fold},auc={auc}")

    df_valid.loc[:,"rf_svd_pred"] = preds

    return df_valid[["id","sentiment","kfold","rf_svd_pred"]]

if __name__ == "__main__":
    dfs=[]
    for i in range(5):
        temp_df = run_training(i)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../models_pred/rf_svd.csv",index=False)
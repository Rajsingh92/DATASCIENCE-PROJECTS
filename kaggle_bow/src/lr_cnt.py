import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

def run_training(fold):
    df = pd.read_csv("../input/train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = CountVectorizer()
    tfv.fit(df_train.review.values)

    x_train = tfv.transform(df_train.review.values)
    x_valid = tfv.transform(df_valid.review.values)

    y_train = df_train.sentiment.values
    y_valid = df_valid.sentiment.values

    clf = linear_model.LogisticRegression()
    clf.fit(x_train,y_train)
    preds = clf.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(y_valid,preds)
    print(f"fold = {fold},auc={auc}")

    df_valid.loc[:,"lr_cnt_pred"] = preds

    return df_valid[["id","sentiment","kfold","lr_cnt_pred"]]

if __name__ == "__main__":
    dfs=[]
    for i in range(5):
        temp_df = run_training(i)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../models_pred/lr_cnt.csv",index=False)
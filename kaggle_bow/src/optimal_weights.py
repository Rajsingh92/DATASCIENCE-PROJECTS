import pandas as pd
import glob
from sklearn import metrics
import numpy as np
from scipy.optimize import fmin
from functools import partial

class OptimizedAUC:
    def __init__(self):
        self.coef_ =0

    def _auc(self,coef,X,y):
        x_coef =X*coef
        predictions = np.sum(x_coef,axis=1)
        auc_score = metrics.roc_auc_score(y,predictions)
        return -1.0*auc_score

    def fit(self,x,y):
        partial_loss = partial(self._auc,X=x,y=y)
        init_coef = np.random.dirichlet(np.ones(x.shape[1]))
        self.coef_ = fmin(partial_loss,init_coef,disp=True)

    def predict(self,X):
        x_coef =X*self.coef_
        predictions = np.sum(x_coef,axis=1)
        return predictions


def run_training(pred_df,fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values

    opt = OptimizedAUC()
    opt.fit(xtrain,train_df.sentiment.values)
    preds = opt.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values,preds)
    print(f"{fold},{auc}")
    
    return opt.coef_



if __name__ == "__main__":
    files = glob.glob("../models_pred/*.csv")
    df=None

    for f in files:
        if df is None:
            df=pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df =df.merge(temp_df,on="id",how="left")
    #print(df.head())
    targets = df.sentiment.values

    pred_cols = ["lr_pred","lr_cnt_pred","rf_svd_pred"]
    coefs = []
    for i in range(5):
        coefs.append(run_training(df,i))

    coefs = np.array(coefs)
    #print(coefs)
    coefs = np.mean(coefs,axis=0)
    print(coefs)

    wt_avg = (
        coefs[0]*df.lr_pred.values
        +  coefs[1]*df.lr_cnt_pred.values
        +  coefs[2]*df.lr_svd_pred.values
    )

    print("optimal auc after finding coefs:")

    print(metrics.roc_auc_score(targets,wt_avg))
   





    


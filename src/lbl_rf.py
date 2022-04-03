import joblib
from matplotlib.pyplot import axis
import pandas as pd
from sklearn import linear_model, metrics, preprocessing, ensemble
import os
import config
def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]
    #fill nan values with none
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    #encode features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        #encode all the data
        lbl.fit(df[col])
        #transform the data
        df.loc[:, col] = lbl.transform(df[col])
    #get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #validation data with folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #get training data
    x_train = df_train[features].values
    #get validation data
    x_valid = df_valid[features].values
    #mode selection
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    #model fitting
    model.fit(x_train, df_train.target.values)
    #model prediction on validation
    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold: {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
    

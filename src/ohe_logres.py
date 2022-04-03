import joblib
from matplotlib.pyplot import axis
import pandas as pd
from sklearn import linear_model, metrics, preprocessing
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
    #train
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #validation
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #now one hot encode
    ohe = preprocessing.OneHotEncoder()
    #fit full data
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit_transform(full_data[features])
    #one hot encode train data
    x_train = ohe.transform(df_train[features])
    #one hot encode validation data
    x_validation = ohe.transform(df_valid[features])
    #basic logic regression model
    model = linear_model.LogisticRegression()
    #fit model on training data
    model.fit(x_train, df_train.target.values)
    #predict on validation data
    validation_preds = model.predict_proba(x_validation)[:, 1]
    #get roc and auc score
    auc = metrics.roc_auc_score(df_valid.target.values, validation_preds)
    print(f"Fold = {fold}, AUC= {auc}")
   

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
    

import pandas as pd
import xgboost as xgb

from sklearn import metrics, preprocessing

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    #features
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]
    #fill nan values
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    #label encode now
    for col in features:
        #initalize the label encoder
        lbl = preprocessing.LabelEncoder()
        #fit
        lbl.fit(df[col])
        #transform
        df.loc[:, col] = lbl.transform(df[col])
    #get training data folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #get validation data folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #get training data
    x_train = df_train[features].values
    #get validation data
    x_valid = df_valid[features].values
    #model selection
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200)
    #model fitting
    model.fit(x_train, df_train.target.values)
    #predictions = predicting the probability of getting 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    #auc
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    #print auc score
    print(f"print Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)


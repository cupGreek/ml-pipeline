import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import giraffez as gi
import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm
from xgboost import XGBClassifier
from sklearn.metrics import *

from r6ssql_0204 import *
from configuration import *


def main():
    global data_path
    global model_path
    global eval_path
    global prediction_path
    global offset
    global n_jobs
    global train
    global DED
    global cv
    
    print('Checking Parameters in Configuration...')
    if not check_date(DED):
        exit() 
    if not check_params(cv, offset, n_jobs):
        exit()
    
    print('Checking and Creating Required Paths...')
    ensure_dir(data_path)
    ensure_dir(model_path)
    ensure_dir(eval_path)
    ensure_dir(prediction_path)
    print('Paths have Successfully Created or Ensured!')

    if train: # do both training and predicting
        print('---Start the TRAINING process---')
        training_start_time = time.time()

        print('Creating Train & Prediction Table...')
        predict_table, train_table = load(DED=DED, train=True, offset=offset)

        print('Preprocessing Data...')
        predict_table = preprocess(predict_table)
        train_table = preprocess(train_table, train=True)
        delta_data = round((time.time() - training_start_time)/60, 2)
        print('Finished in {} minutes!'.format(delta_data))

        print('Saving Training & Prediction Data...')
        save_data(train_table, train=True, data_path = data_path)
        save_data(predict_table, data_path = data_path)
        print('Done!')

        print('Building Model...')
        modeling_start_time = time.time()
        X_train, X_test, y_train, y_test = set_model_matrices(train_table)
        models = build_models(X_train, y_train, X_test, y_test, n_jobs=n_jobs, cv_fold=cv)

        delta_model = round((time.time() - modeling_start_time)/60, 2)
        print('Finished Modeling in {} minutes!'.format(delta_model))

        print('Saving Model...')
        save_models(models, model_path=model_path)
        print('Done!')

        print('Model Evaluating...')
        pred_labels, pred_probs =  ensemble_model_predict(models, X_test)
        model_eval(pred_labels, pred_probs, y_test, eval_path=eval_path)
        print('Done!')

        delta_train = round((time.time() - training_start_time)/60, 2)
        print('---Finished Training in {} minutes---'.format(delta_train))
    
        # predict
        pred_start_time = time.time()
        print('---Start the PREDICTING process---')

        print('Loading Model...')
        models = load_models(path = model_path)
        
        print('Predicting...')
        if 'label' in predict_table.columns:
            X_pred = predict_table.drop('label', axis=1)
        else:
            X_pred = predict_table
        prediction, _ =  ensemble_model_predict(models, X_pred)
        predict_table['label_pred'] = prediction
        
        print('Saving Prediction...')
        pred_path = prediction_path + 'prediction.csv'
        predict_table.to_csv(pred_path, header=True)
        
        print("Exporting result to DB...")
        
        delta_pred = round((time.time() - pred_start_time)/60, 2)
        print('---Finished Predicting in {} minutes---'.format(delta_pred))
        
    else:
        # Predicting only
        pred_start_time = time.time()
        print('---Start the PREDICTING process---')
        
        print('Creating prediction table...')
        predict_table = load(DED=DED)
        
        print('Preprocessing Data...')
        predict_table = preprocess(predict_table)
        
        print('Saving Prediction Data...')
        save_data(predict_table, data_path = data_path)
        print('Done!')
        
        print('Loading existed Model...')
        models = load_models(path=model_path)

        print('Predicting...')
        if 'label' in predict_table.columns:
            X_pred = predict_table.drop('label', axis=1)
        else:
            X_pred = predict_table
        prediction, _ = ensemble_model_predict(models, X_pred)
        predict_table['label_pred'] = prediction
        
        print('Saving Prediction...')
        pred_path = prediction_path + 'prediction.csv'
        predict_table.to_csv(pred_path, header=True)
        
        print("Exporting result to DB...")
        
        delta_pred = round((time.time() - pred_start_time)/60, 2)
        print('---Finished Predicting in {} minutes---'.format(delta_pred))


# Step1: Create and Load Data

def load(DED=None, train=False, offset=None):
    """
    Using all functions from r6ssql to create table in database, 
    then load data to disk.
    
    ------
    Inputs:
        DED: Feature end day
        train: if True do both train and predict, else predict only
        offest: how many days of delay
    
    ------
    Returns:
        predict_table: pd.Dataframe
            The train or predict table  
    """
    create_train_sample(DED, sample_size=3000000)
    predict_table = load_train(DED)
    if train:
        year = DED[:4]
        month = DED[4:6]
        day = DED[6:]
        today_time = datetime.date(int(year), int(month), int(day))
        train_time = str(today_time-datetime.timedelta(offset)).replace("-", "")# offest +2 for delay?
        create_train_sample(train_time, sample_size=3000000)
        train_table = load_train(train_time)
        return predict_table, train_table
    else:
        return predict_table


# Step2: Data Preprocessing


def preprocess(data, train=False):
    """
    Preprocess the data for future train or predict,
    fill missing values, scaling, select features.
    
    ------
    Input: 
        data: pd.Dataframe
            The raw data table
    ------
    Return: cleaned table
    """
    data = data.pivot_table(index='user_key',
                            columns=['feature_type', 'interval_no'],
                            values='feature_value')
    data.columns = [
        f'{i}_{j}' if j not in ('',-1,-2) else f'{i}'
        for i,j in data.columns]
    if train:
        # we don't need the index for training
        data = data.reset_index(drop=True)

    data = data.fillna(0)
    
    ply_amt_feat_drop = [s for s in data.columns if 'ply_amt' in s][3:]
    data.drop(ply_amt_feat_drop, axis=1, inplace=True)
    
    time_features = ['ply_amt_1', 'ply_amt_2', 'ply_amt_3']   
    for timeft in time_features:
        data[timeft] /= 60

    return data

# Step3: Data Saving


def save_data(data, train=False, data_path = None):
    """
    Save the processed data as backup.
    The 'to_redict.csv' may have label for further backtesting.
    If we are predicting future: no label in general.
    If we set the DED as someday in the past, it might have label.
    
    ------
    Input:
        data: pd.Dataframe 
             The cleaned data table
        data_path: string
             The path to save the data
    """
    if train:
        train_path = data_path + 'train.csv'
        data.to_csv(train_path, index=None, header=True)
    else:
        to_predict_path = data_path + 'to_predict.csv'
        data.to_csv(to_predict_path, header=True)


# Step4: Train Model and Save


# 4.1 Training
def set_model_matrices(train_table):
    """
    Set up the model matrics for modeling
    
    ------
    Input: 
        train table: pd.Dataframe
    ------    
    Return: train/validation splits
    """
    X = train_table.drop('label', axis=1)
    y = train_table['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def logistic_cv_fit(X_train, y_train, n_jobs=8, cv_fold=3):
    """
    Fit a logistic regression using CV.
    
    Return: a sklearn grid search object
    """
    rc_scorer = make_scorer(recall_score, pos_label=1)
    clf = LogisticRegression(n_jobs=n_jobs)
    scaler = StandardScaler()
    pipeline = Pipeline(steps=[('scaler', scaler), ('logistic', clf)])

    param_grid = {
        'logistic__class_weight': [None, "balanced"]
    }

    gs_cv = GridSearchCV(
        pipeline,
        param_grid,
        scoring=rc_scorer,
        iid=False,
        cv=cv_fold,
        return_train_score=False
    )
    gs_cv.fit(X_train, y_train)

    return gs_cv


def random_forest_cv_fit(X_train, y_train, n_jobs=8, cv_fold=3):
    """
    Fit a random forest using CV.
    
    Return: a sklearn grid search object
    """
    rc_scorer = make_scorer(recall_score, pos_label=1)
    rf = RandomForestClassifier(n_jobs=n_jobs)
    scaler = StandardScaler()
    pipe = Pipeline(steps=[('scaler', scaler), ('rf', rf)])
    
    param_grid = {
        'rf__max_depth':[5, 10, 15],
        'rf__n_estimators':[50, 100],
        'rf__max_features': ['auto'],
        'rf__class_weight':["balanced"]
    }
    
    gs_cv = GridSearchCV(
        pipe,
        param_grid,
        scoring=rc_scorer,
        iid=False,
        cv=3,
        return_train_score=False)
    gs_cv.fit(X_train, y_train)
    
    return gs_cv


def mlp_pipe_fit(X_train, y_train):
    """
    Fit a multi-layer perceptron without using grid serach
    cross validation for hyper-parameters, because big data
    and the network structure is not so complicated.
    
    Return: a sklearn pipeline object
    """
    buy_count = X_train[y_train==1].shape[0]
    nonbuy_count = X_train[y_train==0].shape[0]
    if nonbuy_count > buy_count:
        # down sample the nonbuy group
        buy = X_train[y_train==1]
        nobuy = X_train[y_train==0].sample(buy_count)
        new_train_nn = buy.append(nobuy)
        ynew = list([1]*buy_count+[0]*buy_count)
        new_y = pd.Series(ynew)
    else:
        new_train_nn = X_train
        new_y = y_train

    clf = MLPClassifier(solver='adam', activation='logistic', alpha=1e-5, hidden_layer_sizes=(8,4))
    scaler = StandardScaler()
    mlp = Pipeline(steps=[('scaler',scaler),('mlp', clf)])
    mlp.fit(new_train_nn, new_y)
    
    return mlp


def xgb_pipe_fit(X_train, y_train, n_jobs=8):
    """
    Fit a xg-boost model.
    
    Return: a sklearn pipeline object
    """
    buy_count = X_train[y_train==1].shape[0]
    nonbuy_count = X_train[y_train==0].shape[0]
    scale = nonbuy_count/buy_count
    
    clf = XGBClassifier(
        scale_pos_weight=scale,
        max_depth=6,
        n_estimators=100,
        n_jobs=n_jobs)
    scaler = StandardScaler()
    xgb = Pipeline(steps=[('scaler',scaler),('xgb', clf)])
    xgb.fit(X_train, y_train)
    
    return xgb


def light_fit(X_train, y_train, X_test, y_test):
    """
    Fit a light-gbm model
    
    Return: a light-gbm base model object
    """
    # Create the LightGBM data containers
    train_data = lightgbm.Dataset(X_train, label=y_train)
    test_data = lightgbm.Dataset(X_test, label=y_test)
    # Train the model
    parameters = {
        'application': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 32,
        'feature_fraction': 0.25,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
    }

    lgbm = lightgbm.train(parameters,
                          train_data,
                          valid_sets=test_data,
                          num_boost_round=1000,
                          early_stopping_rounds=100,
                          verbose_eval=False)
    return lgbm


def build_models(X_train, y_train, X_test, y_test, n_jobs=8, cv_fold=3):
    """
    Build 5 models: logistic regression, random forest,
    multi-layer perceptron, xgboost and light-gbm.
    
    Input: splited training data
    Return: model list with 5 models of different classes
    """
    print("Building logistic regression model")
    lr = logistic_cv_fit(X_train, y_train, n_jobs=n_jobs, cv_fold=cv_fold)
    print("Building random forest model")
    rf = random_forest_cv_fit(X_train, y_train, n_jobs=n_jobs, cv_fold=cv_fold)
    print("Building multi-layer perceptron model")
    mlp = mlp_pipe_fit(X_train, y_train)
    print("Building xgboost model")
    xgb = xgb_pipe_fit(X_train, y_train, n_jobs=n_jobs)
    print("Building lightgbm model")
    lgbm = light_fit(X_train, y_train, X_test, y_test)
    models = [lr, rf, mlp, xgb, lgbm]
    return models


def ensemble_model_predict(models, X_test):
    """
    Use the models in model list to make prediction, ensemble those models
    by taking the majority vote as final label, average of probabilities
    as final probabilities of monetization 
    
    Input: 
        models: model list with 5 models of different classes
    Return:
        pred_labels: np.array
            predicted labels by the ensemble model
        pred_probs: np.array
            predicted probabilities by the ensemble model
    """
    pred_labels = np.zeros(X_test.shape[0])
    pred_probs = np.zeros(X_test.shape[0])
    for model in models:
        if 'lightgbm' not in str(type(model)):
            pred_labels += model.predict(X_test)
            pred_probs += model.predict_proba(X_test)[:,1]
        else:
            pred_probs += model.predict(X_test)
            prediction = [round(prob) for prob in model.predict(X_test)]
            pred_labels += prediction           
    # take majority vote as label
    for i in range(len(pred_labels)):
        if pred_labels[i] >= 3:
            pred_labels[i] = 1
        else:
            pred_labels[i] = 0
    # take average probabilites
    pred_probs /= 5
    return pred_labels, pred_probs
    

# 4.2 Save the model


def save_models(models, model_path=None):
    """
    Take a models list, save them to model path
    based on the model class type
    
    Input:
        models: list of all 5 models
    """
    for i, model in enumerate(models):
        file_path = model_path + '{}model.sav'.format(i)
        if 'GridSearchCV' in str(type(model)):
            joblib.dump(model.best_estimator_, file_path)
        elif 'Pipeline' in str(type(model)):
            joblib.dump(model, file_path)
        else:
            path = model_path + '{}model.txt'.format(i)
            model.save_model(path)
            
# 4.3 Evaluation: 


def model_eval(pred_labels, pred_probs, y_test, eval_path=None):
    """
    Do model evaluation on test data, return some common model 
    metrics including confusion matrix, accuracy, auc, roc...
    Plot the ROC curve, class probability distribution and 
    Decision/Recall curve.
    Save the result to eval_path.
    """
    # Confusion Matrix
    test_labels = {
        0: 'No(actual)',
        1: 'Buy(actual)'
    }
    y_test_relabel = np.vectorize(test_labels.get)(y_test)

    predict_labels = {
        0: 'No(predict)',
        1: 'Buy(predict)'
    }
    pred_relabel = np.vectorize(predict_labels.get)(pred_labels)
    
    y_actu = pd.Series(y_test_relabel, name='Actual')
    y_pred = pd.Series(pred_relabel, name='Predicted')
    confusion = pd.crosstab(y_actu, y_pred)
    
    # AUC
    y_scores = pred_probs
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    # ROC
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(311)
    ax1.set_title('ROC Curve')
    ax1.plot(fpr, tpr, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.axis([-0.005, 1, 0, 1.005])
    ax1.set_xticks(np.arange(0, 1, 0.05))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Recall)')

    # Recall_Precision VS Decision Threshold Plot
    ax2 = fig.add_subplot(312)
    ax2.set_title('Precision and Recall vs decision threshold')
    ax2.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    ax2.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Decision Threshold')
    ax2.legend(loc='best')

    # Class Probability Distribution
    ax3 = fig.add_subplot(313)
    ax3.set_title('Class Probability Distribution')
    buy = ax3.hist(pred_probs[y_test == 1], bins=40,
                   density=True, alpha=0.5)
    nonbuy = ax3.hist(pred_probs[y_test == 0], bins=40,
                      density=True, alpha=0.5)

    # Save the plots
    plot_path = eval_path + 'multiple_metrics_plot.png'
    fig.savefig(
        plot_path,
        bbox_inches='tight'
    )

    # Write result into a txt
    output = '\n'.join([
        'Model Evaluation:',
        '\tAccuracy: {accuracy:.3f}',
        '\tRecall for monetizers: {recall:.3f}',
        '\tPrecision for monetizers: {precision:.3f}',
        '\tF1 score: {f1:.3f}',
        '\tAUC: {auc:.3f}',
        '\n',
        'Confusion Matrix:',
        '{confusion}'
    ]).format(
        accuracy  = accuracy_score(pred_labels, y_test),
        recall    = recall_score(y_test, pred_labels, pos_label=1),
        precision = precision_score(y_test, pred_labels, pos_label=1),
        f1        = f1_score(y_test, pred_labels, pos_label=1),
        auc       = auc(fpr, tpr),
        confusion = confusion.to_csv(
            sep=' ', index=True, header=True, index_label='Confusion')
    )
    result_path = eval_path + 'output.txt'
    with open(result_path, 'w+') as f:
        f.write(output)

        
# Step5: Predict
# 5.1 Load the model

def load_models(path = None):
    """
    Collect 5 saved models from the model path.
    
    Return: 
        models: list of 5 trained models
    """
    model_paths = [path + file for file in os.listdir(path)]
    if len(model_paths) != 5:
        print('Missing {} models out of 5'.format(str(5-len(model_paths))))
        exit()
    else:
        models =[]
        for i in range(5):
            if 'txt' not in model_paths[i]:
                model = joblib.load(model_paths[i])
            else:
                model = lightgbm.Booster(model_file=model_paths[i])
            models.append(model)
    return models


def ensure_dir(file_path):
    """
    Helper function to create/ensure the path.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_date(DED):
    if len(DED) != 8:
        print('Error: The entered date is not 8 digits')
        return False
    try:
        date = int(DED)
    except:
        print('Error: Please enter a number')
        return False

    year = DED[:4]
    month = DED[4: 6]
    day = DED[-2:]

    if (int(month) < 1) or (int(month) > 12):
        print('Error: The entered month out of range')
        return False

    if (int(day) < 1) or (int(day) > 31):
        print('Error: The entered day out of range')
        return False
    else:
        return True

def check_params(cv, offset, n_jobs):
    """
    Helper function to check the global parameters.
    """
    commands = [int(cv),
                int(offset),
                int(n_jobs)]
    
    for com in commands:
        try:
            com
        except:
            print("Please enter a integer for cv, offest or n_jobs")
            return False
    
    if cv < 1 or cv > 5:
        print("CV is too large/small")
        return False
    else:
        return True
    
    
if __name__ == '__main__':
    main()
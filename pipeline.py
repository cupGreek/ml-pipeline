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

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
From sql import *
from configuration import *


def main():
    # global param in config	
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

    if train:  # do both training and predicting
        print('---Start the TRAINING process---')
        training_start_time = time.time()

        print('Creating Train & Prediction Table...')
        predict_table, train_table = load(train=True)

        print('Preprocessing Data...')
        predict_table = preprocess(predict_table)
        train_table = preprocess(train_table, train=True)
        delta_data = round((time.time() - training_start_time)/60, 2)
        print('Finished in {} minutes!'.format(delta_data))

        print('Saving Training & Prediction Data...')
        save_data(train_table, train=True, data_path=data_path)
        save_data(predict_table, data_path=data_path)
        print('Done!')

        print('Building Model...')
        modeling_start_time = time.time()
        X_train, X_test, y_train, y_test = set_model_matrices(train_table)
        pipeline = random_forest_cv_fit(X_train, y_train,
                                        n_jobs=n_jobs, cv_fold=cv)

        delta_model = round((time.time() - modeling_start_time)/60, 2)
        print('Finished Modeling in {} minutes!'.format(delta_model))

        print('Saving Model...')
        save_model(pipeline, model_path=model_path)
        print('Done!')

        print('Model Evaluating...')
        pipeline = load_model()
        model_eval(pipeline, X_test, y_test, eval_path=eval_path)
        print('Done!')

        delta_train = round((time.time() - training_start_time)/60, 2)
        print('---Finished Training in {} minutes---'.format(delta_train))

        # predict
        pred_start_time = time.time()
        print('---Start the PREDICTING process---')

        print('Loading Model...')
        model_path = model_path + 'model.sav'
        pipeline = load_model(path=model_path)

        print('Predicting...')
        if 'label' in predict_table.columns:
            X_test = predict_table.drop('label', axis=1)
        else:
            X_test = predict_table
        prediction = pipeline.predict(X_test)
        predict_table['label_pred'] = prediction

        print('Saving Prediction...')
        pred_path = prediction_path + 'prediction.csv'
        predict_table.to_csv(pred_path, header=True)

        print("Exporting result to DB...")
        #export_data(new_table=predict_table, df = predict_table)
        
        delta_pred = round((time.time() - pred_start_time)/60, 2)
        print('---Finished Predicting in {} minutes---'.format(delta_pred))

    else:
        # Predicting only
        pred_start_time = time.time()
        print('---Start the PREDICTING process---')

        print('Creating prediction table...')
        predict_table = load()

        print('Preprocessing Data...')
        predict_table = preprocess(predict_table)

        print('Saving Prediction Data...')
        save_data(predict_table, data_path=data_path)
        print('Done!')

        print('Loading existed Model...')
        model_path = model_path + 'model.sav'
        pipeline = load_model(path=model_path)

        print('Predicting...')
        if 'label' in predict_table.columns:
            X_test = predict_table.drop('label', axis=1)
        else:
            X_test = predict_table
        prediction = pipeline.predict(X_test)
        predict_table['label_pred'] = prediction

        print('Saving Prediction...')
        pred_path = prediction_path + 'prediction.csv'
        predict_table.to_csv(pred_path, header=True)

        print("Exporting result to DB...")
        #export_data(new_table=predict_table, df = predict_table)
        
        delta_pred = round((time.time() - pred_start_time)/60, 2)
        print('---Finished Predicting in {} minutes---'.format(delta_pred))


# Step1: Create and Load Data
# Use all functions from r6ssql.py

def load(today=DED, train=False, offset=offset):
    """
    Using all functions from `sql` to create table in database, 
    then load data to memory.
    
    Inputs:
        DED: Feature end day
        train: if true do both train and predict, else predict only
        offest: how many days of delay
    Returns:
        train/predict tables  
    """
    create_train_sample(today, sample_size=3000000)
    predict_table = load_train(today)
    if train:
        year = today[:4]
        month = today[4:6]
        day = today[6:]
        today_time = datetime.date(int(year), int(month), int(day))
        # add offset to below code once Jes done with sql
        train_time = str(today_time-datetime.timedelta(14)).replace("-", "")
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
    
    Input: raw data table
    Return: cleaned table
    """
    data = data.pivot_table(index='user_key',
                            columns=['feature_type', 'interval_no'],
                            values='feature_value')
    data.columns = [
        f'{i}_{j}' if j not in ('', -1, -2) else f'{i}'
        for i, j in data.columns]
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


def save_data(data, train=False, data_path=None):
    """
    Save the processed data as backup.
    The 'to_redict.csv' may have label for further backtesting.
    If we are predicting future: no label in general.
    If we set the DED as someday in the past, it might have label.
    
    Input:
        data: cleaned data table
        data_path: path to save the data
    """
    if train:
        train_path = data_path + 'train.csv'
        data.to_csv(train_path, index=None, header=True)
    else:
        to_predict_path = data_path + 'to_predict.csv'
        data.to_csv(to_predict_path, header=True)


# Step4: Train Model and Save


# 4.1 Training
def set_model_matrices(data):
    """
    Set up the model matrics for modeling
    
    Input: train table
    Return: train/validation splits
    """
    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# def logistic_pipeline_fit(X_train, y_train, n_jobs=8, cv_fold=3):
#     """
#     Fit a logistic regression using CV.
    
#     Return: a sklearn grid search object
#     """
#     rc_scorer = make_scorer(recall_score, pos_label=1)
#     clf = LogisticRegression(n_jobs=n_jobs)
#     scaler = StandardScaler()
#     pipeline = Pipeline(steps=[('scaler', scaler), ('logistic', clf)])

#     param_grid = {
#         'logistic__class_weight': [None, "balanced"]
#     }

#     gs_cv = GridSearchCV(
#         pipeline,
#         param_grid,
#         scoring=rc_scorer,
#         iid=False,
#         cv=cv_fold,
#         return_train_score=False
#     )
#     gs_cv.fit(X_train, y_train)

#     return gs_cv


# dev 0.1
def random_forest_cv_fit(X_train, y_train, n_jobs=8, cv_fold=3):
    """
    Fit a random forest using CV.
    
    Return: a sklearn grid search object
    """
    rc_scorer = make_scorer(recall_score, pos_label=1)
    rf = RandomForestClassifier(n_jobs=n_jobs)
    pipe = Pipeline(steps=[('rf', rf)])
    
    param_grid = {
        'rf__max_depth':[5, 10, 15],
        'rf__n_estimators':[50, 100, 150],
        'rf__max_features': ['auto', 10],
        'rf__class_weight':["balanced"]
    }
    
    gs_cv = GridSearchCV(
        pipe,
        param_grid,
        scoring=rc_scorer,
        iid=False,
        cv=cv_fold,
        return_train_score=False)
    gs_cv.fit(X_train, y_train)
    
    return gs_cv

# 4.2 Save the model


def save_model(pipeobj, model_path=None):
    file_path = model_path + 'model.sav'
    joblib.dump(pipeobj.best_estimator_, file_path)


# 4.3 Evaluation: Logistic Regression


def model_eval(pipeline, X_test, y_test, eval_path=None):
    """
    Do model evaluation on test data, return some common model 
    metrics including confusion matrix, accuracy, auc, roc...
    Plot the ROC curve, class probability distribution and 
    Decision/Recall curve.
    Save the result to eval_path.
    """
    prediction = pipeline.predict(X_test)

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
    pred_relabel = np.vectorize(predict_labels.get)(prediction)

    y_actu = pd.Series(y_test_relabel, name='Actual')
    y_pred = pd.Series(pred_relabel, name='Predicted')
    confusion = pd.crosstab(y_actu, y_pred)

    # AUC
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    pred_prob = pipeline.predict_proba(X_test)

    # ROC
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(411)
    ax1.set_title('ROC Curve')
    ax1.plot(fpr, tpr, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.axis([-0.005, 1, 0, 1.005])
    ax1.set_xticks(np.arange(0, 1, 0.05))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Recall)')

    # Recall_Precision VS Decision Threshold Plot
    ax2 = fig.add_subplot(412)
    ax2.set_title('Precision and Recall vs decision threshold')
    ax2.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    ax2.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Decision Threshold')
    ax2.legend(loc='best')

    # Class Probability Distribution
    ax3 = fig.add_subplot(413)
    ax3.set_title('Class Probability Distribution')
    buy = ax3.hist(pred_prob[y_test == 1][:, 1], bins=40,
                   density=True, alpha=0.5)
    nonbuy = ax3.hist(pred_prob[y_test == 0][:, 1], bins=40,
                      density=True, alpha=0.5)
    # Feature importance
    ax4 = fig.add_subplot(414)
    ax4.set_title('Feature Imprtance')
    feature_importance = pipeline.steps[0][1].feature_importances_
    pd.Series(feature_importance, index=X_test.columns).nlargest(X_test.shape[1]).plot(kind='barh')
    
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
        accuracy  = accuracy_score(prediction, y_test),
        recall    = recall_score(y_test, prediction, pos_label=1),
        precision = precision_score(y_test, prediction, pos_label=1),
        f1        = f1_score(y_test, prediction, pos_label=1),
        auc       = auc(fpr, tpr),
        confusion = confusion.to_csv(
            sep=' ', index=True, header=True, index_label='Confusion')
    )
    result_path = eval_path + 'output.txt'
    with open(result_path, 'w+') as f:
        f.write(output)


# Step5: Predict
# 5.1 Load the model

def load_model(path='./r6s_14/models/model.sav'):
    try:
        pipeline = joblib.load(path)
    except IOError:
        print('An error occured trying to read the model file, may not exist')
    return pipeline

# 5.2 export to db

def export_data(new_table, df=None): 
    with gi.BulkLoad(new_table, cleanup=True) as load:
        load.columns = []
        for row in df.values.tolist():
            load.put(row)

def ensure_dir(file_path):
    """
    Helper function to create/ensure the path.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_date(DED):
    """
    Helper function to check the date in configuration.
    """
    if len(DED) != 8:
        print('Error: The entered date is not 8 digits')
        return False
    try:
        date = int(DED)
    except ValueError:
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
        except ValueError:
            print("Please enter a integer for cv, offest or n_jobs")
            return False

    if cv < 1 or cv > 5:
        print("CV is too large/small")
        return False
    else:
        return True


if __name__ == '__main__':
    main()
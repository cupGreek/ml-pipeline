# ml-pipeline
A end-to-end machine learning model pipeline in game industry

### Pipeline structure

*configuration.py*: user-defined variables like cv-fold, number of threads, file_path..

*sql.py*: sql query to load data from database

*pipeline.py*:

The pipeline does the following:

for **training**:

* 1. Load data from database: `load_data()` function
* 2. Preprocess data: `preprocessing_data()`
* 3. Save the preprocessed data to `data` folder
* 4. Build models using `build_models()`, return model list of common API ML models
* 5. Save the models to `model` folder
* 6. Make prediction of test data using `ensemble_model_predict()`
* 7. Evaluate the model performance using `model_eval()`

for **predicting**:

* 1. Load data from database: `load_data()` by setting `train=False`
* 2. Preprocess data: `preprocessing_data()`by setting `train=False`
* 3. Save the preprocessed data to `prediction` folder called `to_predict.csv`
* 4. Load existing models using `load_models()`
* 5. Make prediction of data to be predicted using `ensemble_model_predict()`
* 6. Save the prediction to `prediction` folder called `prediction`
* 7. Export the prediction to database 

### Usage

```bash
$ python3 pipeline.py
```

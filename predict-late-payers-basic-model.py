# Import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Preprocessing and Pipeline libraries
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import pickle

print("\nLoading training data...")
# load training data
test_data = pd.read_csv("data/peerLoanTraining.csv", engine='python', header=0)

# Separate out X and y
X_train = test_data.loc[:, test_data.columns != 'is_late']
y_train = test_data['is_late']

# load test data
test_data = pd.read_csv("data/peerLoanTest.csv", engine='python', header=0)

# Separate out X and y
X_test = test_data.loc[:, test_data.columns != 'is_late']
y_test = test_data['is_late']

# Preprocessing Steps
numeric_features = ['loan_amnt',
                    'int_rate', 'annual_inc', 'revol_util',
                    'dti', 'delinq_2yrs'
                   ]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ])

categorical_features = ['purpose','grade', 'emp_length', 'home_ownership']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ]
    )

rf = RandomForestClassifier(random_state=42)


# Random Search for best hyperparameters

# number of trees in randomized cv
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# number of features considered at every split
max_features =['auto', 'sqrt']

# max levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=10)]
max_depth.append(None)

# min number of samples required to split a node
min_samples_split = [2, 5, 10]

# min number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# method of selecting samples for training each tree
bootstrap = [True, False]

# Make the random grid
rand_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap': bootstrap}


# Combine preprocessing with classifier
latePaymentsModel = make_pipeline(
    preprocess,
    RandomizedSearchCV(estimator = rf, param_distributions= rand_grid, n_iter = 100, cv = 3, verbose =2, random_state = 42, n_jobs = -1))

# Fit the pipeline to the training data (fit is for both the preprocessing and the classifier)
print("\nTraining model ...")
latePaymentsModel.fit(X_train, y_train)

# Save the trained model as a pickle file
print("\nSaving model ...")
file = open('latePaymentsModel.pkl', 'wb')
pickle.dump(latePaymentsModel, file)
file.close()

# load the pickled model
print("\nLoading saved model to make example predictions...")
pickledModel = pickle.load(open('models/latePaymentsModel.pkl','rb'))

# Save the data columns from training
model_columns = list(X_train.columns)
print("\nSaving model columns ...")
file = open('model_columns.pkl','wb')
pickle.dump(model_columns, file)
file.close()

# Make a prediction for a likely on time payer
payOnTimePrediction = {
    'loan_amnt': [100],
    'int_rate': [0.02039],
    'purpose': ['credit_card'],
    'grade': ['A'],
    'annual_inc': [80000.00],
    'revol_util': [0.05],
    'emp_length': ['10+ years'],
    'dti': [1.46],
    'delinq_2yrs': [0],
    'home_ownership': ['RENT']
    }
payOnTimePredictionDf = pd.DataFrame.from_dict(payOnTimePrediction)

print("\nPredicting class probabilities for likely on-time payer:")
print(pickledModel.predict_proba(payOnTimePredictionDf))

# Prediction for a likely late payer
payLatePrediction = {
    'loan_amnt': [5000],
    'int_rate': [0.13],
    'purpose': ['other'],
    'grade': ['C'],
    'annual_inc': [45000.00],
    'revol_util': [0.202],
    'emp_length': ['6 years'],
    'dti': [9.92],
    'delinq_2yrs': [0],
    'home_ownership': ['MORTGAGE']
    }
payLatePredictionDf = pd.DataFrame.from_dict(payLatePrediction)

print("\nPredicting class probabilities for a likely late payer:")
print(payLatePredictionDf)
print(pickledModel.predict_proba(payLatePredictionDf))

# Predict class probabilities for a set of records using the test set
print("\nPredicting class probabilities for the test data set:")
print(pickledModel.predict_proba(X_test))

from sklearn.metrics import accuracy_score
print("Accuracy:\n%s" % accuracy_score(y_test, pickledModel.predict(X_test)))

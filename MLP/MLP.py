import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate

df = pd.read_csv('../final_data.csv')

y = df.Active
data = df.drop(['Active'], axis=1)
column = data.columns
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(data), columns=column)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
mcc = make_scorer(matthews_corrcoef, greater_is_better=True)

param = {'hidden_layer_sizes':[(100,), (100,100),(100,100,100),(100,100,100,100)],
        'activation':['identity','logistic','tanh','relu'],
        'solver':['lbfgs','sgd','adam'],
         'max_iter':[100, 200, 300, 400, 500],
         'batch_size':[64, 128, 'auto'],
        'learning_rate':['constant','invscaling','adaptive']}
        
        
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mlp = MLPClassifier(random_state=42)
grid = GridSearchCV(estimator=mlp,
                   param_grid=param,
                   scoring=mcc,
                   n_jobs=12,
                   cv=cv)
                   
grid.fit(x_train, y_train)

print(grid.best_params_)

dnn = MLPClassifier(hidden_layer_sizes=(100,100,100,100), activation='relu', learning_rate='constant',
                      batch_size=128, max_iter=100, solver='adam', random_state=42, shuffle=True)
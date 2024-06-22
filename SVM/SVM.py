import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../final_data.csv')
y = df.Active
y.replace(0, -1, inplace=True)
data = df.drop(['Active'], axis=1)
column = data.columns
svm_scaler = StandardScaler()
x = data
mcc = make_scorer(matthews_corrcoef, greater_is_better=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
x_train = svm_scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train, columns=column)
x_test = svm_scaler.transform(x_test)
x_test = pd.DataFrame(x_test, columns=column)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
parm_grid = {'C':np.arange(1.0, 20, 0.1),
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma':['scale', 'auto']}
svc = SVC(random_state=42)

grid = GridSearchCV(estimator=svc,
                   param_grid=parm_grid,
                   scoring=mcc,
                   n_jobs=12,
                   cv=cv
                   )
                   
grid.fit(x_train, y_train)

print(grid.best_params_)

svc2 = SVC(C=3.4, gamma='scale', kernel='rbf', random_state=42)
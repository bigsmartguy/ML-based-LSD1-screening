import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


df = pd.read_csv('../final_data.csv')
x = df.drop(['Active'], axis=1)
y = df.Active
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
mcc = make_scorer(matthews_corrcoef, greater_is_better=True)
rfc = RandomForestClassifier(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

score = {'accuracy':'accuracy', # 准确度
        'recall':'recall',  # 灵敏度
         'precision':'precision', # 精确度
        'f1':'f1', 'mcc':mcc
        }
		
validation_score = cross_validate(rfc,x_train,y_train,
                                    scoring=score,
                                    cv=cv,
                                    verbose=False,
                                    n_jobs=6,
                                    error_score='raise')
									
rfc.fit(x_train, y_train)

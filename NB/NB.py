# 标准化数据，然后合并
import pandas as pd
from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score, accuracy_score, precision_score,recall_score, confusion_matrix

continuous_data = pd.read_csv('./continuous.csv')
y = continuous_data.Active
x_con = continuous_data.drop(['Active'], axis=1)
scaler = StandardScaler()
x_continu = scaler.fit_transform(x_con)
x_continu = pd.DataFrame(x_continu, columns=x_con.columns)
x_discrete = pd.read_excel('./discrete.xlsx')
x = pd.concat([x_discrete, x_continu], axis=1)

mcc = make_scorer(matthews_corrcoef, greater_is_better=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
indices = [x for x in range(0, 88,1)]
bayes = MixedNB(categorical_features=indices)
score = {
    'accuracy':'accuracy',
    'recall':'recall',
    'precision':'precision',
    'f1':'f1', 'mcc':mcc
}
validation_score = cross_validate(bayes, x_train, y_train,
                                  scoring=score,
                                  cv=cv,
                                  verbose=False,
                                  n_jobs=6,
                                  error_score='raise'
)
# validation_score
# validation_score['test_accuracy'].mean()
# validation_score['test_precision'].mean()
# validation_score['test_recall'].mean()
# validation_score['test_f1'].mean()
# validation_score['test_mcc'].mean()
## 测试集指标
bayes.fit(x_train, y_train)
accuracy_score(y_test, bayes.predict(x_test))
precision_score(y_test, bayes.predict(x_test))
recall_score(y_test, bayes.predict(x_test))
f1_score(y_test, bayes.predict(x_test))
matthews_corrcoef(y_test, bayes.predict(x_test))
confusion_matrix(y_test, bayes.predict(x_test))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
import pandas as pd
import copy
import numpy as np
#In this file, run divexplorer without remedy methods

# Need to install divexplorer for experiments

# pip install divexplorer

from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence

# Read and process the data

# url is the location of the data
url = "https://raw.githubusercontent.com/niceIrene/remedy/main/datasets/bar_pass.csv"
data = pd.read_csv(url)

bar_y = 'bar1'
columns_all =['decile1b', 'decile3', 'decile1', 'sex', 'lsat', 'ugpa', 'grad',
       'fulltime', 'fam_inc', 'gender', 'race1', 'tier']
columns_bar = ['fam_inc', 'race1', 'lsat', 'ugpa', 'sex', 'fulltime']

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def get_train_test(data, split, list_cols, y_label):
  all_list = copy.deepcopy(list_cols)
  all_list.append(y_label)
  data = pd.DataFrame(data, columns = all_list)
  train_set,test_set = split_train_test(data,split)
  print(len(train_set), "train +", len(test_set), "test")
  train_x = pd.DataFrame(train_set, columns = list_cols)
  train_label = train_set[y_label]
  test_x = pd.DataFrame(test_set, columns = list_cols)
  test_label = test_set[y_label]
  return train_x, test_x, train_label, test_label, train_set, test_set

train_x, test_x, train_label, test_label, train_set, test_set  = get_train_test(data, 0.3, columns_all, bar_y)


scoring = make_scorer(accuracy_score)

# #####################

#Logisitic Regression Settings

# #####################
param_gridlg = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100]
}
logreg = LogisticRegression(random_state=42, max_iter=1000)
gridlg = GridSearchCV(logreg, param_grid=param_gridlg, scoring=scoring, cv=5)


# #####################

#Decision Tree Classifier Settings

# #####################
param_griddt = {
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeClassifier(random_state=42)
griddt = GridSearchCV(dt, param_grid=param_griddt, scoring=scoring, cv=5)

# #####################

#Random Forest Classifier Settings

# #####################
param_gridrf = {'criterion': ['gini', 'entropy'], 'max_depth': [10, 20, 30, 40, 50, 100], 'random_state':[17]}
rf = RandomForestClassifier(random_state=42)
gridrf = GridSearchCV(rf, param_grid=param_gridrf, scoring=scoring, cv=5)

# #####################

# SVM Settings

# #####################

clf = SVC(kernel='rbf', C=1.0, gamma = 'scale', random_state =42)


# #####################

# Run DivExplorer Results

# #####################

# fit the model, get the predicted results
clf.fit(train_x, train_label)
test_predict = clf.predict(test_x)
test_set['predicted'] = test_predict

#Print accuracy without remedy
accuracy = accuracy_score(test_label, test_set['predicted'])
print("accuracy is " , accuracy)

# Preprocessing for divexplorer
class_map={'N': 0, 'P': 1}
columns_bar.extend([bar_y, "predicted"])
df = pd.DataFrame(test_set, columns = columns_bar)

columns_bar.remove(bar_y)
columns_bar.remove('predicted')

min_sup=0.1

# Computes the fairness scoring in terms of the support for unfair group
def fairness_score_computation(d, metrics):
    sum_of_score = 0
    for idx, row in d.iterrows():
      sum_of_score += row['support'] * row[metrics]
    return sum_of_score

# Start DivExplorer results generation
fp_diver=FP_DivergenceExplorer(df, bar_y, "predicted", class_map=class_map)
FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_fpr", "d_fnr", "d_accuracy"])

fp_divergence_fpr=FP_Divergence(FP_fm, "d_fpr")
fp_divergence_fnr=FP_Divergence(FP_fm, "d_fnr")
fp_divergence_acc=FP_Divergence(FP_fm, "d_accuracy")

INFO_VIZ=["support", "itemsets",  fp_divergence_fpr.metric, fp_divergence_fpr.t_value_col]
INFO_VIZ2=["support", "itemsets",  fp_divergence_fnr.metric, fp_divergence_fnr.t_value_col]
INFO_VIZ3=["support", "itemsets",  fp_divergence_acc.metric, fp_divergence_acc.t_value_col]
eps=0.01
K=1000
d = fp_divergence_fpr.getDivergence(th_redundancy=eps)[INFO_VIZ].head(K)
d2 = fp_divergence_fnr.getDivergence(th_redundancy=eps)[INFO_VIZ2].head(K)
d3 = fp_divergence_acc.getDivergence(th_redundancy=eps)[INFO_VIZ3].head(K)

pd.options.display.max_rows = 200
d = fp_divergence_fpr.getDivergence(th_redundancy=0)[INFO_VIZ].head(K)
# summerization

d = fp_divergence_fpr.getDivergence(th_redundancy=eps)[INFO_VIZ].head(K)
d= d[d['d_fpr'] > 0]
d2= d2[d2['d_fnr'] > 0]
d3= d3[d3['d_accuracy'] > 0]

dfpr = fairness_score_computation(d, 'd_fpr')
dfnr = fairness_score_computation(d2, 'd_fnr')
dacc = fairness_score_computation(d3, 'd_accuracy')

print("dfpr: ", dfpr)
print("dfnr: ", dfnr)
print("dacc: ", dacc)


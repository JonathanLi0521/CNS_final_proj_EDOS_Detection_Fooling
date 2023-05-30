# Data Processing
from pathlib import Path
import pandas as pd
import numpy as np
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
# Tree Visualisation
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

mode = 'multiclass' # mode = ['binary', 'multiclass']

# Data
def load_data(preprocess_dir, split, label='label'):
    data = pd.read_csv(preprocess_dir / f'{split}.csv')
    X = data.drop(['id', 'label', 'attack_cat'], axis=1)
    y = data[label]
    return data, X, y

preprocess_dir = Path('./preprocess/')
label = 'label' if mode[0] ==  'b' else 'attack_cat'
_, X_train, y_train = load_data(preprocess_dir, 'train', label)
data_test, X_test, y_test = load_data(preprocess_dir, 'test', label)
# Split validation set
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.7, random_state=1)

# Training
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# Hyperparameter Tuning
param_dist = {'n_estimators': randint(10,100),
              'max_depth': randint(5,20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(
    rf, 
    param_distributions = param_dist, 
    n_iter=10, 
    cv=5,
    random_state=1,
    return_train_score=True
)
rand_search.fit(X_train, y_train)
results_all = pd.DataFrame.from_dict(rand_search.cv_results_)
results = results_all[['params', 'mean_train_score', 'mean_test_score']]
results = results.sort_values(by=['mean_test_score'], ascending=False)
results = results.round(decimals=4)
results.to_csv(f'./rf_results_{mode}.csv')

# Evaluation
best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)
print('Best score:', rand_search.best_score_)
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
precision_avg = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average=None)
recall_avg = recall_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Precision (weighted):", precision_avg)
print("Recall:", recall)
print("Recall (weighted):", recall_avg)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap=plt.cm.Blues
)
plt.show()

feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances[:20].plot.bar()

# Fooling cases
data_fool = data_test[np.logical_and(y_pred == 0, y_test != 0)]
data_fool.to_csv(f'./rf_fool_{mode}.csv', index=False)

# Visualizing results
# rf_dir = Path('./RF/')
# rf_dir.mkdir(parents=True, exist_ok=True)
# n = 3
# for i in range(n):
#     tree = best_rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=5, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     graph.render(format='png', outfile=rf_dir / f'rf_result{i}.png')

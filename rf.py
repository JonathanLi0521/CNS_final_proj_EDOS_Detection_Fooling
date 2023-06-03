# Data Processing
from pathlib import Path
import pandas as pd
import numpy as np
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
# # Tree Visualisation
# from sklearn.tree import export_graphviz
# import graphviz

preprocess_dir = Path('./preprocess/')
output_dir = Path('./rf/')
output_dir.mkdir(parents=True, exist_ok=True)
mode = 'multiclass' # mode = ['binary', 'multiclass']
feature_selection = False
weighted = False
tuning = False
tuning_adv = False

# Data
label = 'label' if mode[0] ==  'b' else 'attack_cat'
def load_data(split):
    data = pd.read_csv(preprocess_dir / f'{split}.csv')
    X = data.drop(['id', 'label', 'attack_cat'], axis=1)
    y = data[label]
    return data, X, y

data_train, X_train, y_train = load_data('train')
data_test, X_test, y_test = load_data('test')

# Feature selection
if feature_selection:
    correlation = data_train.corr()
    correlation_target = abs(correlation[label])
    relevant_features = correlation_target[correlation_target > 0.3]
    relevant_features = relevant_features.drop(['id', 'label', 'attack_cat']).index.tolist()
    X_train = X_train[relevant_features]
    X_test = X_test[relevant_features]

# Split validation dataset
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.7, random_state=1)

# Training
def hyperparameter_tuning(X_train, y_train):
    param_dist = {'n_estimators': randint(10,100),
                  'max_depth': randint(5,20)}
    rf = RandomForestClassifier(class_weight='balanced' if weighted else None)
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
    return rand_search, results

if tuning:
    rand_search, results = hyperparameter_tuning(X_train, y_train)
    # results.to_csv(output_dir / f'rf_results_{mode}.csv')
    
    best_rf = rand_search.best_estimator_
    print('Best hyperparameters:',  rand_search.best_params_)
    print('Best score:', rand_search.best_score_)
else:
    best_rf = RandomForestClassifier(n_estimators=38, max_depth=17, class_weight='balanced' if weighted else None)
    best_rf.fit(X_train, y_train)

# Visualizing results
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
#     graph.render(format='png', outfile=output_dir / f'rf_tree{i}.png')

# Evaluation
# Feature importance
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances[:20].plot.bar()

# Model performance
def evaluate(y_test, y_pred, cm_title='Confusion matrix'):
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
    # ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        cmap=plt.cm.Blues,
        ax=ax
    )
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.title(cm_title, fontsize=20)
    plt.show()
    
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_avg': precision_avg,
        'recall': recall,
        'recall_avg': recall_avg,
        'cm': cm
    }
    return evaluation

# y_pred = best_rf.predict(X_test)
# print('Test evaluation:')
# evaluation_test = evaluate(y_test, y_pred, 'Testing set')
# data_correct = data_test[np.logical_and(y_test != 0, y_pred == y_test)]
# data_fool = data_test[np.logical_and(y_pred == 0, y_test != 0)]
# data_correct.to_csv(output_dir / f'rf_correct_{mode}.csv', index=False)
# data_fool.to_csv(output_dir / f'rf_fool_{mode}.csv', index=False)

# Fooling case analysis
# Split testing set
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, train_size=0.5, random_state=1)

# Model performance
y_pred1 = best_rf.predict(X_test1)
print('test1 evaluation:')
evaluation_test1 = evaluate(y_test1, y_pred1, 'Prediction on testing set 1')

# Adversarial attack
# Label: fooled or not
y_fool1 = np.logical_and(y_pred1 == 0, y_test1 != 0)

# Training
if tuning_adv:
    rand_search_adv, _ = hyperparameter_tuning(X_test1, y_fool1)
    best_rf_adv = rand_search_adv.best_estimator_
    print('Best hyperparameters:',  rand_search_adv.best_params_)
    print('Best score:', rand_search_adv.best_score_)
else:
    best_rf_adv = RandomForestClassifier(n_estimators=38, max_depth=17, class_weight='balanced' if weighted else None)
    best_rf_adv.fit(X_test1, y_fool1)

# Attack efficiency: model performance reduction
# Model performance before attack
y_pred2 = best_rf.predict(X_test2)
print('Evaluation before attack:')
evaluation_test2 = evaluate(y_test2, y_pred2, 'Prediction on testing set 2')

X_test2_attack = X_test2[y_test2 != 0]
y_test2_attack = y_test2[y_test2 != 0]
y_pred2_attack = best_rf.predict(X_test2_attack)
print('Evaluation before attack (attack only):')
evaluation_test2_attack = evaluate(y_test2_attack, y_pred2_attack, 'Prediction on testing set (attack only)')

# Model performance after attack
y_fool2_pred = best_rf_adv.predict(X_test2)
X_test2_adv = X_test2[y_fool2_pred]
y_test2_adv = y_test2[y_fool2_pred]
y_pred2_adv = best_rf.predict(X_test2_adv)
print('Evaluation after attack:')
evaluation_test2_adv = evaluate(y_test2_adv, y_pred2_adv, 'Prediction on adversarial testing set')

y_fool2_attack_pred = best_rf_adv.predict(X_test2_attack)
X_test2_attack_adv = X_test2_attack[y_fool2_attack_pred]
y_test2_attack_adv = y_test2_attack[y_fool2_attack_pred]
y_pred2_attack_adv = best_rf.predict(X_test2_attack_adv)
print('Evaluation after attack (attack only):')
evaluation_test2_adv_attack = evaluate(y_test2_attack_adv, y_pred2_attack_adv, 'Prediction on adversarial testing set (attack only)')

# Fooling case prediction performance
y_fool2 = np.logical_and(y_pred2 == 0, y_test2 != 0)
print('Fooling case evaluation:')
evaluation_fool2 = evaluate(y_fool2, y_fool2_pred, 'Fooling case prediction')

y_fool2_attack = np.logical_and(y_pred2_attack == 0, y_test2_attack != 0)
print('Fooling case evaluation:')
evaluation_fool2_attack = evaluate(y_fool2_attack, y_fool2_attack_pred, 'Fooling case prediction (attack only)')

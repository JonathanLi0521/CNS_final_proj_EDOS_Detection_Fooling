# Data Processing
from pathlib import Path
import json
import pandas as pd
import numpy as np
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import randint
import matplotlib.pyplot as plt
import joblib

import warnings
warnings.filterwarnings("ignore")
# # Tree Visualisation
# from sklearn.tree import export_graphviz
# import graphviz

preprocess_dir = Path('./preprocess/')
output_dir = Path('./rf_svm/')
output_dir.mkdir(parents=True, exist_ok=True)
mode = 'multiclass' # mode = ['binary', 'multiclass']
feature_selection = False
weighted = False
tuning = False
tuning_adv = False
tuning_retrain = False

# Data
attack2idx_path = preprocess_dir / 'attack2idx.json'
attack2idx = json.loads(attack2idx_path.read_text())
attack_name = list(attack2idx.keys())

label = 'label' if mode[0] ==  'b' else 'attack_cat'
def load_data(split):
    data = pd.read_csv(preprocess_dir / f'{split}.csv')
    X = data.drop(['id', 'label', 'attack_cat'], axis=1)
    y = data[label]
    return data, X, y

data_train, X_train, y_train = load_data('train')
data_test, X_test, y_test = load_data('test')


best_rf = joblib.load('./rf/best_rf.model') # trained detection model

# Evaluation

# Model performance
def evaluate(y_test, y_pred, cm_title='Confusion matrix', display_labels=attack_name, return_fool_ratio=True, count = 0):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    precision_avg = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average=None)
    recall_avg = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average=None)
    fl_avg = f1_score(y_test, y_pred, average='weighted')
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Precision (weighted):", precision_avg)
    # print("Recall:", recall)
    # print("Recall (weighted):", recall_avg)
    # print("F1 Score:", f1)
    # print("F1 Score (weighted):", fl_avg)
    
    cm = confusion_matrix(y_test, y_pred)    
    # ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=display_labels,
        xticks_rotation=45 if len(display_labels) > 2 else 0,
        values_format='d',
        cmap=plt.cm.Blues,
        ax=ax
    )
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.title(cm_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./rf_svm/fig{count}.png")
    
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_avg': precision_avg,
        'recall': recall,
        'recall_avg': recall_avg,
        'f1': f1,
        'fl_avg': fl_avg,
        'cm': cm
    }
    # Fooling ratio
    if return_fool_ratio:
        evaluation['fool_ratio'] = np.array([sum(np.logical_and(y_pred == 0, y_test == attack_cat)) / (sum(y_test == attack_cat) + 1e-5)
                      for attack_cat in range(1, 10)])
        evaluation['fool_ratio_avg'] = np.array(sum(np.logical_and(y_pred == 0, y_test != 0)) / len(y_test))
        print("========================================================================")
        print(evaluation['fool_ratio_avg'])
        print(evaluation['fool_ratio'])
        print("========================================================================")
        
    return evaluation

# Fooling case analysis
# Split testing set
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, train_size=0.5, random_state=1)

# Adversarial attack
# Label: fooled or not
X_test1_attack = X_test1[y_test1 != 0]
y_test1_attack = y_test1[y_test1 != 0]
y_pred1_attack = best_rf.predict(X_test1_attack)
y_fool1 = np.logical_and(y_pred1_attack == 0, y_test1_attack != 0)

best_svm_adv = joblib.load("./svm/best_svm_adv.model") # trained svm adv model (trained by svm detection model)

def attack_efficiency(X_test1, y_test1, X_test2, y_test2, best_rf, tag='', count = 0):
    # Model performance
    y_pred1 = best_rf.predict(X_test1)
    print('test1 evaluation:')
    evaluation_test1 = evaluate(y_test1, y_pred1, f'Test1{tag}', count=1+count)
    
    # Attack efficiency: model performance reduction
    # Model performance before attack
    # y_pred2 = best_svm.predict(X_test2)
    # print('Evaluation before attack:')
    # evaluation_test2 = evaluate(y_test2, y_pred2, f'Test2{tag}')
    
    X_test2_attack = X_test2[y_test2 != 0]
    y_test2_attack = y_test2[y_test2 != 0]
    y_pred2_attack = best_rf.predict(X_test2_attack)
    print('Evaluation before attack (attack only):')
    evaluation_test2_attack = evaluate(y_test2_attack, y_pred2_attack, f'Test2{tag}', count = 2+count)
    
    # Model performance after attack
    # y_fool2_pred = best_svm_adv.predict(X_test2)
    # X_test2_adv = X_test2[y_fool2_pred]
    # y_test2_adv = y_test2[y_fool2_pred]
    # y_pred2_adv = best_svm.predict(X_test2_adv)
    # print('Evaluation after attack:')
    # evaluation_test2_adv = evaluate(y_test2_adv, y_pred2_adv, f'Adversarial test2{tag}')
    
    y_fool2_attack_pred = best_svm_adv.predict(X_test2_attack)
    X_test2_attack_adv = X_test2_attack[y_fool2_attack_pred]
    y_test2_attack_adv = y_test2_attack[y_fool2_attack_pred]
    y_pred2_attack_adv = best_rf.predict(X_test2_attack_adv)
    print('Evaluation after attack (attack only):')
    evaluation_test2_adv_attack = evaluate(y_test2_attack_adv, y_pred2_attack_adv, f'Adversarial test2{tag}', count = 3+count)
    
    # Fooling case prediction performance
    # y_fool2 = np.logical_and(y_pred2 == 0, y_test2 != 0)
    # print('Fooling case evaluation:')
    # evaluation_fool2 = evaluate(y_fool2, y_fool2_pred, f'Fooling case{tag}', ['Not fool', 'Fool'], return_fool_ratio=False)
    
    y_fool2_attack = np.logical_and(y_pred2_attack == 0, y_test2_attack != 0)
    print('Fooling case evaluation (attack only):')
    evaluation_fool2_attack = evaluate(y_fool2_attack, y_fool2_attack_pred, f'Fooling case{tag}', ['Not fool', 'Fool'], return_fool_ratio=False, count = 4+count)
    
    evaluation = {
        'evaluation_test1': evaluation_test1,
        # 'evaluation_test2': evaluation_test2,
        'evaluation_test2_attack': evaluation_test2_attack,
        # 'evaluation_test2_adv': evaluation_test2_adv,
        'evaluation_test2_adv_attack': evaluation_test2_adv_attack,
        # 'evaluation_fool2': evaluation_fool2,
        'evaluation_fool2_attack': evaluation_fool2_attack
    }
    
    return evaluation

print('Attack without retraining:')
evaluation_attack = attack_efficiency(X_test1, y_test1, X_test2, y_test2, best_rf, count = 0)

best_rf_retrain = joblib.load("./rf/best_rf_retrain.model") # trained retrain rf detection model

print('Attack with retraining:')
evaluation_attack_retrain = attack_efficiency(X_test1, y_test1, X_test2, y_test2, best_rf_retrain, tag=' after retraining', count = 4)

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def pyriemann_pipeline_plot(auc, methods, X_balanced_data, y_balanced, clfs):
    results = pd.DataFrame(data=auc, columns=['AUC'])
    results['Method'] = methods
    fig = plt.figure(figsize=[8,4])
    sns.barplot(data=results, x='AUC', y='Method')
    plt.xlim(0.3, 0.95)
    sns.despine()
    X_train, X_test, y_train, y_test = train_test_split(X_balanced_data, y_balanced==0, test_size=0.30, random_state=42)
    plt.figure(figsize=(16, 8))  
    for i, (m, clf) in enumerate(clfs.items()):
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.subplot(3, 3, i+1)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["target", "distractors"], yticklabels=["target", "distractors"])
            plt.title(f'Confusion Matrix - {m}')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
        except Exception as e:
            print(f"Error occurred for {m}: {str(e)}")
    plt.tight_layout()
    plt.show()

def classification_pipeline_plot(accuracy, method, classifiers, classifiers_names, X_resampled, y_resampled):
    result = pd.DataFrame(data=accuracy, columns=['AUC'])
    result['Method'] = method
    fig = plt.figure(figsize=[8,4])
    sns.barplot(data=result, x='AUC', y='Method')
    plt.xlim(0.3, 1.0)
    sns.despine()
    num_rows = 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    for i, (clf, clf_name) in enumerate(zip(classifiers, classifiers_names)):
        if i >= num_rows * num_cols:
            break  
        ax = axes.flatten()[i]
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        if np.sum(cm) > 0:
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax, xticklabels=["target", "distractors"], yticklabels=["target", "distractors"])
            ax.set_title(f'Confusion Matrix - {clf_name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        else:
            ax.axis('off')
    for j in range(i+1, num_rows * num_cols):
        axes.flatten()[j].axis('off')
    plt.tight_layout()
    plt.show()
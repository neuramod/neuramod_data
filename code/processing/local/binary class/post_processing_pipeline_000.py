## Binary Classification ##
# Import libraries
import mne
import os
import numpy as np
import pandas as pd

# Sklearn library
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Pyriemann library
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import ERPCovariances, XdawnCovariances


# Other imports

from imblearn.under_sampling import InstanceHardnessThreshold
mne.viz.set_browser_backend('matplotlib', verbose=None)
from collections import OrderedDict
from mne.decoding import Vectorizer

#%%
def pyriemann_pipeline(epoch):
    clfs = OrderedDict()
    clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(with_mean=True, with_std=True), LogisticRegression(C=0.1, tol=0.0001,max_iter=100,
                                                multi_class='ovr', solver='liblinear'))
    clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
    clfs['ERPCov + TS + SVC'] = make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(),StandardScaler(), SVC()) #LogisticRegression()
    clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(estimator='oas', classes=[0]), MDM())
    clfs['XdawnCov + TS+ SVC'] = make_pipeline(XdawnCovariances(nfilter=2, applyfilters=True, classes=None, 
                estimator='lwf', xdawn_estimator='lwf', baseline_cov=None), TangentSpace(),StandardScaler(), SVC())
    clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(nfilter=2, applyfilters=True, classes=[0], 
               estimator='lwf', xdawn_estimator='lwf',  baseline_cov=None), MDM())
    clfs['XdawnCov + Vect + LR'] = make_pipeline(XdawnCovariances(nfilter=2, applyfilters=True, classes=None, 
                estimator='lwf', xdawn_estimator='lwf', baseline_cov=None), Vectorizer(), StandardScaler(with_mean=True, with_std=True), LogisticRegression(C=0.1, tol=0.0001,max_iter=100,
                                                                multi_class='ovr', solver='liblinear'))
    clfs["tgsp + svm"] = make_pipeline(
            Covariances("lwf"), TangentSpace(metric="riemann"), StandardScaler(), SVC())

    # format data
    X = epoch.get_data()
    epoch.pick_types(eeg=True)
    X = X * 1e6
    times = epoch.times
    y = epoch.events[:, -1]
    y = LabelEncoder().fit_transform(y)
    cv = StratifiedShuffleSplit(n_splits=10, 
                            random_state=42, test_size=0.30) 
    epoch, channels , time = X.shape
    X_reshaped = X.reshape(epoch, channels* time)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(X_reshaped)
    iht = InstanceHardnessThreshold(random_state=0,
                      estimator=LogisticRegression(
                         solver='lbfgs', multi_class='auto'))
    X_balanced, y_balanced = iht.fit_resample(X_reshaped, y)
    X_balanced_data = X_balanced.reshape(-1, channels, time)
    pr = np.zeros(len(y_balanced))
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.30, 
                            random_state=42)
    auc = []
    methods = []
    for m in clfs:
        try:
            res = cross_val_score(clfs[m], X_balanced_data, y_balanced==0, scoring='roc_auc', 
                              cv=cv, n_jobs=-1)
            auc.extend(res)
            methods.extend([m]*len(res))
        except:
            pass
    return auc, methods, X_balanced_data, y_balanced, clfs
#%%
def classifier_pipeline(epochs):
    tmin, tmax = 0.3, 0.6
    epochs_crop = epochs.get_data()[:, :, int(epochs.time_as_index(tmin)):int(epochs.time_as_index(tmax))]
    channels_of_interest = ['P3', 'P4', 'Pz','P7','P8','PO3','POz','PO4','PO7','PO8','Oz','O1','O2']
    peak_amplitudes = {ch: [] for ch in channels_of_interest}
    
    for ch in channels_of_interest:
        ch_idx = epochs.info['ch_names'].index(ch)  
        for epoch_data in epochs_crop:
            time_max = np.argmax(epoch_data[ch_idx, :])
            peak_amplitude = epoch_data[ch_idx, time_max]
            peak_amplitudes[ch].append(peak_amplitude)
    label = epochs.events[:, -1]
    le= LabelEncoder()
    y_label = le.fit_transform(label)
    feature_matrix = np.vstack([peak_amplitudes[ch] for ch in channels_of_interest]).T

    # normalize the feature matrix
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    # apply class balance
    iht = InstanceHardnessThreshold(random_state=0,
                      estimator=LogisticRegression(
                         solver='lbfgs', multi_class='auto'))
    X_resampled, y_resampled = iht.fit_resample(normalized_features, y_label)
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    clf4 = SVC()
    clf5 = DecisionTreeClassifier (max_depth =6)
    clf6 = LDA(solver='lsqr',shrinkage=0.5)
    clf7 = KNeighborsRegressor(n_neighbors=1)
    clf8 = QuadraticDiscriminantAnalysis()
    classifiers_names = ['Logistic Regression','Random Forest Classifier','GaussianNB','SVC', 'Decision Tree Classifier', 
                         'LDA', 'KNeighborsRegressor','Quadratic Discriminant Analysis']
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.30, 
                            random_state=42)
    classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]
    accuracies=[]
    method=[]
    accuracy =[]
    for clf, clfs_name in zip(classifiers, classifiers_names):
        acc = cross_val_score(clf, X_resampled, y_resampled==0, scoring='roc_auc',cv=cv)
        accuracies.append(acc)
        accuracy.extend(acc)
        method.extend([clfs_name]*len(acc))
        for train_index, test_index in cv.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
    for acc, clf_name in zip(accuracies, classifiers_names):
        print(f"Classifier: {clf_name}")
        print(f"Mean accuracy: : {acc.mean():.2f}")
    return accuracy, method, classifiers, classifiers_names, X_resampled, y_resampled

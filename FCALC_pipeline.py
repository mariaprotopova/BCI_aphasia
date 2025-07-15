import glob
import mne
import argparse
import numpy as np
import pandas as pd
import fcalc
from config import model_test_CV, temporal_train_test_split_custom

from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('subject')
parser.add_argument('run')
args = parser.parse_args()
subj = args.subject
run = args.run

epochs = mne.read_epochs(f'DATA/BCI_HSE{subj}/epochs/{subj}_run{run}_clean-epo.fif', preload = True)#.crop(tmin = 0, tmax = None)

X = np.concatenate([epochs['stim'].pick('eeg').get_data(copy = True), epochs['distr'].pick('eeg').get_data(copy = True)], axis = 0)
y = np.concatenate([np.array([1]*len(epochs['stim'].pick('eeg').get_data(copy = True))),
                   np.array([0]*len(epochs['distr'].pick('eeg').get_data(copy = True)))], axis = 0)

print('TRAIN-TEST SPLIT')
X_train, X_test, y_train, y_test, mask = temporal_train_test_split_custom(X = X.mean(axis = 1), y = y, train_size = .8)

print('performing PCA...')
pca = PCA(n_components=.99, random_state=123)
pca.fit(X_train)

X_train_dec = pca.transform(X_train)
X_test_dec = pca.transform(X_test)
print(f'PCA: n_components = {X_train_dec.shape[1]}')

result = []
best_params = dict()
best_params['roc_auc_score'] = 0
best_params['alpha'] = None
best_params['method'] = None
alphas = np.arange(0, 1, .1)
methods = ['standard', 'standard-support', 'ratio-support']
for method in methods:
    for alpha in alphas:
        res = model_test_CV(X_train_dec,y_train,randomize=True, method = method, alpha = alpha)
        result.append(res.loc["mean"].values)
        if res['ROC-AUC'].mean() > best_params['roc_auc_score']:
            best_params['roc_auc_score'] = res['ROC-AUC'].mean()
            best_params['alpha'] = alpha
            best_params['method'] = method

pat_cls = fcalc.classifier.PatternClassifier(context=X_train_dec, labels=y_train, 
                                            method = best_params['method'], randomize=True,
                                            alpha = best_params['alpha']) # other methods: standard-support, ratio-support
# Computing support
pat_cls.predict(X_test_dec)
probs = pat_cls.predictions

print(f'ROC-AUC on a test sample: {roc_auc_score(y_test, probs[:, 1])}')

fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

with open(f'reports_clean/{subj}_run{run}_FCALC_clean_pca.txt', 'w') as f:
    f.write(f'subject: {subj}' + '\n')
    f.write('='*55 + '\n')
    lines = [
        "Best parameter (CV score=%0.3f):" % best_params['roc_auc_score'],
        repr(best_params),
        f'ROC-AUC on a test sample: {roc_auc_score(y_test, probs[:, 1])}',
        'Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]),
        classification_report(y_test, probs[:, 1] > thresholds[ix])
    ]
    for line in lines:
        f.write(line + '\n')

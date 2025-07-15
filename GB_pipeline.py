import glob
import mne
import argparse
import numpy as np
from config import temporal_train_test_split_custom, auc_scorer

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument('subject')
args = parser.parse_args()
subj = args.subject

epochs_fb = mne.read_epochs(f'DATA/BCI_HSE{subj}/epochs/{subj}_run04_clean_fb-epo.fif', preload = True)
epochs_wofb = mne.read_epochs(f'DATA/BCI_HSE{subj}/epochs/{subj}_run04_clean_wofb-epo.fif', preload = True)

X = np.concatenate([epochs_fb.pick('eeg').get_data(copy = True), epochs_wofb.pick('eeg').get_data(copy = True)], axis = 0)
y = np.concatenate([np.array([1]*len(epochs_fb.pick('eeg').get_data(copy = True))),
                   np.array([0]*len(epochs_wofb.pick('eeg').get_data(copy = True)))], axis = 0)

print('TRAIN-TEST SPLIT')
X_train, X_test, y_train, y_test, mask = temporal_train_test_split_custom(X = X.mean(axis = 1), y = y, train_size = .8)

print('performing PCA...')
pca = PCA(n_components=.99, random_state=123)
pca.fit(X_train)

X_train_dec = pca.transform(X_train)
X_test_dec = pca.transform(X_test)
print(f'PCA: n_components = {X_train_dec.shape[1]}')

sample_weight = np.array([4 if i == 1 else 1 for i in y_train])
estimators = [('gbc', GradientBoostingClassifier(random_state = 42))] #('rf', RandomForestClassifier(random_state = 42))]
param_grid = dict(gbc__learning_rate = [.1, 1, 10, 100],
                  gbc__n_estimators = [5, 50, 100],
                  gbc__max_depth = [3, 10, 30],
                  )
pipe = Pipeline(estimators)
grid_search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    # scoring = 'roc_auc',
    scoring=auc_scorer,
    verbose=10,
    n_jobs = -2
)
grid_search.fit(X_train_dec, y_train)

print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)

best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train_dec, y_train)
y_pred = best_estimator.predict_proba(X_test_dec)
print(f'ROC-AUC on a test sample: {roc_auc_score(y_test, y_pred[:, 1])}')

fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

with open(f'reports_clean/{subj}_GB_fb-wofb_clean_pca.txt', 'w') as f:
    f.write(f'subject: S{subj}' + '\n')
    f.write('='*55 + '\n')
    lines = [
        "Best parameter (CV score=%0.3f):" % grid_search.best_score_,
        repr(grid_search.best_params_),
        f'ROC-AUC on a test sample: {roc_auc_score(y_test, y_pred[:, 1])}',
        'Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]),
        classification_report(y_test, y_pred[:, 1] > thresholds[ix])
    ]
    for line in lines:
        f.write(line + '\n')

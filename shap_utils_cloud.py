import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score
import inspect

def return_model(mode, **kwargs):
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode == 'logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs,
                                   max_iter=max_iter, random_state=666)
    elif mode == 'Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode == 'RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'SVC':
        kernel = kwargs.get('kernel', 'rbf')
        model = SVC(kernel=kernel, random_state=666)
    elif mode == 'LinearSVC':
        model = LinearSVC(loss='hinge', random_state=666)
    elif mode == 'GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode == 'KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode == 'NB':
        model = MultinomialNB()
    elif mode == 'linear':
        model = LinearRegression(random_state=666)
    elif mode == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    else:
        raise ValueError("Invalid model!")
    return model


def error(mem):
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


def my_accuracy_score(clf, X, y):
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))


def my_f1_score(clf, X, y):
    predictions = clf.predict(x)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')


def my_auc_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)


def my_xe_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)

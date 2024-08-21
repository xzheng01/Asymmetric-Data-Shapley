from shap_utils import *
from sklearn.base import clone
import warnings
import os
import numpy as np
import pickle as pkl
from sklearn.metrics import f1_score

class Monte_Carlo_Asymmetric_Shapley(object):

    def __init__(self, X, y, X_test, y_test, directory, social_class, continued,
                 task='classification', model_family='logistic', metric='accuracy',
                 seed=None, **kwargs):

        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test covariates
            y_test: Test labels
            directory: Directory to save results and figures.
            task: "Classification" or "Regression"
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting same permutations.
            social_class: The partition of the grand coalition which can be interpreted as the "social class" of players
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)

        self.task = task
        self.model_family = model_family
        self.n_neighbors = 3
        self.metric = metric
        self.directory = directory
        self.social_class = social_class
        self.continued = continued
        self.model = return_model(self.model_family, **kwargs)
        self.initialize_instance(X, y, X_test, y_test)
        self.random_score = self.init_score(self.metric)

        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'

    def initialize_instance(self, X, y, X_test, y_test):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.create_results_placeholder()

    def create_results_placeholder(self):
        tmc_dir = os.path.join(self.directory, 'tmc_asymmetric_shapley_model={}_metric={}.pkl'.format(self.model_family, self.metric))
        if os.path.exists(tmc_dir) and self.continued:
          with open(tmc_dir, "rb") as fp:
            mydict = pkl.load(fp)
          self.marginals_tmc = mydict['marginals_tmc']
          self.idxs_tmc = mydict['idxs_tmc']
          print('existing self.marginals_tmc.shape', self.marginals_tmc.shape, 
          'existing self.idxs_tmc.shape', self.idxs_tmc.shape)
        else:
          n_instances = len(self.X)
          self.marginals_tmc = np.zeros((0, n_instances))
          self.idxs_tmc = np.zeros((0, n_instances), int)
          pkl.dump({'marginals_tmc': self.marginals_tmc, 'idxs_tmc': self.idxs_tmc}, open(tmc_dir, 'wb'))

    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float) / len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            self.model.fit(self.X, rnd_y)
            random_scores.append(self.value(self.model))
        return np.mean(random_scores)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
        """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')


    def run(self, save_every, err, tolerance=0, loo_run=True):
        """Approximate data points' Shapley values by Monte Carlo Sampling.
        Args:
            save_every: save marginal contributions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            tmc_run: If True, computes TMC-asymmetric-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """

        loo_dir = os.path.join(self.directory, 'loo_model={}_metric={}.pkl'.format(self.model_family, self.metric))
        if (not loo_run):
          print("LOO values already calculated! Don't need to compute again.")
        else:
          print("Now compute LOO values...")
          self.vals_loo = self.calculate_loo_vals()
          self.save_loo_results()

        tmc_run = True
        while tmc_run:
            print('error(self.marginals_tmc):', error(self.marginals_tmc))
            if error(self.marginals_tmc) < err:
                tmc_run = False
            else:
                self.tmc_asym_shap(save_every, tolerance=tolerance) # save marginal contributions every n iterations
                self.vals_tmc = np.mean(self.marginals_tmc, 0)
                self.save_asymmetric_shapley_results()

    def save_loo_results(self, overwrite=True):
        """Saves results computed so far."""
        loo_dir = os.path.join(self.directory, 'loo_model={}_metric={}.pkl'.format(self.model_family, self.metric))
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))

    def save_asymmetric_shapley_results(self):
        tmc_dir = os.path.join(self.directory, 'tmc_asymmetric_shapley_model={}_metric={}.pkl'.format(self.model_family, self.metric))
        pkl.dump({'vals_tmc':self.vals_tmc, 'marginals_tmc': self.marginals_tmc,
                  'idxs_tmc': self.idxs_tmc}, open(tmc_dir, 'wb'))

    def tmc_asym_shap(self, iterations, tolerance):
        """Runs TMC-Shapley algorithm.
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        try:
            self.mean_score
        except:
            self.tol_mean_score()
        for iteration in range(iterations):
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance)
            self.marginals_tmc = np.concatenate([self.marginals_tmc, np.reshape(marginals, (1, -1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1, -1))])

    def tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X, self.y)

            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model,
                    metric=self.metric,
                    X=self.X_test[bag_idxs],
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)

    def one_iteration(self, tolerance):
        """Runs one iteration of TMC-Asymmetric-Shapley algorithm."""
        idxs = np.zeros((0,))
        marginal_contribs = np.zeros(len(self.X))
        for i_class in sorted(self.social_class.keys()):
            i_class_idxs = np.random.permutation(self.social_class[i_class]).astype('int')
            idxs = np.append(idxs, i_class_idxs)
        idxs = idxs.astype('int')
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        truncation_counter = 0
        new_score = self.random_score
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, np.reshape(self.X[idx], (1,-1))])
            y_batch = np.append(y_batch, self.y[idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.task=='regression' or len(set(y_batch)) == len(set(self.y_test))):
                    if self.model_family=='KNN':
                        if len(y_batch) >= self.n_neighbors:
                            self.restart_model()
                            self.model.fit(X_batch, y_batch)
                            new_score = self.value(self.model, self.metric, self.X_test, self.y_test)
                    else:
                        self.restart_model()
                        self.model.fit(X_batch, y_batch)
                        new_score = self.value(self.model, self.metric, self.X_test, self.y_test)
            marginal_contribs[idx] = (new_score - old_score)
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def restart_model(self):
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)

    def calculate_loo_vals(self, metric=None):
        """Calculated leave-one-out values for the given metric.
        Args: metric: If None, it will use the objects default metric.
        Returns: Leave-one-out scores
        """
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric
        self.restart_model()
        self.model.fit(self.X, self.y)

        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in range(len(self.X)):
            X_batch = np.delete(self.X, i, axis=0)
            y_batch = np.delete(self.y, i, axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.value(self.model, metric=self.metric)
            vals_loo[i] = baseline_value - removed_value
        return vals_loo

    def merge_parallel_results(self, file_names):
        """Helper method for 'merge_results' method."""
        marginals = np.zeros((0, self.X.shape[0]))
        idxs = np.zeros((0, len(self.X)), int)
        vals = np.zeros(len(self.X))
        counter = 0.
        for file in file_names:
            samples_dir = os.path.join(self.directory, file)
            dict = pkl.load(open(samples_dir, 'rb'))
            marginals = np.concatenate([marginals, dict['marginals_tmc']])
            idxs = np.concatenate([idxs, dict['idxs_tmc']])
            counter += len(dict['marginals_tmc'])
            vals *= (counter - len(dict['marginals_tmc'])) / counter
            vals += len(dict['marginals_tmc']) / counter * np.mean(marginals, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory,
            'merged_tmc_asymmetric_shapley_model={}_metric={}.pkl'.format(self.model, self.metric)
        )
        pkl.dump({'marginals_tmc': marginals, 'idxs_tmc': idxs, 'vals_tmc': vals}, open(merged_dir, 'wb'))
        return marginals, idxs, vals

    def merge_results(self, file_names):
        """Merge all the results from different runs.
        Returns: combined marginals, sampled indexes and values calculated
        """
        tmc_results = self.merge_parallel_results(file_names)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results

import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from cyvlfeat.fisher import fisher


class FVClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, n_mixtures=16, n_pc=512, clf_cls=LinearSVC,
		_seed=None, _rnd=None, **clf_params):
		super(FVClassifier, self).__init__()

		self.n_mixtures = n_mixtures
		self.n_pc = n_pc
		self._seed = _seed
		self.clf_cls = clf_cls
		self.clf_params = clf_params

		self._rnd = _rnd or np.random.RandomState(_seed)

		logging.info("PCA will have {} components".format(n_pc))

		self.pca = PCA(
			n_components=n_pc,
			whiten=True,
			random_state=self._rnd.randint(2**32-1))

		logging.info("GMM will have {} gaussians".format(n_mixtures))
		self.gmm = GaussianMixture(
			n_mixtures,
			covariance_type="diag",
			random_state=self._rnd.randint(2**32-1))

		self.clf = clf_cls(
			random_state=self._rnd.randint(2**32-1),
			**clf_params)

	def _transform(self, X):
		res = [self._fisher_vector(x) for x in X]
		return np.stack(res)

	def _fisher_vector(self, x):
		params = self.gmm.means_, self.gmm.covariances_, self.gmm.weights_
		# convert to float32 and transpose
		params = [p.astype(np.float32).T for p in params]
		return fisher(x.astype(np.float32).T, *params, improved=True)

	def _data_stats(self, X):
		assert X.ndim in [2, 3], "Data should have either 2 or 3 dimensions!"
		if X.ndim == 3:
			n_samples, n_parts, feat_size = X.shape
		else:
			n_samples, feat_size = X.shape
			n_parts = 1
		return n_samples, n_parts, feat_size


	def fit(self, X, y=None, gmm_ratio=1):
		assert y is not None, "No GT labels given!"
		n_samples, n_parts, feat_size = self._data_stats(X)

		X_reduced = self.pca.fit_transform(X.reshape(-1, feat_size))
		logging.info("Reducing dimensions with the PCA: {} -> {}".format(
			feat_size, self.n_pc
		))
		n_pairs = X_reduced.shape[0]
		if 0 < gmm_ratio < 1:
			n = int(n_pairs * gmm_ratio)
			idxs = self._rnd.choice(n_pairs, n, replace=False)
			logging.info("Training GMM on {} out of {} part-samples pairs".format(n, n_pairs))
			self.gmm.fit(X_reduced[idxs])
		else:
			logging.info("Training GMM on all {} part-samples pairs".format(n_samples))
			self.gmm.fit(X_reduced)

		X_fisher = self._transform(X_reduced.reshape(n_samples, n_parts, -1))
		logging.info("Resulting Fisher Vector shape: {}".format(X_fisher.shape))

		logging.info("Training {}...".format(self.clf_cls.__name__))
		self.clf.fit(X_fisher, y)

		return self

	def predict(self, X):
		n_samples, n_parts, feat_size = self._data_stats(X)

		X_reduced = self.pca.transform(X.reshape(-1, feat_size))
		X_fisher = self._transform(X_reduced.reshape(n_samples, n_parts, -1))
		return self.clf.predict(X_fisher)



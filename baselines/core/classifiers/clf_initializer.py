import joblib
import logging
import numpy as np

from os.path import join, isfile

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from .fv_classifier import FVClassifier


class ClfInitializer(object):
	def __init__(self, clf_class, name, load=None, **params):
		super(ClfInitializer, self).__init__()
		self.clf_class = clf_class
		self.name = name
		self.params = params
		if load is None or not isfile(load):
			self.coef_ = self.intercept_ = None
		else:
			logging.info("pre-trained initial weights will be loaded from \"{}\"".format(load))
			cont = np.load(load)
			self.coef_ = cont["weights"]
			self.intercept_ = cont["bias"]

	def __call__(self, n_parts, **kwargs):

		_params = dict(self.params)
		_params.update(kwargs)

		clf = self.clf_class(**_params)
		if self.coef_ is not None:
			n_classes, feat_size = self.coef_.shape
			clf.classes_ = np.arange(n_classes).astype(np.int32)
			clf.intercept_ = self.intercept_.copy()

			# if n_parts == 1:
			# 	clf.coef_ = self.coef_.copy()
			# else:
			new_shape = (n_classes, n_parts, feat_size)
			new_coef_ = np.expand_dims(self.coef_, 1)
			new_coef_ = np.broadcast_to(new_coef_, new_shape)
			clf.coef_ = new_coef_.reshape(n_classes, -1)

		return clf

	def dump(self, output, key, suffix):

		fpath = join(output, "clf_{}_{}_{}.npz".format(self.name, key, suffix))
		logging.info("Dumping {} classifier to \"{}\"".format(
			self.__class__.__name__, fpath))

		return joblib.dump(self, fpath)

	@classmethod
	def new(cls, opts):
		kwargs = dict(
			name=opts.classifier,
			C=opts.C, max_iter=opts.max_iter)

		if opts.sparse:
			return cls(LinearSVC,
				load=opts.load,
				penalty="l1", dual=False,
				**kwargs)

		elif opts.classifier == "svm":
			return cls(LinearSVC,
				load=opts.load,
				**kwargs)

		elif opts.classifier == "logreg":
			return cls(LogisticRegression,
				load=opts.load,
				solver='lbfgs',
				multi_class='multinomial',
				warm_start=bool(opts.load),
				**kwargs)

		elif opts.classifier == "fv":
			return cls(FVClassifier,
				load=opts.load,
				n_mixtures=4,
				n_pc=512,
				**kwargs)

		else:
			raise ValueError("Unknown Classifier: {}".format(opts.classifier))

import joblib
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from functools import partial
from os.path import join, isfile

def _dump(opts, clf, key, suffix):
	dump_name = "clf_{}_{}_{}.npz".format(opts.classifier, key, suffix)
	fpath = join(opts.output, dump_name)
	logging.info("Dumping {0.__class__.__name__} classifier to \"{1}\"".format(
		clf, fpath))

	return joblib.dump(clf, fpath)


def evaluate(opts, train, val, key):
	n_parts = train.features.shape[1]

	if n_parts == 1:
		evaluate_global(opts, train, val, key)
	else:
		evaluate_parts(opts, train, val, key)
		if opts.shuffle_part_features:
			evaluate_parts(opts, train, val, key, shuffle=True)

class ClfInitializer(object):
	def __init__(self, clf_class, load=None, **params):
		super(ClfInitializer, self).__init__()
		self.clf_class = clf_class
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

def _get_clf(opts):
	kwargs = dict(C=opts.C, max_iter=opts.max_iter)

	if opts.sparse:
		return ClfInitializer(LinearSVC, load=opts.load,
			penalty="l1", dual=False,
			**kwargs)

	elif opts.classifier == "svm":
		return ClfInitializer(LinearSVC, load=opts.load,
			**kwargs)

	elif opts.classifier == "logreg":
		return ClfInitializer(LogisticRegression, load=opts.load,
			solver='lbfgs',
			multi_class='multinomial',
			warm_start=bool(opts.load),
			**kwargs)

def evaluate_parts(opts, train, val, key, shuffle=False):

	assert not opts.sparse, "Sparsity is not supported here!"

	train_feats = train.features
	val_feats = val.features

	def inner(X, y, X_val, y_val, suffix):
		if shuffle:
			suffix += "_shuffled"

		clf_class = _get_clf(opts)

		clf, score = train_score(X, y, X_val, y_val, clf_class,
			n_parts=train_feats.shape[1],
			scale=opts.scale_features)

		logging.info("Accuracy {}: {:.4%}".format(suffix, score))

		if not opts.no_dump:
			_dump(opts, clf, key=key, suffix=suffix)

	if shuffle:
		logging.info("Shuffling features")
		train_feats = train_feats.copy()
		val_feats = val_feats.copy()

		for f in train_feats:
			np.random.shuffle(f[:-1])

		for f in val_feats:
			np.random.shuffle(f[:-1])

	y, y_val = train.labels, val.labels

	X = train_feats.reshape(len(train), -1)
	X_val = val_feats.reshape(len(val), -1)
	inner(X, y, X_val, y_val, "all_parts")

	if opts.eval_local_parts:
		X = train_feats[:, :-1, :].reshape(len(train), -1)
		X_val = val_feats[:, :-1, :].reshape(len(val), -1)
		inner(X, y, X_val, y_val, "local_parts")

def evaluate_global(opts, train, val, key):

	X, y = train.features[:, -1, :], train.labels
	X_val, y_val = val.features[:, -1, :], val.labels
	suffix = "glob_only"

	if opts.sparse:
		suffix += "_sparse_coefs"
		opts.classifier = "svm"

	clf_class = _get_clf(opts)

	clf, score = train_score(X, y, X_val, y_val, clf_class,
		n_parts=1,
		scale=opts.scale_features)

	logging.info("Accuracy {}: {:.2%}".format(suffix, score))

	if not opts.no_dump:
		_dump(opts, clf, key=key, suffix=suffix)

def train_score(X, y, X_val, y_val, clf_class, n_parts=1, scale=False, **kwargs):
	logging.debug("Training")
	logging.debug(X.shape)
	logging.debug(y)

	logging.debug("Validation")
	logging.debug(X_val.shape)
	logging.debug(y_val)

	if scale:
		logging.info("Scaling Data...")
		scaler = MinMaxScaler()
		X = scaler.fit_transform(X)
		X_val = scaler.transform(X_val)
	clf = clf_class(n_parts=n_parts, **kwargs)

	logging.info("Training {0.__class__.__name__} Classifier...".format(clf))
	clf.fit(X, y)
	logging.info("Training Accuracy: {:.4%}".format(clf.score(X, y)))
	return clf, clf.score(X_val, y_val)

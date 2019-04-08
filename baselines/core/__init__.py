import joblib
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from functools import partial
from os.path import join

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
		evaluate_parts(opts, train, val, key, shuffle=True)

def _get_clf(opts):
	if opts.sparse:
		return partial(LinearSVC, penalty="l1", C=0.1, dual=False, max_iter=200)

	elif opts.classifier == "svm":
		return LinearSVC

	elif opts.classifier == "logreg":
		return partial(LogisticRegression,
			C=0.01,
			solver='lbfgs',
			multi_class='multinomial',
			max_iter=500)

def evaluate_parts(opts, train, val, key, shuffle=False):

	assert not opts.sparse, "Sparsity is not supported here!"


	def inner(X, y, X_val, y_val, suffix):
		if shuffle:
			suffix += "_shuffled"

		clf_class = _get_clf(opts)

		clf, score = train_score(X, y, X_val, y_val, clf_class,
			scale=opts.scale_features)

		logging.info("Accuracy {}: {:.2%}".format(suffix, score))

		if not opts.no_dump:
			_dump(opts, clf, key=key, suffix=suffix)

	train_feats = train.features
	val_feats = val.features

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
		scale=opts.scale_features)

	logging.info("Accuracy {}: {:.2%}".format(suffix, score))

	if not opts.no_dump:
		_dump(opts, clf, key=key, suffix=suffix)

def train_score(X, y, X_val, y_val, clf_class=LinearSVC, scale=False, **kwargs):
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
	clf = clf_class(**kwargs)

	logging.info("Training {0.__class__.__name__} Classifier...".format(clf))
	clf.fit(X, y)
	return clf, clf.score(X_val, y_val)

import logging

from sklearn.preprocessing import MinMaxScaler

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

	logging.info(f"Training {clf.__class__.__name__} Classifier...".format(clf))
	clf.fit(X, y)
	train_accu = clf.score(X, y)
	logging.info(f"Training Accuracy: {train_accu:.4%}")
	return clf, clf.score(X_val, y_val)

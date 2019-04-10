#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import socket
if socket.gethostname() != "sigma25":
	import matplotlib
	matplotlib.use('Agg')

from os.path import join
import logging
import joblib

from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MinMaxScaler


from nabirds.annotations import AnnotationType
from baselines.utils import arguments, visualization

def main(args):
	KEY = "{}.{}".format(args.parts, args.model_type)
	logging.info("===== Setup key: {} =====".format(KEY))

	clf = joblib.load(args.weights)
	logging.info("Loaded \"{}\" classifier from \"{}\"".format(
		clf.__class__.__name__,
		args.weights))

	annot_cls = AnnotationType.get(args.dataset).value

	annot = annot_cls(args.data,
		parts=args.parts,
		feature_model=args.model_type)

	data = annot.new_dataset(args.subset)
	logging.info("Loaded {} {} images".format(len(data), args.subset))

	feats = data.features

	assert feats is not None, "No features found!"

	logging.info("Feature shape: {}".format(feats.shape))
	X = feats.reshape(len(data), -1)

	if args.scale_features:
		logging.info("Scaling Data...")
		train_data = annot.new_dataset("train")
		scaler = MinMaxScaler()
		scaler.fit(train_data.features.reshape(len(train_data), -1))
		X = scaler.transform(X)

	if args.evaluate:
		logging.info("Evaluating classifier...")
		accu = clf.score(X, data.labels)
		logging.info("Accuracy: {:.4%}".format(accu))


	logging.info("Predicting labels...")
	with open(args.output, "w") as out:
		writer = partial(print, sep=",", file=out)

		writer("id", "predicted")
		for i, feat in tqdm(zip(data.uuids, X), total=len(data)):
			y = clf.predict(feat[None])
			writer(i, int(y))

	logging.info("Ready")



main(arguments.predict_args())

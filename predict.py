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

from chainer_addons.models import ModelType
from chainer_addons.links import PoolingType

from nabirds.annotations import AnnotationType
from baselines.utils import arguments, visualization

class CNN_Wrapper(object):
	def __init__(self, model):
		super(CNN_Wrapper, self).__init__()
		self.clf = model.clf_layer

	def predict(self, feat):
		logit = self.clf(feat).array
		return logit.argmax(axis=1)

	def score(self, X, y):
		logits = self.clf(X).array
		return (logits.argmax(axis=1) == y).mean()


def load_CNN(args, annot):

	logging.info("Loading CNN model from \"{}\"...".format(args.weights))
	data_info = annot.info
	model_info = data_info.MODELS[args.model_type]
	part_info = data_info.PARTS[args.parts]

	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=0,
		pooling=PoolingType.G_AVG,
		aux_logits=False,
	)
	model.load_for_inference(
		n_classes=part_info.n_classes + args.label_shift,
		weights=args.weights)

	return CNN_Wrapper(model)

def load_sklearn(args, annot):
	return joblib.load(args.weights)


def load_clf(args, annot):

	for loader in [load_sklearn, load_CNN]:
		try:
			clf = loader(args, annot)
			break
		except Exception as e:
			pass

	else:
		raise ValueError("Unable to load weights \"{}\"!".format(args.weights))

	logging.info("Loaded \"{}\" classifier from \"{}\"".format(
		clf.__class__.__name__,
		args.weights))

	return clf

def main(args):
	KEY = "{}.{}".format(args.parts, args.model_type)
	logging.info("===== Setup key: {} =====".format(KEY))


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

	clf = load_clf(args, annot)

	if args.evaluate:
		logging.info("Evaluating classifier...")
		accu = clf.score(X, data.labels)
		logging.info("Accuracy: {:.4%}".format(accu))


	if not args.no_export:
		logging.info("Predicting labels...")
		with open(args.output, "w") as out:
			writer = partial(print, sep=",", file=out)

			writer("id", "predicted")
			for i, feat in tqdm(zip(data.uuids, X), total=len(data)):
				y = clf.predict(feat[None])
				writer(i, int(y))

	logging.info("Ready")

import chainer

with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(arguments.predict_args())

#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import numpy as np
import joblib
import logging

from os.path import join
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from functools import partial
from contextlib import contextmanager

from cvdatasets.annotations import AnnotationType
from cvdatasets.utils import new_iterator

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType
from chainer_addons.links import PoolingType
from chainer_addons.utils.imgproc import _center_crop

from chainer.cuda import to_cpu
from chainer.dataset.convert import concat_examples

from l1_svm_parts.utils import arguments, IdentityScaler
from l1_svm_parts.core.visualization import show_feature_saliency, visualize_coefs
from l1_svm_parts.core.extraction import parts_to_file, extract_parts
from l1_svm_parts.core.propagator import Propagator

from cluster_parts.utils import ClusterInitType

@contextmanager
def outputs(args):
	if args.extract:
		assert args.extract is not None, \
			"For extraction output files are required!"
		outputs = [open(out, "w") for out in args.extract]
		yield outputs
		[out.close for out in outputs]
	else:
		logging.warning("Extraction is disabled!")
		yield None, None



def new_model(args, model_info, n_classes):

	logging.info("Creating and loading model ...")

	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=args.input_size,
		pooling=PoolingType.G_AVG,
		aux_logits=False,
	)
	size = model.meta.input_size
	if not isinstance(size, tuple):
		size = (size, size)

	_prepare = PrepareType[args.prepare_type](model)

	if args.no_center_crop_on_val:
		prepare = lambda im: _prepare(im,
			swap_channels=args.swap_channels,
			keep_ratio=False)
	else:
		prepare = lambda im: _center_crop(
				_prepare(im,
					size=size,
					swap_channels=args.swap_channels), size)


	logging.info("Created {} model with \"{}\" prepare function. Image input size: {}"\
		.format(
			model.__class__.__name__,
			args.prepare_type,
			size
		)
	)

	if args.weights:
		weights = args.weights
	else:
		weight_subdir, _, _ = model_info.weights.rpartition(".")
		weights = join(
			data_info.BASE_DIR,
			data_info.MODEL_DIR,
			model_info.folder,
			"ft_{}".format(args.dataset),
			"rmsprop.g_avg_pooling",
			weight_subdir,
			"model_final.npz"
		)

	logging.info("Loading \"{}\" weights from \"{}\"".format(
		model_info.class_key, weights))
	model.load_for_inference(n_classes=n_classes, weights=weights)

	return model, prepare

def topk_decision(X, y, clf, topk):
	decs = clf.decision_function(X)
	topk_preds = np.argsort(decs)[:, -topk:]
	topk_accu = (topk_preds == np.expand_dims(y, 1)).max(axis=1).mean()
	return topk_preds, topk_accu

def evaluate_batch(feats, gt, clf, topk):

	topk_preds, topk_accu = topk_decision(feats, gt, clf=clf, topk=topk)

	logging.debug("Batch Accuracy: {:.4%} (Top1) | {:.4%} (Top{}) {: 3d} / {: 3d}".format(

		np.mean(topk_preds[:, -1] == gt),
		topk_accu,

		topk,
		np.sum(topk_preds[:, -1] == gt),
		len(feats)
	))

	return topk_preds


def evaluate_data(clf, data, subset, topk, scaler):

	X = scaler.transform(data.features[:, -1])
	y = data.labels
	pred = clf.decision_function(X).argmax(axis=1)
	logging.info("Accuracy on {} subset: {:.4%}".format(subset, (pred == y).mean()))

	topk_preds, topk_accu = topk_decision(X, y, clf=clf, topk=topk)
	logging.info("Top{}-Accuracy on {} subset: {:.4%}".format(topk, subset, topk_accu))

def load_svm(args):

	logging.info("Loading SVM from \"{}\"".format(args.trained_svm))
	clf = joblib.load(args.trained_svm)

	if args.visualize_coefs:
		logging.info("Visualizing coefficients...")
		visualize_coefs(clf.coef_, figsize=(16, 9*3))

	return clf

def init_data(args, clf=None):

	annot_cls = AnnotationType.get(args.dataset).value
	parts_key = "{}_{}".format(args.dataset, "GLOBAL")

	logging.info("Loading {} annotations from \"{}\"".format(
		annot_cls, args.data))
	logging.info("Using \"{}\"-parts".format(parts_key))

	annot = annot_cls(root_or_infofile=args.data, parts=parts_key, feature_model=args.model_type)

	data_info = annot.info
	model_info = data_info.MODELS[args.model_type]
	part_info = data_info.PARTS[parts_key]

	n_classes = part_info.n_classes + args.label_shift

	data = annot.new_dataset(subset=None)
	train_data, val_data = map(annot.new_dataset, ["train", "test"])

	if annot.labels.max() > n_classes:
		_, annot.labels = np.unique(annot.labels, return_inverse=True)

	logging.info("Minimum label value is \"{}\"".format(data.labels.min()))

	assert train_data.features is not None and val_data.features is not None, \
		"Features are not loaded!"

	assert val_data.features.ndim == 2 or val_data.features.shape[1] == 1, \
		"Only GLOBAL part features are supported here!"

	if args.scale_features:
		logging.info("Scaling data on training set!")
		scaler = MinMaxScaler()
		scaler.fit(train_data.features[:, -1])
	else:
		scaler = IdentityScaler()

	it, n_batches = new_iterator(data,
		args.n_jobs, args.batch_size,
		repeat=False, shuffle=False
	)
	it = tqdm(enumerate(it), total=n_batches)

	if clf is not None:
		for _data, subset in [(train_data, "training"), (val_data, "validation")]:
			evaluate_data(clf, _data, subset, args.topk, scaler)

	return scaler, data, it, model_info, n_classes


feature_composition = [
	"coords",
	# "grad",
	# "RGB"
]

def main(args):
	global feature_composition

	clf = load_svm(args)

	scaler, data, it, *model_args = init_data(args, clf)

	model, prepare = new_model(args, *model_args)

	GPU = args.gpu[0]
	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()
		model.to_gpu(GPU)

	logging.info("Using following feature composition: {}".format(feature_composition))

	with outputs(args) as (pred_out, full_out):

		for batch_i, batch in it:

			batch = [(prepare(im), lab) for im, _, lab in batch]
			X, y = concat_examples(batch, device=GPU)

			ims = chainer.Variable(X)
			feats = model(ims, layer_name=model.meta.feature_layer)

			if isinstance(feats, tuple):
				feats = feats[0]

			_feats = scaler.transform(to_cpu(feats.array))

			topk_preds = evaluate_batch(_feats, to_cpu(y), clf=clf, topk=args.topk)

			kwargs = dict(

				xp=model.xp,
				peak_size=None, #int(h * 0.35 / 2),
				swap_channels=args.swap_channels,

				gamma=args.gamma,
				sigma=args.sigma,
				K=args.K,
				alpha=1,
				thresh_type=args.thresh_type,
				cluster_init=ClusterInitType.MAXIMAS,

				feature_composition=feature_composition
			)

			propagator = Propagator(model, feats, ims, y, clf.coef_, topk_preds)

			if args.extract:

				for i, parts in extract_parts(propagator, **kwargs):

					im_idx = i + batch_i * args.batch_size
					im_uuid = data.uuids[im_idx]

					for pred_part, full_part in zip(*parts):
						parts_to_file(im_uuid, *pred_part, out=pred_out)
						parts_to_file(im_uuid, *full_part, out=full_out)

			else:
				show_feature_saliency(propagator, **kwargs)
				break



np.seterr(all="raise")
chainer.global_config.cv_resize_backend = "PIL"
with chainer.using_config("train", False):
	main(arguments.parse_args())

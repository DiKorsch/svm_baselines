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

from nabirds.annotations import AnnotationType

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType
from chainer_addons.links import PoolingType

from chainer.cuda import to_cpu
from chainer.dataset.convert import concat_examples

from l1_svm_parts.utils import arguments, IdentityScaler
from l1_svm_parts.core.visualization import show_feature_saliency, visualize_coefs
from l1_svm_parts.core.extraction import extract_parts, parts_to_file


def main(args):
	annot_cls = AnnotationType.get(args.dataset).value
	parts_key = "{}_{}".format(args.dataset, "GLOBAL")

	logging.info("Loading {} annotations from \"{}\"".format(
		annot_cls.__name__, args.data))
	logging.info("Using \"{}\"-parts".format(parts_key))

	annot = annot_cls(args.data,
		parts=parts_key,
		feature_model=args.model_type
	)

	data_info = annot.info
	model_info = data_info.MODELS[args.model_type]
	part_info = data_info.PARTS[parts_key]

	data = annot.new_dataset(subset=args.subset)

	assert data.features is not None, \
		"Features are not loaded!"

	assert data.features.ndim == 2 or data.features.shape[1] == 1, \
		"Only GLOBAL part features are supported here!"


	if args.scale_features:
		logging.info("Scaling data on training set!")
		train_data = annot.new_dataset("train")
		scaler = MinMaxScaler()
		scaler.fit(train_data.features[:, -1])
	else:
		scaler = IdentityScaler()

	X = scaler.transform(data.features[:, -1])
	y = data.labels

	trained_svm = "".format(
		args.dataset, args.model_type)

	if args.trained_svm:
		trained_svm = args.trained_svm

	logging.info("Loading SVM from \"{}\"".format(args.trained_svm))
	clf = joblib.load(args.trained_svm)

	COEFS = clf.coef_

	if args.visualize_coefs:
		logging.info("Visualizing coefficients...")
		visualize_coefs(clf.coef_, figsize=(16, 9*3))

	logging.info("Accuracy on subset ({}): {:.4%}".format(
		args.subset, clf.score(X, y)))

	decs = clf.decision_function(X)
	topk_preds = np.argsort(decs)[:, -args.topk:]
	topk_accu = (topk_preds == np.expand_dims(y, 1)).max(axis=1).mean()
	logging.info("Validation Accuracy (Top{}): {:.4%}".format(
		args.topk, topk_accu))

	model_cls = ModelType
	logging.info("Creating and loading model ...")

	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=args.input_size,
		pooling=PoolingType.G_AVG,
		aux_logits=False,
	)

	prepare = PrepareType[args.prepare_type](model)

	logging.info("Using \"{}\" prepare function".format(
		args.prepare_type))

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
	model.load_for_inference(n_classes=201, weights=weights)

	GPU = args.gpu[0]
	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()
		model.to_gpu(GPU)


	it, n_batches = data.new_iterator(
		n_jobs=args.n_jobs,
		batch_size=args.batch_size,
		repeat=False, shuffle=False
	)

	for batch_i, batch in tqdm(enumerate(it), total=n_batches):
		batch = [(prepare(im), lab) for im, _, lab in batch]
		X, y = concat_examples(batch, device=GPU)
		ims = chainer.Variable(X)

		feats = model(ims, layer_name=model.meta.feature_layer)
		if isinstance(feats, tuple):
			feats = feats[0]

		f = scaler.transform(to_cpu(feats.array))
		gt = to_cpu(y)

		decs = clf.decision_function(f)
		sorted_pred = np.argsort(decs)
		topk_preds = sorted_pred[:, -args.topk:]
		topk_accu = (topk_preds == np.expand_dims(gt, 1)).max(axis=1).mean()

		#preds = clf.predict(f)

		logging.debug("Batch Accuracy: {:.4%} (Top1) | {:.4%} (Top{}) {: 3d} / {: 3d}".format(

			np.mean(topk_preds[:, -1] == gt),
			topk_accu,

			args.topk,
			np.sum(topk_preds[:, -1] == gt),
			len(batch)
		))


		kwargs = dict(
			model=model, coefs=COEFS,
			ims=ims, labs=y,
			feats=feats, topk_preds=topk_preds,

			peak_size=None, #int(h * 0.35 / 2),

			gamma=args.gamma,
			sigma=args.sigma,
			K=args.K,
			init_from_maximas=args.init_from_maximas
		)


		if args.extract:
			extract_iter = extract_parts(**kwargs)
			for i, parts in enumerate(extract_iter):
				im_idx = i + batch_i * it.batch_size
				for pred_part, full_part in zip(*parts):
					parts_to_file(im_idx, *pred_part, out=None)#pred_out)
					parts_to_file(im_idx, *full_part, out=None)#full_out)

			raise ValueError("Open output files!")
		else:
			show_feature_saliency(**kwargs)
		break

with chainer.using_config("train", False):
	main(arguments.parse_args())

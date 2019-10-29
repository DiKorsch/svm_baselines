#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import joblib
import logging
import numpy as np

from contextlib import contextmanager

from chainer.cuda import to_cpu
from chainer.dataset.convert import concat_examples

from l1_svm_parts.core import Data
from l1_svm_parts.core import Model
from l1_svm_parts.core import Propagator
from l1_svm_parts.core import extract_parts
from l1_svm_parts.core import parts_to_file
from l1_svm_parts.core import show_feature_saliency
from l1_svm_parts.core import visualize_coefs
from l1_svm_parts.utils import arguments
from l1_svm_parts.utils import topk_decision

from cluster_parts.core import BoundingBoxPartExtractor
from cluster_parts.core import Corrector
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


def load_svm(args):

	logging.info("Loading SVM from \"{}\"".format(args.trained_svm))
	clf = joblib.load(args.trained_svm)

	if args.visualize_coefs:
		logging.info("Visualizing coefficients...")
		visualize_coefs(clf.coef_, figsize=(16, 9*3))

	return clf


def main(args):

	clf = load_svm(args)

	scaler, data, it, *model_args = Data.new(args, clf)

	model, prepare = Model.new(args, *model_args)

	GPU = args.gpu[0]
	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()
		model.to_gpu(GPU)

	logging.info("Using following feature composition: {}".format(args.feature_composition))

	kwargs = dict(
		extractor=BoundingBoxPartExtractor(
			corrector=Corrector(gamma=args.gamma, sigma=args.sigma),

			K=args.K,
			thresh_type=args.thresh_type,
			cluster_init=ClusterInitType.MAXIMAS,

			feature_composition=args.feature_composition,
		),
		xp=model.xp,
		swap_channels=args.swap_channels,
	)
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

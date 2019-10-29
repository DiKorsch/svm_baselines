#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np

from contextlib import contextmanager
from tqdm import tqdm

from chainer.cuda import to_cpu
from chainer.dataset.convert import concat_examples

from l1_svm_parts.core import Data
from l1_svm_parts.core import Model
from l1_svm_parts.core import Propagator
from l1_svm_parts.core import extract_parts
from l1_svm_parts.core import show_feature_saliency
from l1_svm_parts.utils import arguments

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


def main(args):
	GPU = args.gpu[0]

	clf = Model.load_svm(args.trained_svm, args.visualize_coefs)
	scaler, it, n_batches, *model_args = Data.new(args, clf)
	model, prepare = Model.new(args, *model_args)

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

	propagator = Propagator(model, clf, scaler=scaler, topk=args.topk)

	with outputs(args) as files:
		for batch_i, batch in tqdm(enumerate(it), total=n_batches):

			batch = [(prepare(im), lab) for im, _, lab in batch]
			X, y = concat_examples(batch, device=GPU)

			ims = chainer.Variable(X)
			feats = model(ims, layer_name=model.meta.feature_layer)

			if isinstance(feats, tuple):
				feats = feats[0]

			with propagator(feats, ims, y) as prop_iter:

				if args.extract:
					extract_parts(prop_iter, it, batch_i, files, **kwargs)

				else:
					show_feature_saliency(prop_iter, **kwargs)
					break


np.seterr(all="raise")
chainer.global_config.cv_resize_backend = "PIL"
with chainer.using_config("train", False):
	main(arguments.parse_args())

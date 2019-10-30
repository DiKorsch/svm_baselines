#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np

from contextlib import contextmanager
from multiprocessing.pool import Pool
from tqdm import tqdm

from chainer.cuda import to_cpu
from chainer.dataset.convert import concat_examples

from l1_svm_parts.core import Data
from l1_svm_parts.core import Model
from l1_svm_parts.core import Propagator
from l1_svm_parts.core import ExtractionPipeline
from l1_svm_parts.core import VisualizationPipeline
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
	scaler, it, *model_args = Data.new(args, clf)
	model, prepare = Model.new(args, *model_args)

	logging.info("Using following feature composition: {}".format(args.feature_composition))

	propagator = Propagator(model, clf,
		scaler=scaler,
		topk=args.topk,
		swap_channels=args.swap_channels,
		n_jobs=1,
	)

	extractor = BoundingBoxPartExtractor(
		corrector=Corrector(gamma=args.gamma, sigma=args.sigma),

		K=args.K,
		thresh_type=args.thresh_type,
		cluster_init=ClusterInitType.MAXIMAS,

		feature_composition=args.feature_composition
	)

	with outputs(args) as files, Pool(it.batch_size // 2) as pool:

		kwargs = dict(
			model=model,
			extractor=extractor,
			propagator=propagator,
			iterator=it,
			prepare=prepare,
			device=GPU,
		)
		if args.extract:
			pipeline = ExtractionPipeline(
				files=files,
				**kwargs
			)
		else:
			pipeline = VisualizationPipeline(
				**kwargs
			)


		pipeline.run(pool)


		# for batch_i, batch in tqdm(enumerate(it), total=n_batches):

		# 	batch = [(prepare(im), lab) for im, _, lab in batch]
		# 	X, y = concat_examples(batch, device=GPU)

		# 	ims = chainer.Variable(X)
		# 	feats = model(ims, layer_name=model.meta.feature_layer)

		# 	if isinstance(feats, tuple):
		# 		feats = feats[0]

		# 	with propagator(feats, ims, y) as prop_iter:

		# 		if args.extract:
		# 			pipeline(prop_iter, batch_i, pool=pool)

		# 		else:
		# 			show_feature_saliency(prop_iter, extractor=extractor)
		# 			break


np.seterr(all="raise")
chainer.global_config.cv_resize_backend = "PIL"
with chainer.using_config("train", False):
	main(arguments.parse_args())

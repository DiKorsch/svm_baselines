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
		fit_object=True,

		thresh_type=args.thresh_type,
		cluster_init=ClusterInitType.MAXIMAS,
		feature_composition=args.feature_composition,

	)


	kwargs = dict(
		model=model,
		extractor=extractor,
		propagator=propagator,
		iterator=it,
		prepare=prepare,
		device=args.gpu[0],
	)
	if args.extract:
		with outputs(args) as files:
			pipeline = ExtractionPipeline(
				files=files,
				**kwargs
			)
	else:
		pipeline = VisualizationPipeline(
			**kwargs
		)


	pipeline.run()


np.seterr(all="raise")
chainer.global_config.cv_resize_backend = "PIL"
with chainer.using_config("train", False):
	main(arguments.parse_args())

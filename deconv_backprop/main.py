#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")
import os
import logging
import chainer
import numpy as np

from tqdm import tqdm
from functools import partial

from cvargparse import Arg, ArgFactory, GPUParser
from cvdatasets.utils import read_info_file
from cvdatasets import AnnotationType

from chainer_addons.utils.imgproc import Size
from chainer.dataset.convert import concat_examples
from chainer.backends import cuda

from diffprop.core.models import InvCeptionV3


import matplotlib.pyplot as plt

def _norm_grad(grad):
	res = np.abs(grad).sum(axis=-1)

	res -= res.min()
	res /= res.max()

	return res

def main(args):

	model_kwargs=dict(
		pretrained_model=args.load,
		n_classes=201
	)
	model = InvCeptionV3(**model_kwargs)

	GPU = args.gpu[0]

	if GPU >= 0:
		cuda.get_device_from_id(GPU).use()
		model.to_gpu(GPU)

	annot_cls = AnnotationType.get(args.dataset).value
	annot = annot_cls(args.data)

	data = annot.new_dataset(args.subset)

	logging.info(f"Loadded {len(data)} samples from \"{args.dataset}\" dataset")
	kwargs = dict(
		n_jobs=args.n_jobs,
		batch_size=args.batch_size,
		shuffle=False,
		repeat=False,
	)
	it, n_batches = data.new_iterator(**kwargs)

	size = Size(args.input_size)
	prepare = partial(model.meta.prepare_func, size=size, keep_ratio=False)

	for batch in tqdm(it, total=n_batches):
		_batch = [(prepare(im), lab) for im, _, lab in batch]
		X, y = concat_examples(_batch, device=GPU)

		feat, _ = model(X, layer_name=model.meta.feature_layer)
		# feat.grad = model.xp.ones_like(feat.array)
		# feat.backward()

		feat.unchain_backward()
		grads = model.backward(model.xp.ones_like(feat.array))
		grads.unchain_backward()
		# for x, grad in zip(X, grads):
		# 	fig, axs = plt.subplots(1, 2)
		# 	_x = cuda.to_cpu(x)[::-1].transpose(1,2,0)
		# 	axs[0].imshow(_x)
		# 	axs[0].axis("off")

		# 	_g = cuda.to_cpu(grad.array)[::-1].transpose(1,2,0)
		# 	axs[1].imshow(_norm_grad(_g))
		# 	axs[1].axis("off")

		# 	plt.show()
		# 	plt.close()


DEFAULT_INFO_FILE=os.environ.get("DATA", "/home/korsch/Data/info.yml")
info_file = read_info_file(DEFAULT_INFO_FILE)

parser = GPUParser(ArgFactory([
		Arg("data", default=DEFAULT_INFO_FILE),

		Arg("dataset", choices=info_file.DATASETS.keys()),
		Arg("parts", choices=info_file.PARTS.keys()),

		Arg("--subset", "-s", default=None, choices=["train", "test"]),

		Arg("--input_size", type=int, nargs="+", default=0,
			help="overrides default input size of the model, if greater than 0"),

		Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),

		Arg("--load", type=str, help="load already fine-tuned model"),

	])\

	.batch_size()\
	.debug()\
)


chainer.global_config.cv_resize_backend = "PIL"
parser.init_logger()

with chainer.using_config("train", False):
	main(parser.parse_args())

import numpy as np
from chainer.cuda import to_cpu

from functools import partial

from l1_svm_parts.core.propagator import Propagator
from l1_svm_parts.utils import prepare_back

from cluster_parts import BoundingBoxParts
from cluster_parts.utils import image

def extract_parts(propagator, xp=np, swap_channels=True, **kwargs):


	for i, grads in propagator:
		im = prepare_back(propagator.ims[i], swap_channels=swap_channels)

		parts = BoundingBoxParts(im, xp=xp, **kwargs)

		pred_grad, full_grad = [prepare_back(image.saliency_to_im(grad, xp=xp), swap_channels=swap_channels)
			for grad in grads]

		pred_parts = parts(pred_grad)
		full_parts = parts(full_grad)
		yield i, (pred_parts, full_parts)

def parts_to_file(im_id, part_id, box, out):
	(x, y), w, h = box

	print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
		file=out
	)


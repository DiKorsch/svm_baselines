
from chainer.cuda import to_cpu

from functools import partial

from l1_svm_parts.core.propagator import Propagator

from cluster_parts import BoundingBoxParts
from cluster_parts.utils import image

def extract_parts(propagator, swap_channels=True, **kwargs):


	for i, (full_grad, pred_grad) in propagator:
		im = image.prepare_back(propagator.ims[i])

		parts = BoundingBoxParts(im, swap_channels=swap_channels, **kwargs)
		pred_parts = parts(pred_grad)
		full_parts = parts(full_grad)
		yield i, (pred_parts, full_parts)

def parts_to_file(im_id, part_id, box, out):
	(x, y), w, h = box

	print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
		file=out
	)


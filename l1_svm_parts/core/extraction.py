
from chainer.cuda import to_cpu

from functools import partial

from l1_svm_parts.core.propagator import Propagator
from l1_svm_parts.core.parts_and_bboxes import get_parts
from l1_svm_parts.utils.image import prepare_back

def extract_parts(propagator, swap_channels=True, **kwargs):

	_get_parts = partial(get_parts, swap_channels=swap_channels, **kwargs)

	for i, (full_grad, pred_grad) in propagator:
		im = prepare_back(propagator.ims[i])

		pred_parts = _get_parts(im, pred_grad)
		full_parts = _get_parts(im, full_grad)
		yield i, (pred_parts, full_parts)

def parts_to_file(im_id, part_id, box, out):
	(x, y), w, h = box

	print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
		file=out
	)


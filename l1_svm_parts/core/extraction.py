import numpy as np

from chainer.cuda import to_cpu

from functools import partial

from cluster_parts.core import BoundingBoxPartExtractor

def extract_parts(propagator, it, batch_i, files, extractor):
	pred_out, full_out = files

	for i, im, (pred_grad, full_grad), _ in propagator:

		parts = [extractor(im, pred_grad), extractor(im, full_grad)]

		im_idx = i + batch_i * it.batch_size
		im_uuid = it.dataset.uuids[im_idx]

		for pred_part, full_part in zip(*parts):
			parts_to_file(im_uuid, *pred_part, out=pred_out)
			parts_to_file(im_uuid, *full_part, out=full_out)

def parts_to_file(im_id, part_id, box, out):
	(x, y), w, h = box

	print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
		file=out
	)


#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np

from tqdm import tqdm

from cvargparse import BaseParser, Arg
from cvdatasets import AnnotationType, ImageWrapperDataset

def main(args):
	annot_cls = AnnotationType.get(args.dataset).value
	parts_name = "{}_GLOBAL".format(args.dataset.upper())
	annot = annot_cls(args.info_file, parts=parts_name)

	dataset = annot.new_dataset(dataset_cls=ImageWrapperDataset)

	min_w, min_h = np.inf, np.inf
	with open(args.parts_file) as f:
		for line, im_obj in zip(f, tqdm(dataset)):
			im_id, part_id, x, y, w, h = line.strip().split()

			x, y, w, h = map(int, [x, y, w, h])
			min_h, min_w = min(h, min_h), min(w, min_w)
			if 0 in (w,h):
				import pdb; pdb.set_trace()

			size = im_obj.im.size
			size = im_obj.im_array.shape[:-1][::-1]
			# print(size)
			if x >= size[0] or y >= size[1]:
				import pdb; pdb.set_trace()

	print(min_w, min_h)


parser = BaseParser([
	Arg("info_file"),
	AnnotationType.as_arg("dataset"),
	Arg("parts_file"),
])

parser.init_logger()
main(parser.parse_args())

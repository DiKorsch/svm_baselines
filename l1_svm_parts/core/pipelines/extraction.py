import numpy as np

from chainer.cuda import to_cpu

from functools import partial

from cluster_parts.core import BoundingBoxPartExtractor

from l1_svm_parts.core.pipelines.base import BasePipeline

class ExtractionPipeline(BasePipeline):

	def __init__(self, *, files, iterator, **kwargs):
		super(ExtractionPipeline, self).__init__(iterator=iterator, **kwargs)
		assert None not in files

		self.pred_out, self.full_out = files
		self.uuids = iterator.dataset.uuids
		self.batch_size = iterator.batch_size

	def to_out(self, im_id, part_id, box, out):
		(x, y), w, h = box

		print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
			file=out)

	def to_pred_out(self, *args, **kwargs):
		self.to_out(*args, **kwargs, out=self.pred_out)

	def to_full_out(self, *args, **kwargs):
		self.to_out(*args, **kwargs, out=self.full_out)

	def __getstate__(self):
		# self_dict = self.__dict__.copy()
		# del self_dict['pred_out']
		# del self_dict['full_out']
		return dict(
			extractor=self.extractor
		)

	def __setstate__(self, state):
		self.__dict__.update(state)

	def extract(self, args):
		i, im, (pred_grad, full_grad), _ = args

		try:
			return i, self.extractor(im, [pred_grad, full_grad])
		except KeyboardInterrupt:
			pass

	def __call__(self, propagator, pool=None):

		if pool is None:
			_map = map
		else:
			_map = pool.map


		for i, parts in _map(self.extract, propagator):

			im_idx = i + self.batch_i * self.batch_size
			im_uuid = self.uuids[im_idx]

			for pred_part, full_part in zip(*parts):
				self.to_pred_out(im_uuid, *pred_part)
				self.to_full_out(im_uuid, *full_part)


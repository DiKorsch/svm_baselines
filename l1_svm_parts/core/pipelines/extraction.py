import numpy as np
import threading
import logging
import time

from chainer.cuda import to_cpu

from functools import partial
from multiprocessing import Pool, Manager

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
			extractor=self.extractor,
			inqueue=self.inqueue,
			outqueue=self.outqueue,
			worker_done=self.worker_done,
			writer_done=self.writer_done,
		)

	def __setstate__(self, state):
		self.__dict__.update(state)

	def error_callback(self, exc):
		print(exc)

	def run(self):
		n_jobs = self.batch_size // 2
		with Pool(n_jobs) as pool, Manager() as m:

			self.worker_done = m.Value("b", False)
			self.writer_done = m.Value("b", False)
			self.inqueue = m.Queue()#maxsize=20)
			self.outqueue = m.Queue()

			self.writer_thread = threading.Thread(target=self.write_result)
			self.writer_thread.deamon = True
			self.writer_thread._state = 0
			self.writer_thread.start()

			results = [pool.apply_async(self.extract, error_callback=self.error_callback) for _ in range(n_jobs)]

			super(ExtractionPipeline, self).run()

			self.worker_done.value = True
			for result in results:
				result.wait()

			self.writer_done.value = True
			self.writer_thread.join()

	def __call__(self, prop_iter):

		for i, im, grads, _ in prop_iter:

			im_idx = i + self.batch_i * self.batch_size
			im_uuid = self.uuids[im_idx]

			self.inqueue.put([im_uuid, im, grads])

	def extract(self):
		while True:
			if self.worker_done.value and self.inqueue.empty():
				break

			if self.inqueue.empty():
				time.sleep(0.1)
				continue

			im_uuid, im, grads = self.inqueue.get()
			# print("Processing item {}".format(im_uuid))
			result = im_uuid, self.extractor(im, grads)
			self.outqueue.put(result)

		# print("Exiting worker", self.worker_done.value, self.inqueue.qsize())

	def write_result(self):
		while True:
			if self.writer_done.value and self.outqueue.empty():
				break

			if self.outqueue.empty():
				time.sleep(0.1)
				continue

			im_uuid, parts = self.outqueue.get()

			for pred_part, full_part in zip(*parts):
				self.to_pred_out(im_uuid, *pred_part)
				self.to_full_out(im_uuid, *full_part)

		# print("Writer exists... {} | {} | {}".format(self.writer_done.value, self.inqueue.qsize(), self.outqueue.qsize()))

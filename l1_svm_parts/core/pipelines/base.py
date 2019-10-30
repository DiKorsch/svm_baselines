import abc
import numpy as np
import chainer

from tqdm import tqdm

from chainer.dataset.convert import concat_examples


class BasePipeline(abc.ABC):

	def __init__(self, *, model, iterator, extractor, propagator, prepare, device=-1):
		super(BasePipeline, self).__init__()
		self.model = model
		self.extractor = extractor
		self.propagator = propagator
		self.iterator = iterator
		self.prepare = prepare
		self.device = device

		self.n_batches = int(np.ceil(len(self.iterator.dataset.uuids) / self.iterator.batch_size))

	def run(self, *args, **kwargs):

		for self.batch_i, batch in tqdm(enumerate(self.iterator), total=self.n_batches):

			batch = [(self.prepare(im), lab) for im, _, lab in batch]
			X, y = concat_examples(batch, device=self.device)

			ims = chainer.Variable(X)
			feats = self.model(ims, layer_name=self.model.meta.feature_layer)

			if isinstance(feats, tuple):
				feats = feats[0]

			with self.propagator(feats, ims, y) as prop_iter:

				self(prop_iter, *args, **kwargs)

	@abc.abstractmethod
	def __call__(self, iter, *args, **kwargs):
		raise NotImplementedError

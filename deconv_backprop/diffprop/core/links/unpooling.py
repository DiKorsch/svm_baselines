import chainer
import chainer.functions as F

from chainer.functions.pooling.average_pooling_2d import AveragePooling2D, AveragePooling2DGrad
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D, MaxPooling2DGrad
from chainer_addons.links import GlobalAveragePooling

from functools import partial

from .base import BaseWrapper


class BaseUnpooling(BaseWrapper):
	# wraps a partial function
	wraps = partial

	def __init__(self, partial_func):
		super(BaseUnpooling, self).__init__(partial_func)
		self.kwargs = partial_func.keywords

	def forward(self, gy):
		_pooling = self.pooling_classes[0](**self.kwargs)
		y = _pooling.apply(self.inputs)[0]
		_pooling_grad = self.pooling_classes[1](_pooling)
		try:
			return _pooling_grad.apply((gy,))[0]
		except Exception as e:
			import pdb; pdb.set_trace()
			raise e


class MaxUnPooling2D(BaseUnpooling):
	pooling_classes = (MaxPooling2D, MaxPooling2DGrad)


class AverageUnPooling2D(BaseUnpooling):
	pooling_classes = (AveragePooling2D, AveragePooling2DGrad)

class GlobalAverageUnPooling2D(BaseWrapper):
	wraps = GlobalAveragePooling

	def __init__(self, g_avg_func):
		super(GlobalAverageUnPooling2D, self).__init__(g_avg_func)

	def forward(self, gy):
		_x = self.inputs[0]
		n, c, h, w = _x.shape
		return F.broadcast_to(gy[..., None, None] / (h*w), _x.shape)

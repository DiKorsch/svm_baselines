import chainer
import chainer.functions as F

from .base import BaseWrapper


class UnLinear(BaseWrapper):
	wraps = chainer.links.Linear

	def __init__(self, layer):
		super(UnLinear, self).__init__(layer)
		self.W = layer.W

	def forward(self, gy):
		X, = self.inputs
		return F.sum(self.W, axis=0).reshape(X.shape[1:])


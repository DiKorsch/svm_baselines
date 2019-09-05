import chainer
import chainer.links as L
import chainer.functions as F

from chainer_addons.links import Conv2D_BN

from .base import BaseWrapper
from .batch_denorm import BatchDeNorm


class DeConv2D(BaseWrapper):
	wraps = chainer.links.Convolution2D

	def __init__(self, conv):
		super(DeConv2D, self).__init__(conv)

		W = conv.W.array
		_out, _in, *ksize = conv.W.shape

		with self.init_scope():
			self.deconv = L.Deconvolution2D(
				in_channels=_out,
				out_channels=_in,
				ksize=ksize,
				stride=conv.stride,
				pad=conv.pad,
				initialW=W,
				nobias=True)

	def _wrap(self, func):
		"""Do not wrap the Convolution layers"""
		return func

	def forward(self, gy):
		return self.deconv(gy)


class DeConv2D_BN(BaseWrapper):
	wraps = Conv2D_BN

	def __init__(self, conv_bn):
		super(DeConv2D_BN, self).__init__(conv_bn)

		with self.init_scope():
			self.deconv = DeConv2D(conv_bn.conv)
			self.debn = BatchDeNorm(conv_bn.bn)

		self.activation = conv_bn.activation

	def forward(self, gy):

		if self.activation == F.relu:
			gy *= F.sign(self.outputs)

		return self.deconv(self.debn(gy))

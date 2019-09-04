import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np

from functools import wraps, partial

from utils import BaseTestCase


class Conv2D_BN(chainer.Chain):
	def __init__(self, insize, outsize, ksize, stride=1, pad=0, activation=F.relu):
		super(Conv2D_BN, self).__init__()
		assert callable(activation)

		with self.init_scope():
			self.conv = L.Convolution2D(insize, outsize,
				ksize=ksize, stride=stride, pad=pad, nobias=True)
			self.bn = L.BatchNormalization(outsize,
				use_gamma=False, eps=2e-5)

		self.activation = activation

	def __call__(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return self.activation(x)

class LayerComparisons(BaseTestCase):


	def _new_conv(self):

		kwargs = dict(ksize=self.ksize, stride=1, pad=0, nobias=True)
		return L.Convolution2D(self._in, self._out, **kwargs)

	def _new_conv_bn(self, activation=F.identity):
		kwargs = dict(ksize=self.ksize, stride=1, pad=0, activation=activation)
		return Conv2D_BN(self._in, self._out, **kwargs)

	def _new_deconv_bn(self, conv_bn):
		deconv = self._new_deconv(conv_bn.conv)

		inv_var = np.reciprocal(conv_bn.bn.avg_var + conv_bn.bn.eps)
		bn_grad = F.normalization.batch_normalization.FixedBatchNormalizationGrad(
			eps=conv_bn.bn.eps,
			expander=(None, slice(None, None, None), None, None),
			axis=(0,2,3),

			inv_var=inv_var,
			inv_std=np.sqrt(inv_var),
		)
		_x = conv_bn.conv(self.X)

		gamma = conv_bn.xp.ones_like(conv_bn.bn.avg_mean)

		@wraps(bn_grad)
		def inner(y):

			gy = y.grad
			if conv_bn.activation == F.relu:
				gy *= F.sign(y)
			gx, *_ = bn_grad(_x, gamma, conv_bn.bn.avg_mean, conv_bn.bn.avg_var, gy)
			return deconv(gx)

		return inner

	def _new_deconv(self, conv):
		kwargs = dict(ksize=conv.ksize, stride=conv.stride, pad=conv.pad, nobias=True)
		W = conv.W.array
		return L.Deconvolution2D(self._out, self._in, initialW=W, **kwargs)

	def test_conv_layer(self):

		conv = self._new_conv()
		y = self._call_and_backprop(conv)

		deconv = self._new_deconv(conv)
		res = deconv(y.grad)
		self.assertGradCloseTo(res.array)

	def test_linear_layer(self):
		linear = L.Linear(self._out)

		y = linear(self.X)
		y = self._call_and_backprop(linear)

		G = linear.W.array.sum(axis=0).reshape(self.X.shape[1:])

		self.assertGradCloseTo(np.broadcast_to(G, self.X.shape))


	def test_relu(self):

		y = self._call_and_backprop(F.relu)
		self.assertGradCloseTo(F.sign(y).array)

	def test_conv_relu(self):
		conv = self._new_conv()
		y = self._call_and_backprop(lambda x: F.relu(conv(x)))

		deconv = self._new_deconv(conv)
		res = deconv(y.grad * F.sign(y))

		self.assertGradCloseTo(res.array)

	def test_conv_bn(self):
		conv_bn = self._new_conv_bn(F.identity)

		with chainer.using_config("train", False):
			y = self._call_and_backprop(conv_bn)

		deconv_bn_grad = self._new_deconv_bn(conv_bn)
		res = deconv_bn_grad(y)

		self.assertGradCloseTo(res.array)

	def test_conv_bn_relu(self):
		conv_bn = self._new_conv_bn(F.relu)

		with chainer.using_config("train", False):
			y = self._call_and_backprop(conv_bn)

		deconv_bn_grad = self._new_deconv_bn(conv_bn)
		res = deconv_bn_grad(y)

		self.assertGradCloseTo(res.array)

	def test_max_pooling(self):
		kwargs = dict(ksize=2, stride=2)
		pool = lambda x: F.max_pooling_2d(x, **kwargs)
		y = self._call_and_backprop(pool)

		pool_grad = F.pooling.max_pooling_2d.MaxPooling2DGrad(y.creator)

		gx, = pool_grad.apply((y.grad,))
		self.assertGradCloseTo(gx.array)


	def test_avg_pooling(self):
		kwargs = dict(ksize=2, stride=2)
		pool = lambda x: F.average_pooling_2d(x, **kwargs)
		y = self._call_and_backprop(pool)

		pool_grad = F.pooling.average_pooling_2d.AveragePooling2DGrad(y.creator)

		gx, = pool_grad.apply((y.grad,))

		self.assertGradCloseTo(gx.array)

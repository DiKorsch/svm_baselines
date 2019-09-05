import numpy as np
import unittest

import chainer
import chainer.functions as F
import chainer.links as L

from chainer_addons.links import Conv2D_BN

from chainer import Variable
from functools import wraps

from diffprop.core import DeConv2D, BatchDeNorm

class BaseTestCase(unittest.TestCase):
	N = 5
	h, w = 15, 15
	ksize = 3
	_in, _out = 3, 32

	atol = 1e-7


	def assertAllClose(self, x0, x1):
		return self.assertTrue(np.allclose(x0, x1, atol=self.atol))

	def assertGradCloseTo(self, other):
		return self.assertAllClose(self.X.grad, other)

	def _call_and_backprop(self, layer):
		y = layer(self.X)
		y.grad = np.ones_like(y.array)
		y.backward()
		y.unchain_backward()

		return y

	def setUp(self):
		self.X = Variable(np.random.randn(self.N, self._in, self.h, self.w).astype(np.float32))

	def _new_conv(self):
		kwargs = dict(ksize=self.ksize, stride=1, pad=0, nobias=True)
		return L.Convolution2D(self._in, self._out, **kwargs)

	def _new_conv_bn(self, activation=F.identity):
		kwargs = dict(ksize=self.ksize, stride=1, pad=0, activation=activation)
		return Conv2D_BN(self._in, self._out, **kwargs)

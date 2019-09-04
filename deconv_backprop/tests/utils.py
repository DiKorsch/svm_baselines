import numpy as np
import unittest

from chainer import Variable

class BaseTestCase(unittest.TestCase):

	def assertAllClose(self, x0, x1):
		return self.assertTrue(np.allclose(x0, x1))

	def assertGradCloseTo(self, other):
		return self.assertAllClose(self.X.grad, other)

	def _call_and_backprop(self, layer):
		y = layer(self.X)
		y.grad = np.ones_like(y.array)
		y.backward()

		return y

	def setUp(self):
		N = 5
		h, w = 15, 15
		self._in, self._out = 3, 32
		self.ksize = 3
		self.X = Variable(np.random.randn(N, self._in, h, w).astype(np.float32))

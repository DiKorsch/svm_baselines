import chainer.functions as F
import chainer.links as L

import numpy as np
import unittest

from functools import partial
from functools import wraps

from tests.utils import BaseTestCase

from diffprop.core import AverageUnPooling2D
from diffprop.core import BatchDeNorm
from diffprop.core import DeConv2D
from diffprop.core import DeConv2D_BN
from diffprop.core import MaxUnPooling2D
from diffprop.core import UnLinear

class TestSimpleLayers(BaseTestCase):


	def test_conv_layer(self):

		conv = self._new_conv()
		y = self._call_and_backprop(conv)

		deconv = DeConv2D(conv)
		res = deconv(y.grad)
		self.assertGradCloseTo(res.array)

	def test_linear_layer(self):
		layer = L.Linear(self._out)
		unlinear = UnLinear(layer)

		y = self._call_and_backprop(layer)
		res = unlinear(y.grad)

		self.assertGradCloseTo(res.array)

	def test_bn_layer(self):
		layer = L.BatchNormalization(self._in)
		denorm = BatchDeNorm(layer)

		y = self._call_and_backprop(layer)
		res = denorm(y.grad)

		self.assertGradCloseTo(res.array)


	def test_relu(self):

		y = self._call_and_backprop(F.relu)
		self.assertGradCloseTo(F.sign(y).array)

class TestSimpleLayerCombinations(BaseTestCase):


	def test_conv_relu(self):
		conv = self._new_conv()
		y = self._call_and_backprop(lambda x: F.relu(conv(x)))

		deconv = DeConv2D(conv)
		res = deconv(y.grad * F.sign(y))

		self.assertGradCloseTo(res.array)

	def test_conv_bn(self):
		conv_bn = self._new_conv_bn(F.identity)
		deconv_bn_grad = DeConv2D_BN(conv_bn)

		y = self._call_and_backprop(conv_bn)

		res = deconv_bn_grad(y.grad)
		self.assertGradCloseTo(res.array)

	def test_conv_bn_relu(self):
		conv_bn = self._new_conv_bn(F.relu)
		deconv_bn_grad = DeConv2D_BN(conv_bn)

		y = self._call_and_backprop(conv_bn)

		res = deconv_bn_grad(y.grad)

		self.assertGradCloseTo(res.array)

	# @unittest.skip
	def test_max_pooling(self):
		pool = partial(F.max_pooling_2d, ksize=2, stride=2)
		unpool = MaxUnPooling2D(pool)

		y = self._call_and_backprop(pool)

		gx = unpool(y.grad)

		self.assertGradCloseTo(gx.array)


	# @unittest.skip
	def test_avg_pooling(self):
		pool = partial(F.average_pooling_2d, ksize=2, stride=2)
		unpool = AverageUnPooling2D(pool)

		y = self._call_and_backprop(pool)
		gx = unpool(y.grad)

		self.assertGradCloseTo(gx.array)

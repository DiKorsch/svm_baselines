import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np
import unittest

from chainer_addons.models import InceptionV3
from diffprop.core import AverageUnPooling2D, MaxUnPooling2D
from diffprop.core.models import InvCeptionV3
from diffprop.core.models.invception.blocks import InvCeptionHead
from diffprop.core.models.invception.blocks import InvCeption1
from diffprop.core.models.invception.blocks import InvCeption2
from diffprop.core.models.invception.blocks import InvCeption3
from diffprop.core.models.invception.blocks import InvCeption4
from diffprop.core.models.invception.blocks import InvCeption5

from utils import BaseTestCase


class BlockComparison(BaseTestCase):
	h, w = 299, 299

	def setUp(self):
		super(BlockComparison, self).setUp()
		self.model = InceptionV3()

	def test_inception_head(self):
		inv_head = InvCeptionHead(self.model.head)

		y = self._call_and_backprop(self.model.head)

		res = inv_head(y.grad)
		self.assertGradCloseTo(res.array)


	def test_inception_block1(self):
		_X = self.model(self.X, layer_name="head")[0]
		self.X = chainer.Variable(np.random.randn(*_X.shape).astype(np.float32))

		inv_block = InvCeption1(self.model.mixed00, AverageUnPooling2D)
		y = self._call_and_backprop(self.model.mixed00)

		res = inv_block(y.grad)
		self.assertGradCloseTo(res.array)


	def test_inception_block2(self):
		_X = self.model(self.X, layer_name="mixed02")[0]
		self.X = chainer.Variable(np.random.randn(*_X.shape).astype(np.float32))

		inv_block = InvCeption2(self.model.mixed03, MaxUnPooling2D)
		y = self._call_and_backprop(self.model.mixed03)

		res = inv_block(y.grad)
		self.assertGradCloseTo(res.array)


	def test_inception_block3(self):
		_X = self.model(self.X, layer_name="mixed03")[0]
		self.X = chainer.Variable(np.random.randn(*_X.shape).astype(np.float32))

		inv_block = InvCeption3(self.model.mixed04, AverageUnPooling2D)
		y = self._call_and_backprop(self.model.mixed04)

		res = inv_block(y.grad)
		self.assertGradCloseTo(res.array)


	def test_inception_block4(self):
		_X = self.model(self.X, layer_name="mixed07")[0]
		self.X = chainer.Variable(np.random.randn(*_X.shape).astype(np.float32))

		inv_block = InvCeption4(self.model.mixed08, MaxUnPooling2D)
		y = self._call_and_backprop(self.model.mixed08)

		res = inv_block(y.grad)
		self.assertGradCloseTo(res.array)


	def test_inception_block5(self):
		_X = self.model(self.X, layer_name="mixed08")[0]
		self.X = chainer.Variable(np.random.randn(*_X.shape).astype(np.float32))

		inv_block = InvCeption5(self.model.mixed09, AverageUnPooling2D)
		y = self._call_and_backprop(self.model.mixed09)

		res = inv_block(y.grad)
		self.assertGradCloseTo(res.array)

class ModelComparison(BaseTestCase):
	h, w = 299, 299

	def test_whole_model(self):
		invmodel = InvCeptionV3()

		# first part of the model
		model = lambda x: invmodel(x, layer_name=invmodel.meta.feature_layer)[0]

		y = self._call_and_backprop(model)
		res = invmodel.backward(y.grad)

		self.assertGradCloseTo(res.array)

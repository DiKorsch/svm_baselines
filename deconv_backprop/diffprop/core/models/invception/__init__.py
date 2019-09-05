import chainer
from chainer_addons.models import InceptionV3
from chainer_addons.links import PoolingType

from collections import OrderedDict
from functools import partial

from diffprop.core import AverageUnPooling2D, GlobalAverageUnPooling2D, MaxUnPooling2D
from diffprop.core.models.invception.blocks import InvCeptionHead
from diffprop.core.models.invception.blocks import InvCeption1
from diffprop.core.models.invception.blocks import InvCeption2
from diffprop.core.models.invception.blocks import InvCeption3
from diffprop.core.models.invception.blocks import InvCeption4
from diffprop.core.models.invception.blocks import InvCeption5


class InvCeptionV3(InceptionV3):

	def __init__(self, pooling=PoolingType.Default, *args, **kwargs):
		super(InvCeptionV3, self).__init__(pooling=pooling, *args, **kwargs)

		with self.init_scope():
			self.invhead = InvCeptionHead(self.head)

			self.invmixed00 = InvCeption1(self.mixed00, unpooling_cls=AverageUnPooling2D)
			self.invmixed01 = InvCeption1(self.mixed01, unpooling_cls=AverageUnPooling2D)
			self.invmixed02 = InvCeption1(self.mixed02, unpooling_cls=AverageUnPooling2D)

			self.invmixed03 = InvCeption2(self.mixed03, unpooling_cls=MaxUnPooling2D)

			self.invmixed04 = InvCeption3(self.mixed04, unpooling_cls=AverageUnPooling2D)
			self.invmixed05 = InvCeption3(self.mixed05, unpooling_cls=AverageUnPooling2D)
			self.invmixed06 = InvCeption3(self.mixed06, unpooling_cls=AverageUnPooling2D)
			self.invmixed07 = InvCeption3(self.mixed07, unpooling_cls=AverageUnPooling2D)

			self.invmixed08 = InvCeption4(self.mixed08, unpooling_cls=MaxUnPooling2D)

			self.invmixed09 = InvCeption5(self.mixed09, unpooling_cls=AverageUnPooling2D)
			self.invmixed10 = InvCeption5(self.mixed10, unpooling_cls=AverageUnPooling2D)

		if PoolingType.get(pooling) == PoolingType.AVG:
			self.unpool = AverageUnPooling2D(self.pool)

		elif PoolingType.get(pooling) == PoolingType.G_AVG:
			self.unpool = GlobalAverageUnPooling2D(self.pool)

		elif PoolingType.get(pooling) == PoolingType.MAX:
			self.unpool = MaxUnPooling2D(self.pool)

		else:
			raise ValueError(f"{pooling} pooling is not supported yet!")

	@property
	def _backward_links(self):

		names = ["invmixed{:02d}".format(i) for i in reversed(range(11))]
		body = [(name, [getattr(self, name)]) for name in names]
		return [
			("unpool", [self.unpool]),
		] + body + [
			("invhead", [self.invhead]),
		]


	@property
	def backward_functions(self):
		return OrderedDict(self._backward_links)


	def backward(self, gy, start_layer="unpool"):

		bwd_funcs = self.backward_functions
		assert start_layer in bwd_funcs, \
			f"Layer {start_layer} was not found in {bwd_funcs.keys()}"

		gx = gy
		started = False
		for key, funcs in bwd_funcs.items():
			if not started and key != start_layer: continue

			started = True
			for func in funcs:
				gx = func(gx)

		return gx

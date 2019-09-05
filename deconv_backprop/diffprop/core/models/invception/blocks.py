import chainer

from diffprop.core import DeConv2D_BN, MaxUnPooling2D, AverageUnPooling2D

class InvCeptionHead(chainer.Chain):

	def __init__(self, head):
		super(InvCeptionHead, self).__init__()

		with self.init_scope():
			self.deconv1 = DeConv2D_BN(head.conv1)
			self.deconv2 = DeConv2D_BN(head.conv2)
			self.deconv3 = DeConv2D_BN(head.conv3)
			self.unpooling4 = MaxUnPooling2D(head.pool4)

			self.deconv5 = DeConv2D_BN(head.conv5)
			self.deconv6 = DeConv2D_BN(head.conv6)
			self.unpooling7 = MaxUnPooling2D(head.pool7)


	def __call__(self, gy):


		gx = self.unpooling7(gy)
		gx = self.deconv6(gx)
		gx = self.deconv5(gx)


		gx = self.unpooling4(gx)
		gx = self.deconv3(gx)
		gx = self.deconv2(gx)
		gx = self.deconv1(gx)

		return gx

class BaseBlock(chainer.Chain):
	def __init__(self, block, unpooling_cls):
		super(BaseBlock, self).__init__()

		with self.init_scope():
			if hasattr(block, "pool_conv"):
				self.pool_deconv = DeConv2D_BN(block.pool_conv)


		self.unpool = unpooling_cls(block.pool)



class InvCeption1(BaseBlock):
	def __init__(self, block, unpooling_cls):
		super(InvCeption1, self).__init__(block, unpooling_cls)

		with self.init_scope():
			self.deconv1x1 = DeConv2D_BN(block.conv1x1)
			self._out1 = block.conv1x1.conv.W.shape[0]

			self.deconv5x5_1 = DeConv2D_BN(block.conv5x5_1)
			self.deconv5x5_2 = DeConv2D_BN(block.conv5x5_2)
			self._out2 = block.conv5x5_2.conv.W.shape[0]

			self.deconv3x3_1 = DeConv2D_BN(block.conv3x3_1)
			self.deconv3x3_2 = DeConv2D_BN(block.conv3x3_2)
			self.deconv3x3_3 = DeConv2D_BN(block.conv3x3_3)
			self._out3 = block.conv3x3_3.conv.W.shape[0]

			# self._out4 = block.pool_conv.conv.W.shape[0]


	def forward(self, gy):
		rest = gy
		gy1, rest = rest[:, :self._out1], rest[:, self._out1:]
		gy2, rest = rest[:, :self._out2], rest[:, self._out2:]
		gy3, gy4  = rest[:, :self._out3], rest[:, self._out3:]

		gx1 = self.deconv1x1(gy1)
		gx2 = self.deconv5x5_1(self.deconv5x5_2(gy2))
		gx3 = self.deconv3x3_1(self.deconv3x3_2(self.deconv3x3_3(gy3)))
		gx4 = self.unpool(self.pool_deconv(gy4))

		return sum([gx1, gx2, gx3, gx4])


class InvCeption2(BaseBlock):
	def __init__(self, block, unpooling_cls):
		super(InvCeption2, self).__init__(block, unpooling_cls)

		with self.init_scope():
			self.deconv3x3 = DeConv2D_BN(block.conv3x3)
			self._out1 = block.conv3x3.conv.W.shape[0]

			self.deconv3x3_1 = DeConv2D_BN(block.conv3x3_1)
			self.deconv3x3_2 = DeConv2D_BN(block.conv3x3_2)
			self.deconv3x3_3 = DeConv2D_BN(block.conv3x3_3)
			self._out2 = block.conv3x3_3.conv.W.shape[0]


	def forward(self, gy):
		rest = gy

		gy1, rest = rest[:, :self._out1], rest[:, self._out1:]
		gy2, gy3 = rest[:, :self._out2], rest[:, self._out2:]

		gx1 = self.deconv3x3(gy1)
		gx2 = self.deconv3x3_1(self.deconv3x3_2(self.deconv3x3_3(gy2)))
		gx3 = self.unpool(gy3)

		return sum([gx1, gx2, gx3])


class InvCeption3(BaseBlock):
	def __init__(self, block, unpooling_cls):
		super(InvCeption3, self).__init__(block, unpooling_cls)

		with self.init_scope():

			self.deconv1x1 = DeConv2D_BN(block.conv1x1)
			self._out1 = block.conv1x1.conv.W.shape[0]

			self.deconv7x7_1 = DeConv2D_BN(block.conv7x7_1)
			self.deconv7x7_2 = DeConv2D_BN(block.conv7x7_2)
			self.deconv7x7_3 = DeConv2D_BN(block.conv7x7_3)
			self._out2 = block.conv7x7_3.conv.W.shape[0]

			self.deconv7x7x2_1 = DeConv2D_BN(block.conv7x7x2_1)
			self.deconv7x7x2_2 = DeConv2D_BN(block.conv7x7x2_2)
			self.deconv7x7x2_3 = DeConv2D_BN(block.conv7x7x2_3)
			self.deconv7x7x2_4 = DeConv2D_BN(block.conv7x7x2_4)
			self.deconv7x7x2_5 = DeConv2D_BN(block.conv7x7x2_5)
			self._out3 = block.conv7x7x2_5.conv.W.shape[0]


	def forward(self, gy):

		rest = gy
		gy1, rest = rest[:, :self._out1], rest[:, self._out1:]
		gy2, rest = rest[:, :self._out2], rest[:, self._out2:]
		gy3, gy4  = rest[:, :self._out3], rest[:, self._out3:]

		gx1 = self.deconv1x1(gy1)
		gx2 = self.deconv7x7_1(self.deconv7x7_2(self.deconv7x7_3(gy2)))
		gx3 = self.deconv7x7x2_1(self.deconv7x7x2_2(self.deconv7x7x2_3(self.deconv7x7x2_4(self.deconv7x7x2_5(gy3)))))
		gx4 = self.unpool(self.pool_deconv(gy4))

		return sum([gx1, gx2, gx3, gx4])


class InvCeption4(BaseBlock):
	def __init__(self, block, unpooling_cls):
		super(InvCeption4, self).__init__(block, unpooling_cls)

		with self.init_scope():

			self.deconv3x3_1 = DeConv2D_BN(block.conv3x3_1)
			self.deconv3x3_2 = DeConv2D_BN(block.conv3x3_2)
			self._out1 = block.conv3x3_2.conv.W.shape[0]

			self.deconv7x7_1 = DeConv2D_BN(block.conv7x7_1)
			self.deconv7x7_2 = DeConv2D_BN(block.conv7x7_2)
			self.deconv7x7_3 = DeConv2D_BN(block.conv7x7_3)
			self.deconv7x7_4 = DeConv2D_BN(block.conv7x7_4)
			self._out2 = block.conv7x7_4.conv.W.shape[0]


	def forward(self, gy):
		rest = gy
		gy1, rest = rest[:, :self._out1], rest[:, self._out1:]
		gy2, gy3 = rest[:, :self._out2], rest[:, self._out2:]

		gx1 = self.deconv3x3_1(self.deconv3x3_2(gy1))
		gx2 = self.deconv7x7_1(self.deconv7x7_2(self.deconv7x7_3(self.deconv7x7_4(gy2))))
		gx3 = self.unpool(gy3)
		return sum([gx1, gx2, gx3])


class InvCeption5(BaseBlock):
	def __init__(self, block, unpooling_cls):
		super(InvCeption5, self).__init__(block, unpooling_cls)

		with self.init_scope():

			self.deconv1x1 = DeConv2D_BN(block.conv1x1)
			self._out1 = block.conv1x1.conv.W.shape[0]

			self.deconv3x3_1 = DeConv2D_BN(block.conv3x3_1)
			self.deconv3x3_2 = DeConv2D_BN(block.conv3x3_2)
			self.deconv3x3_3 = DeConv2D_BN(block.conv3x3_3)
			self._out2 = block.conv3x3_3.conv.W.shape[0]

			self.deconv3x3x2_1 = DeConv2D_BN(block.conv3x3x2_1)
			self.deconv3x3x2_2 = DeConv2D_BN(block.conv3x3x2_2)
			self.deconv3x3x2_3 = DeConv2D_BN(block.conv3x3x2_3)
			self.deconv3x3x2_4 = DeConv2D_BN(block.conv3x3x2_4)
			self._out3 = block.conv3x3x2_4.conv.W.shape[0]

	def forward(self, gy):
		rest = gy
		gy1, rest = rest[:, :self._out1], rest[:, self._out1:]

		gy2_1, rest = rest[:, :self._out2], rest[:, self._out2:]
		gy2_2, rest = rest[:, :self._out2], rest[:, self._out2:]

		gy3_1, rest = rest[:, :self._out3], rest[:, self._out3:]
		gy3_2, gy4  = rest[:, :self._out3], rest[:, self._out3:]

		gx1 = self.deconv1x1(gy1)

		gx2_1 = sum([self.deconv3x3_2(gy2_1), self.deconv3x3_3(gy2_2)])
		gx2 = self.deconv3x3_1(gx2_1)

		gx3_1 = sum([self.deconv3x3x2_3(gy3_1), self.deconv3x3x2_4(gy3_2)])
		gx3 = self.deconv3x3x2_1(self.deconv3x3x2_2(gx3_1))

		gx4 = self.unpool(self.pool_deconv(gy4))


		return sum([gx1, gx2, gx3, gx4])

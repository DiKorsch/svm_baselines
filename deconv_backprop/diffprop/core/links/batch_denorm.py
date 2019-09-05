import chainer
from chainer.functions.normalization.batch_normalization import FixedBatchNormalizationGrad

from .base import BaseWrapper


class BatchDeNorm(BaseWrapper):
	wraps = chainer.links.BatchNormalization

	def __init__(self, bn):
		super(BatchDeNorm, self).__init__(bn)
		self.eps = bn.eps
		self.expander = (None, slice(None, None, None), None, None)
		self.axis = (0, 2, 3)

		self.bn = bn

	def forward(self, gy):
		X, = self.inputs

		mean = self.bn.avg_mean
		var = self.bn.avg_var
		inv_var = self.xp.reciprocal(var + self.eps)

		if self.bn.gamma is None:
			gamma = self.xp.ones_like(mean)
		else:
			gamma = self.bn.gamma

		gx, *_ = FixedBatchNormalizationGrad(
			eps=self.eps,
			expander=self.expander,
			axis=self.axis,
			inv_var=inv_var,
			inv_std=self.xp.sqrt(inv_var),
		)(X, gamma, mean, var, gy)

		return gx

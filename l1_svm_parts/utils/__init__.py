import numpy as np
import chainer.functions as F

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from cvargparse.utils.enumerations import BaseChoiceType

class ClusterInitType(BaseChoiceType):
	NONE = 0
	MAXIMAS = 0
	MIN_MAX = 2

	Default = MAXIMAS

	def __call__(self, grad, K=None):

		if self == ClusterInitType.MAXIMAS:
			return peak_local_max(grad, num_peaks=K).T

		elif self == ClusterInitType.MIN_MAX:

			max_loc = np.unravel_index(grad.argmax(), grad.shape)
			min_loc = np.unravel_index(grad.argmin(), grad.shape)

			return np.vstack([min_loc, max_loc]).T

			# this may result in multiple extremas
			# max_init = np.where(grad == grad.max())
			# min_init = np.where(grad == grad.min())

			# import pdb; pdb.set_trace()
			# return np.hstack([max_init, min_init])

		else:
			return None

class ThresholdType(BaseChoiceType):
	NONE = 0
	MEAN = 1
	PRECLUSTER = 2
	OTSU = 3

	Default = PRECLUSTER

	def __call__(self, im, grad):
		if self == ThresholdType.MEAN:
			return np.abs(grad).mean()

		elif self == ThresholdType.PRECLUSTER:
			from .clustering import cluster_gradient
			centers, labs = cluster_gradient(im, grad,
				K=2, thresh=None,
				cluster_init=ClusterInitType.MIN_MAX,
				# small fix, since it does not work with only one dimension
				# or at least, it has to be fixed
				feature_composition=["grad", "grad"]
			)


			# 1th cluster represents the cluster around the maximal peak
			return labs == 1

		elif self == ThresholdType.OTSU:
			thresh = threshold_otsu(grad)
			return grad > thresh
		else:
			return None


def prop_back(model, from_, to, coefs=None):
	to.grad = None
	model.cleargrads()

	if coefs is None:
		F.sum(from_).backward()
	else:
		F.sum(from_[np.where(coefs)]).backward()

	assert to.grad is not None, "Backprop mode is off?"
	return to.grad


class IdentityScaler(object):
	"""
		Do not scale the data, just return itself
	"""
	transform = lambda self, x: x

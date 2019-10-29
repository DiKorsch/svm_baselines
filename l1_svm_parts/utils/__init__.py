import numpy as np
import chainer.functions as F

from chainer.backends import cuda

def normalize(im, axis=(1,2)):
	im = im - im.min(axis=axis, keepdims=True)
	chan_max = im.max(axis=axis, keepdims=True)
	if 0 in chan_max:
		return im
	else:
		return im / chan_max

def prepare_back(im, swap_channels=True):
	im = im.array if hasattr(im, "array") else im
	im = normalize(cuda.to_cpu(im))
	if swap_channels:
		im = im[::-1]
	return im.transpose(1, 2, 0)

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

import numpy as np
import chainer.functions as F

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

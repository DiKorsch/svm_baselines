import numpy as np
from chainer.cuda import to_cpu


def normalize(im, axis=(1,2)):
	im = im - im.min(axis=axis, keepdims=True)
	chan_max = im.max(axis=axis, keepdims=True)
	if 0 in chan_max:
		return im
	else:
		return im / chan_max


def prepare_back(im):
	im = im.array if hasattr(im, "array") else im
	return normalize(to_cpu(im))[::-1].transpose(1, 2, 0)

def grad_to_im(grad, xp=np, keepdims=True):
	return xp.abs(grad).mean(axis=0, keepdims=keepdims)

def grad_correction(grad, xp=np, sigma=None, gamma=1.):

	grad = prepare_back(grad_to_im(grad, xp=xp))

	if sigma is None:
		grad = grad.squeeze()
	else:
		grad = gaussian_filter(grad, sigma=sigma).squeeze()

	return grad**gamma

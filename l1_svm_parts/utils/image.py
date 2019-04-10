import numpy as np

from chainer.cuda import to_cpu
from scipy.ndimage.filters import gaussian_filter

def normalize(im, axis=(1,2)):
	im = im - im.min(axis=axis, keepdims=True)
	chan_max = im.max(axis=axis, keepdims=True)
	if 0 in chan_max:
		return im
	else:
		return im / chan_max


def prepare_back(im, swap_channels=True):
	im = im.array if hasattr(im, "array") else im
	im = normalize(to_cpu(im))
	if swap_channels:
		im = im[::-1]
	return im.transpose(1, 2, 0)

def grad_to_im(grad, xp=np, keepdims=True):
	return xp.abs(grad).mean(axis=0, keepdims=keepdims)

def grad_correction(grad, xp=np, sigma=None, gamma=1., swap_channels=True):

	grad = prepare_back(grad_to_im(grad, xp=xp), swap_channels)

	if sigma is None:
		grad = grad.squeeze()
	else:
		grad = gaussian_filter(grad, sigma=sigma).squeeze()

	return grad**gamma

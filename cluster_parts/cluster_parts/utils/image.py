import numpy as np

from scipy.ndimage.filters import gaussian_filter


def saliency_to_im(saliency, chan_axis=0, xp=np, keepdims=True):
	"""Compute absolute mean over the channel axis"""
	return xp.abs(saliency).mean(axis=chan_axis, keepdims=keepdims)

def correction(saliency, xp=np, sigma=None, gamma=1.):
	"""Apply an optional gaussian filter and gamma correction"""

	if sigma is None:
		saliency = saliency.squeeze()
	else:
		saliency = gaussian_filter(saliency, sigma=sigma).squeeze()

	return saliency**gamma

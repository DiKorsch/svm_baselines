import numpy as np

from scipy.ndimage.filters import gaussian_filter


def saliency_to_im(saliency, xp=np, keepdims=True):
	return xp.abs(saliency).mean(axis=0, keepdims=keepdims)

def correction(saliency, xp=np, sigma=None, gamma=1.):

	# saliency = prepare_back(saliency_to_im(saliency), swap_channels)

	if sigma is None:
		saliency = saliency.squeeze()
	else:
		saliency = gaussian_filter(saliency, sigma=sigma).squeeze()

	return saliency**gamma

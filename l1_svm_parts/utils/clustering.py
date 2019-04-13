import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max

def _norm(arr):
	arr = arr - arr.min()
	arr_max = arr.max()
	if arr_max == 0:
		return arr
	else:
		return arr / arr_max

def _as_cluster_feats(im, grad, coords):
	ys, xs = coords
	_im = im[ys, xs]
	return np.stack([
		_norm(ys),
		_norm(xs),
		_norm(grad[ys, xs].ravel()),
		_norm(_im[:, 0].ravel()),
		_norm(_im[:, 1].ravel()),
		_norm(_im[:, 2].ravel()),
	]).transpose()

def cluster_gradient(im, grad, K=4, thresh=None, init_from_maximas=False):
	assert K is not None and K > 0, "Positive K is required!"

	### get x,y coordinates
	if thresh is None:
		idxs = np.arange(np.multiply(*grad.shape))
		coords = np.unravel_index(idxs, grad.shape)
	else:
		coords = np.where(np.abs(grad) >= thresh)

	if init_from_maximas:
		init_coords = peak_local_max(grad, num_peaks=K).T
		init = _as_cluster_feats(im, grad, init_coords)
		clf = KMeans(K, init=init, n_init=1)
	else:
		clf = KMeans(K)

	data = _as_cluster_feats(im, grad, coords)
	clf.fit(data)
	labels = np.full(grad.shape, np.nan)
	labels[coords] = clf.labels_
	centers = clf.cluster_centers_.copy()
	centers[:, 0] *= grad.shape[0]
	centers[:, 1] *= grad.shape[1]

	return centers, labels

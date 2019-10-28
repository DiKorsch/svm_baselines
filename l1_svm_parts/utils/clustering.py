raise ImportError("DO NOT IMPORT ME!")
import numpy as np
from sklearn.cluster import KMeans

from l1_svm_parts.utils import ClusterInitType

def _norm(arr):
	arr = arr - arr.min()
	arr_max = arr.max()
	if arr_max == 0:
		return arr
	else:
		return arr / arr_max

def _as_cluster_feats(im, grad, coords, feature_composition=None):
	ys, xs = coords
	_im = im[ys, xs]

	composition = dict(
		coords=[_norm(ys), _norm(xs)],
		grad=[_norm(grad[ys, xs].ravel())],
		RGB=[_norm(_im[:, i].ravel()) for i in range(3)],
	)

	if feature_composition is None:

		feature_composition = ["coords", "grad", "RGB"]

	cluster_feats = []
	for key in feature_composition:
		assert key in composition
		cluster_feats.extend(composition[key])

	return np.stack(cluster_feats).transpose()



def cluster_gradient(im, grad, K=4,
	thresh=None, cluster_init=ClusterInitType.Default,
	feature_composition=["coords"]):
	assert K is not None and K > 0, "Positive K is required!"


	cluster_init = ClusterInitType.get(cluster_init)
	init_coords = cluster_init(grad, K)

	if init_coords is None:
		clf = KMeans(K)
	else:
		init = _as_cluster_feats(im, grad, init_coords, feature_composition)
		clf = KMeans(K, init=init, n_init=1)


	### get x,y coordinates
	if isinstance(thresh, (int, float)):
		coords = np.where(np.abs(grad) >= thresh)
	elif isinstance(thresh, np.ndarray):
		# thresh is a mask
		coords = np.where(thresh)
	else:
		idxs = np.arange(np.multiply(*grad.shape))
		coords = np.unravel_index(idxs, grad.shape)
	data = _as_cluster_feats(im, grad, coords, feature_composition)


	clf.fit(data)

	labels = np.full(grad.shape, np.nan)
	labels[coords] = clf.labels_
	centers = clf.cluster_centers_.copy()
	centers[:, 0] *= grad.shape[0]
	centers[:, 1] *= grad.shape[1]

	return centers, labels

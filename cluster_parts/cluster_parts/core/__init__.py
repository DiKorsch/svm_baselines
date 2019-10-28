import logging
import numpy as np

from functools import partial
from functools import wraps
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import ThresholdType
from cluster_parts.utils import image


def _param_check(param, default):
	return default if param is None else param

def _norm(arr):
	arr = arr - arr.min()
	arr_max = arr.max()
	if arr_max == 0:
		return arr
	else:
		return arr / arr_max

def _as_cluster_feats(im, saliency, coords, feature_composition=None):
	ys, xs = coords
	_im = im[ys, xs]

	composition = dict(
		coords=[_norm(ys), _norm(xs)],
		saliency=[_norm(saliency[ys, xs].ravel())],
		RGB=[_norm(_im[:, i].ravel()) for i in range(3)],
	)

	if feature_composition is None:
		feature_composition = ["coords", "saliency", "RGB"]

	cluster_feats = []
	for key in feature_composition:
		assert key in composition
		cluster_feats.extend(composition[key])

	return np.stack(cluster_feats).transpose()

def _check_min_bbox(bbox, min_bbox):
	y0, x0, y1, x1 = bbox
	h, w = y1 - y0, x1 - x0

	# if bbox is greater that min_bbox in both, the width and height
	if min(h, w) >= min_bbox:
		return bbox

	old_bbox = bbox.copy()
	dy, dx = max(min_bbox - h, 0), max(min_bbox - w, 0)

	bbox[0] -= int(dy / 2)
	bbox[1] -= int(dx / 2)
	bbox[2] += int(dy / 2)
	bbox[3] += int(dx / 2)

	if (bbox < 0).any():
		dy, dx, _, _ = np.minimum(bbox, 0)
		bbox[0] -= dy
		bbox[1] -= dx
		bbox[2] -= dy
		bbox[3] -= dx

	text = "Adjusted bbox from {} to {}".format(old_bbox, bbox)
	logging.debug("=" * len(text))
	logging.debug(text)
	logging.debug("=" * len(text))
	return bbox


class BoundingBoxParts(object):
	"""Extracts bounding box parts from a saliency map and an image

		Arguments:
			image
				- input image

			optimal (default: True)
				- ...

			gamma, sigma
				- saliency correction parameters

	"""

	def __init__(self, image, xp=np, *,
		K=4, optimal=True, gamma=1.0, sigma=1.0, swap_channels=True,
		min_bbox=64, fit_object=False,
		thresh_type=ThresholdType.Default,
		cluster_init=ClusterInitType.Default,
		feature_composition=["coords"]):
		super(BoundingBoxParts, self).__init__()

		self.image = image

		self.gamma = gamma
		self.sigma = sigma
		self.swap_channels = swap_channels

		assert K is not None and K > 0, "Positive K is required!"
		self.K = K
		self.optimal = optimal
		self.thresh_type = ThresholdType.get(thresh_type)
		self.cluster_init = ClusterInitType.get(cluster_init)
		self.feature_composition = feature_composition
		self.fit_object = fit_object
		self.min_bbox = min_bbox
		self.xp = xp


	def __call__(self, saliency, *, xp=None, gamma=None, sigma=None, swap_channels=None, **kwargs):
		if (saliency == 0).all():
			import pdb; pdb.set_trace()

		xp = _param_check(xp, self.xp)
		gamma = _param_check(gamma, self.gamma)
		sigma = _param_check(sigma, self.sigma)
		swap_channels = _param_check(swap_channels, self.swap_channels)

		saliency = image.correction(saliency, xp, sigma, gamma, swap_channels)

		centers, labs = self.cluster_saliency(saliency)
		boxes = self.get_boxes(centers, labs, saliency)
		return boxes


	def get_boxes(self, centers, labels, saliency):
		saliency = saliency if self.optimal else None

		values = labels[np.logical_not(np.isnan(labels))]
		obj_box = None

		res = []
		for i in range(self.K):
			mask = labels == i
			if mask.sum() == 0:
				# if there is no cluster for this label,
				# then take the extend of the whole object
				if obj_box is None:
					# lazy init this box to speed up the things a bit
					obj_mask = np.logical_not(np.isnan(labels))
					obj_box = self.fit_bbox(obj_mask, saliency)

				y0, x0, y1, x1 = obj_box
			else:
				y0, x0, y1, x1 = self.fit_bbox(mask, saliency)

			h, w = y1 - y0, x1 - x0
			res.append([i, ((x0, y0), w, h)])

		if self.fit_object:
			if obj_box is None:
				obj_mask = np.logical_not(np.isnan(labels))
				obj_box = self.fit_bbox(obj_mask, saliency)

			y0, x0, y1, x1 = obj_box
			h, w = y1 - y0, x1 - x0
			res.append([i + 1, ((x0, y0), w, h)])

		return res

	def fit_bbox(self, mask, saliency):
		ys, xs = np.where(mask)
		bbox = np.array([min(ys), min(xs), max(ys), max(xs)])

		bbox = _check_min_bbox(bbox, self.min_bbox)
		if not self.optimal:
			return bbox

		y0, x0, y1, x1 = bbox
		h, w = y1 - y0, x1 - x0
		search_area = mask[y0:y1, x0:x1].astype(np.float32)

		# the search area is weighted with the saliency values
		if saliency is not None:
			assert 0.0 <= saliency.max() <= 1.0
			search_area *= saliency[y0:y1, x0:x1]

		scaler = np.array([h, w, h, w])

		# (1) Our search area is [(x0,y0), (x1,y1)].
		# (2) If we  shift it with (x0,y0) it becomes [(0,0), (w,h)]
		# (3) We see it in normalized way, so it changes to [(0,0), (1,1)]
		# (4) The initial bbox is then always [(0.25, 0.25), (0.75, 0.75)]  with width and height 0.5

		init_bbox = np.array([0.25, 0.25, 0.75, 0.75])

		def _measures(b, mask):
			# scale back to the area mentioned in (2)
			y0, x0, y1, x1 = map(int, b * scaler)
			area = mask[y0:y1, x0:x1]
			_h, _w = y1-y0, x1-x0

			if _h < _w:
				ratio = (_h/_w) if _w != 0 else -100
			else:
				ratio = (_w/_h) if _h != 0 else -100

			TP = area.sum()
			FP = (1-area).sum()
			FN = mask.sum() - TP
			TN = (1-mask).sum() - FP

			# ratio = 1
			return TP, FP, FN, TN, ratio

		def Recall(b, mask):
			TP, FP, FN, TN, ratio = _measures(b, mask)
			# if TP + FN == 0:
			# 	return 0
			return -(TP / (TP + FN)) * ratio

		def Precision(b, mask):
			TP, FP, FN, TN, ratio = _measures(b, mask)
			# if TP + FP == 0:
			# 	return 0
			return -(TP / (TP + FP)) * ratio

		def Fscore(b, mask, beta=1):
			TP, FP, FN, TN, ratio = _measures(b, mask)
			recall = TP / (TP + FN)
			prec = TP / (TP + FP)

			return -((1 + beta**2) * (recall * prec) / (recall + beta**2 * prec) )

		F2 = partial(Fscore, beta=2)
		F1 = partial(Fscore, beta=1)
		F0_5 = partial(Fscore, beta=0.5)

		res = minimize(Recall, init_bbox,
					   args=(search_area,),
					   options=dict(eps=1e-2, gtol=1e-1),
					   bounds=Bounds(0, 1),
					   )
		# scale back to (2) and shift to original values (1)
		bbox = res.x * scaler + np.array([y0, x0, y0, x0])
		bbox = _check_min_bbox(bbox, self.min_bbox)
		return bbox


	def cluster_saliency(self, saliency,):
		im = self.image
		thresh = self.thresh_type(self.image, saliency)
		init_coords = self.cluster_init(saliency, self.K)

		if init_coords is None:
			clf = KMeans(self.K)
		else:
			init = _as_cluster_feats(self.image, saliency, init_coords,
				self.feature_composition)
			clf = KMeans(self.K, init=init, n_init=1)

		### get x,y coordinates of pixels to cluster
		if isinstance(thresh, (int, float)):
			coords = np.where(np.abs(saliency) >= thresh)

		elif isinstance(thresh, np.ndarray):
			# thresh is a mask
			coords = np.where(thresh)

		else:
			idxs = np.arange(np.multiply(*saliency.shape))
			coords = np.unravel_index(idxs, saliency.shape)

		data = _as_cluster_feats(self.image, saliency, coords,
			self.feature_composition)

		clf.fit(data)

		labels = np.full(saliency.shape, np.nan)
		labels[coords] = clf.labels_
		centers = clf.cluster_centers_.copy()
		centers[:, 0] *= saliency.shape[0]
		centers[:, 1] *= saliency.shape[1]

		return centers, labels


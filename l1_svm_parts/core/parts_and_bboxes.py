import numpy as np
import logging

from functools import partial, wraps
from scipy.optimize import minimize, Bounds

from l1_svm_parts.utils import ClusterInitType
from l1_svm_parts.utils import ThresholdType
from l1_svm_parts.utils.image import grad_correction
from l1_svm_parts.utils.clustering import cluster_gradient


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

	text = f"Adjusted bbox from {old_bbox} to {bbox}"
	logging.debug("=" * len(text))
	logging.debug(text)
	logging.debug("=" * len(text))
	return bbox

class EnlargeBbox(object):
	def __init__(self, factor):
		self.factor = factor

	def __call__(self, func):

		@wraps(func)
		def inner(mask, *args, **kwargs):
			im_h, im_w = mask.shape
			bbox = func(mask, *args, **kwargs)
			if self.factor <= 0:
				return bbox

			y0, x0, y1, x1 = bbox

			w, h = x1 - x0, y1 - y0
			dx = w * self.factor / 2
			dy = h * self.factor / 2

			y0, x0 = max(y0 - dy, 0), max(x0 - dx, 0)
			y1, x1 = min(y1 + dy, im_h), min(x1 + dx, im_w)
			return y0, x0, y1, x1
		return inner



@EnlargeBbox(factor=0.2)
def fit_bbox(mask, grad=None, optimize=False, min_bbox=64):
	ys, xs = np.where(mask)
	bbox = np.array([min(ys), min(xs), max(ys), max(xs)])

	bbox = _check_min_bbox(bbox, min_bbox)
	if not optimize:
		return bbox

	y0, x0, y1, x1 = bbox
	h, w = y1 - y0, x1 - x0
	search_area = mask[y0:y1, x0:x1].astype(np.float32)
	if grad is not None:
		assert 0.0 <= grad.max() <= 1.0
		search_area *= grad[y0:y1, x0:x1]
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
		return -(TP / (TP + FN)) * ratio

	def Precision(b, mask):
		TP, FP, FN, TN, ratio = _measures(b, mask)
		return -(TP / (TP + FP))

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
	bbox = _check_min_bbox(bbox, min_bbox)
	return bbox

def get_boxes(centers, labels, **kwargs):
	values = labels[np.logical_not(np.isnan(labels))]
	res = []
	for i in np.unique(values):
		y0, x0, y1, x1 = fit_bbox(labels == i, **kwargs)
		h, w = y1 - y0, x1 - x0
		res.append([i, ((x0, y0), w, h)])
	return res

def _boxes(im, grad, optimal=True,
	thresh_type=ThresholdType.Default,
	min_bbox=64,
	**kwargs):

	thresh_type = ThresholdType.get(thresh_type)
	centers, labs = cluster_gradient(
		im, grad,
		thresh=thresh_type(im, grad),
		**kwargs)

	if optimal:
		# Boxes optimized for maximum recall
		return get_boxes(centers, labs, optimize=True, grad=grad, min_bbox=min_bbox), centers, labs
	else:
		return get_boxes(centers, labs, optimize=False, min_bbox=min_bbox), centers, labs

def optimal_boxes(im, grad, **kwargs):
	return _boxes(im, grad, optimal=True, **kwargs)

def simple_boxes(im, grad, **kwargs):
	return _boxes(im, grad, optimal=False, **kwargs)

def get_parts(im, grad, xp=np,
	swap_channels=True,
	alpha=0.5, gamma=1.0, sigma=1,
	peak_size=None, **kwargs):

	# hack if the gradient is not present
	if (grad == 0).all():
		h, w, c = im.shape
		middle = (h//2, w//2)
		return [[i, (middle, h//2, w//2)] for i in range(kwargs["K"])]

	grad = grad_correction(grad, xp, sigma, gamma, swap_channels)
	boxes, centers, labs = optimal_boxes(im, grad, **kwargs)

	return boxes

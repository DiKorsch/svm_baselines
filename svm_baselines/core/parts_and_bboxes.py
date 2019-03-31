import numpy as np
from functools import partial
from scipy.optimize import minimize, Bounds

from svm_baselines.utils.image import grad_correction
from svm_baselines.utils.clustering import cluster_gradient


def fit_bbox(mask, grad=None, optimize=False):
	ys, xs = np.where(mask)
	bbox = np.array([min(ys), min(xs), max(ys), max(xs)])

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
		TP = area.sum()
		FP = (1-area).sum()
		FN = mask.sum() - TP
		TN = (1-mask).sum() - FP
		return TP, FP, FN, TN

	def Recall(b, mask):
		TP, FP, FN, TN = _measures(b, mask)
		return -(TP / (TP + FN))

	def Precision(b, mask):
		TP, FP, FN, TN = _measures(b, mask)
		return -(TP / (TP + TN))

	def Fscore(b, mask, beta=1):
		TP, FP, FN, TN = _measures(b, mask)
		recall = TP / (TP + FN)
		prec = TP / (TP + TN)

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
	return res.x * scaler + np.array([y0, x0, y0, x0])

def get_boxes(centers, labels, **kwargs):
	values = labels[np.logical_not(np.isnan(labels))]
	res = []
	for i in np.unique(values):
		y0, x0, y1, x1 = fit_bbox(labels == i, **kwargs)
		h, w = y1 - y0, x1 - x0
		res.append([i, ((x0, y0), w, h)])
	return res

def _boxes(im, grad, optimal=True, **kwargs):
	thresh = np.abs(grad).mean()
	centers, labs = cluster_gradient(im, grad, **kwargs)
	if optimal:
		# Boxes optimized for maximum recall
		return get_boxes(centers, labs, optimize=True, grad=grad)
	else:
		return get_boxes(centers, labs, optimize=False)

def optimal_boxes(im, grad, **kwargs):
	return _boxes(im, grad, optimal=True, **kwargs)

def simple_boxes(im, grad, **kwargs):
	return _boxes(im, grad, optimal=False, **kwargs)

def get_parts(im, grad, xp=np,
	alpha=0.5, gamma=1.0, sigma=1,
	peak_size=None, K=None, init_from_maximas=False):

	grad = grad_correction(grad, xp, sigma, gamma)
	return optimal_boxes(im, grad,
		K=K, thresh=thresh,
		init_from_maximas=init_from_maximas)

import numpy as np
import logging

from chainer.cuda import to_cpu

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from functools import partial

from l1_svm_parts.core.propagator import Propagator
from l1_svm_parts.utils import prop_back, prepare_back, saliency_to_im

from cluster_parts.core import BoundingBoxPartExtractor
from cluster_parts.utils import ThresholdType
from cluster_parts.utils import ClusterInitType

def imshow(im, ax=None, title=None, figsize=(32, 18), **kwargs):
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.imshow(im, **kwargs)
	ax.axis("off")

	if title is not None:
		ax.set_title(title)

	return ax


def visualize_coefs(coefs, **kwargs):
	fig, ax = plt.subplots(3, 1)
	ax[0].set_title("Selected features per class")
	ax[0].set_xlabel("Feature Dimension")
	ax[0].set_ylabel("Class")
	ax[0].imshow(coefs != 0, aspect="auto")

	ax[1].set_title("Selections per feature dimension")
	ax[1].set_xlabel("Feature Dimension")
	ax[1].set_ylabel("# of selections")
	# ax[1].imshow((coefs != 0).sum(axis=0, keepdims=True), aspect="auto")
	ax[1].scatter(range(coefs.shape[1]), (coefs != 0).sum(axis=0))

	ax[2].set_title("Number of selected features per class")
	ax[2].set_xlabel("Class")
	ax[2].set_ylabel("# of features")
	ax[2].bar(range(len(coefs)), (coefs != 0).sum(axis=1))

	plt.show()
	plt.close()


def plot_gradient(extractor, im, grad, peak_size=None, spec=None):

	ax1 = plt.subplot(spec[2:4, 0:2])
	ax2 = plt.subplot(spec[0:2, 2:4])

	grad = extractor.corrector(grad)

	ax2 = imshow(im, ax=ax2, title="Gradient")
	ax2 = imshow(grad, ax=ax2, alpha=0.7)

	thresh_mask = extractor.thresh_type(im, grad)
	new_grad = np.zeros_like(grad)
	new_grad[thresh_mask] = grad[thresh_mask]

	# new_grad = grad.copy()
	ax1 = imshow(new_grad, ax=ax1, cmap=plt.cm.gray, alpha=1.0)
	# ax = imshow(thresh_mask, ax=ax, cmap=plt.cm.Reds, alpha=0.4)

	if extractor.K is None or extractor.K <= 0:
		return

	cmap = plt.cm.viridis_r

	ys, xs = init_coords = extractor.cluster_init(grad, extractor.K)
	ax1.scatter(xs, ys, marker="x", color="black")

	centers, labs = extractor.cluster_saliency(im, grad)
	boxes = extractor.get_boxes(centers, labs, grad)

	for c, box in boxes:
		ax1.add_patch(Rectangle(
			*box, fill=False,
			linewidth=3,
			color=cmap(c / len(boxes))))

	imshow(labs, ax1, cmap=cmap, alpha=0.3)

	for i in range(extractor.K):
		row, col = np.unravel_index(i, (2, 2))
		_ax = plt.subplot(spec[row + 2, col + 2])
		_c, ((x, y), w, h) = boxes[i]
		x,y,w,h = map(int, [x,y,w,h])
		imshow(im[y:y+h, x:x+w], _ax, title="Part #{}".format(i+1))

	if peak_size is not None:
		peaks = peak_local_max(grad, min_distance=peak_size, exclude_border=False)

		ys, xs = peaks.T
		ax.scatter(xs, ys, marker="x", c="blue")


	return ax1


def show_feature_saliency(propagator, extractor, xp=np, swap_channels=True,
	plot_topk_grads=False,
	plot_sel_feats_grad=False):

	for i, (full_grad, pred_grad) in propagator:
		pred, gt = propagator.topk_preds[i, -1], propagator.labs[i]
		logging.debug("predicted class: {}, GT class: {}".format(pred, gt))

		spec = GridSpec(4, 4)
		fig = plt.figure(figsize=(16, 9))

		ax0 = plt.subplot(spec[0:2, 0:2])

		im = prepare_back(propagator.ims[i], swap_channels=swap_channels)
		title ="Original Image [predicted: {}, GT: {}]".format(pred, gt)

		imshow(im, ax=ax0, title=title)
		grad = prepare_back(saliency_to_im(pred_grad, xp=xp), swap_channels=swap_channels)
		plot_gradient(extractor, im, grad, spec=spec)

		plt.tight_layout()
		plt.show()
		plt.close()

		############### Plots Top-k Gradients ###############
		if plot_topk_grads:
			grads_coefs_preds = list(zip(
				topk_pred_im_grad,
				topk_pred_coefs,
				topk_preds[i]
			))

			_coefs = np.array([c[i] for c in topk_pred_coefs])
			fig, ax = plt.subplots(figsize=(16, 9))
			# reverse first axis, so that top prediction is displayed first
			ax.imshow(_coefs[::-1] != 0, aspect="auto")
			for k, (_grad, _coef, _pred) in enumerate(reversed(grads_coefs_preds), 1):
				_plot_gradient(
					im, _grad[i],
					ax=None,
					title="Pred #{} Gradient (Class {})".format(k, _pred),
				)
			plt.show()
			plt.close()
		#####################################################

import numpy as np
import logging

from chainer.cuda import to_cpu

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from functools import partial

from l1_svm_parts.core.parts_and_bboxes import optimal_boxes, simple_boxes
from l1_svm_parts.utils import ThresholdType
from l1_svm_parts.utils import ClusterInitType
from l1_svm_parts.utils import prop_back
from l1_svm_parts.utils.image import prepare_back, grad_correction

def imshow(im, ax=None, figsize=(32, 18), **kwargs):
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.imshow(im, **kwargs)
	ax.axis("off")
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


def plot_gradient(im, grad, xp=np, ax=None, title="",
	swap_channels=True,
	alpha=0.5, gamma=1.0, sigma=1,
	peak_size=None, K=None, **kwargs):
	grad = grad_correction(grad, xp, sigma, gamma, swap_channels)

	if ax is None:
		_, ax = plt.subplots(figsize=(16, 9))

	#ax = imshow(im, ax=ax)
	ax = imshow(np.zeros_like(im), ax=ax)

	thresh_type = ThresholdType.get(kwargs["thresh_type"])
	thresh_mask = thresh_type(im, grad)
	new_grad = np.zeros_like(grad)
	new_grad[thresh_mask] = grad[thresh_mask]

	# new_grad = grad.copy()
	ax = imshow(new_grad, ax=ax,
				cmap=plt.cm.gray,
				alpha=alpha)

	# ax = imshow(thresh_mask, ax=ax,
	# 			cmap=plt.cm.Reds,
	# 			alpha=0.4)

	if K is not None and K > 0:
		cmap = plt.cm.viridis_r

		cluster_init = ClusterInitType.get(kwargs["cluster_init"])
		init_coords = cluster_init(grad, K)

		ax.scatter(*init_coords[::-1], marker="x", color="K")

		boxes, centers, labs = optimal_boxes(im, grad,
			K=K, **kwargs)

		for c, box in boxes:
			ax.add_patch(Rectangle(
				*box, fill=False,
				linewidth=3,
				color=cmap(c / len(boxes))))

		# sim_boxes, centers, labs = simple_boxes(im, grad,
		# 	K=K, **kwargs)

		# for c, box in sim_boxes:
		# 	ax.add_patch(Rectangle(
		# 		*box, fill=False,
		# 		color=cmap(c / len(sim_boxes)),
		# 		linewidth=3,
		# 		linestyle="--",
		# 		alpha=1
		# 	))

		# ys, xs = centers[:, :2].T
		# ax.scatter(xs, ys, marker="D", c=range(K), cmap=cmap)
		ax.imshow(labs, cmap=cmap, alpha=0.3)


		# _idx = 0
		# _fig, _ax = plt.subplots()
		# _labs = labs.copy()
		# _labs[labs != _idx] = np.nan

		# _sim_c, _sim_box = sim_boxes[_idx]
		# _c, _box = boxes[_idx]

		# _ax = imshow(np.zeros_like(_labs), ax=_ax, cmap=plt.cm.gray)
		# _ax.imshow(_labs, vmin=labs.min(), vmax=labs.max(),
		# 	cmap=cmap, alpha=0.3)
		# _ax.add_patch(Rectangle(
		# 	*_sim_box, fill=False,
		# 	color=cmap(_sim_c / len(sim_boxes)),
		# 	linewidth=3,
		# 	linestyle="--",
		# 	alpha=0.5
		# ))
		# _ax.add_patch(Rectangle(
		# 	*_box, fill=False,
		# 	linewidth=3,
		# 	color=cmap(_c / len(sim_boxes)),
		# ))


		_fig, _axs = plt.subplots(2, 2)

		for i in range(K):
			_ax = _axs[np.unravel_index(i, _axs.shape)]
			_c, ((x, y), w, h) = boxes[i]
			x,y,w,h = map(int, [x,y,w,h])
			_ax.axis("off")
			_ax.imshow(im[y:y+h, x:x+w])
			_ax.set_title(f"Part #{i+1}")

	if peak_size is not None:
		peaks = peak_local_max(
				grad,
				min_distance=peak_size,
				exclude_border=False)

		ys, xs = peaks.T
		ax.scatter(xs, ys, marker="x", c="blue")


	ax.set_title(title)

	return ax


def show_feature_saliency(model, coefs, ims, labs, feats, topk_preds,
	swap_channels=True,
	normalize_grads=False,
	plot_topk_grads=False,
	plot_sel_feats_grad=False,
	**kwargs):

	preds = topk_preds[:, -1]

	gt_coefs = coefs[to_cpu(labs)]
	gt_im_grad = prop_back(model, feats, ims, gt_coefs != 0)

	topk_pred_coefs = [coefs[to_cpu(p)] for p in topk_preds.T]
	topk_pred_im_grad = [prop_back(model, feats, ims, p != 0) for p in topk_pred_coefs]

	pred_coefs = topk_pred_coefs[-1]
	pred_im_grad = topk_pred_im_grad[-1]

	full_im_grad = prop_back(model, feats, ims)


	# for g in [full_im_grad, pred_im_grad]:
	# 	logging.debug(g.min(), g.max())

	_plot_gradient = partial(plot_gradient,
		xp=model.xp, swap_channels=swap_channels, **kwargs)

	for i, (gt_coef, pred_coef) in enumerate(zip(gt_coefs, pred_coefs)):
		# if i != 2:
		# 	continue
		logging.debug(
			"predicted class: {}, GT class: {}".format(
			preds[i], labs[i]))
		fig, _axs = plt.subplots(1, 2, figsize=(16, 9))

		axs = lambda i: _axs[np.unravel_index(i, _axs.shape)]
		[axs(_i).axis("off") for _i in range(_axs.size)]

		im = prepare_back(ims[i], swap_channels=swap_channels)
		ax = imshow(im, ax=axs(0))
		ax.set_title("Original Image [predicted: {}, GT: {}]".format(
			preds[i], labs[i]))

		# _plot_gradient(im, gt_im_grad[i],
		# 			   ax=axs(1),
		# 			   title="GT Gradient",
		# 			  )

		_plot_gradient(im, pred_im_grad[i],
					   ax=axs(1),
					   title="Fitted Boxes",
					  )

		# _plot_gradient(im, full_im_grad[i],
		# 			   ax=axs(3),
		# 			   title="Full Gradient",
		# 			  )

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



		############### Plots Gradients of the selected Features ###############
		if plot_sel_feats_grad:
			for feat_idx in np.where(pred_coef)[0]:
				mask = np.zeros_like(pred_coefs).astype(bool)
				mask[i, feat_idx] = True
				_grad = prop_back(feats, ims, mask)
				_plot_gradient(
					np.zeros_like(im), _grad[i],
					ax=None,#axs[0],
					title="Feat Gradient #{}".format(feat_idx),
					**kwargs
				)

				plt.show()
				plt.close()
		########################################################################

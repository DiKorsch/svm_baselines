import numpy as np
import logging

from chainer.cuda import to_cpu

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from scipy.ndimage.filters import gaussian_filter
from functools import partial

from svm_baselines.core.parts_and_bboxes import get_boxes
from svm_baselines.utils import prop_back
from svm_baselines.utils.image import prepare_back, grad_to_im
from svm_baselines.utils.clustering import cluster_gradient

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
	alpha=0.5, gamma=1.0, sigma=1,
	peak_size=None, K=None, init_from_maximas=False):

	# refactor this code! ########################################
	# its duplicate from core.parts_and_bboxes.get_parts #########
	grad = prepare_back(grad_to_im(grad, xp=xp))

	if sigma is None:
		grad = grad.squeeze()
	else:
		grad = gaussian_filter(grad, sigma=sigma).squeeze()

	grad = grad**gamma
	##############################################################

	if ax is None:
		_, ax = plt.subplots(figsize=(16, 9))

	#ax = imshow(im, ax=ax)
	ax = imshow(np.zeros_like(im), ax=ax)

	ax = imshow(grad, ax=ax,
				cmap=plt.cm.viridis, alpha=alpha)

	cmap = plt.cm.jet
	if K is not None and K > 0:

		# refactor this code! ########################################
		# its duplicate from core.parts_and_bboxes.get_parts #########
		thresh = np.abs(grad).mean()
		centers, labs = cluster_gradient(im, grad,
			K=K, thresh=thresh,
			init_from_maximas=init_from_maximas)
		boxes = get_boxes(centers, labs, optimize=True, grad=grad)
		##############################################################
		for c, box in boxes:
			ax.add_patch(Rectangle(
				*box, fill=False,
				color=cmap(c / len(boxes))))

		boxes = get_boxes(centers, labs, optimize=False)
		for c, box in boxes:
			ax.add_patch(Rectangle(
				*box, fill=False,
				color=cmap(c / len(boxes)),
				linestyle="--", alpha=0.5
			))

		ys, xs = centers[:, :2].T
		ax.scatter(xs, ys, marker="D", c=range(K), cmap=cmap)
		ax.imshow(labs, cmap=cmap, alpha=0.3)

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

	_plot_gradient = partial(plot_gradient, xp=model.xp, **kwargs)

	for i, (gt_coef, pred_coef) in enumerate(zip(gt_coefs, pred_coefs)):

		logging.debug(
			"predicted class: {}, GT class: {}".format(
			preds[i], labs[i]))
		fig, axs = plt.subplots(2, 2, figsize=(16, 9))

		im = prepare_back(ims[i])
		ax = imshow(im, ax=axs[0,0])
		ax.set_title("Original Image [predicted: {}, GT: {}]".format(
			preds[i], labs[i]))

		_plot_gradient(im, gt_im_grad[i],
					   ax=axs[np.unravel_index(1, axs.shape)],
					   title="GT Gradient",
					  )

		_plot_gradient(im, pred_im_grad[i],
					   ax=axs[np.unravel_index(2, axs.shape)],
					   title="Pred Gradient",
					  )

		_plot_gradient(im, full_im_grad[i],
					   ax=axs[np.unravel_index(3, axs.shape)],
					   title="Full Gradient",
					  )

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


def show_conv_maps(coefs, ims, labs, conv_maps, topk_preds, alpha=0.7, order=1):
	raise ValueError("DEPRECATED!")
	preds = topk_preds[:, -1]

	gt_coefs = coefs[to_cpu(labs)]

	topk_pred_coefs = [coefs[to_cpu(p)] for p in topk_preds.T]
	pred_coefs = topk_pred_coefs[-1]

	for i in range(len(ims)):
		im, conv, gt_c, pred_c = ims[i], conv_maps[i], gt_coefs[i], pred_coefs[i]
		pred_y, gt_y = preds[i], labs[i]

		_resize = partial(resize,
						  order=order,
						  output_shape=im.shape[1:],
						  mode="edge",
						  anti_aliasing=True,
						 )

		def show_conv(conv, ax, title, coef=None, best_n=10):
			if coef is None:
				_conv = np.abs(conv).sum(axis=0)
			else:
				best_coef = np.argsort(np.abs(coef))[-best_n:]
				_conv = conv * coef.reshape(-1, 1, 1)
				_conv = np.abs(_conv[best_coef]).sum(axis=0)

			_conv = _resize(normalize(_conv, axis=None))
			ax.imshow(_conv, alpha=alpha)
			ax.set_title(title)

		fig, ax = plt.subplots(1,3, figsize=(16, 9))
		for a in ax:
			a.imshow(prepare_back(im))

		show_conv(conv, ax=ax[0],
				  title="Full Conv Map")
		show_conv(conv, ax=ax[1],
				  title="Pred Conv Map (class {})".format(pred_y),
				  coef=pred_c)
		show_conv(conv, ax=ax[2],
				  title="GT Conv Map (class {})".format(gt_y),
				  coef=gt_c)

		continue
		c0 = (conv * gt_c.reshape(-1,1,1))[gt_c != 0]
		c1 = (conv * pred_c.reshape(-1,1,1))[pred_c != 0]
		for gt_conv, pred_conv in zip(c0, c1):
			fig, axs = plt.subplots(figsize=(16, 9))

			# gt_conv = _resize(normalize(gt_conv, axis=None))
			# axs[0].imshow(prepare_back(im))
			# axs[0].imshow(gt_conv, alpha=alpha)#, interpolation="bilinear")

			pred_conv = _resize(normalize(pred_conv, axis=None))
			axs.imshow(prepare_back(im))
			axs.imshow(pred_conv, alpha=alpha)#, interpolation="bilinear")

			plt.show()
			plt.close()

		break



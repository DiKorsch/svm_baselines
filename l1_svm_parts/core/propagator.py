import chainer.functions as F
import logging
import numpy as np

from chainer.backends import cuda
from contextlib import contextmanager

from l1_svm_parts.utils import topk_decision
from l1_svm_parts.utils import prepare_back
from l1_svm_parts.utils import saliency_to_im

class ImageGradient(object):
	"""
		Computes image gradients from given features w.r.t. image
		based on a model and an optional coeffiecient mask
	"""

	def __init__(self, model, feats, ims):
		super(ImageGradient, self).__init__()
		self.model = model
		self.feats = feats
		self.ims = ims

	def __call__(self, coefs=None):

		self.ims.grad = None
		self.model.cleargrads()

		if coefs is None:
			F.sum(self.feats).backward()
		else:
			F.sum(self.feats[np.where(coefs)]).backward()

		assert self.ims.grad is not None, "Backprop mode is off?"
		return self.ims.grad


class Propagator(object):

	def __init__(self, model, clf, scaler, topk, swap_channels=True):
		super(Propagator, self).__init__()
		self.model = model
		self.clf = clf
		self.topk = topk
		self.swap_channels = swap_channels
		self.scaler = scaler

		self.reset()


	@property
	def coefs(self):
		return self.clf.coef_

	def reset(self):
		self.ims = self.labs = self.topk_preds = None
		self.full_im_grad = self.pred_im_grad = None

	@contextmanager
	def __call__(self, feats, ims, labs):

		self.ims = ims
		self.labs = labs

		_feats = cuda.to_cpu(feats.array)
		labs = cuda.to_cpu(labs)

		self.topk_preds = topk_preds = self.evaluate_batch(_feats, labs)

		gt_coefs = self.coefs[labs]
		topk_pred_coefs = [self.coefs[p] for p in topk_preds.T]
		pred_coefs = topk_pred_coefs[-1]

		im_grad = ImageGradient(self.model, feats, ims)

		self.full_im_grad = im_grad()

		# gt_im_grad = im_grad(gt_coefs != 0)

		topk_pred_im_grad = [im_grad(p != 0) for p in topk_pred_coefs]
		self.pred_im_grad = topk_pred_im_grad[-1]

		yield self

		self.reset()

	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.i >= len(self.ims):
			raise StopIteration

		i = self.i
		self.i += 1
		im = self.prepare_back(self.ims[i])
		pred_grad = self.prepare_back(self.pred_im_grad[i], is_grad=True)
		full_grad = self.prepare_back(self.full_im_grad[i], is_grad=True)

		pred, gt = self.topk_preds[i, -1], self.labs[i]
		return i, im, (pred_grad, full_grad), (pred, gt)


	def prepare_back(self, im, is_grad=False):
		if is_grad:
			im = saliency_to_im(im, xp=self.model.xp)
		return prepare_back(im, swap_channels=self.swap_channels)

	def evaluate_batch(self, feats, gt):

		feats = self.scaler.transform(feats)
		topk_preds, topk_accu = topk_decision(feats, gt,
			clf=self.clf, topk=self.topk)

		logging.debug("Batch Accuracy: {:.4%} (Top1) | {:.4%} (Top{}) {: 3d} / {: 3d}".format(

			np.mean(topk_preds[:, -1] == gt),
			topk_accu,

			self.topk,
			np.sum(topk_preds[:, -1] == gt),
			len(feats)
		))

		return topk_preds

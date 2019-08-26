from chainer.cuda import to_cpu

from l1_svm_parts.utils import prop_back

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
		return prop_back(self.model, self.feats, self.ims, coefs)


class Propagator(object):

	def __init__(self, model, feats, ims, labs, coefs, topk_preds):
		super(Propagator, self).__init__()

		# preds = topk_preds[:, -1]

		self.gt_coefs = coefs[to_cpu(labs)]
		self.topk_pred_coefs = [coefs[to_cpu(p)] for p in topk_preds.T]
		self.pred_coefs = self.topk_pred_coefs[-1]

		im_grad = ImageGradient(model, feats, ims)

		self.full_im_grad = im_grad()

		self.gt_im_grad = im_grad(self.gt_coefs != 0)

		self.topk_pred_im_grad = [im_grad(p != 0) for p in self.topk_pred_coefs]
		self.pred_im_grad = self.topk_pred_im_grad[-1]



	def __iter__(self):
		return enumerate(zip(self.full_im_grad, self.pred_im_grad))

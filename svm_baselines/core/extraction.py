
from chainer.cuda import to_cpu

from functools import partial

from svm_baselines.core.parts_and_bboxes import get_parts
from svm_baselines.utils import prop_back
from svm_baselines.utils.image import prepare_back

def extract_parts(model, coefs, ims, labs, feats, topk_preds,
	**kwargs):

	normalize_grads = False
	preds = topk_preds[:, -1]

	gt_coefs = coefs[to_cpu(labs)]
	gt_im_grad = prop_back(model, feats, ims, gt_coefs != 0)

	topk_pred_coefs = [coefs[to_cpu(p)] for p in topk_preds.T]
	topk_pred_im_grad = [prop_back(model, feats, ims, p != 0) for p in topk_pred_coefs]

	pred_coefs = topk_pred_coefs[-1]
	pred_im_grad = topk_pred_im_grad[-1]

	full_im_grad = prop_back(model, feats, ims)
	_get_parts = partial(get_parts, xp=model.xp, **kwargs)

	for i, (gt_coef, pred_coef) in enumerate(zip(gt_coefs, pred_coefs)):
		im = prepare_back(ims[i])

		#gt_parts = _get_parts(im, pred_im_grad[i])
		pred_parts = _get_parts(im, pred_im_grad[i])
		full_parts = _get_parts(im, full_im_grad[i])
		yield pred_parts, full_parts

def parts_to_file(im_id, part_id, box, out):
	(x, y), w, h = box
	# TODO: remove the x-axis flipping
	#if "inception" in DATASET:
	#    x = 299 - (x + w) # flip x-axis

	print(im_id+1, int(part_id+1), *map(int, map(round, [x,y,w,h])),
		file=out
	)


import abc
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from cvdatasets.annotations import AnnotationType
from cvdatasets.utils import new_iterator

from l1_svm_parts.utils import IdentityScaler
from l1_svm_parts.utils import topk_decision

def evaluate_data(clf, data, subset, topk, scaler):

	X = scaler.transform(data.features[:, -1])
	y = data.labels
	pred = clf.decision_function(X).argmax(axis=1)
	logging.info("Accuracy on {} subset: {:.4%}".format(subset, (pred == y).mean()))

	topk_preds, topk_accu = topk_decision(X, y, clf=clf, topk=topk)
	logging.info("Top{}-Accuracy on {} subset: {:.4%}".format(topk, subset, topk_accu))


class Data(abc.ABC):

	@abc.abstractmethod
	def __init__(self):
		super(Data, self).__init__()

	@classmethod
	def new(self, opts, clf=None):

		annot_cls = AnnotationType.get(opts.dataset).value
		parts_key = "{}_{}".format(opts.dataset, "GLOBAL")

		logging.info("Loading {} annotations from \"{}\"".format(
			annot_cls, opts.data))
		logging.info("Using \"{}\"-parts".format(parts_key))

		annot = annot_cls(root_or_infofile=opts.data, parts=parts_key, feature_model=opts.model_type)

		data_info = annot.info
		model_info = data_info.MODELS[opts.model_type]
		part_info = data_info.PARTS[parts_key]

		n_classes = part_info.n_classes + opts.label_shift

		data = annot.new_dataset(subset=None)
		train_data, val_data = map(annot.new_dataset, ["train", "test"])

		if annot.labels.max() > n_classes:
			_, annot.labels = np.unique(annot.labels, return_inverse=True)

		logging.info("Minimum label value is \"{}\"".format(data.labels.min()))

		assert train_data.features is not None and val_data.features is not None, \
			"Features are not loaded!"

		assert val_data.features.ndim == 2 or val_data.features.shape[1] == 1, \
			"Only GLOBAL part features are supported here!"

		if opts.scale_features:
			logging.info("Scaling data on training set!")
			scaler = MinMaxScaler()
			scaler.fit(train_data.features[:, -1])
		else:
			scaler = IdentityScaler()

		it, n_batches = new_iterator(data,
			opts.n_jobs, opts.batch_size,
			repeat=False, shuffle=False
		)
		it = tqdm(enumerate(it), total=n_batches)

		if clf is not None:
			for _data, subset in [(train_data, "training"), (val_data, "validation")]:
				evaluate_data(clf, _data, subset, opts.topk, scaler)

		return scaler, data, it, model_info, n_classes



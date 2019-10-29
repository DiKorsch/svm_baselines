import abc
import logging

from os.path import join

from chainer_addons.links import PoolingType
from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType
from chainer_addons.utils.imgproc import _center_crop


class Model(abc.ABC):

	@abc.abstractmethod
	def __init__(self):
		super(Model, self).__init__()

	@classmethod
	def new(self, opts, model_info, n_classes):
		logging.info("Creating and loading model ...")

		model = ModelType.new(
			model_type=model_info.class_key,
			input_size=opts.input_size,
			pooling=PoolingType.G_AVG,
			aux_logits=False,
		)
		size = model.meta.input_size
		if not isinstance(size, tuple):
			size = (size, size)

		_prepare = PrepareType[opts.prepare_type](model)

		if opts.no_center_crop_on_val:
			prepare = lambda im: _prepare(im,
				swap_channels=opts.swap_channels,
				keep_ratio=False)
		else:
			prepare = lambda im: _center_crop(
					_prepare(im,
						size=size,
						swap_channels=opts.swap_channels), size)


		logging.info("Created {} model with \"{}\" prepare function. Image input size: {}"\
			.format(
				model.__class__.__name__,
				opts.prepare_type,
				size
			)
		)

		if opts.weights:
			weights = opts.weights
		else:
			weight_subdir, _, _ = model_info.weights.rpartition(".")
			weights = join(
				data_info.BASE_DIR,
				data_info.MODEL_DIR,
				model_info.folder,
				"ft_{}".format(opts.dataset),
				"rmsprop.g_avg_pooling",
				weight_subdir,
				"model_final.npz"
			)

		logging.info("Loading \"{}\" weights from \"{}\"".format(
			model_info.class_key, weights))
		model.load_for_inference(n_classes=n_classes, weights=weights)

		return model, prepare

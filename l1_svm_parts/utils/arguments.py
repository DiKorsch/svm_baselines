from cvargparse import GPUParser, ArgFactory, Arg

from chainer_addons.models import PrepareType

from cvdatasets.utils import read_info_file

import os
DEFAULT_INFO_FILE=os.environ.get("DATA", "/home/korsch/Data/info.yml")

info_file = read_info_file(DEFAULT_INFO_FILE)


from cluster_parts.utils import ThresholdType, ClusterInitType

def parse_args():
	parser = GPUParser(ArgFactory([
			Arg("data", default=DEFAULT_INFO_FILE),
			Arg("dataset", choices=info_file.DATASETS.keys()),

			Arg("trained_svm", type=str,
				help="Trained L1 SVM"),


			Arg("--n_jobs", type=int, default=2),

			Arg("--scale_features", action="store_true"),
			Arg("--visualize_coefs", action="store_true"),

			Arg("--topk", type=int, default=5),
			Arg("--extract", type=str, nargs=2,
				help="outputs to store extracted part locations"),

			Arg("--model_type", "-mt",
				default="resnet", choices=info_file.MODELS.keys(),
				help="type of the model"),

			Arg("--weights", "-w"),

			Arg("--input_size", type=int, default=0,
				help="overrides default input size of the model, if greater than 0"),

			Arg("--label_shift", type=int, default=1),

			Arg("--no_center_crop_on_val", action="store_true"),
			Arg("--swap_channels", action="store_true",
				help="preprocessing option: swap channels from RGB to BGR"),

			PrepareType.as_arg("prepare_type",
				help_text="type of image preprocessing"),

			ThresholdType.as_arg("thresh_type",
				help_text="type of gradient thresholding"),

			Arg("--K", type=int, default=4),

			Arg("--gamma", type=float, default=0.7,
				help="Gamma-Correction of the gradient intesities"),

			Arg("--sigma", type=float, default=5,
				help="Gaussian smoothing strength"),



		])\
	.batch_size()
	)

	parser.init_logger()
	return parser.parse_args()

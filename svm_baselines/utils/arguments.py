from cvargparse import GPUParser, ArgFactory, Arg

from chainer_addons.models import PrepareType

from nabirds.utils import read_info_file

DEFAULT_INFO_FILE="/home/korsch/Data/info.yml"

info_file = read_info_file(DEFAULT_INFO_FILE)

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
			Arg("--extract", action="store_true"),


			Arg("--model_type", "-mt",
				default="resnet", choices=info_file.MODELS.keys(),
				help="type of the model"),

			Arg("--input_size", type=int, default=0,
				help="overrides default input size of the model, if greater than 0"),

			PrepareType.as_arg("prepare_type",
				help_text="type of image preprocessing"),

			Arg("--subset", "-sub",
				default="train", choices=["train", "test"],
				help="Dataset subsets. (\"test\" is equivalent to \"val\")"),

			Arg("--K", type=int, default=4),
			Arg("--init_from_maximas", action="store_true"),

			Arg("--gamma", type=float, default=0.7,
				help="Gamma-Correction of the gradient intesities"),
			Arg("--sigma", type=float, default=5,
				help="Gaussian smoothing strength"),



		])\
	.batch_size()
	)

	parser.init_logger()
	return parser.parse_args()

from chainer_addons.models import PrepareType

from cvargparse import BaseParser, Arg

from nabirds.utils import read_info_file

DEFAULT_INFO_FILE="/home/korsch/Data/info.yml"

info_file = read_info_file(DEFAULT_INFO_FILE)

def parse_args():
	parser = BaseParser([
		Arg("data", default=DEFAULT_INFO_FILE),

		Arg("dataset", choices=info_file.DATASETS.keys()),
		Arg("parts", choices=info_file.PARTS.keys()),

		Arg("--model_type", "-mt",
			default="resnet", choices=info_file.MODELS.keys(),
			help="type of the model"),

		Arg("--classifier", "-clf", default="svm",
			choices=["svm", "logreg"]),

		Arg("--C", type=float, default=0.1,
			help="classifier regularization parameter"),

		Arg("--max_iter", type=int, default=200,
			help="maximum number of training iterations"),

		Arg("--show_feature_stats", action="store_true"),

		Arg("--sparse", action="store_true",
			help="Use LinearSVC with L1 regularization for sparse feature selection"),
		Arg("--scale_features", action="store_true"),
		Arg("--no_dump", action="store_true"),


		Arg("--output", default=".out"),

	])

	parser.init_logger()

	return parser.parse_args()

from chainer_addons.models import PrepareType

from cvargparse import BaseParser, Arg

from cvdatasets.utils import read_info_file

DEFAULT_INFO_FILE="/home/korsch/Data/info.yml"

info_file = read_info_file(DEFAULT_INFO_FILE)

def train_args():
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

		Arg("--load", type=str,
			help="initial weights and biases to use as initialization"),

		Arg("--show_feature_stats", action="store_true"),
		Arg("--shuffle_part_features", action="store_true"),

		Arg("--sparse", action="store_true",
			help="Use LinearSVC with L1 regularization for sparse feature selection"),
		Arg("--scale_features", action="store_true"),
		Arg("--eval_local_parts", action="store_true"),
		Arg("--no_dump", action="store_true"),


		Arg("--output", default=".out"),

	])

	parser.init_logger()

	return parser.parse_args()


def predict_args():
	parser = BaseParser([
		Arg("data", default=DEFAULT_INFO_FILE),

		Arg("dataset", choices=info_file.DATASETS.keys()),
		Arg("parts", choices=info_file.PARTS.keys()),
		Arg("weights"),

		Arg("--model_type", "-mt",
			default="resnet", choices=info_file.MODELS.keys(),
			help="type of the model"),

		Arg("--subset", choices=[
			"train", "test"
		]),

		Arg("--evaluate", action="store_true"),
		Arg("--scale_features", action="store_true"),

		Arg("--no_export", action="store_true"),

		Arg("--label_shift", type=int, default=0),

		Arg("--output", default="predictions.csv"),

	])

	parser.init_logger()

	return parser.parse_args()

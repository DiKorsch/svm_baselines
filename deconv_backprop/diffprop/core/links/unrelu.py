from .base import BaseWrapper

#
# DOES NOT MAKE SENSE HERE!
#



class UnReLu(BaseWrapper):
	# wraps a python function, hence we cannot
	# check for the correct instance
	wraps = None

	def __init__(self, relu):
		super(UnReLu, self).__init__(relu)

	def forward(self, gy):
		import pdb; pdb.set_trace()

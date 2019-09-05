import chainer

from functools import wraps, partial


class BaseWrapper(chainer.Chain):

	def __init__(self, layer):
		super(BaseWrapper, self).__init__()

		if hasattr(self, "wraps") and self.wraps is not None:
			assert isinstance(layer, self.wraps), \
				f"The layer should an instance of \"{self.wraps}\", but was \"{type(layer)}\""

		assert callable(layer), \
			"The wrapped layer should be callable!"

		if hasattr(layer, "forward"):
			layer.forward = self._wrap(layer.forward)

		elif isinstance(layer, partial):
			_, _, (func, args, kwargs, namespace) = layer.__reduce__()
			state = (self._wrap(layer.func), args, kwargs, namespace)
			layer.__setstate__(state)

		else:
			raise ValueError(f"Could not wrap the callable: {layer}")

		self.reset()

	def _wrap(self, fwd_func):

		@wraps(fwd_func)
		def inner(*inputs, **kwargs):
			self.inputs = inputs
			self.kwargs = kwargs
			self.outputs = fwd_func(*inputs, **kwargs)
			return self.outputs

		return inner

	def __call__(self, *args, **kwargs):
		res = super(BaseWrapper, self).__call__(*args, **kwargs)
		self.reset()
		return res

	def reset(self):
		self.inputs = self.outputs = None






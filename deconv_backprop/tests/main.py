#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import sys
from os.path import abspath, dirname, join

root_dir = abspath(join(abspath(dirname(__file__)), ".."))
sys.path.insert(0, root_dir)

import chainer
import unittest

from tests.layers import TestSimpleLayers, TestSimpleLayerCombinations
from tests.models import BlockComparison, ModelComparison


with chainer.using_config("train", False):
	unittest.main()

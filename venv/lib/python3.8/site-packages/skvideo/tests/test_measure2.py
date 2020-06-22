import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

def test_measure_Li3DDCT():
    vidpaths = skvideo.datasets.fullreferencepair()
    vidpaths = skvideo.datasets.bigbuckbunny()
    dis = skvideo.io.vread(vidpaths, as_grey=True)
    dis = dis[:10, :200, :200]

    #dis = skvideo.io.vread(vidpaths[1], as_grey=True)[:12]
    Li_array = skvideo.measure.Li3DDCT_features(dis)

    print(Li_array)


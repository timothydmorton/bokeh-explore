import numpy as np
import param
import holoviews as hv
from holoviews.operation import histogram

class QAhistogram(histogram):

    def _process(self, *args, **kwargs):
        hist = super(QAhistogram, self)._process(*args, **kwargs)

        heights = hist.dimension_values('Frequency')
        
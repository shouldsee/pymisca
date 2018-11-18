import matplotlib.ticker as mticker
import matplotlib.axes
import numpy as np
def matshow(self, Z, **kwargs):
    """
    Plot a matrix or array as an image.

    The matrix will be shown the way it would be printed, with the first
    row at the top.  Row and column numbering is zero-based.

    Parameters
    ----------
    Z : array_like shape (n, m)
        The matrix to be displayed.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.axes.Axes.imshow` arguments
        Sets `origin` to 'upper', 'interpolation' to 'nearest' and
        'aspect' to equal.

    See also
    --------
    imshow : plot an image

    """
    Z = np.asanyarray(Z)
    nr, nc = Z.shape[:2]
    kw = {'origin': 'upper',
          'interpolation': 'nearest',
          'aspect': 'equal'}          # (already the imshow default)
    kw.update(kwargs)
    im = self.imshow(Z, **kw)
    self.title.set_y(1.05)
    self.xaxis.tick_top()
    self.xaxis.set_ticks_position('both')
    self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                             steps=[1, 2, 5, 10],
                                             integer=True))
    self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                             steps=[1, 2, 5, 10],
                                             integer=True))
    return im
matplotlib.axes.Axes.matshow = matshow
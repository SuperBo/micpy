from __future__ import division, absolute_import, print_function

from .numeric import asarray as asmicarray


def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    .. note:: Previous to NumPy 1.13.0, neither ``axis < -a.ndim - 1`` nor
       ``axis > a.ndim`` raised errors or put the new axis where documented.
       Those axis values are now deprecated and will raise an AxisError in the
       future.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray
        Output array. The number of dimensions is one greater than that of
        the input array.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    doc.indexing, atleast_1d, atleast_2d, atleast_3d

    Examples
    --------
    >>> x = mp.array([1,2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[mp.newaxis,:]`` or ``x[mp.newaxis]``:

    >>> y = mp.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    >>> y = mp.expand_dims(x, axis=1)  # Equivalent to x[:,newaxis]
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    Note that some examples may use ``None`` instead of ``mp.newaxis``.  These
    are the same objects:

    >>> mp.newaxis is None
    True

    """
    a = asmicarray(a)
    shape = a.shape
    if axis > a.ndim or axis < -a.ndim - 1:
        # 2017-05-17, 1.13.0
        warnings.warn("Both axis > a.ndim and axis < -a.ndim - 1 are "
                      "deprecated and will raise an AxisError in the future.",
                      DeprecationWarning, stacklevel=2)
    # When the deprecation period expires, delete this if block,
    if axis < 0:
        axis = axis + a.ndim + 1
    # and uncomment the following line.
    # axis = normalize_axis_index(axis, a.ndim + 1)
    return a.reshape(shape[:axis] + (1,) + shape[axis:])

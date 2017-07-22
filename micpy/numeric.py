from numpy.core.multiarray import normalize_axis_index, array
from numpy.core.numeric import normalize_axis_tuple
from numpy import AxisError
from . import multiarray
from .multiarray import empty, empty_like
from .multiarray import array as micarray


def _wrapfunc(obj, method, *args, **kwds):
    try:
        return getattr(obj, method)(*args, **kwds)

    # An AttributeError occurs if the object does not have
    # such a method in its class.

    # A TypeError occurs if the object does have such a method
    # in its class, but its signature is not identical to that
    # of NumPy's. This situation has occurred in the case of
    # a downstream library like 'pandas'.
    except (AttributeError, TypeError):
        return None


def full(shape, fill_value, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, `None`, means
         `np.array(fill_value).dtype`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    full_like : Fill an array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> mp.full((2, 2), np.inf)
    micarray([[ inf,  inf],
           [ inf,  inf]])
    >>> mp.full((2, 2), 10)
    micarray([[10, 10],
           [10, 10]])

    """
    if dtype is None:
        dtype = array(fill_value).dtype
    a = empty(shape, dtype, order)
    multiarray.copyto(a, fill_value, casting='unsafe')
    return a


def full_like(a, fill_value, dtype=None, order='K', subok=True):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    full : Fill a new array.

    Examples
    --------
    >>> x = mp.arange(6, dtype=np.int)
    >>> mp.full_like(x, 1)
    micarray([1, 1, 1, 1, 1, 1])
    >>> mp.full_like(x, 0.1)
    micarray([0, 0, 0, 0, 0, 0])
    >>> mp.full_like(x, 0.1, dtype=np.double)
    micarray([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
    >>> mp.full_like(x, np.nan, dtype=np.double)
    micarray([ nan,  nan,  nan,  nan,  nan,  nan])

    >>> y = mp.arange(6, dtype=np.double)
    >>> mp.full_like(y, 0.1)
    micarray([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])

    """
    res = empty_like(a, dtype=dtype, order=order, subok=subok)
    multiarray.copyto(res, fill_value, casting='unsafe')
    return res


def asarray(a, dtype=None, order=None):
    """Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> mp.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = mp.array([1, 2])
    >>> mp.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = mp.array([1, 2], dtype=np.float32)
    >>> mp.asarray(a, dtype=np.float32) is a
    True
    >>> mp.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.matrix, np.ndarray)
    True
    >>> a = mp.matrix([[1, 2]])
    >>> mp.asarray(a) is a
    False
    >>> mp.asanyarray(a) is a
    True

    """
    return micarray(a, dtype, copy=False, order=order)


def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    Parameters
    ----------
    a : micpy.ndarray
        Input array.
    axis : int
        The axis to roll backwards.  The positions of the other axes do not
        change relative to one another.
    start : int, optional
        The axis is rolled until it lies before this position.  The default,
        0, results in a "complete" roll.

    Returns
    -------
    res : micpy.ndarray
        a view of `a` is always returned.

    See Also
    --------
    moveaxis : Move array axes to new positions.
    roll : Roll the elements of an array by a number of positions along a
        given axis.

    Examples
    --------
    >>> a = mp.ones((3,4,5,6))
    >>> mp.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> mp.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> mp.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        return a.view()
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)


def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    .. versionadded::1.11.0

    Parameters
    ----------
    a : np.ndarray
        The array whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : np.ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    transpose: Permute the dimensions of an array.
    swapaxes: Interchange two axes of an array.

    Examples
    --------

    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    These all achieve the same result:

    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    try:
        # allow duck-array types if they define transpose
        transpose = a.transpose
    except AttributeError:
        a = asarray(a)
        transpose = a.transpose

    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = transpose(order)
    return result


def argmax(a, axis=None, out=None):
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    ndarray.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> a = mp.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mp.argmax(a)
    5
    >>> mp.argmax(a, axis=0)
    array([1, 1, 1])
    >>> mp.argmax(a, axis=1)
    array([2, 2])

    >>> b = mp.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> mp.argmax(b) # Only the first occurrence is returned.
    1

    """
    return _wrapfunc(a, 'argmax', axis=axis, out=out)


def argmin(a, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    ndarray.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> a = mp.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mp.argmin(a)
    0
    >>> mp.argmin(a, axis=0)
    array([0, 0, 0])
    >>> mp.argmin(a, axis=1)
    array([0, 0])

    >>> b = mp.arange(6)
    >>> b[4] = 0
    >>> b
    array([0, 1, 2, 3, 0, 5])
    >>> mp.argmin(b) # Only the first occurrence is returned.
    0

    """
    return _wrapfunc(a, 'argmin', axis=axis, out=out)

from __future__ import division, print_function

import os
import re
import struct
import sys
import textwrap

Zero = "PyUFunc_Zero"
One = "PyUFunc_One"
None_ = "PyUFunc_None"
AllOnes = "PyUFunc_MinusOne"
ReorderableNone = "PyUFunc_ReorderableNone"

# Sentinel value to specify using the full type description in the
# function name
class FullTypeDescr(object):
    pass

class FuncNameSuffix(object):
    """Stores the suffix to append when generating functions names.
    """
    def __init__(self, suffix):
        self.suffix = suffix

class TypeDescription(object):
    """Type signature for a ufunc.

    Attributes
    ----------
    type : str
        Character representing the nominal type.
    func_data : str or None or FullTypeDescr or FuncNameSuffix, optional
        The string representing the expression to insert into the data
        array, if any.
    in_ : str or None, optional
        The typecode(s) of the inputs.
    out : str or None, optional
        The typecode(s) of the outputs.
    astype : dict or None, optional
        If astype['x'] is 'y', uses PyUFunc_x_x_As_y_y/PyUFunc_xx_x_As_yy_y
        instead of PyUFunc_x_x/PyUFunc_xx_x.
    simd: list
        Available SIMD ufunc loops, dispatched at runtime in specified order
        Currently only supported for simples types (see make_arrays)
    """
    def __init__(self, type, f=None, in_=None, out=None, astype=None, simd=None):
        self.type = type
        self.func_data = f
        if astype is None:
            astype = {}
        self.astype_dict = astype
        if in_ is not None:
            in_ = in_.replace('P', type)
        self.in_ = in_
        if out is not None:
            out = out.replace('P', type)
        self.out = out
        self.simd = simd

    def finish_signature(self, nin, nout):
        if self.in_ is None:
            self.in_ = self.type * nin
        assert len(self.in_) == nin
        if self.out is None:
            self.out = self.type * nout
        assert len(self.out) == nout
        self.astype = self.astype_dict.get(self.type, None)

_fdata_map = dict(e='mpy_%sf', f='mpy_%sf', d='mpy_%s', g='mpy_%sl',
                  F='nc_%sf', D='nc_%s', G='nc_%sl')
def build_func_data(types, f):
    func_data = []
    for t in types:
        d = _fdata_map.get(t, '%s') % (f,)
        func_data.append(d)
    return func_data

def TD(types, f=None, astype=None, in_=None, out=None, simd=None):
    if f is not None:
        if isinstance(f, str):
            func_data = build_func_data(types, f)
        else:
            assert len(f) == len(types)
            func_data = f
    else:
        func_data = (None,) * len(types)
    if isinstance(in_, str):
        in_ = (in_,) * len(types)
    elif in_ is None:
        in_ = (None,) * len(types)
    if isinstance(out, str):
        out = (out,) * len(types)
    elif out is None:
        out = (None,) * len(types)
    tds = []
    for t, fd, i, o in zip(types, func_data, in_, out):
        # [(simd-name, list of types)]
        if simd is not None:
            simdt = [k for k, v in simd if t in v]
        else:
            simdt = []
        tds.append(TypeDescription(t, f=fd, in_=i, out=o, astype=astype, simd=simdt))
    return tds

class Ufunc(object):
    """Description of a ufunc.

    Attributes
    ----------
    nin : number of input arguments
    nout : number of output arguments
    identity : identity element for a two-argument function
    docstring : docstring for the ufunc
    type_descriptions : list of TypeDescription objects
    """
    def __init__(self, nin, nout, identity, docstring, typereso,
                 *type_descriptions):
        self.nin = nin
        self.nout = nout
        if identity is None:
            identity = None_
        self.identity = identity
        self.docstring = docstring
        self.typereso = typereso
        self.type_descriptions = []
        for td in type_descriptions:
            self.type_descriptions.extend(td)
        for td in self.type_descriptions:
            td.finish_signature(self.nin, self.nout)

# String-handling utilities to avoid locale-dependence.

import string
if sys.version_info[0] < 3:
    UPPER_TABLE = string.maketrans(string.ascii_lowercase,
                                   string.ascii_uppercase)
else:
    UPPER_TABLE = bytes.maketrans(bytes(string.ascii_lowercase, "ascii"),
                                  bytes(string.ascii_uppercase, "ascii"))

def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy.lib.utils import english_upper
    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_upper(s)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


#each entry in defdict is a Ufunc object.

#name: [string of chars for which it is defined,
#       string of characters using func interface,
#       tuple of strings giving funcs for data,
#       (in, out), or (instr, outstr) giving the signature as character codes,
#       identity,
#       docstring,
#       output specification (optional)
#       ]

chartoname = {'?': 'bool',
              'b': 'byte',
              'B': 'ubyte',
              'h': 'short',
              'H': 'ushort',
              'i': 'int',
              'I': 'uint',
              'l': 'long',
              'L': 'ulong',
              'q': 'longlong',
              'Q': 'ulonglong',
              'e': 'half',
              'f': 'float',
              'd': 'double',
              'g': 'longdouble',
              'F': 'cfloat',
              'D': 'cdouble',
              'G': 'clongdouble',
              'M': 'datetime',
              'm': 'timedelta',
              'O': 'OBJECT',
              # '.' is like 'O', but calls a method of the object instead
              # of a function
              'P': 'OBJECT',
              }

all = '?bBhHiIlLqQefdgFDGOMm'
O = 'O'
P = 'P'
ints = 'bBhHiIlLqQ'
times = 'Mm'
timedeltaonly = 'm'
intsO = ints + O
bints = '?' + ints
bintsO = bints + O
flts = 'efdg'
fltsO = flts + O
fltsP = flts + P
cmplx = 'FDG'
cmplxO = cmplx + O
cmplxP = cmplx + P
inexact = flts + cmplx
inexactvec = 'fd'
inexactnoevec = 'g' + cmplx
noint = inexact+O
nointP = inexact+P
allP = bints+times+flts+cmplxP
nobool = all[1:]
noobj = all[:-3]+all[-2:]
nobool_or_obj = all[1:-3]+all[-2:]
nobool_or_datetime = all[1:-2]+all[-1:]
intflt = ints+flts
intfltcmplx = ints+flts+cmplx
nocmplx = bints+times+flts
nocmplxO = nocmplx+O
nocmplxP = nocmplx+P
notimes_or_obj = bints + inexact
nodatetime_or_obj = bints + inexact

# Find which code corresponds to int64.
int64 = ''
uint64 = ''
for code in 'bhilq':
    if struct.calcsize(code) == 8:
        int64 = code
        uint64 = english_upper(code)
        break

# This dictionary describes all the ufunc implementations, generating
# all the function names and their corresponding ufunc signatures.  TD is
# an object which expands a list of character codes into an array of
# TypeDescriptions.
defdict = {
'add':
    Ufunc(2, 1, Zero,
          '',
          'PyUFunc_AdditionTypeResolver',
          TD(notimes_or_obj, simd=[('imci', ints)]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'mM', 'M'),
          ],
          ),
'subtract':
    Ufunc(2, 1, None, # Zero is only a unit to the right, not the left
          '',
          'PyUFunc_SubtractionTypeResolver',
          TD(notimes_or_obj, simd=[('imci', ints)]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'MM', 'm'),
          ],
          ),
'multiply':
    Ufunc(2, 1, One,
          '',
          'PyUFunc_MultiplicationTypeResolver',
          TD(notimes_or_obj, simd=[('imci', ints)]),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'qm', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'dm', 'm'),
          ],
          ),
'divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          '',
          'PyUFunc_MixedDivisionTypeResolver',
          TD(intfltcmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          ),
'floor_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          '',
          'PyUFunc_DivisionTypeResolver',
          TD(intfltcmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           #TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          ),
'true_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          '',
          'PyUFunc_DivisionTypeResolver',
          TD('bBhH', out='d'),
          TD('iIlLqQ', out='d'),
          TD(flts+cmplx),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'mm', 'd'),
          ],
          ),
'conjugate':
    Ufunc(1, 1, None,
          '',
          None,
          TD(ints+flts+cmplx, simd=[('imci', ints)]),
          ),
'fmod':
    Ufunc(2, 1, None,
          '',
          None,
          TD(ints),
          TD(flts, f='fmod', astype={'e':'f'}),
          ),
'square':
    Ufunc(1, 1, None,
          '',
          None,
          TD(ints+inexact, simd=[('imci', ints)]),
          ),
'reciprocal':
    Ufunc(1, 1, None,
          '',
          None,
          TD(ints+inexact, simd=[('imci', ints)]),
          ),
# This is no longer used as numpy.ones_like, however it is
# still used by some internal calls.
'_ones_like':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_OnesLikeTypeResolver',
          TD(noobj),
          ),
'power':
    Ufunc(2, 1, None,
          '',
          None,
          TD(ints),
          TD(inexact, f='pow', astype={'e':'f'}),
          ),
'float_power':
    Ufunc(2, 1, None,
          '',
          None,
          TD('dgDG', f='pow'),
          ),
'absolute':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_AbsoluteTypeResolver',
          TD(bints+flts+timedeltaonly),
          TD(cmplx, out=('f', 'd', 'g')),
          ),
# '_arg':
#     Ufunc(1, 1, None,
#           '',
#           None,
#           TD(cmplx, out=('f', 'd', 'g')),
#           ),
'negative':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_NegativeTypeResolver',
          TD(bints+flts+timedeltaonly, simd=[('imci', ints)]),
          TD(cmplx, f='neg'),
          ),
'positive':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_SimpleUnaryOperationTypeResolver',
          TD(ints+flts+timedeltaonly),
          TD(cmplx, f='pos'),
          ),
'sign':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_SimpleUnaryOperationTypeResolver',
          TD(intfltcmplx),
          ),
'greater':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'greater_equal':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'less':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'less_equal':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'equal':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'not_equal':
    Ufunc(2, 1, None,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(noobj, out='?', simd=[('imci', ints)]),
          ),
'logical_and':
    Ufunc(2, 1, One,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('imci', ints)]),
          ),
'logical_not':
    Ufunc(1, 1, None,
          '',
          None,
          TD(nodatetime_or_obj, out='?', simd=[('imci', ints)]),
          ),
'logical_or':
    Ufunc(2, 1, Zero,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('imci', ints)]),
          ),
'logical_xor':
    Ufunc(2, 1, Zero,
          '',
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?'),
          ),
'maximum':
    Ufunc(2, 1, ReorderableNone,
          '',
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          ),
'minimum':
    Ufunc(2, 1, ReorderableNone,
          '',
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          ),
'fmax':
    Ufunc(2, 1, ReorderableNone,
          '',
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          ),
'fmin':
    Ufunc(2, 1, ReorderableNone,
          '',
          'PyUFunc_SimpleBinaryOperationTypeResolver',
          TD(noobj),
          ),
'logaddexp':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts, f="logaddexp", astype={'e':'f'})
          ),
'logaddexp2':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts, f="logaddexp2", astype={'e':'f'})
          ),
'bitwise_and':
    Ufunc(2, 1, AllOnes,
          '',
          None,
          TD(bints, simd=[('imci', ints)]),
          ),
'bitwise_or':
    Ufunc(2, 1, Zero,
          '',
          None,
          TD(bints, simd=[('imci', ints)]),
          ),
'bitwise_xor':
    Ufunc(2, 1, Zero,
          '',
          None,
          TD(bints, simd=[('imci', ints)]),
          ),
'invert':
    Ufunc(1, 1, None,
          '',
          None,
          TD(bints, simd=[('imci', ints)]),
          ),
'left_shift':
    Ufunc(2, 1, None,
          '',
          None,
          TD(ints, simd=[('imci', ints)]),
          ),
'right_shift':
    Ufunc(2, 1, None,
          '',
          None,
          TD(ints, simd=[('imci', ints)]),
          ),
'heaviside':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts, f='heaviside', astype={'e':'f'}),
          ),
'degrees':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='degrees', astype={'e':'f'}),
          ),
'rad2deg':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='rad2deg', astype={'e':'f'}),
          ),
'radians':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='radians', astype={'e':'f'}),
          ),
'deg2rad':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='deg2rad', astype={'e':'f'}),
          ),
'arccos':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='acos', astype={'e':'f'}),
          ),
'arccosh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='acosh', astype={'e':'f'}),
          ),
'arcsin':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='asin', astype={'e':'f'}),
          ),
'arcsinh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='asinh', astype={'e':'f'}),
          ),
'arctan':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='atan', astype={'e':'f'}),
          ),
'arctanh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='atanh', astype={'e':'f'}),
          ),
'cos':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='cos', astype={'e':'f'}),
          ),
'sin':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='sin', astype={'e':'f'}),
          ),
'tan':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='tan', astype={'e':'f'}),
          ),
'cosh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='cosh', astype={'e':'f'}),
          ),
'sinh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='sinh', astype={'e':'f'}),
          ),
'tanh':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='tanh', astype={'e':'f'}),
          ),
'exp':
    Ufunc(1, 1, None,
          '',
          None,
          TD('e', f='exp', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexactnoevec, f='exp'),
          ),
'exp2':
    Ufunc(1, 1, None,
          '',
          None,
          TD('e', f='exp2', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexactnoevec, f='exp2'),
          ),
'expm1':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='expm1', astype={'e':'f'}),
          ),
'log':
    Ufunc(1, 1, None,
          '',
          None,
          TD('e', f='log', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexactnoevec, f='log'),
          ),
'log2':
    Ufunc(1, 1, None,
          '',
          None,
          TD('e', f='log2', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexactnoevec, f='log2'),
          ),
'log10':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='log10', astype={'e':'f'}),
          ),
'log1p':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='log1p', astype={'e':'f'}),
          ),
'sqrt':
    Ufunc(1, 1, None,
          '',
          None,
          TD('e', f='sqrt', astype={'e':'f'}),
          TD(inexactvec),
          TD(inexactnoevec, f='sqrt'),
          ),
'cbrt':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='cbrt', astype={'e':'f'}),
          ),
'ceil':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='ceil', astype={'e':'f'}),
          ),
'trunc':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='trunc', astype={'e':'f'}),
          ),
'fabs':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='fabs', astype={'e':'f'}),
       ),
'floor':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, f='floor', astype={'e':'f'}),
          ),
'rint':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, f='rint', astype={'e':'f'}),
          ),
'arctan2':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts, f='atan2', astype={'e':'f'}),
          ),
'remainder':
    Ufunc(2, 1, None,
          '',
          None,
          TD(intflt),
          ),
'divmod':
    Ufunc(2, 2, None,
          '',
          None,
          TD(intflt),
          ),
'hypot':
    Ufunc(2, 1, Zero,
          '',
          None,
          TD(flts, f='hypot', astype={'e':'f'}),
          ),
'isnan':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, out='?'),
          ),
'isnat':
    Ufunc(1, 1, None,
          '',
          'PyUFunc_IsNaTTypeResolver',
          TD(times, out='?'),
          ),
'isinf':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, out='?'),
          ),
'isfinite':
    Ufunc(1, 1, None,
          '',
          None,
          TD(inexact, out='?'),
          ),
'signbit':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts, out='?'),
          ),
'copysign':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts),
          ),
'nextafter':
    Ufunc(2, 1, None,
          '',
          None,
          TD(flts),
          ),
'spacing':
    Ufunc(1, 1, None,
          '',
          None,
          TD(flts),
          ),
'modf':
    Ufunc(1, 2, None,
          '',
          None,
          TD(flts),
          ),
'ldexp' :
    Ufunc(2, 1, None,
          '',
          None,
          [TypeDescription('e', None, 'ei', 'e'),
          TypeDescription('f', None, 'fi', 'f'),
          TypeDescription('e', FuncNameSuffix('long'), 'el', 'e'),
          TypeDescription('f', FuncNameSuffix('long'), 'fl', 'f'),
          TypeDescription('d', None, 'di', 'd'),
          TypeDescription('d', FuncNameSuffix('long'), 'dl', 'd'),
          TypeDescription('g', None, 'gi', 'g'),
          TypeDescription('g', FuncNameSuffix('long'), 'gl', 'g'),
          ],
          ),
'frexp' :
    Ufunc(1, 2, None,
          '',
          None,
          [TypeDescription('e', None, 'e', 'ei'),
          TypeDescription('f', None, 'f', 'fi'),
          TypeDescription('d', None, 'd', 'di'),
          TypeDescription('g', None, 'g', 'gi'),
          ],
          )
}

if sys.version_info[0] >= 3:
    # Will be aliased to true_divide in umathmodule.c.src:InitOtherOperators
    del defdict['divide']

def indent(st, spaces):
    indention = ' '*spaces
    indented = indention + st.replace('\n', '\n'+indention)
    # trim off any trailing spaces
    indented = re.sub(r' +$', r'', indented)
    return indented

chartotype1 = {'e': 'e_e',
               'f': 'f_f',
               'd': 'd_d',
               'g': 'g_g',
               'F': 'F_F',
               'D': 'D_D',
               'G': 'G_G',
               'O': 'O_O',
               'P': 'O_O_method'}

chartotype2 = {'e': 'ee_e',
               'f': 'ff_f',
               'd': 'dd_d',
               'g': 'gg_g',
               'F': 'FF_F',
               'D': 'DD_D',
               'G': 'GG_G',
               'O': 'OO_O',
               'P': 'OO_O_method'}
#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function.

def make_arrays(funcdict):
    # functions array contains an entry for every type implemented NULL
    # should be placed where PyUfunc_ style function will be filled in
    # later
    # NOTE: PyMUFunc_ is placed directly instead of fill in later to
    # avoid compiler stripping unused MIC function
    code1list = []
    code2list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        funclist = []
        datalist = []
        siglist = []
        k = 0
        sub = 0

        if uf.nin > 1:
            assert uf.nin == 2
            thedict = chartotype2  # two inputs and one output
        else:
            thedict = chartotype1  # one input and one output

        for t in uf.type_descriptions:
            if (t.func_data not in (None, FullTypeDescr) and
                    not isinstance(t.func_data, FuncNameSuffix)):
                #funclist.append('NULL')
                astype = ''
                if not t.astype is None:
                    astype = '_As_%s' % thedict[t.astype]
                # astr = ('%s_functions[%d] = PyMUFunc_%s%s;' %
                #            (name, k, thedict[t.type], astype))
                # code2list.append(astr)
                funclist.append('PyMUFunc_%s%s'%(thedict[t.type], astype))
                if t.type == 'O':
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                elif t.type == 'P':
                    datalist.append('(void *)"%s"' % t.func_data)
                else:
                    #astr = ('%s_data[%d] = (void *) %s;' %
                    #           (name, k, t.func_data))
                    #code2list.append(astr)
                    #datalist.append('(void *)NULL')
                    datalist.append('(void *)%s' % t.func_data)
                sub += 1
            elif t.func_data is FullTypeDescr:
                tname = english_upper(chartoname[t.type])
                datalist.append('(void *)NULL')
                funclist.append(
                        '%s_%s_%s_%s' % (tname, t.in_, t.out, name))
            elif isinstance(t.func_data, FuncNameSuffix):
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append(
                        '%s_%s_%s' % (tname, name, t.func_data.suffix))
            else:
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append('%s_%s' % (tname, name))
                if t.simd is not None:
                    for vt in t.simd:
                        code2list.append("""\
#ifdef HAVE_ATTRIBUTE_TARGET_{ISA}
if (MPY_CPU_SUPPORTS_{ISA}) {{
    {fname}_functions[{idx}] = {type}_{fname}_{isa};
}}
#endif
""".format(ISA=vt.upper(), isa=vt, fname=name, type=tname, idx=k))

            for x in t.in_ + t.out:
                siglist.append('NPY_%s' % (english_upper(chartoname[x]),))

            k += 1

        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = {%s};"
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = {%s};"
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = {%s};"
                         % (name, signames))
    return "\n".join(code1list), "\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        mlist = []
        docstring = textwrap.dedent(uf.docstring).strip()
        if sys.version_info[0] < 3:
            docstring = docstring.encode('string-escape')
            docstring = docstring.replace(r'"', r'\"')
        else:
            docstring = docstring.encode('unicode-escape').decode('ascii')
            docstring = docstring.replace(r'"', r'\"')
            # XXX: I don't understand why the following replace is not
            # necessary in the python 2 case.
            docstring = docstring.replace(r"'", r"\'")
        # Split the docstring because some compilers (like MS) do not like big
        # string literal in C code. We split at endlines because textwrap.wrap
        # do not play well with \n
        docstring = '\\n\"\"'.join(docstring.split(r"\n"))
        mlist.append(\
r"""f = PyMUFunc_FromFuncAndData(%s_functions, %s_data, %s_signatures, %d,
                                %d, %d, %s, "%s",
                                "%s", 0);""" % (name, name, name,
                                                len(uf.type_descriptions),
                                                uf.nin, uf.nout,
                                                uf.identity,
                                                name, docstring))
        if uf.typereso is not None:
            mlist.append(
                r"((PyUFuncObject *)f)->type_resolver = &%s;" % uf.typereso)
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);""" % name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))
    return '\n'.join(code3list)


def make_code(funcdict, filename):
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2, 4)
    code3 = indent(code3, 4)
    code = r"""

/** Warning this file is autogenerated!!!

    Please make changes to the code generator program (%s)
**/

%s

static void
InitOperators(PyObject *dictionary) {
    PyObject *f;

%s
%s
}
""" % (filename, code1, code2, code3)
    return code


if __name__ == "__main__":
    filename = __file__
    fid = open('__umath_generated.c', 'w')
    code = make_code(defdict, filename)
    fid.write(code)
    fid.close()

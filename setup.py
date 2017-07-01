#!/usr/bin/env python

from __future__ import absolute_import, print_function
import os.path
from os.path import join
from distutils.dep_util import newer
import sys

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

# This is a bit hackish: we are setting a global variable so that the main
# micpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being ussed.
builtins.__MICPY_SETUP__ = True

from micpy.distutils.build_offload import build_ext as build_ext_offload
from micpy.distutils.build_offload_lib import build_clib as build_clib_offload
from micpy.code_generators import generate_umath as gen_umath


numpy_private_dir = join('numpy', 'private')
multiarray_dir = join('micpy', 'multiarray')
mpymath_dir = join('micpy', 'mpymath')
codegen_dir = join('micpy', 'code_generators')


private_npy_defines = [('HAVE_ENDIAN_H', 1),
            ('HAVE_COMPLEX_H', 1),
            ('HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE', 1)]


def add_multiarray_ext(config):
    multiarray_sources = ['alloc.c', 'array_assign.c', 'arrayobject.c',
            'cblasfuncs.c', 'common.c', 'calculation.c', 'convert.c',
            'number.c', 'conversion_utils.c', 'creators.c', 'getset.c',
            'methods.c', 'shape.c', 'scalar.c', 'item_selection.c',
            'convert_datatype.c', 'dtype_transfer.c', 'mpymem_overlap.c',
            'nditer_templ.c.src', 'nditer_constr.c', 'nditer_api.c',
            'arraytypes.c.src', 'mpy_lowlevel_strided_loops.c.src',
            'temp_elide.c' ,'multiarraymodule.c']
    multiarray_sources = [join(multiarray_dir, f) for f in multiarray_sources]

    #Add numpy/private/mem_overlap.c to sources
    multiarray_sources += [
            join(numpy_private_dir, 'templ_common.h.src')]

    #TODO: find a better way to define NMAXDEVICES
    #hint: use micinfo to get number of devices
    config.add_extension('multiarray',
                        sources=multiarray_sources,
                        define_macros=private_npy_defines +
                                [('NMAXDEVICES', '2')],
                        include_dirs=[multiarray_dir],
                        libraries=['mpymath'],
                        extra_link_args=['-mkl'])


def add_mpymath_lib(config):
    sources = ['non_standards.h.src', 'ieee754.c.src',
                'mpy_math_complex.c.src', 'halffloat.c']
    mpymath_sources = [join(mpymath_dir, f) for f in sources]

    config.add_library("mpymath", sources=mpymath_sources,
                        include_dirs=[mpymath_dir],
                        macros=private_npy_defines)


def add_umath_ext(config):
    def generate_umath_c(ext, build_dir):
        config.add_include_dirs(build_dir)
        target = join(build_dir, '__umath_generated.c')
        dir = os.path.dirname(target)
        if not os.path.exists(dir):
            os.makedirs(dir)
        script = join(codegen_dir, 'generate_umath.py')
        if newer(script, target):
            f = open(target, 'w')
            f.write(gen_umath.make_code(gen_umath.defdict,
                                        gen_umath.__file__))
            f.close()
        return []

    umath_dir = join('micpy', 'umath')

    umath_sources = ['umathmodule.c', 'mufunc_object.c',
                     'output_creators.c', 'reduction.c',
                     'funcs.inc.src', 'loops.h.src', 'loops.c.src',
                     'simd.inc.src']
    umath_sources = [join(umath_dir, f) for f in umath_sources]
    umath_sources.append(join(mpymath_dir, 'non_standards.h.src'))
    umath_sources.append(join(multiarray_dir, 'mpymem_overlap.c'))
    umath_sources.append(generate_umath_c)

    config.add_extension('umath',
                        sources=umath_sources,
                        include_dirs=[umath_dir, mpymath_dir],
                        libraries=['mpymath'],
                        define_macros=private_npy_defines)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import \
            Configuration, get_numpy_include_dirs

    config = Configuration('micpy', parent_package, top_path)

    config.add_include_dirs([numpy_private_dir, 'micpy'])

    add_multiarray_ext(config)
    add_mpymath_lib(config)
    add_umath_ext(config)

    config.add_subpackage('random',
                          subpackage_path=join('micpy', 'random'))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name='micpy',
          version='0.2',
          description='Numpy like interface for MIC device',
          author='Super Bo',
          author_email='supernbo@gmail.com',
          url='https://github.com/SuperBo/micpy',
          packages=['micpy'],
          configuration=configuration,
          cmdclass={'build_ext': build_ext_offload,
                    'build_clib': build_clib_offload})
    del builtins.__MICPY_SETUP__

#!/usr/bin/env python

from __future__ import absolute_import, print_function
from os.path import join

from micpy.distutils.build_offload import build_ext as build_ext_offload

numpy_private_dir = join('numpy', 'private')
multiarray_dir = join('micpy', 'multiarray')

def add_multiarray_ext(config):
    multiarray_sources = ['alloc.c', 'array_assign.c', 'arrayobject.c',
            'common.c', 'calculation.c', 'convert.c',
            'conversion_utils.c', 'creators.c', 'getset.c',
            'methods.c', 'shape.c', 'scalar.c', 'item_selection.c',
            'convert_datatype.c',
            'multiarraymodule.c']
    multiarray_sources = [join(multiarray_dir, f) for f in multiarray_sources]

    #Add numpy/private/mem_overlap.c to sources
    multiarray_sources += [join(numpy_private_dir, 'mem_overlap.c'),
            join(numpy_private_dir, 'templ_common.h.src')]

    #TODO: find a better way to define NMAXDEVICES
    #hint: use micinfo to get number of devices
    config.add_extension('multiarray',
                        sources=multiarray_sources,
                        define_macros=[('NMAXDEVICES', '2')])


def add_umath_ext(config):
    umath_dir = join('micpy', 'umath')
    umath_sources = ['umathmodule.c', 'mufunc_object.c',
            'output_creators.c', 'reduction.c',
            'loops.c.src', 'simd.inc.src']
    umath_sources = [join(umath_dir, f) for f in umath_sources]

    config.add_extension('umath',
                        sources=umath_sources,
                        include_dirs=[])


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import \
            Configuration, get_numpy_include_dirs

    config = Configuration('micpy', parent_package, top_path)

    config.add_include_dirs([numpy_private_dir, 'micpy'])

    add_multiarray_ext(config)
    add_umath_ext(config)

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
          cmdclass={'build_ext': build_ext_offload})

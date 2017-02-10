#!/usr/bin/env python

from __future__ import absolute_import, print_function
from os.path import join

from micpy.distutils.build_offload import build_ext as build_ext_offload

numpy_private_dir = join('numpy', 'private')
multiarray_dir = join('micpy', 'multiarray')

multiarray_sources = ['alloc.c', 'array_assign.c', 'arrayobject.c',
                      'common.c', 'convert.c', 'conversion_utils.c',
                      'creators.c', 'getset.c', 'shape.c',
                      'scalar.c', 'multiarraymodule.c']
multiarray_sources = list(map(lambda f: join(multiarray_dir, f),
                         multiarray_sources))
# Add numpy/private/mem_overlap.c to sources
multiarray_sources += [join(numpy_private_dir, 'mem_overlap.c'),
                       join(numpy_private_dir, 'templ_common.h.src')]

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import \
            Configuration, get_numpy_include_dirs

    config = Configuration('micpy', parent_package, top_path)

    config.add_extension('multiarray',
                        sources=multiarray_sources,
                        define_macros=[('NMAXDEVICES', '2')],
                        include_dirs=[multiarray_dir,
                                      numpy_private_dir],
                        runtime_library_dirs=[])

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

from __future__ import division, print_function

from os.path import join, split, dirname
import os
import sys
from distutils.dep_util import newer
from distutils.msvccompiler import get_build_version as get_msvc_build_version

def gen_mpyrandom(ext, build_dir):
    from Cython.Compiler.Main import compile_single, CompilationOptions, default_options
    target = join(build_dir, 'mpyrandom.c')
    src = join('micpy', 'random', 'mpyrandom.pyx')
    header = join('micpy', 'random', 'mpyrandom.pxd')
    if newer(src, target) or newer(header, target):
        # create target file
        options = CompilationOptions(default_options)
        options.output_file = target
        result = compile_single(src, options)
    return target

def needs_mingw_ftime_workaround():
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw ...
    # ... but we can't easily detect compiler version outside distutils command
    # context, so we will need to detect in randomkit whether we build with gcc
    msver = get_msvc_build_version()
    if msver and msver >= 8:
        return True

    return False

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random', parent_package, top_path)

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    if sys.platform[:3] == "aix":
        defs = [('_LARGE_FILES', None)]
    else:
        defs = [('_FILE_OFFSET_BITS', '64'),
                ('_LARGEFILE_SOURCE', '1'),
                ('_LARGEFILE64_SOURCE', '1')]
    if needs_mingw_ftime_workaround():
        defs.append(("NPY_NEEDS_MINGW_TIME_WORKAROUND", None))

    #TODO: find a better way to get max devices
    defs.append(('NMAXDEVICES', 2))

    libs = []
    # Configure mtrand
    config.add_extension('mpyrandom',
                         sources=[gen_mpyrandom, 'randomkit.c', 'distributions.c'],
                         include_dirs=['./'],
                         depends=[join('./', '*.h'),
                                  join('./', '*.pyx'),
                                  join('./', '*.pxi')],
                         define_macros=defs,
                         extra_link_args=['-mkl'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    from micpy.distutils.build_offload import build_ext as build_ext_offload
    setup(configuration=configuration,
          cmdclass={'build_ext': build_ext_offload})
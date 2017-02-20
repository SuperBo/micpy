from __future__ import division, absolute_import, print_function

import platform

from distutils.unixccompiler import UnixCCompiler


class IntelOffloadCCompiler(UnixCCompiler):
    """A modified Intel compiler support MIC offloading compatible with GCC-built Python."""
    compiler_type = 'intel_offload'
    cc_exe = 'icc'
    cc_args = 'fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        self.cc_exe = ('icc -fPIC -fp-model precise -O3 '
                       '-fomit-frame-pointer -qopenmp '
                       '-Wall -Wimplicit '
                       '-qoffload-option,mic,compiler,'
                       '\'"-fp-model fast=2"\'')
        compiler = self.cc_exe
        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             archiver='xiar' + ' cru',
                             linker_exe=compiler + ' -shared-intel',
                             linker_so=compiler + ' ' + shared_flag +
                             ' -shared-intel')


if platform.system() == 'Windows':
    from numpy.distutils.intelcompiler import IntelCCompilerW

    class IntelOffloadCCompilerW(IntelCCompilerW):
        """A modified Intel compiler support MIC offloading compatible with an MSVC-built Python."""
        compiler_type = 'intel_offloadw'
        compiler_cxx = 'icl'

        def initialize(self, plat_name=None):
            IntelCCompilerW.initialize(plat_name)
            self.compile_options = ['/nologo', '/O3', '/MD', '/W3',
                                    '/Qstd=c99', '/qopenmp']


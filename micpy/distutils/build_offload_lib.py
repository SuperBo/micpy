""" Modified version of build_ext that handles Offload source files.

"""
from __future__ import division, absolute_import, print_function

import platform
import shutil

from numpy.distutils.command.build_clib import build_clib as old_build_clib
from numpy.distutils.misc_util import filter_sources, has_f_sources, \
     has_cxx_sources, get_ext_source_files, \
     get_numpy_include_dirs, is_sequence, get_build_architecture, \
     msvc_version

from . import offloadcompiler

class build_clib(old_build_clib):
    description = "build C/C++/F Offload libraries used by Python extensions"

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        languages = []

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        for (lib_name, build_info) in self.libraries:
            l = build_info.get('language', None)
            if l and l not in languages: languages.append(l)

        # Initialize C compiler:
        if platform.system() == 'Windows':
            compiler_type = 'intel_offloadw'
            self.compiler = offloadcompiler.IntelOffloadCCompilerW(
                                dry_run=self.dry_run,
                                force=self.force)
        else:
            compiler_type = 'intel_offload'
            self.compiler = offloadcompiler.IntelOffloadCCompiler(
                                dry_run=self.dry_run,
                                force=self.force)

        self.compiler.customize(self.distribution,
                                need_cxx=self.have_cxx_sources())

        libraries = self.libraries
        self.libraries = None
        self.compiler.customize_cmd(self)
        self.libraries = libraries

        self.compiler.show_customization()

        if self.have_f_sources():
            from numpy.distutils.fcompiler import new_fcompiler
            self._f_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90='f90' in languages,
                                               c_compiler=self.compiler)
            if self._f_compiler is not None:
                self._f_compiler.customize(self.distribution)

                libraries = self.libraries
                self.libraries = None
                self._f_compiler.customize_cmd(self)
                self.libraries = libraries

                self._f_compiler.show_customization()
        else:
            self._f_compiler = None

        self.build_libraries(self.libraries)

        if self.inplace:
            for l in  self.distribution.installed_libraries:
                libname = self.compiler.library_filename(l.name)
                source = os.path.join(self.build_clib, libname)
                target =  os.path.join(l.target_dir, libname)
                self.mkpath(l.target_dir)
                shutil.copy(source, target)


""" Modified version of build_ext that handles Offload source files.

"""
from __future__ import division, absolute_import, print_function

import platform

from numpy.distutils import log
from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.misc_util import filter_sources, has_f_sources, \
     has_cxx_sources, get_ext_source_files, \
     get_numpy_include_dirs, is_sequence, get_build_architecture, \
     msvc_version

from . import offloadcompiler

class build_ext(old_build_ext):
    description = "build C offload extensions (compile/link to build directory)"

    def run(self):
        if not self.extensions:
            return

        self.run_command('build_src')

        if self.distribution.has_c_libraries():
            if self.inplace:
                if self.distribution.have_run.get('build_clib'):
                    log.warn('build_clib already run, it is too late to ' \
                            'ensure in-place build of build_clib')
                    build_clib = self.distribution.get_command_obj('build_clib')
                else:
                    build_clib = self.distribution.get_command_obj('build_clib')
                    build_clib.inplace = 1
                    build_clib.ensure_finalized()
                    build_clib.run()
                    self.distribution.have_run['build_clib'] = 1

            else:
                self.run_command('build_clib')
                build_clib = self.get_finalized_command('build_clib')
            self.library_dirs.append(build_clib.build_clib)
        else:
            build_clib = None

        # Initialize C compiler:
        if platform.system() == 'Windows':
            compiler_type = 'intel_offloadw'
            self.compiler = offloadcompiler.IntelOffloadCCompilerW(
                                verbose=self.verbose,
                                dry_run=self.dry_run,
                                force=self.force)
        else:
            compiler_type = 'intel_offload'
            self.compiler = offloadcompiler.IntelOffloadCCompiler(
                                verbose=self.verbose,
                                dry_run=self.dry_run,
                                force=self.force)

        self.compiler.customize(self.distribution)
        self.compiler.customize_cmd(self)
        self.compiler.show_customization()

        # Create mapping of libraries built by build_clib:
        clibs = {}
        if build_clib is not None:
            for libname, build_info in build_clib.libraries or []:
                if libname in clibs and clibs[libname] != build_info:
                    log.warn('library %r defined more than once,'\
                             ' overwriting build_info\n%s... \nwith\n%s...' \
                             % (libname, repr(clibs[libname])[:300], repr(build_info)[:300]))
                clibs[libname] = build_info
        # .. and distribution libraries:
        for libname, build_info in self.distribution.libraries or []:
            if libname in clibs:
                # build_clib libraries have a precedence before distribution ones
                continue
            clibs[libname] = build_info

        # Determine if C++/Fortran 77/Fortran 90 compilers are needed.
        # Update extension libraries, library_dirs, and macros.
        # Borrowed from numpy.distutils.command.build_ext
        all_languages = set()
        for ext in self.extensions:
            ext_languages = set()
            c_libs = []
            c_lib_dirs = []
            macros = []
            for libname in ext.libraries:
                if libname in clibs:
                    binfo = clibs[libname]
                    c_libs += binfo.get('libraries', [])
                    c_lib_dirs += binfo.get('library_dirs', [])
                    for m in binfo.get('macros', []):
                        if m not in macros:
                            macros.append(m)

                for l in clibs.get(libname, {}).get('source_languages', []):
                    ext_languages.add(l)
            if c_libs:
                new_c_libs = ext.libraries + c_libs
                log.info('updating extension %r libraries from %r to %r'
                         % (ext.name, ext.libraries, new_c_libs))
                ext.libraries = new_c_libs
                ext.library_dirs = ext.library_dirs + c_lib_dirs
            if macros:
                log.info('extending extension %r defined_macros with %r'
                         % (ext.name, macros))
                ext.define_macros = ext.define_macros + macros

            # determine extension languages
            if has_f_sources(ext.sources):
                ext_languages.add('f77')
            if has_cxx_sources(ext.sources):
                ext_languages.add('c++')
            l = ext.language or self.compiler.detect_language(ext.sources)
            if l:
                ext_languages.add(l)
            # reset language attribute for choosing proper linker
            if 'c++' in ext_languages:
                ext_language = 'c++'
            elif 'f90' in ext_languages:
                ext_language = 'f90'
            elif 'f77' in ext_languages:
                ext_language = 'f77'
            else:
                ext_language = 'c' # default
            if l and l != ext_language and ext.language:
                log.warn('resetting extension %r language from %r to %r.' %
                         (ext.name, l, ext_language))
            ext.language = ext_language
            # global language
            all_languages.update(ext_languages)

        need_f90_compiler = 'f90' in all_languages
        need_f77_compiler = 'f77' in all_languages
        need_cxx_compiler = 'c++' in all_languages

        # Initialize C++ compiler:
        if need_cxx_compiler:
            self._cxx_compiler = new_compiler(compiler=compiler_type,
                                             verbose=self.verbose,
                                             dry_run=self.dry_run,
                                             force=self.force)
            compiler = self._cxx_compiler
            compiler.customize(self.distribution, need_cxx=need_cxx_compiler)
            compiler.customize_cmd(self)
            compiler.show_customization()
            self._cxx_compiler = compiler.cxx_compiler()
        else:
            self._cxx_compiler = None

        # Initialize Fortran 77 compiler:
        if need_f77_compiler:
            ctype = self.fcompiler
            self._f77_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=False,
                                               c_compiler=self.compiler)
            fcompiler = self._f77_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f77_compiler=%s is not available.' %
                          (ctype))
                self._f77_compiler = None
        else:
            self._f77_compiler = None

        # Initialize Fortran 90 compiler:
        if need_f90_compiler:
            ctype = self.fcompiler
            self._f90_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=True,
                                               c_compiler = self.compiler)
            fcompiler = self._f90_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f90_compiler=%s is not available.' %
                          (ctype))
                self._f90_compiler = None
        else:
            self._f90_compiler = None

        # Build extensions
        self.build_extensions()


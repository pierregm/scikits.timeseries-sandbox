#!/usr/bin/env python
import fileinput
import glob
import os
import shutil
import sys

def check_build():
    build_dirs = ['build', 'build/doctrees', 'build/html', 'build/latex',
                  '_static', '_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass

def html():
    check_build()

    if os.system('sphinx-build -b html -d build/doctrees . build/html'):
        raise SystemExit("Building HTML failed.")

def latex():
    check_build()

    # LaTeX format.
    if os.system('sphinx-build -b latex -d build/doctrees . build/latex'):
        raise SystemExit("Building LaTeX failed.")

    # Produce pdf.
    os.chdir('build/latex')

    # first call to pdflatex used to generate .idx files
    os.system('pdflatex scikitstimeseries.tex')

    # convert .idx files to indices
    os.system('makeindex -s python.ist scikitstimeseries.idx')
    os.system('makeindex -s python.ist modscikitstimeseries.idx')

    # regenerate pdf with proper indices
    os.system('pdflatex scikitstimeseries.tex')

    os.chdir('../..')

def clean():
    if os.path.exists('build'):
        shutil.rmtree('build')

def all():
    html()
    latex()


funcd = {'html':html,
         'latex':latex,
         'clean':clean,
         'all':all,
         }


if len(sys.argv)>1:
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are'%(
                    arg, funcd.keys()))
        func()
else:
    all()

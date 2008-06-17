"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

version = '0.67.0'

classifiers = ['Development Status :: 0.67.0 - Beta',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Programming Language :: C',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS']
               
distname = 'scikits.timeseries'

import os
import sys
import setuptools
from numpy.distutils.core import setup, Extension


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(
                None, parent_package, top_path,
                namespace_packages=['scikits'])

    config.set_options(
                ignore_setup_xxx_py=True,
                assume_default_configuration=True,
                delegate_options_to_subpackages=True,
                quiet=True,
    )

    config.add_subpackage('scikits')
    config.add_subpackage(distname)
    config.add_data_files('scikits/__init__.py')

    return config



def setup_package():

    setup(
          install_requires='numpy',
          namespace_packages=['scikits'],
          packages=setuptools.find_packages(),
          test_suite = 'nose.collector',
          name = distname,
          version = version,
          description = "Time series manipulation",
          author = "Pierre GF GERARD-MARCHANT & Matt KNOX",
          author_email = "pgmdevlist_AT_gmail_DOT_com & "\
                         "mattknox_ca_at_hotmail_dot_com",
          url = "http://scipy.org/scipy/scikits/wiki/TimeSeries",
          classifiers = classifiers, 
          platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
          configuration = configuration,
    )

    return

if __name__ == '__main__':
    setup_package()

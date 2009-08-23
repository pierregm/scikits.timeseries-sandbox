"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

version = '0.91.2'

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: C',
               'Programming Language :: Python',
               'Topic :: Education',
               'Topic :: Scientific/Engineering',
               'Topic :: Software Development',
              ]

distname = 'scikits.timeseries'

long_description = """
The scikits.timeseries module provides classes and functions for manipulating,
reporting, and plotting time series of various frequencies. The focus is on
convenient data access and manipulation while leveraging the existing
mathematical functionality in Numpy and SciPy.
"""

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
          install_requires='numpy >= 1.3.0',
          namespace_packages=['scikits'],
          packages=setuptools.find_packages(),
          test_suite = 'nose.collector',
          name = distname,
          version = version,
          description = "Time series manipulation",
          long_description = long_description,
          license = "BSD",
          author = "Pierre GF GERARD-MARCHANT & Matt KNOX",
          author_email = "pierregmcode_AT_gmail_DOT_com & "\
                         "mattknox.ca_AT_gmail_DOT_com",
          maintainer = "Pierre GERARD-MARCHANT",
          maintainer_email = "pierregmcode_AT_gmail_DOT_com",
          url = "http://pytseries.sourceforge.net",
          classifiers = classifiers,
          platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
          configuration = configuration,
    )

    return

if __name__ == '__main__':
    setup_package()

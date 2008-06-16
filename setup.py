"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""
import setuptools
from scikits.timeseries.version import version

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

    config.add_subpackage('scikits.timeseries')
    config.add_data_files('scikits/__init__.py')
    config.get_version('scikits/timeseries/version.py')

    return config

def setup_package():

    from numpy.distutils.core import setup

    setup(
        name = 'scikits.timeseries',
        version = version,
        description = "Time series manipulation",
        configuration = configuration,
    )

    return

if __name__ == '__main__':
    setup_package()

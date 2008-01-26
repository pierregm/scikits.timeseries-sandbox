# install setuptools on user's pc if not installed already
import ez_setup
ez_setup.use_setuptools()

import setuptools

subpackage = 'timeseries'
version = '0.5'

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

    return config

def setup_package():

    from numpy.distutils.core import setup

    setup(
        name = 'scikits.' + subpackage,
        version = version,
        description = "Time series manipulation",
        configuration = configuration,
    )

    return

if __name__ == '__main__':
    setup_package()

from pkg_resources import require, DistributionNotFound

try:
    __version__ = require('scikits.timeseries')[0].version
except DistributionNotFound:
    # package hasn't actually been installed. Importing directly from source
    # folder. Assign a dummy value for __version__ .
    # This should only happen for developers of the package
    __version__ = '0.0.0 - dev'

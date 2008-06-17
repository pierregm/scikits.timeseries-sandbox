#version='0.67.0'
#release=False
#module_name = None
#
#if not release:
#    version += '.dev'
#    import os
#    current_dir = os.path.dirname(__file__)
#    module_name = module_name or os.path.split(current_dir)[-1]
#    svn_version_file = os.path.join(current_dir,'__svn_version__.py')
#    if os.path.isfile(svn_version_file):
#        import imp
#        svn = imp.load_module(module_name + '.__svn_version__',
#                              open(svn_version_file),
#                              svn_version_file,
#                              ('.py','U',1))
#        version += svn.version
#
#__all__ = str(version)

#from pkg_resources import require
#__version__ = require('scikits.timeseries')[0].version

__version__ = '0.67.0'

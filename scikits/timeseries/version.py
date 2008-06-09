version='0.67.0'
release=False

if not release:
    version += '.dev'
    import os
    svn_version_file = os.path.join(os.path.dirname(__file__),
                                   'timeseries','__svn_version__.py')
    if os.path.isfile(svn_version_file):
        import imp
        svn = imp.load_module('scikits.timeseries.__svn_version__',
                              open(svn_version_file),
                              svn_version_file,
                              ('.py','U',1))
        version += svn.version

"""
Collection of bug-fixes, add-ons and other modifications that are not yet part
of a stable numpy/scipy release.
"""

__all__ = ['LineSplitter', 'NameValidator', 'StringConverter',
           'flatten_dtype', 'genfromtxt', 'mafromtxt', 'ndfromtxt', 'recfromtxt',
           'recfromcsv', 'str2bool']


import itertools
import numpy as np
import numpy.ma as ma


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def _to_filehandle(fname, flag='r', return_opened=False):
    """
    Returns the filehandle corresponding to a string or a file.
    If the string ends in '.gz', the file is automatically unzipped.
    
    Parameters
    ----------
    fname : string, filehandle
        Name of the file whose filehandle must be returned.
    flag : string, optional
        Flag indicating the status of the file ('r' for read, 'w' for write).
    return_opened : boolean, optional
        Whether to return the opening status of the file.
    """
    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fhd = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            import bz2
            fhd = bz2.BZ2File(fname)
        else:
            fhd = file(fname, flag)
        opened = True
    elif hasattr(fname, 'seek'):
        fhd = fname
        opened = False
    else:
        raise ValueError('fname must be a string or file handle')
    if return_opened:
        return fhd, opened
    return fhd


def flatten_dtype(ndtype):
    """
    Unpack a structured data-type.

    """
    names = ndtype.names
    if names is None:
        return [ndtype]
    else:
        types = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            flat_dt = flatten_dtype(typ)
            types.extend(flat_dt)
        return types


#def nested_masktype(datatype):
#    """
#    Construct the dtype of a mask for nested elements.
#
#    """
#    names = datatype.names
#    if names:
#        descr = []
#        for name in names:
#            (ndtype, _) = datatype.fields[name]
#            descr.append((name, nested_masktype(ndtype)))
#        return descr
#    # Is this some kind of composite a la (np.float,2)
#    elif datatype.subdtype:
#        mdescr = list(datatype.subdtype)
#        mdescr[0] = np.dtype(bool)
#        return tuple(mdescr)
#    else:
#        return np.bool



class LineSplitter:
    """
    Defines a function to split a string at a given delimiter or at given places.
    
    Parameters
    ----------
    comment : {'#', string}
        Character used to mark the beginning of a comment.
    delimiter : var, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    autostrip : boolean, optional
        Whether to strip each individual fields
    """

    def autostrip(self, method):
        "Wrapper to strip each member of the output of `method`."
        return lambda input: [_.strip() for _ in method(input)]
    #
    def __init__(self, delimiter=None, comments='#', autostrip=True):
        self.comments = comments
        # Delimiter is a character
        if (delimiter is None) or _is_string_like(delimiter):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0]+list(delimiter))
            delimiter = [slice(i,j) for (i,j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
    #
    def _delimited_splitter(self, line):
        line = line.split(self.comments)[0].strip()
        if not line:
            return []
        return line.split(self.delimiter)
    #
    def _fixedwidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i+fixed) for i in range(len(line))[::fixed]]
        return [line[s] for s in slices]
    #
    def _variablewidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]
    #
    def __call__(self, line):
        return self._handyman(line)



class NameValidator:
    """
    Validates a list of strings to use as field names.
    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by `_`. If the optional input parameter `case_sensitive`
    is False, the strings are set to upper case.

    During instantiation, the user can define a list of names to exclude, as 
    well as a list of invalid characters. Names in the exclude list are appended
    a '_' character.

    Once an instance has been created, it can be called with a list of names
    and a list of valid names will be created.
    The `__call__` method accepts an optional keyword, `default`, that sets
    the default name in case of ambiguity. By default, `default = 'f'`, so
    that names will default to `f0`, `f1`

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : string, optional
        A string combining invalid characters that must be deleted from the names.
    casesensitive : boolean, optional
        Whether the field names are case sensitive or not. If not, then fields
        like `'date'` and `'DATE'` are assumed to be the same.
    """
    #
    defaultexcludelist = ['return','file','print']
    defaultdeletechars = set("""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
    #
    def __init__(self, excludelist=None, deletechars=None, case_sensitive=True):
        #
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        #
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        self.case_sensitive = case_sensitive
    #
    def validate(self, names, default='f'):
        #
        if names is None:
            return
        #
        validatednames = []
        seen = dict()
        #
        deletechars = self.deletechars
        excludelist = self.excludelist
        casesensitive = self.case_sensitive
        for i, item in enumerate(names):
            if not casesensitive:
                item = item.upper()
            item = item.strip().replace(' ', '_')
            item = ''.join([c for c in item if c not in deletechars])
            if not len(item):
                item = '%s%d' % (default, i)
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt+1
        return validatednames
    #
    def __call__(self, names, default='f'):
        return self.validate(names, default)



def str2bool(value):
    """
    Tries to transform a string supposed to represent a boolean to a boolean.
    
    Raises
    ------
    ValueError
        If the string is not 'True' or 'False' (case independent)
    """
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")



class StringConverter:
    """
    Factory class for function transforming a string into another object (int,
    float).

    After initialization, an instance can be called to transform a string 
    into another object. If the string is recognized as representing a missing
    value, a default value is returned.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        Input data type, used to define a basic function and a default value
        for missing data. For example, when `dtype` is float, the :attr:`func`
        attribute is set to ``float`` and the default value to `np.nan`.
        Alternatively, function used to convert a string to another object.
        In that later case, it is recommended to give an associated default
        value as input.
    default : {None, var}, optional
        Value to return by default, that is, when the string to be converted
        is flagged as missing.
    missing_values : {sequence}, optional
        Sequence of strings indicating a missing value.
    locked : {boolean}, optional
        Whether the StringConverter should be locked to prevent automatic 
        upgrade or not.

    Attributes
    ----------
    func : function
        Function used for the conversion
    default : var
        Default value to return when the input corresponds to a missing value.
    _status : integer
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in order.
    _locked : boolean
        Whether the StringConverter is locked, thereby preventing automatic any
        upgrade or not.

    """
    #
    _mapper = [(np.bool_, str2bool, None),
               (np.integer, int, -1),
               (np.floating, float, np.nan),
               (np.complex, complex, np.nan+0j),
               (np.string_, str, '???')]
    (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
    #
    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        return np.array(val).dtype.type
    #
    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
    Upgrade the mapper of a StringConverter by adding a new function and its
    corresponding default.
    
    The input function (or sequence of functions) and its associated default 
    value (if any) is inserted in penultimate position of the mapper.
    The corresponding type is estimated from the dtype of the default value.
    
    Parameters
    ----------
    func : var
        Function, or sequence of functions

    Examples
    --------
    >>> import dateutil.parser
    >>> import datetime
    >>> dateparser = datetutil.parser.parse
    >>> defaultdate = datetime.date(2000, 1, 1)
    >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
        # Func is a single functions
        if hasattr(func, '__call__'):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func)-len(default)))
            for (fct, dft) in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))
    #
    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # Defines a lock for upgrade
        self._locked = bool(locked)
        # No input dtype: minimal initialization
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default
        else:
            # Is the input a np.dtype ?
            try:
                self.func = None
                ttype = np.dtype(dtype_or_func).type
            except TypeError:
                # dtype_or_func must be a function, then
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = "The input argument `dtype` is neither a function"\
                             " or a dtype (got '%s' instead)"
                    raise TypeError(errmsg % type(dtype_or_func))
                # Set the function
                self.func = dtype_or_func
                # If we don't have a default, try to guess it or set it to None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                ttype = self._getsubdtype(default)
            # Set the status according to the dtype
            for (i, (deftype, func, default_def)) in enumerate(self._mapper):
                if np.issubdtype(ttype, deftype):
                    self._status = i
                    self.default = default or default_def
                    break
            # If the input was a dtype, set the function to the last we saw
            if self.func is None:
                self.func = func
            # If the status is 1 (int), change the function to smthg more robust
            if self.func == self._mapper[1][1]:
                self.func = lambda x : int(float(x))
        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = set([''])
        else:
            self.missing_values = set(list(missing_values) + [''])
        #
        self._callingfunction = self._strict_call
    #
    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default
    #
    def _strict_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            if value.strip() in self.missing_values:
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)
    #
    def __call__(self, value):
        return self._callingfunction(value)
    #
    def upgrade(self, value):
        """
    Tries to find the best converter for `value`, by testing different
    converters in order.
    The order in which the converters are tested is read from the
    :attr:`_status` attribute of the instance.
        """
        try:
            self._strict_call(value)
        except ValueError:
            # Raise an exception if we locked the converter...
            if self._locked:
                raise ValueError("Converter is locked and cannot be upgraded")
            _statusmax = len(self._mapper)
            # Complains if we try to upgrade by the maximum
            if self._status == _statusmax:
                raise ValueError("Could not find a valid conversion function")
            elif self._status < _statusmax - 1:
                self._status += 1
            (_, self.func, self.default) = self._mapper[self._status]
            self.upgrade(value)
    #
    def update(self, func, default=None, missing_values='', locked=False):
        """
    Sets the :attr:`func` and :attr:`default` attributes directly.

    Parameters
    ----------
    func : function
        Conversion function.
    default : {var}, optional
        Default value to return when a missing value is encountered.
    missing_values : {var}, optional
        Sequence of strings representing missing values.
    locked : {False, True}, optional
        Whether the status should be locked to prevent automatic upgrade.
        """
        self.func = func
        self._locked = locked
        # Don't reset the default to None if we can avoid it
        if default is not None:
            self.default = default
        # Add the missing values to the existing set
        if missing_values is not None:
            if _is_string_like(missing_values):
                self.missing_values.add(missing_values)
            elif hasattr(missing_values, '__iter__'):
                for val in missing_values:
                    self.missing_values.add(val)
        else:
            self.missing_values = []




def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None, usecols=None,
               names=None, excludelist=None, deletechars=None,
               case_sensitive=True, unpack=None, usemask=False, loose=True):
    """
    Load data from a text file.


    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is `.gz` or `.bz2`,
        the file is first decompressed.
    dtype : data-type
        Data type of the resulting array.  If this is a flexible data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array. In this case, the number
        of columns used must match the number of fields in the data-type,
        and the names of each field will be set by the corresponding name
        of the dtype.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : {string}, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded
    delimiter : {string}, optional
        The string used to separate values.  By default, any consecutive
        whitespace act as delimiter.
    skiprows : {int}, optional
        Numbers of lines to skip at the beginning of the file.
    converters : {None, dictionary}, optional
        A dictionary mapping column number to a function that will convert
        values in the column to a number. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    missing : {string}, optional
        A string representing a missing value, irrespective of the column where
        it appears (e.g., `'missing'` or `'unused'`).
    missing_values : {None, dictionary}, optional
        A dictionary mapping a column number to a string indicating whether the
        corresponding field should be masked.
    usecols : {None, sequence}, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, string, sequence}, optional
        If `names` is True, the field names are read from the first valid line
        after the first `skiprows` lines.
        If `names` is a sequence or a single-string of comma-separated names,
        the names will be used to define the field names in a flexible dtype.
        If `names` is None, the names of the dtype fields will be used, if any.
    excludelist : {sequence}, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : {string}, optional
        A string combining invalid characters that must be deleted from the names.
    case_sensitive : {True, False], optional
        Whether names are case sensitive. If not, names are transformed to 
        upper case.
    unpack : {bool}, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``
    usemask : {bool}, optional
        If True, returns a masked array.
        If False, return a regular standard array.

    Returns
    -------
    out : MaskedArray
        Data read from the text file.

    Notes
    --------
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When the variable are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a :exc:ValueError exception
      is raised).


    """
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
                 "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))
    # Check the input dictionary of missing values
    user_missing_values = missing_values or {}
    if not isinstance(user_missing_values, dict):
        errmsg = "The input argument 'missing_values' should be a valid "\
                 "dictionary (got '%s' instead)"
        raise TypeError(errmsg % type(missing_values))
    defmissing = [_.strip() for _ in missing.split(',')] + ['']

    # Initialize the filehandle, the LineSplitter and the NameValidator
#    fhd = _to_filehandle(fname)
    if isinstance(fname, basestring):
        fhd = np.lib._datasource.open(fname)
    elif not hasattr(fname, 'read'):
        raise TypeError("The input should be a string or a filehandle. "\
                        "(got %s instead)" % type(fname))
    else:
        fhd = fname
    split_line = LineSplitter(delimiter=delimiter, comments=comments, 
                              autostrip=False)._handyman
    validate_names = NameValidator(excludelist=excludelist,
                                   deletechars=deletechars,
                                   case_sensitive=case_sensitive)

    # Get the first valid lines after the first skiprows ones
    for i in xrange(skiprows):
        fhd.readline()
    first_values = None
    while not first_values:
        first_line = fhd.readline()
        if first_line == '':
            raise IOError('End-of-file reached before encountering data.')
        first_values = split_line(first_line)

    # Check the columns to use
    if usecols is not None:
        usecols = list(usecols)
    nbcols = len(usecols or first_values)

    # Check the names and overwrite the dtype.names if needed
    if dtype is not None:
        dtype = np.dtype(dtype)
    dtypenames = getattr(dtype, 'names', None)
    if names is True:
        names = validate_names([_.strip() for _ in first_values])
        first_line =''
    elif _is_string_like(names):
        names = validate_names([_.strip() for _ in names.split(',')])
    elif names:
        names = validate_names(names)
    elif dtypenames:
        dtype.names = validate_names(dtypenames)
    if names and dtypenames:
        dtype.names = names

    # If usecols is a list of names, convert to a list of indices
    if usecols:
        for (i, current) in enumerate(usecols):
            if _is_string_like(current):
                usecols[i] = names.index(current)

    # If user_missing_values has names as keys, transform them to indices
    missing_values = {}
    for (key, val) in user_missing_values.iteritems():
        # If val is a list, flatten it. In any case, add missing &'' to the list
        if isinstance(val, (list, tuple)):
            val = [str(_) for _ in val]
        else:
            val = [str(val),]
        val.extend(defmissing)
        if _is_string_like(key):
            try:
                missing_values[names.index(key)] = val
            except ValueError:
                pass
        else:
            missing_values[key] = val


    # Initialize the default converters
    if dtype is None:
        # Note: we can't use a [...]*nbcols, as we would have 3 times the same
        # ... converter, instead of 3 different converters.
        converters = [StringConverter(None,
                              missing_values=missing_values.get(_, defmissing))
                      for _ in range(nbcols)]
    else:
        flatdtypes = flatten_dtype(dtype)
        # Initialize the converters
        if len(flatdtypes) > 1:
            # Flexible type : get a converter from each dtype
            converters = [StringConverter(dt,
                              missing_values=missing_values.get(i, defmissing),
                              locked=True)
                          for (i, dt) in enumerate(flatdtypes)]
        else:
            # Set to a default converter (but w/ different missing values)
            converters = [StringConverter(dtype,
                              missing_values=missing_values.get(_, defmissing),
                              locked=True)
                          for _ in range(nbcols)]
    missing_values = [_.missing_values for _ in converters]

    # Update the converters to use the user-defined ones
    for (i, conv) in user_converters.iteritems():
        # If the converter is specified by column names, use the index instead
        if _is_string_like(i):
            i = names.index(i)
        if usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i].update(conv, default=None, 
                             missing_values=missing_values[i],
                             locked=True)

    # Reset the names to match the usecols
    if (not first_line) and usecols:
        names = [names[_] for _ in usecols]

    rows = []
    append_to_rows = rows.append
    if usemask:
        masks = []
        append_to_masks = masks.append
    # Parse each line
    for line in itertools.chain([first_line,], fhd):
        values = split_line(line)
        # Skip an empty line
        if len(values) == 0:
            continue
        # Select only the columns we need
        if usecols:
            values = [values[_] for _ in usecols]
        # Check whether we need to update the converter
        if dtype is None:
            for (converter, item) in zip(converters, values):
                converter.upgrade(item)
        # Store the values
        append_to_rows(tuple(values))
        if usemask:
            append_to_masks(tuple([val.strip() in mss 
                                   for (val, mss) in zip(values,
                                                         missing_values)]))

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
    if loose:
        conversionfuncs = [conv._loose_call for conv in converters]
    else:
        conversionfuncs = [conv._strict_call for conv in converters]
    for (i, vals) in enumerate(rows):
        rows[i] = tuple([convert(val)
                         for (convert, val) in zip(conversionfuncs, vals)])


    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the first row
        coldtypes = [np.array(val).dtype for val in data[0]]
        # Find the columns with strings, and take the largest number of chars.
        strcolidx = [i for (i, v) in enumerate(coldtypes) if v.char == 'S']
        for i in strcolidx:
            coldtypes[i] = "|S%i" % max(len(row[i]) for row in data)
        
        if names is None:
            # If the dtype is uniform, don't define names, else use ''
            base = coldtypes[0]
            if np.all([(dt == base) for dt in coldtypes]):
                (ddtype, mdtype) = (base, np.bool)
            else:
                ddtype = [('', dt) for dt in coldtypes]
                mdtype = [('', np.bool) for dt in coldtypes]
        else:
            ddtype = zip(names, coldtypes)
            mdtype = zip(names, [np.bool] * len(coldtypes))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names:
            dtype.names = names
        # Check whether we have a nested dtype
        flatdtypes = flatten_dtype(dtype)
        if len(flatdtypes) > 1:
            # Nested dtype, eg  [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            rows = np.array(data, dtype=[('', t) for t in flatdtypes])
            output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(masks,
                                    dtype=np.dtype([('', np.bool)
                                                    for t in flatdtypes]))
                # Construct the new dtype
                mdtype = ma.make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        else:
            output = np.array(data, dtype)
            if usemask:
                if dtype.names:
                    mdtype = [(_, np.bool) for _ in dtype.names]
                else:
                    mdtype = np.bool
                outputmask = np.array(masks, dtype=mdtype)
    # Try to take care of the missing data we missed
    if usemask and output.dtype.names:
        for (name, conv) in zip(names or (), converters):
            missing_values = [conv(_) for _ in conv.missing_values if _ != '']
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    # Construct the final array
    if usemask:
        from numpy.ma import MaskedArray
        output = output.view(MaskedArray)
        output._mask = outputmask
    if unpack:
        return output.squeeze().T
    return output.squeeze()



def ndfromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
             converters=None, missing='', missing_values=None,
             usecols=None, unpack=None, names=None,
             excludelist=None, deletechars=None, case_sensitive=True,):
    """
    Load ASCII data stored in fname and returns a ndarray.
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    genfromtxt : generic function
    
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=False)
    return genfromtxt(fname, **kwargs)

def mafromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
              converters=None, missing='', missing_values=None,
              usecols=None, unpack=None, names=None,
              excludelist=None, deletechars=None, case_sensitive=True,):
    """
    Load ASCII data stored in fname and returns a MaskedArray.
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    genfromtxt : generic function
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive,
                  usemask=True)
    return genfromtxt(fname, **kwargs)


def recfromtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None, names=None,
               excludelist=None, deletechars=None, case_sensitive=True,
               usemask=False):
    """
    Load ASCII data stored in fname and returns a standard recarray (if 
    `usemask=False`) or a MaskedRecords (if `usemask=True`).
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    genfromtxt : generic function

    Warnings
    --------
    * by default, `dtype=None`, which means that the dtype of the output array
      will be determined from the data.
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=usemask)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, dtype=None, comments='#', skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None, names=None,
               excludelist=None, deletechars=None, case_sensitive=True,
               usemask=False):
    """
    Load ASCII data stored in comma-separated file and returns a recarray (if 
    `usemask=False`) or a MaskedRecords (if `usemask=True`).
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    genfromtxt : generic function
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=",", 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=usemask)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output



def _getconv(dtype):
    typ = dtype.type
    if issubclass(typ, np.bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, np.integer):
        return lambda x: int(float(x))
    elif issubclass(typ, np.floating):
        return float
    elif issubclass(typ, np.complex):
        return complex
    else:
        return str


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is ``.gz`` or
        ``.bz2``, the file is first decompressed.
    dtype : data-type
        Data type of the resulting array.  If this is a record data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array.   In this case, the number
        of columns used must match the number of fields in the data-type.
    comments : string, optional
        The character used to indicate the start of a comment.
    delimiter : string, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : {}
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    skiprows : int
        Skip the first `skiprows` lines.
    usecols : sequence
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    unpack : bool
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    scipy.io.loadmat : reads Matlab(R) data files

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])

    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x,y = np.loadtxt(c, delimiter=',', usecols=(0,2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """
    user_converters = converters

    if usecols is not None:
        usecols = list(usecols)

    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
        elif fname.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []


    def split_line(line):
        """Chop off comments, strip, and split at delimiter."""
        line = line.split(comments)[0].strip()
        if line:
            return line.split(delimiter)
        else:
            return []

    # Make sure we're dealing with a proper dtype
    dtype = np.dtype(dtype)
    defconv = _getconv(dtype)

    # Skip the first `skiprows` lines
    for i in xrange(skiprows):
        fh.readline()

    # Read until we find a line with some values, and use
    # it to estimate the number of columns, N.
    first_vals = None
    while not first_vals:
        first_line = fh.readline()
        if first_line == '': # EOF reached
            raise IOError('End-of-file reached before encountering data.')
        first_vals = split_line(first_line)
    N = len(usecols or first_vals)

    dtype_types = flatten_dtype(dtype)
    if len(dtype_types) > 1:
        # We're dealing with a structured array, each field of
        # the dtype matches a column
        converters = [_getconv(dt) for dt in dtype_types]
    else:
        # All fields have the same dtype
        converters = [defconv for i in xrange(N)]

    # By preference, use the converters specified by the user
    for i, conv in (user_converters or {}).iteritems():
        if usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i] = conv

    # Parse each line, including the first
    for i, line in enumerate(itertools.chain([first_line], fh)):
        vals = split_line(line)
        if len(vals) == 0:
            continue

        if usecols:
            vals = [vals[i] for i in usecols]

        # Convert each value according to its column and store
        X.append(tuple([conv(val) for (conv, val) in zip(converters, vals)]))

    if len(dtype_types) > 1:
        # We're dealing with a structured array, with a dtype such as
        # [('x', int), ('y', [('s', int), ('t', float)])]
        #
        # First, create the array using a flattened dtype:
        # [('x', int), ('s', int), ('t', float)]
        #
        # Then, view the array using the specified dtype.
        X = np.array(X, dtype=np.dtype([('', t) for t in dtype_types]))
        X = X.view(dtype)
    else:
        X = np.array(X, dtype)

    X = np.squeeze(X)
    if unpack:
        return X.T
    else:
        return X


def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save an array to file.

    Parameters
    ----------
    fname : filename or a file handle
        If the filename ends in .gz, the file is automatically saved in
        compressed gzip format.  The load() command understands gzipped
        files transparently.
    X : array_like
        Data.
    fmt : string or sequence of strings
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case delimiter is ignored.
    delimiter : str
        Character separating columns.

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to preceed result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This is not an exhaustive specification.



    Examples
    --------
    >>> savetxt('test.out', x, delimiter=',') # X is an array
    >>> savetxt('test.out', (x,y,z)) # x,y,z equal sized 1D arrays
    >>> savetxt('test.out', x, fmt='%1.4e') # use exponential notation

    """

    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname,'wb')
        else:
            fh = file(fname,'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    X = np.asarray(X)

    # Handle 1-dimensional arrays
    if X.ndim == 1:
        # Common case -- 1d array of numbers
        if X.dtype.names is None:
            X = np.atleast_2d(X).T
            ncol = 1

        # Complex dtype -- each field indicates a separate column
        else:
            ncol = len(X.dtype.descr)
    else:
        ncol = X.shape[1]

    # `fmt` can be a string with multiple insertion points or a list of formats.
    # E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
        format = delimiter.join(fmt)
    elif type(fmt) is str:
        if fmt.count('%') == 1:
            fmt = [fmt,]*ncol
            format = delimiter.join(fmt)
        elif fmt.count('%') != ncol:
            raise AttributeError('fmt has wrong number of %% formats.  %s'
                                 % fmt)
        else:
            format = fmt

    for row in X:
        fh.write(format % tuple(row) + '\n')

import re
def fromregex(inputfile, regexp, dtype):
    """
    Construct an array from a text file, using regular-expressions parsing.

    Array is constructed from all matches of the regular expression
    in the file. Groups in the regular expression are converted to fields.

    Parameters
    ----------
    file : str or file
        File name or file object to read.
    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or dtype list
        Dtype for the structured array

    Examples
    --------
    >>> f = open('test.dat', 'w')
    >>> f.write("1312 foo\\n1534  bar\\n444   qux")
    >>> f.close()
    >>> np.fromregex('test.dat', r"(\\d+)\\s+(...)",
    ...              [('num', np.int64), ('key', 'S3')])
    array([(1312L, 'foo'), (1534L, 'bar'), (444L, 'qux')],
          dtype=[('num', '<i8'), ('key', '|S3')])

    """
    if not hasattr(inputfile, "read"):
        inputfile = open(inputfile,'r')
    if not hasattr(regexp, 'match'):
        regexp = re.compile(regexp)
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    seq = regexp.findall(inputfile.read())
    if seq and not isinstance(seq[0], tuple):
        # make sure np.array doesn't interpret strings as binary data
        # by always producing a list of tuples
        seq = [(x,) for x in seq]
    output = np.array(seq, dtype=dtype)
    return output

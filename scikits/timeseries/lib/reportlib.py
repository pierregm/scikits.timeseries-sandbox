"""
Reporting functions

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id$

Ideas borrowed from:

- George Sakkis
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/267662

- Mike Brown
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/148061

:Examples:

    import numpy as np
    from scikits import timeseries as ts
    from scikits.timeseries.lib.reportlib import Report, wrap_onspace
    from numpy import ma

    series1 = ts.time_series(np.random.uniform(-100,100,15), start_date=ts.thisday('b')-15)
    series2 = ts.time_series(np.random.uniform(-100,100,13), start_date=ts.thisday('b')-10)
    series3 = ts.time_series(['string1', 'another string', 'yet another string']*3, start_date=ts.thisday('b')-10)

    darray = ts.date_array(start_date=ts.thisday('b')-8, end_date=ts.thisday('b')-3)

    txt_o = open('myfile.txt', 'w')
    html_o = open('myfile.html', 'w')

    # report containing only numerical series, showing 2 decimal places
    num_report = Report(series1, series2, fmt_func=lambda x:'%.2f' % x)

    # report containing some string and numerical data
    mixed_report = Report(series1, series2, series3)

    # output a csv report suitable for excel to sys.stdout, show masked values as "N/A"
    num_report(delim=', ', mask_rep='N/A')

    # format one column one with 2 decimal places, and column two with 4.
    # Add a sum footer. Write the output to txt_o
    num_report(fmt_func=[(lambda x:'%.2f' % x), (lambda x:'%.4f' % x)],
                 footer_func=ma.sum, footer_label='sum', output=txt_o)

    # create an html table of the data over a specified range.
    # Wrap text in cells to width 10. Output to html_o
    html_o.write("<table>")
    mixed_report(series1, series2, series3, dates=darray,
               delim="</td><td>", prefix="<tr><td>", postfix="</td></tr>",
               wrap_func=wrap_onspace(10, nls='<BR>'), output=html_o)
    html_o.write("</table>")

"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author$)"
__revision__ = "$Revision$"
__date__     = '$Date$'

import sys
import re
import math
import operator, types, copy
from scikits import timeseries as ts
from numpy import ma

__all__ = [
    'Report', 'wrap_onspace', 'wrap_onspace_strict',
    'wrap_always']

class fmt_func_wrapper:
    """wraps a formatting function such that it handles masked values

:IVariables:
    - `fmt_func` : formatting function.
    - `mask_rep` : string to use for masked values
    """
    def __init__ (self, fmt_func, mask_rep):
        if fmt_func is None:
            self.f = str
        else:
            self.f = fmt_func
        self.mr = mask_rep

    def __call__ (self, item):
        "Execute the call behavior."

        if item is ma.masked:
            return self.mr
        else:
            return self.f(item)


_default_options = {
    'dates':None,
    'header_row':None,
    'header_char':'-',
    'header_justify':None,
    'row_char':None,
    'footer_label':None,
    'footer_char':'-',
    'footer_func':None,
    'delim':' | ',
    'justify':None,
    'prefix':'',
    'postfix':'',
    'mask_rep':'--',
    'datefmt':None,
    'fmt_func':str,
    'wrap_func':lambda x:x,
    'col_width':None,
    'nls':'\n',
    'output':sys.stdout,
    'fixed_width':True
}

class Report(object):
    """
    Create a tabular TimeSeries report with dates in the left column. All
    parameters are optional and simply serve as the defaults when calling the
    report. Parameters for calling the report are the exact same as for
    initialization. When calling the report, new options specified will not be
    saved to the instance.

    Parameters
    ----------
    *tseries : {TimeSeries objects}, optional
        The time series objects to generate a report for. Must all be the same
        frequency, but do not need to be aligned.

    dates : {DateArray}, optional
        dates at which values of all the series will be output. If not
        specified, data will be output from the minimum start_date to the
        maximum end_date of all the time series objects

    header_row : {list of strings}, optional
        List of column headers. Specifying the header for the date column is
        optional.

    header_char : {"-", str}, optional
        Character to be used for the row separator line between the header and
        first row of data. Specify None for no separator. This is ignored if
        `header_row` is not specified.

    header_justify : {'left', 'right', 'center', list of these values}, optional
        Determines how headers are justified. If not specified, all headers
        are left justified. If a single string is specified all headers will
        be justified the same way. If a list is specified, each header will be
        justified according to the specification for that header in the list.
        Specifying the justification for the date column header is optional.

    row_char : {str}, optional
        Character to be used for the row separator line between each row of data.
        By default, there is no separator between rows.

    footer_func : {list of aggregation functions or single function}, optional
        A function or list of functions for summarizing each data column in
        the report. For example, ma.sum to get the sum of the column. If a
        list of functions is provided there must be exactly one function for
        each column. Be careful to ensure that the function properly handles
        masked values. Do not specify a function for the Date column.

    footer_char : {'-', str}
        Character to be used for the row separator line between the last row
        of data and the footer. Specify None for no separator. This is ignored
        if `footer_func` is not specified.

    footer_label : {str}
        label for the footer row. This goes at the end of the date column.
        This is ignored if footer_func is not specified.

    justify : {'left', 'right', 'center', list of these values}, optional
        Determines how data are justified in each column. If not specified,
        the date column and string columns are left justified, and everything
        else is right justified. If a single string is specified, all columns
        will be justified the same way. If a list is specified, each column
        will be justified according to the specification for that column in
        the list. Specifying the justification for the date column is
        optional.

    prefix : {str}, optional
        A string prepended to each printed row.

    postfix`: {str}, optional
        A string appended to each printed row.

    mask_rep : {'--', str}, optional
        String used to represent masked values in output

    datefmt : {str}, optional
        Formatting string used for displaying the. If not specified, the
        default format for the given frequency is used.

    fmt_func : {list of functions or single function}, optional
        A function or list of functions for formatting each data column in the
        report. If not specified, str() is simply called on each item. If a
        list of functions is provided, there must be exactly one function for
        each column. Do not specify a function for the Date column, that is
        handled by the datefmt argument.

    wrap_func : {list of functions or single function}, optional
        A function for wrapping text; each element in the column is first
        wrapped by this function. Instances of wrap_onspace,
        wrap_onspace_strict, and wrap_always (which are part of this module)
        work well for this. For example, wrap_func=wrap_onspace(10). If a list
        of functions is specified, each column will be wrapped according to
        the specification for that column in the list. Specify None for
        columns you do not wish to wrap. Specifying a function for the Date
        column is optional

    col_width : {list of integers or single integer}, optional
        Use this to specify a width for all columns (single integer), or each
        column individually (list of integers). The column will be at least as
        wide as col_width, but may be larger if cell contents exceed
        col_width. If specifying a list, you may optionally specify the width
        for the Date column as the first entry

    output : {sys.stdout, file like object}, optional
        file like object to send output to. Only requirement is that it has a
        write method that accepts a single string.

    fixed_width : {True, False}, optional
        If True, columns are fixed width (ie. cells will be padded with spaces
        to ensure all cells in a given column are the same width). If False,
        `col_width` will be ignored and cells will not be padded.
    """

    def __init__(self, *tseries, **kwargs):

        self.options = {}
        self.tseries = None
        if len(tseries) > 0:
            self.tseries = tseries
        self.options = self.__make_dict(**kwargs)

    def __make_dict(self, **kwargs):

        option_dict = {}
        option_list = list(_default_options)

        for x in [kw for kw in kwargs if kw in option_list]:
            option_dict[x] = kwargs.pop(x)

        if len(kwargs) > 0:
            raise KeyError("Unrecognized keyword(s): %s" % (", ".join(kwargs.keys())))

        return option_dict

    def set_options(self, *tseries, **kwargs):
        """
        set new options or modify options in the report

        Parameters
        ----------
        Accepts same parameters as __init__ method of Report class
        """

        if len(tseries) > 0:
            self.tseries = tseries
        self.options.update(self.__make_dict(**kwargs))


    def __call__(self, *tseries, **kwargs):
        """
        generate a report. Parameter values are not saved to the Report instance.

        Parameters
        ----------
        Accepts same parameters as __init__ method of Report class
        """

        option_dict = copy.copy(self.options)
        option_dict.update(self.__make_dict(**kwargs))
        if len(tseries) == 0:
            tseries = self.tseries

        def option(kw):
            return option_dict.get(kw, _default_options[kw])

        dates = option('dates')
        header_row = option('header_row')
        header_char = option('header_char')
        header_justify = option('header_justify')
        row_char = option('row_char')
        footer_label = option('footer_label')
        footer_char = option('footer_char')
        footer_func = option('footer_func')
        delim = option('delim')
        justify = option('justify')
        prefix = option('prefix')
        postfix = option('postfix')
        mask_rep = option('mask_rep')
        datefmt = option('datefmt')
        fmt_func = option('fmt_func')
        wrap_func = option('wrap_func')
        col_width = option('col_width')
        nls=option('nls')
        output=option('output')
        fixed_width=option('fixed_width')

        if header_row is not None:
            has_header=True
            if len(header_row) == len(tseries)+1:
                # label for date column included
                rows = [header_row]
            elif len(header_row) == len(tseries):
                # label for date column not included
                rows = [['']+header_row]
            else:
                raise ValueError("mismatch with number of headers and series")
        else:
            has_header=False
            rows=[]

        if fixed_width:

            def _standardize_justify(userspec):
                if isinstance(userspec, str):
                    # justify all columns the the same way
                    return [userspec for x in range(len(tseries)+1)]
                elif isinstance(userspec, list):
                    if len(userspec) == len(tseries):
                        # justification for date column not included, so set that
                        # to left by default
                        return ['left'] + userspec
                else:
                    raise ValueError("invalid `justify` specification")

            if justify is not None:
                justify = _standardize_justify(justify)
            else:
                # default column justification
                justify = ['left']
                for ser in tseries:
                    if ser.dtype.char in 'SUO': justify.append('left')
                    else: justify.append('right')


            if header_justify is not None:
                header_justify = _standardize_justify(header_justify)
            else:
                # default column justification
                header_justify = ['left' for x in range(len(tseries)+1)]
        else:
            justify = [None for x in range(len(tseries)+1)]

        if datefmt is None:
            def datefmt_func(date): return str(date)
        else:
            def datefmt_func(date): return date.strftime(datefmt)

        if dates is None:
            tseries = ts.align_series(*tseries)
            dates = ts.date_array(start_date=tseries[0].start_date,
                                  end_date=tseries[0].end_date)
        else:
            tseries = ts.align_series(start_date=dates[0], end_date=dates[-1], *tseries)

        if isinstance(fmt_func, list):
            fmt_func = [fmt_func_wrapper(f, mask_rep) for f in fmt_func]
        else:
            fmt_func = [fmt_func_wrapper(fmt_func, mask_rep)]*len(tseries)

        def wrap_func_default(func):
            if func is None: return lambda x:x
            else: return func

        if isinstance(wrap_func, list):
            if len(wrap_func) == len(tseries):
                wrap_func = [lambda x: x] + wrap_func
            wrap_func = [wrap_func_default(func) for func in wrap_func]
        else:
            wrap_func = [wrap_func_default(wrap_func) for x in range(len(tseries)+1)]


        if isinstance(col_width, list):
            if len(col_width) == len(tseries):
                col_width = [None] + col_width
        else:
            col_width = [col_width for x in range(len(tseries)+1)]

        _sd = dates[0]

        for d in dates:
            rows.append([datefmt_func(d)]+[fmt_func[i](ser.series[d - _sd]) for i, ser in enumerate(tseries)])

        if footer_func is not None:
            has_footer=True
            if not isinstance(footer_func, list):
                footer_func = [footer_func]*len(tseries)

            if footer_label is None: footer_label = ['']
            else: footer_label = [footer_label]

            footer_data = []
            has_missing = dates.has_missing_dates()

            for i, ser in enumerate(tseries):
                if footer_func[i] is None:
                    footer_data.append('')
                else:
                    if has_missing: _input = ser[dates]
                    else:           _input = ser.series
                    footer_data.append(fmt_func[i](footer_func[i](_input)))

            rows.append(footer_label + footer_data)
        else:
            has_footer=False


        def rowWrapper(row):
            newRows = [wrap_func[i](item).split('\n') for i, item in enumerate(row)]
            return [[(substr or '') for substr in item] for item in map(None,*newRows)]
        # break each logical row into one or more physical ones
        logicalRows = [rowWrapper(row) for row in rows]
        numLogicalRows = len(logicalRows)
        # columns of physical rows
        columns = map(None,*reduce(operator.add,logicalRows))
        numCols = len(columns)
        colNums = list(range(numCols))

        # get the maximum of each column by the string length of its items
        maxWidths = [max(col_width[i], *[len(str(item)) for item in column])
                        for i, column in enumerate(columns)]

        def getSeparator(char, separate):
            if char is not None and separate:
                return char * (len(prefix) + len(postfix) + sum(maxWidths) + \
                                             len(delim)*(len(maxWidths)-1))
            else:
                return None

        header_separator = getSeparator(header_char, has_header)
        footer_separator = getSeparator(footer_char, has_footer)
        row_separator = getSeparator(row_char, True)

        # select the appropriate justify method
        justify_funcs = {'center':str.center, 'right':str.rjust, 'left':str.ljust,
                          'none':(lambda text, width: text)}

        if has_header and has_footer:
            data_start = 1
            data_end = numLogicalRows-3
        elif has_header:
            data_start = 1
            data_end = numLogicalRows-2
        elif has_footer:
            data_start = 0
            data_end = numLogicalRows-3
        else:
            data_start = 0
            data_end = numLogicalRows-2

        for rowNum, physicalRows in enumerate(logicalRows):
            for row in physicalRows:
                if rowNum == 0 and header_separator:
                    _justify = header_justify
                else:
                    _justify = justify

                output.write(prefix \
                           + delim.join([justify_funcs[str(_justify[colNum]).lower()](str(item),width) for (colNum,item,width) in zip(colNums,row,maxWidths)]) \
                           + postfix + nls)

            if row_separator and (data_start <= rowNum <= data_end):
                output.write(row_separator + nls)
            elif header_separator and rowNum < data_start:
                output.write(header_separator + nls)
            elif footer_separator and rowNum == data_end + 1:
                output.write(footer_separator + nls)


class wrap_onspace(object):
    """
    A callable word-wrap class that preserves existing line breaks and most
    spaces in the text.

    Parameters
    ----------
    width : {int}
        width to wrap at. Won't split up words wider than `width`
    nls : {'\n', str}
        New line separator. Assumes existing line breaks use this new line
        separator as well.
    """

    def __init__(self, width, nls='\n'):
        self.width = width
        self.nls = nls

    def __call__(self, text):

        width = self.width
        nls = self.nls

        def break_or_space(line, word, width):
            temp_idx = (
                len(line[line.rfind(nls)+1:]) + \
                len(word.split(nls,1)[0]) >= width
            )
            if temp_idx: return nls
            else:        return ' '

        return reduce(
            lambda line, word, width=width: '%s%s%s' % (
                line, break_or_space(line, word, width), word
            ),
            text.split(' ')
        )


class wrap_onspace_strict(object):
    """
    A callable word-wrap class similar to wrap_onspace, but enforces the width
    constraint: words longer than width are split.

    Parameters
    ----------
    width : {int}
        width to wrap at. Will split up words wider than `width`
    nls : {'\n', str}
        New line separator. Assumes existing line breaks use this new line
        separator as well.
    """

    def __init__(self, width, nls='\n'):
        self.width = width
        self.nls = nls

    def __call__(self, text):

        width = self.width
        nls = self.nls

        wordRegex = re.compile(r'\S{'+str(width)+r',}')
        return wrap_onspace(
            wordRegex.sub(
                lambda m: wrap_always(m.group(),width, nls=nls),text
            ),
            width,
            nls=nls
        )

class wrap_always(object):
    """
    A callable word-wrap class that wraps text on exactly width characters.
    It doesn't split the text into words.

    Parameters
    ----------
    width : {int}
        width to wrap at.
    nls : {'\n', str}
        New line separator.
    """

    def __init__(self, width, nls='\n'):
        self.width = width
        self.nls = nls

    def __call__(self, text):

        width = self.width
        nls = self.nls
        return nls.join([
            text[width*i:width*(i+1)] \
            for i in xrange(int(math.ceil(1.*len(text)/width)))
        ])

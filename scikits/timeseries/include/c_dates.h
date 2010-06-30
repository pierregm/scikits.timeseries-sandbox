#ifndef C_DATES_H
#define C_DATES_H

#include "c_lib.h"
#include "dtypes.h"

#define HIGHFREQ_ORIG 719163

////////////////////////////////////////////////////



PyObject *DateArray_asfreq(PyObject *, PyObject *);
PyObject *DateArray_getDateInfo(PyObject *, PyObject *);


PyObject *c_dates_now(PyObject *, PyObject *);

PyObject *set_callback_DateFromString(PyObject *, PyObject *);
PyObject *set_callback_DateTimeFromString(PyObject *, PyObject *);

void import_c_dates(PyObject *);

#endif

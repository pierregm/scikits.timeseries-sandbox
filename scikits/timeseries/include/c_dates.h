#ifndef C_DATES_H
#define C_DATES_H

#include "c_lib.h"
// #include "arrayscalars.h"
#include "c_types.h"

// #define HIGHFREQ_ORIG 719163;


////////////////////////////////////////////////////

PyTypeObject DatetimeObject_Type;
#define DatetimeObject_Check(op) PyObject_TypeCheck(op, &DatetimeObject_Type)

void init_metadata_from_unit(ts_metadata *, int);
ts_datetime datetimestruct_to_tsdatetime(ts_metadata *, ts_datetimestruct *);
ts_datetime PyString_to_tsdatetime(ts_metadata *, PyObject *);
ts_datetime PyDatetime_to_tsdatetime(ts_metadata *, PyObject *);


DatetimeObject *DatetimeObject_FromFreqAndValue(int, npy_int64);

PyObject *DateArray_asfreq(PyObject *, PyObject *);
PyObject *DateArray_getdateinfo(PyObject *, PyObject *);
PyObject *DateArray_getdatetime(PyObject *, PyObject *);


PyObject *c_dates_now(PyObject *, PyObject *);

PyObject *set_callback_DateFromString(PyObject *, PyObject *);
PyObject *set_callback_DateTimeFromString(PyObject *, PyObject *);

PyObject * PyDateTime_FromFreqAndValue(npy_int64, ts_metadata *);

void import_c_dates(PyObject *);

#endif

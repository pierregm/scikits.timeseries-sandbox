#ifndef C_DATES_H
#define C_DATES_H

#include "c_lib.h"
// #include "arrayscalars.h"
#include "dtypes.h"

#define HIGHFREQ_ORIG 719163
;
typedef npy_int64 ts_datetime;

typedef struct {
    PyObject_HEAD;
    int freq; /* frequency of date */
    int unit;
    ts_datetime value; /* integer representation of date */
} DatetimeObject;


////////////////////////////////////////////////////

PyTypeObject DatetimeObject_Type;
#define DatetimeObject_Check(op) PyObject_TypeCheck(op, &DatetimeObject_Type)

ts_datetime datetimestruct_to_tsdatetime(int, ts_datetimestruct *);
ts_datetime string_to_tsdatetime(int, PyObject *);
ts_datetime PyDatetime_to_tsdatetime(int, PyObject *);


DatetimeObject *DatetimeObject_FromFreqAndValue(int, npy_int64);

PyObject *DateArray_asfreq(PyObject *, PyObject *);
PyObject *DateArray_getdateinfo(PyObject *, PyObject *);
PyObject *DateArray_getdatetime(PyObject *, PyObject *);


PyObject *c_dates_now(PyObject *, PyObject *);

PyObject *set_callback_DateFromString(PyObject *, PyObject *);
PyObject *set_callback_DateTimeFromString(PyObject *, PyObject *);

PyObject * PyDateTime_FromFreqAndValue(npy_int64, int);

void import_c_dates(PyObject *);

#endif

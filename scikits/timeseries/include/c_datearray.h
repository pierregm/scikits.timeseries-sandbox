#ifndef C_DATEARRAY_H
#define C_DATEARRAY_H

#include "c_lib.h"
#include <numpy/ndarrayobject.h>


typedef struct {
    int has_dups;
    int has_missing;
    int is_chrono;
} ts_timestatus;

typedef struct {
    PyArrayObject base;
    ts_metadata obmeta;
    ts_timestatus status;
} DatetimeArrayObject;

#define DatetimeArray_Check(o)                              \
    PyObject_TypeCheck((PyObject*)(o), &DatetimeArray_Type)


void import_c_datearray(PyObject *);

#endif

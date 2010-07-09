#ifndef C_DATEARRAY_H
#define C_DATEARRAY_H

#include "c_lib.h"
#include <numpy/ndarrayobject.h>


typedef struct {
    int has_dups, has_missing, is_chrono;
} timestatus;

typedef struct {
    PyArrayObject base;
//    PyObject *cached_vals;
    int unit;
    npy_int64 timestep;
    timestatus status;
} DateTimeArrayObject;

#define DateTimeArray_Check(o)                              \
    PyObject_TypeCheck((PyObject*)(o), &DateTimeArray_Type)


void import_c_datearray(PyObject *);

#endif

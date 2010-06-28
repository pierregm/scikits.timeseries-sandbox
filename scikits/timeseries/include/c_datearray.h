#ifndef C_DATEARRAY_H
#define C_DATEARRAY_H

#include "c_lib.h"
#include <numpy/ndarrayobject.h>


typedef struct {
    PyArrayObject array;
//    PyObject *cached_vals;
    long unit;
} DateTimeArray;

#define DateTimeArray_Check(o)                               \
    PyObject_TypeCheck((PyObject*)(o), &DateTimeArray_Type)


void import_c_datearray(PyObject *);

#endif

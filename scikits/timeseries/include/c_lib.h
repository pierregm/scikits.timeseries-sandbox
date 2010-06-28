#ifndef C_LIB_H
#define C_LIB_H

#include <Python.h>
#include <structmember.h>
#include "arrayobject.h"

/* c_lib defines generic functions that aren't inherently time series/date
specific but are needed in various parts of the module. */

/* The 2 functions below are borrowed from mx.DateTime version
   2.0.6, and hence this code is subject to the terms of the egenix public
   license version 1.0.0
*/
#define Py_AssertWithArg(x,errortype,errorstr,a1) {if (!(x)) {PyErr_Format(errortype,errorstr,a1); goto onError;}}
#define Py_Error(errortype,errorstr) {PyErr_SetString(errortype,errorstr); goto onError;}

/**/


#define INT_ERR_CODE -999

#define MEM_CHECK(item) if (item == NULL) { return PyErr_NoMemory(); }
#define NULL_CHECK(item) if (item == NULL) { return NULL; }
#define ERR_CHECK(result) if ((result) == INT_ERR_CODE) return NULL

char *str_uppercase(char *);
char *str_replace(const char*, const char*, const char*);

PyObject *np_add(PyObject*, PyObject*);
PyObject *np_multiply(PyObject*, PyObject*);
PyObject *np_subtract(PyObject*, PyObject*);
PyObject *np_sqrt(PyObject*);
int np_greater(PyObject*, PyObject*);
int np_greater_equal(PyObject*, PyObject*);

PyObject *set_callback(PyObject*, PyObject**);

 /* Error Exception objects */
PyObject *DateCalc_Error;
PyObject *DateCalc_RangeError;

void import_c_lib(PyObject*);

#endif


#include <unistd.h>
#define DEBUGPRINTF(fmt, args...) fprintf(stderr, "\n>>> " fmt, ## args);


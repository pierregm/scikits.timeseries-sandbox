#include "c_freqs.h"
#include "c_dates.h"
#include "c_datearray.h"
#include <datetime.h>
#include <time.h>

#include "c_lib.h"

static PyTypeObject DateTimeArray_Type;

static PyObject *
DateTimeArray_new(PyTypeObject *cls, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"object", "unit", 0};

    PyObject *obj;
    PyObject *arr = NULL;
    PyObject *unit = NULL;
//    PyObject *cached_vals = NULL;
    DateTimeArray *self;

    if(!PyArg_ParseTupleAndKeywords(args,kw,"OO",kwlist,
                                    &obj,
                                    &unit))
        return NULL;

    arr = PyArray_FROM_O(obj);
    if(arr == NULL)
        return NULL;
    DEBUGPRINTF("We have an array...");


    self = (DateTimeArray*)PyObject_CallMethod(arr, "view", "O", cls);
//    self = PyArray_View(&arr, NULL, &DateTimeArray_Type);
    Py_DECREF(arr);
    if(self == NULL)
        return NULL;

    if (unit == NULL){
        unit = PyInt_FromLong(FR_UND);
    }
//    self->cached_vals = cached_vals;

    self->unit = check_freq(unit);
    if (self->unit == -1)
        goto onError;

    return (PyObject *)self;

 onError:
    Py_XDECREF((PyObject*)self);
    return NULL;
}

static void
DateTimeArray_dealloc(DateTimeArray *self)
{
//    DEBUGPRINTF("Dropping cache");
//    Py_XDECREF(self->cached_vals);
    DEBUGPRINTF("Dropping object");
    self->array.ob_type->tp_free((PyObject*)self);
}

static PyObject*
DateTimeArray_finalize(DateTimeArray *self, PyObject *args)
{
    DateTimeArray *context;
    if(PyArg_ParseTuple(args,"O",&context))
    {
        if (DateTimeArray_Check(context)){
            DEBUGPRINTF("in context from DTA");
//            self->cached_vals = context->cached_vals;
            self->unit = context->unit;
        } else {
            DEBUGPRINTF("in context from scratch");
//            self->cached_vals = NULL;
            self->unit = FR_UND;
//            Py_XINCREF(self->cached_vals);
        };
//        Py_XINCREF(self->cached_vals);
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}





static PyObject *
DateTimeArray_TryNew(PyObject *list, PyObject *unit)
{
    DEBUGPRINTF("_TryNew got called");
    PyObject *args = PyTuple_Pack((unit != NULL) ? 2:1, list, unit);
    PyObject *res = DateTimeArray_new(&DateTimeArray_Type,args,NULL);
    Py_DECREF(args);
    PyErr_Clear();
    return res;
}

static PyObject *
//DateTimeArray_freqstr(DateTimeArray *self, void *closure) {
DateTimeArray_freqstr(DateTimeArray *self) {
    PyObject *key = PyInt_FromLong(self->unit);
    PyObject *freq_aliases = PyDict_GetItem(freq_dict, key);
    PyObject *main_alias = PyTuple_GET_ITEM(freq_aliases, 0);
    Py_DECREF(key);
    return main_alias;
}




static PyMemberDef DateTimeArray_members[] = {
    {"unit", T_INT, offsetof(DateTimeArray, unit), 0,
     "frequency"},
//     {"cached_vals", T_OBJECT_EX, offsetof(DateTimeArray, cached_vals), 0,
//      "cached_values"},
//    {"freq", T_INT, offsetof(DateObject, freq), 0,
//     "frequency"},
//    {"value", T_INT, offsetof(DateObject, value), 0,
//     "integer representation of the Date"},
    {NULL}  /* Sentinel */
};


/*
 * PROPERTIES
 */

static int
DateTimeArray_ReadOnlyErr(DateTimeArray *self, PyObject *value, void *closure) {
   PyErr_SetString(PyExc_AttributeError, "Cannot set read-only property");
   return -1;
};

static PyGetSetDef DateTimeArray_getseters[] = {
    {"freqstr", (getter)DateTimeArray_freqstr, (setter)DateTimeArray_ReadOnlyErr,
     "Returns the string representation of frequency.", NULL},
    {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/*
 * METHODS
 */

static PyMethodDef DateTimeArray_methods[] = {
    { "__array_finalize__", (PyCFunction)DateTimeArray_finalize, METH_VARARGS,
      "" },
    {0}
};


static PyTypeObject DateTimeArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
    "timeseries.DateTimeArray",                      /* tp_name */
    sizeof(DateTimeArray),                  /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)DateTimeArray_dealloc,          /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "DateTimeArray",                          /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,     /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    DateTimeArray_methods,                    /* tp_methods */
    DateTimeArray_members,                    /* tp_members */
    DateTimeArray_getseters,                  /* tp_getset */
    0,                            /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    DateTimeArray_new,                        /* tp_new */
};







/*
 * */
void import_c_datearray(PyObject *m)
{

    import_array();
    PyDateTime_IMPORT;

    DateTimeArray_Type.tp_base = &PyArray_Type;
    if (PyType_Ready(&DateTimeArray_Type) < 0)
        return;


    Py_INCREF(&DateTimeArray_Type);
    PyModule_AddObject(m, "DateTimeArray", (PyObject *)(&DateTimeArray_Type));
}



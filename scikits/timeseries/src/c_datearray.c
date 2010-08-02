#include "c_freqs.h"
#include "c_dates.h"
#include "c_datearray.h"
#include <datetime.h>
#include <time.h>

#include "c_lib.h"


static PyTypeObject DatetimeArray_Type;

int PyArray_TS_DATETIME;

#define TS_METADATA_DTSTR "timeunit"

typedef struct {
   PyObject_HEAD;
   ts_datetime obval;
   ts_metadata obmeta;
} DatetimeScalarObject;

NPY_NO_EXPORT PyTypeObject DatetimeArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
//    "timeseries.datetime" _THIS_SIZE,                  /* tp_name*/
    "timeseries.datetime",                  /* tp_name*/
    sizeof(DatetimeScalarObject),               /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

#undef _THIS_SIZE
/**/




#if PY_VERSION_HEX >= 0x02070000
#define get_metadata_from_descr(descr)  \
    ((descr->metadata == NULL) ? \
     NULL :                                       \
    ((ts_metadata *)(PyCapsule_GetPointer(                   \
                     PyDict_GetItemString(descr->metadata, TS_METADATA_DTSTR), \
                     NULL))))
#else
#define get_metadata_from_descr(descr)  \
    ((descr->metadata == NULL) ? \
     NULL :                                       \
     ((ts_metadata *)(PyCObject_AsVoidPtr(                    \
                      PyDict_GetItemString(descr->metadata, TS_METADATA_DTSTR)))))
#endif


#define get_descr(self) ( ((PyArrayObject *)self)->descr )
#define get_metadata_from_array(self) (get_metadata_from_descr(get_descr(self)))
//#define get_timestep(self) (get_metadata_from_array(self)->timestep)


#define TS_METADATA_DTSTR "timeunit"

//----------------------------------------------------------------------------
/* from private/npy_3kcompat.h */
#if PY_VERSION_HEX >= 0x02070000

static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *))
{
    PyObject *ret = PyCapsule_New(ptr, NULL, dtor);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static void
simple_capsule_dtor(PyObject *cap)
{
    PyArray_free(PyCapsule_GetPointer(cap, NULL));
}

#else

static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(void *))
{
    return PyCObject_FromVoidPtr(ptr, dtor);
}

static void
simple_capsule_dtor(void *ptr)
{
    PyArray_free(ptr);
}

#endif
/**/


#include "noprefix.h"

static void
init_datetime_descr(PyArray_Descr *descr)
{
    ts_metadata *dt_data;
    PyObject *cobj;

    dt_data = _pya_malloc(sizeof(ts_metadata));
    dt_data->unit = FR_UND;
    dt_data->timestep = 1;
    dt_data->period_end_at = 0;
    dt_data->periods_per_day = -1;
    dt_data->secs_per_period = -1;
    dt_data->convert_to_start = 0;

/* FIXME
 * There is no error check here and no way to indicate an error
 * until the metadata turns up NULL.
 */
    cobj = NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
    descr->metadata = PyDict_New();
    PyDict_SetItemString(descr->metadata, TS_METADATA_DTSTR, cobj);
    Py_DECREF(cobj);

}

static void
update_datetime_metadata(PyArray_Descr *descr, ts_metadata *meta) {
    PyObject *cobj;
    cobj = NpyCapsule_FromVoidPtr((void *)meta, simple_capsule_dtor);
    descr->metadata = PyDict_New();
    PyDict_SetItemString(descr->metadata, TS_METADATA_DTSTR, cobj);
    Py_DECREF(cobj);
}


//-----------------------------------------------------------------------------

static PyObject *
DatetimeArray_new(PyTypeObject *cls, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"object", "unit", 0};

    PyObject *obj;
    PyArrayObject *arr = NULL;
    PyObject *unit = NULL;
    PyObject *timestep = NULL;
    DatetimeArrayObject *self;
    PyArray_Descr *descr;

    if(!PyArg_ParseTupleAndKeywords(args,kw,"OO",kwlist,
                                    &obj,
                                    &unit,&timestep))
        return NULL;

    arr = (PyArrayObject *)PyArray_FROM_O(obj);
    if(arr == NULL)
        return NULL;
//    DEBUGPRINTF("We have an array...");

    descr = PyArray_DescrFromType(PyArray_INT64);
    if (descr == NULL)
        return NULL;
    Py_INCREF(descr);
    init_datetime_descr(descr);
//    DEBUGPRINTF("datetime initialize %i", descr->type_num);
//    DEBUGPRINTF("PyArray_DescrCheck(obj):%i",PyArray_DescrCheck((PyObject *)descr));



    // DEBUGPRINTF("type_num; %i", (long)(&descr->type_num));
    // DEBUGPRINTF("flags: %i", (long)(&descr->flags));
    // descr = &TS_DATETIME_Descr;
    self = (DatetimeArrayObject *)PyArray_NewFromDescr(&DatetimeArray_Type,
                                                       arr->descr,
                                                       arr->nd, arr->dimensions,
                                                       arr->strides,
                                                       arr->data,
                                                       arr->flags,
                                                       (PyObject *)arr);
    if(self == NULL)
        return NULL;
    Py_INCREF(arr);
    PyArray_BASE(self) = (PyObject *)arr;
//    DEBUGPRINTF("Setting the dtype");
    if (PyObject_SetAttrString((PyObject *)self, "dtype", (PyObject *)descr) < 0) {
        Py_DECREF(self);
        return NULL;
    }

    ts_metadata *obmeta = get_metadata_from_descr(descr);

    if (unit == NULL){
        unit = PyInt_FromLong(FR_UND);
    }
//    self->cached_vals = cached_vals;
//    DEBUGPRINTF("Setting the unit");
    int u = check_freq(unit);
    if (u == -1)
        goto onError;
    init_metadata_from_unit(obmeta, u);

//    DEBUGPRINTF("Setting the timestep");
    if (timestep == NULL)
        obmeta->timestep = 1;
    else
        obmeta->timestep = PyInt_AsLong(timestep);

    self->obmeta.unit = obmeta->unit;
    self->obmeta.timestep = obmeta->timestep; // num in numpy...
    self->obmeta.period_end_at = obmeta->period_end_at; //
    self->obmeta.periods_per_day = obmeta->periods_per_day;
    self->obmeta.secs_per_period = obmeta->secs_per_period;
    self->obmeta.convert_to_start = obmeta->convert_to_start;
//    timestatus status;
//    status.has_dups = -1;
//    status.has_missing = -1;
//    status.is_chrono = -1;
//    self->status = status;

    return (PyObject *)self;

 onError:
    DEBUGPRINTF("Dropping it..");
    Py_XDECREF((PyObject*)self);
    return NULL;
}

static void
DatetimeArray_dealloc(DatetimeArrayObject *self)
{
//    DEBUGPRINTF("Dropping cache");
//    Py_XDECREF(self->cached_vals);
    DEBUGPRINTF("Dropping object");
    self->base.ob_type->tp_free((PyObject*)self);
}

static PyObject*
DatetimeArray_finalize(DatetimeArrayObject *self, PyObject *args)
{
    DatetimeArrayObject *context;
    if(PyArg_ParseTuple(args,"O",&context))
    {
        if (DatetimeArray_Check(context)){
            DEBUGPRINTF("in context from DTA");
            ts_metadata *meta_context = get_metadata_from_array(context);
            update_datetime_metadata(get_descr(self), meta_context);
        } else {
            DEBUGPRINTF("in context from scratch");
            init_datetime_descr(get_descr(self));
        };
        self->status.is_chrono = -1;
        self->status.has_dups = -1;
        self->status.has_missing = -1;
        ts_metadata *selfmeta = get_metadata_from_array(self);
        DEBUGPRINTF("self.unit is %i", selfmeta->unit);
    }
    PyErr_Clear();
    DEBUGPRINTF("Returning...");
    Py_RETURN_NONE;
}



// static PyObject *
// DateTimeArray_TryNew(PyObject *list, PyObject *unit)
// {
//     DEBUGPRINTF("_TryNew got called");
//     PyObject *args = PyTuple_Pack((unit != NULL) ? 2:1, list, unit);
//     PyObject *res = DateTimeArray_new(&DateTimeArray_Type,args,NULL);
//     Py_DECREF(args);
//     PyErr_Clear();
//     return res;
// }

static int
_get_unit_from_descr(PyArray_Descr *descr) {
    ts_metadata *meta = get_metadata_from_descr(descr);
    return meta->unit + meta->period_end_at;
}
static int
_get_unit_from_array(DatetimeArrayObject *self) {
    ts_metadata *meta = get_metadata_from_descr(((PyArrayObject *)self)->descr);
    return meta->unit + meta->period_end_at;
}
static PyObject *
DatetimeArray_unit(DatetimeArrayObject *self){
    int unit = _get_unit_from_array(self);
    return PyInt_FromLong(unit);
}
static PyObject *
DatetimeArray_timestep(DatetimeArrayObject *self){
    PyArray_Descr *descr = ((PyArrayObject *)self)->descr;
    ts_metadata *meta = get_metadata_from_descr(descr);
    return PyInt_FromLong(meta->timestep);
}
static PyObject *
DatetimeArray_freqstr(DatetimeArrayObject *self) {
    PyObject *key = DatetimeArray_unit(self);
    PyObject *freq_aliases = PyDict_GetItem(freq_dict, key);
    PyObject *main_alias = PyTuple_GET_ITEM(freq_aliases, 0);
    Py_DECREF(key);
    return main_alias;
}

static PyObject *
DatetimeArray_steps(DatetimeArrayObject *self){
    PyArrayObject *steps=NULL;
    PyArrayIterObject *self_iter, *steps_iter;
    npy_intp size;

    size = PyArray_SIZE(self) - 1;

    steps = (PyArrayObject*)PyArray_ZEROS(1,
                                          &size,
                                          PyArray_INT64, 0);
    NULL_CHECK(steps);

    steps_iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)steps);
    if (steps_iter == NULL) {
        DEBUGPRINTF("can't initialize steps_iter");
        goto fail;
    };
    self_iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (self_iter == NULL) {
        DEBUGPRINTF("can't initialize self_iter");
        goto fail;
    };
    PyObject *val=NULL, *prev=NULL, *diff=NULL;
    prev = PyArray_GETITEM(self, self_iter->dataptr);
    PyArray_ITER_NEXT(self_iter);
    while (steps_iter->index < steps_iter->size) {
        val = PyArray_GETITEM(self, self_iter->dataptr);
        diff = PyNumber_Subtract(val, prev);
        PyArray_SETITEM(steps, steps_iter->dataptr, diff);
        PyArray_ITER_NEXT(self_iter);
        PyArray_ITER_NEXT(steps_iter);
        prev = val;
    };
    Py_DECREF(self_iter);
    Py_DECREF(steps_iter);
    Py_DECREF(prev);
    Py_DECREF(val);
    Py_DECREF(diff);
    return (PyObject *)steps;

 fail:
    DEBUGPRINTF("DatetimeArray.steps: Oops...");
    Py_DECREF(steps);
    return NULL;
}



static int
DatetimeArray_check_status(DatetimeArrayObject *self)
{
    PyArrayIterObject *self_iter;
    npy_int64 timestep, diff;
    int is_chrono = 1, has_dups=0, has_missing=0;

//    timestep = PyInt_FromLong(DatetimeArray_timestep(self));
    timestep = get_metadata_from_array(self)->timestep;
    self_iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (self_iter == NULL) {
        return -1;
    }

    PyObject *val=NULL, *prev=NULL, *odiff=NULL;
    prev = PyArray_GETITEM(self, self_iter->dataptr);
    PyArray_ITER_NEXT(self_iter);
    while (self_iter->index < self_iter->size) {
        val = PyArray_GETITEM(self, self_iter->dataptr);
        odiff = PyNumber_Subtract(val, prev);
        diff = PyInt_AsLong(odiff);
        if (diff < 0)
            is_chrono = 0;
        else if (diff == 0)
            has_dups = 1;
        else if (diff > timestep)
            has_missing = 1;
        if (has_dups && has_missing)
            break;
        PyArray_ITER_NEXT(self_iter);
        prev = val;
    }
    Py_DECREF(self_iter);
    Py_DECREF(odiff);
    Py_DECREF(prev);
    Py_DECREF(val);
    // Set the status
    self->status.is_chrono = is_chrono;
    self->status.has_dups = has_dups;
    self->status.has_missing = has_missing;
    return 0;
}
static PyObject *
DatetimeArray_has_dups(DatetimeArrayObject *self)
{
    if (self->status.has_dups == -1)
        if (DatetimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.has_dups == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DatetimeArray_has_missing(DatetimeArrayObject *self)
{
    if (self->status.has_missing == -1)
        if (DatetimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.has_missing == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DatetimeArray_is_chrono(DatetimeArrayObject *self)
{
    if (self->status.is_chrono == -1)
        if (DatetimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.is_chrono == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DatetimeArray_is_full(DatetimeArrayObject *self)
{
    if (self->status.has_dups == -1)
        if (DatetimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.has_dups)
        Py_RETURN_FALSE;
    if (self->status.has_missing)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DatetimeArray_is_valid(DatetimeArrayObject *self)
{
    ts_timestatus status = self->status;
    if (status.has_dups == -1)
        if (DatetimeArray_check_status(self) < 0)
            return NULL;
    if (status.has_missing)
        Py_RETURN_FALSE;
    if (status.has_dups)
        Py_RETURN_FALSE;
    if (! status.is_chrono)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyMemberDef DatetimeArray_members[] = {
//     {"cached_vals", T_OBJECT_EX, offsetof(DateTimeArray, cached_vals), 0,
//      "cached_values"},
    {NULL}  /* Sentinel */
};


static char *
DEBUGGETTYPE(PyObject *obj){
    char *type_str;
    PyObject *type_repr, *obj_type;
    obj_type = PyObject_Type(obj);
    type_repr = PyObject_Repr(obj_type);
    type_str = PyString_AsString(type_repr);
//    DEBUGPRINTF("get_tsdatetime_from_object got %s [%i]", type_str, meta->unit);
    Py_DECREF(obj_type);
    Py_DECREF(type_repr);
    return type_str;
}


static ts_datetime
get_tsdatetime_from_object(ts_metadata *meta, PyObject *date){
    ts_datetime value;
    DEBUGPRINTF("get_tsdatetime_from_object type %s", DEBUGGETTYPE(date));
    //
    if (PyString_Check(date)) {
        value = PyString_to_tsdatetime(meta, date);
        DEBUGPRINTF("get_tsdatetime_from_object.from string: %ld", value);
    }
    else if (PyDateTime_Check(date) || PyDate_Check(date)) {
        value = PyDatetime_to_tsdatetime(meta, date);
        DEBUGPRINTF("get_tsdatetime_from_object.from datetime.datetime: %ld", value);
    }
    else if (DatetimeObject_Check(date)) {
        value = ((DatetimeObject *)date)->obval;
        DEBUGPRINTF("get_tsdatetime_from_object.from tsdatetime: %ld", value);
    }
    else if (PyInt_Check(date) || PyLong_Check(date) || PyFloat_Check(date)) {
        value = (ts_datetime)PyInt_AsLong(date);
        DEBUGPRINTF("get_tsdatetime_from_object.from number: %ld", value);
    }
    else {
        value = -1;
    }
    return value;
}



//static PyObject *
//DatetimeArray_single_date_to_index(DatetimeArrayObject *self, PyObject *date){
//    ts_datetime value;
//    Py_ssize_t i;
//    int nd = ((PyArrayObject *)self)->nd;
//
//
//    ts_metadata *meta = get_metadata_from_array(self);
//    ts_timestatus status = self->status;
//    int is_valid = ((! status.has_missing) && (! status.has_dups) && (status.is_chrono));
//
//    if (is_valid) {
//        npy_intp size = PyArray_SIZE(self);
//        float timestep = (float)(meta->timestep);
//        void *zero = PyArray_GETPTR1(self, 0);
//        PyObject *pyfirst = PyArray_GETITEM((PyObject *)self, zero);
//        if (pyfirst == NULL)
//            DEBUGPRINTF("Dang...");
//        ts_datetime first = PyInt_AsLong(pyfirst);
//        Py_DECREF(pyfirst);
//        value = get_tsdatetime_from_object(meta, date);
//        DEBUGPRINTF("got value: %ld [%ld] [%ld]", value, first, timestep * size + first);
//        if ((value < first) || (value > timestep * size + first)) {
//            return NULL;
//        }
//        else {
//            DEBUGPRINTF("got: %ld", (value - first) / timestep);
//            return PyInt_FromLong((value - first) / timestep);
//        }
//    }
//    else {
//        PyArrayIterObject *itr = NULL;
//        PyObject *indexlist = PyList_New(0);
//        npy_int k;
//        ts_datetime current;
//
//        value = get_tsdatetime_from_object(meta, date);
//        DEBUGPRINTF("DatetimeArray_single_date_to_index invalid got value: %ld", value);
//
//        itr = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
//        if (itr == NULL)
//            return NULL;
//        k = 0;
//        //
//        if (nd == 1) {
//            while (itr->index < itr->size) {
//                current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
//                if (current == value) {
//                    DEBUGPRINTF("DatetimeArray_single_date_to_index invalid got one at %ld", k);
//                    PyList_Append(indexlist, PyInt_FromLong(k));
//                }
//                k++;
//                PyArray_ITER_NEXT(itr);
//            }
//            Py_DECREF(itr);
//        }
//        else {
//            while (itr->index < itr->size) {
//                current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
//                if (current == value) {
//                    DEBUGPRINTF("DatetimeArray_single_date_to_index invalid got many");
//                    PyList_Append(indexlist, PyArray_IntTupleFromIntp(nd, itr->coordinates));
//                }
//                PyArray_ITER_NEXT(itr);
//            }
//            Py_DECREF(itr);
//        }
//        if (PyList_Size(indexlist) == 0) {
//            Py_DECREF(indexlist);
//            return NULL;
//        }
//        return indexlist;
//    };
//    return NULL;
//}

static PyObject *
DatetimeArray_single_date_to_index(DatetimeArrayObject *self, PyObject *date){
    ts_datetime value, current;
    intp count=0, i, size;
    int nd = ((PyArrayObject *)self)->nd, j;

    PyArrayIterObject *itr = NULL;
    PyObject *result = NULL, *item;
    intp *dptr[MAX_DIMS];

    itr = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (itr == NULL)
        return NULL;

    /*Count the valid elements*/
    ts_metadata *meta = get_metadata_from_array(self);
    value = get_tsdatetime_from_object(meta, date);
    DEBUGPRINTF("DatetimeArray_single_date_to_index invalid got value: %ld", value);

    size = itr->size;
    for (i = 0; i < size; i++) {
        current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
        if (current == value) {
            DEBUGPRINTF("DatetimeArray_single_date_to_index invalid got one at %ld", &count);
            count++;
        }
        PyArray_ITER_NEXT(itr);
    }

    PyArray_ITER_RESET(itr);
    result = PyTuple_New(nd);
    if (result == NULL)
        goto fail;
    for (j = 0; j < nd; j++) {
        item = PyArray_New(Py_TYPE(self), 1, &count,
                           PyArray_INTP, NULL, NULL, 0, 0,
                           (PyObject *)self);
        if (item == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(result, j, item);
        dptr[j] = (intp *)PyArray_DATA(item);
    }
    if (nd == 1) {
        for (i = 0; i < size; i++){
            current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
            if (current == value)
                *(dptr[0])++ = i;
            PyArray_ITER_NEXT(itr);
        }
    }
    else {
        itr->contiguous = 0;
        for (i = 0; i < size; i++){
            current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
            if (current == value) {
                for (j = 0; j < nd; j++)
                    *(dptr[j])++ = itr->coordinates[j];
            }
            PyArray_ITER_NEXT(itr);
        }
    }
    Py_DECREF(itr);
    return result;
 fail:
    Py_XDECREF(result);
    Py_XDECREF(itr);
    return NULL;
}



static PyObject *
DatetimeArray_date_to_index(DatetimeArrayObject *self, PyObject *dateargs){
    PyObject *result=NULL, *date;
    ts_datetime value;
    Py_ssize_t i;
    int nd = ((PyArrayObject *)self)->nd;

    /* Make sure we have at least 1 argument */
    Py_ssize_t nbargs = PyObject_Length(dateargs);
    if (nbargs < 1) {
        DEBUGPRINTF("At least one argument");
        return NULL;
    }

    ts_metadata *meta = get_metadata_from_array(self);
    ts_timestatus status = self->status;
    int is_valid = ((! status.has_missing) && (! status.has_dups) && (status.is_chrono));

    result = PyList_New(0);

    if (is_valid) {
        npy_intp size = PyArray_SIZE(self);
        float timestep = (float)(meta->timestep);
        void *zero = PyArray_GETPTR1(self, 0);
        PyObject *pyfirst = PyArray_GETITEM((PyObject *)self, zero);
        if (pyfirst == NULL) {
            DEBUGPRINTF("Dang...");
            return NULL;
        }
        ts_datetime first = PyInt_AsLong(pyfirst);
        PyObject *index = NULL;
        for (i=0; i < nbargs; i++){
            date = PyTuple_GetItem(dateargs, i);
            value = get_tsdatetime_from_object(meta, date);
            DEBUGPRINTF("got value: %ld [%ld] [%ld]", value, first, timestep * size + first);
            if ((value < first) || (value > timestep * size + first)) {
                index = Py_None;
                DEBUGPRINTF("missed");
            }
            else {
                DEBUGPRINTF("got: %ld", (value - first) / timestep);
                index = PyInt_FromLong((value - first) / timestep);
            }
            PyList_Append(result, index);
        }
        DEBUGPRINTF("DatetimeArray_date_to_index done");
        Py_DECREF(index);
        Py_DECREF(pyfirst);
    }
    else {
        PyArrayIterObject *itr = NULL;
        PyObject *tmp = NULL;
        long k;
        int empty;
        ts_datetime current;
        for (i=0; i < nbargs; i++){
            date = PyTuple_GetItem(dateargs, i);
            value = get_tsdatetime_from_object(meta, date);
            DEBUGPRINTF("got value: %ld", value);

            tmp = PyList_New(0);
            itr = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
            if (itr == NULL)
                return NULL;
            k = 0;
            empty = 1;
            int nd = ((PyArrayObject *)self)->nd;
            while (itr->index < itr->size) {
                current = (ts_datetime)PyInt_AsLong(PyArray_GETITEM(self, itr->dataptr));
                if (current == value) {
//                    PyList_Append(tmp, PyInt_FromLong(k));
                    PyObject *coords = PyObject_GetAttrString(itr, "coords");
//                    PyList_Append(tmp, PyArray_IntTupleFromIntp(nd, itr->coordinates));
                    PyList_Append(tmp, coords);
                    empty = 0;
                }
                k++;
                PyArray_ITER_NEXT(itr);
            }
            if (empty)
                tmp = Py_None;
            PyList_Append(result, tmp);
        };
        Py_DECREF(itr);
        Py_DECREF(tmp);
    };
    if (nbargs == 1)
        return PyList_GetItem(result, 0);
    return result;
}
//
//

//def date_to_index(self, dates):
//    """
//Returns the index corresponding to one given date, as an integer.
//    """
//    values = self.__array__()
//    # Transform a string into a Date
//    if isinstance(dates, basestring):
//        dates = Date(self._unit, dates)
//    # Just one date ?
//    if isinstance(dates, Date):
//        _val = dates.value
//        if _val not in values:
//            raise IndexError("Date '%s' is out of bounds" % dates)
//        if self.is_valid():
//            return (_val - values[0]) / self._timestep
//        else:
//            return np.where(values == _val)[0][0]
//    #
//    _dates = date_array(dates, freq=self.freq).__array__()
//    if self.is_valid():
//        indx = (_dates - values[0]) / self._timestep
//        err_cond = (indx < 0) | (indx > self.size)
//        if err_cond.any():
//            err_indx = np.compress(err_cond, _dates)[0]
//            err_msg = "Date '%s' is out of bounds '%s' <= date <= '%s'"
//            raise IndexError(err_msg % (err_indx, self[0], self[-1]))
//        return indx
//    vals = values.tolist()
//    indx = np.array([vals.index(d) for d in _dates])
//    #
//    return indx



static PyObject *
DatetimeArray_getitem(DatetimeArrayObject *self, PyObject *op)
{
//    int reset_full=1, keep_chrono=0;
//    DEBUGPRINTF("in __getitem__ w %s", DEBUGGETTYPE(op));
    PyObject *idx;

    if (DatetimeObject_Check(op) || PyString_Check(op) || PyDateTime_Check(op)) {
        if (DatetimeObject_Check(op)) {
            DEBUGPRINTF("index is Date");
        }
        else if (PyString_Check(op)) {
            DEBUGPRINTF("index is string");
        }
        else if (PyDateTime_Check(op)) {
            DEBUGPRINTF("index is datetime");
        };
//        PyObject *dateargs = PyTuple_New(1);
//        Py_INCREF(op);
//        PyTuple_SetItem(dateargs, 0, op);
        idx = DatetimeArray_single_date_to_index(self, op);
        if (idx == NULL) {
            PyErr_SetString(PyExc_IndexError, "date out of bounds");
            return NULL;
        }
    }
    else {
        idx = op;
    }

    PyObject *r, *result;
//    r = PyArray_GETITEM(self, indx);
    r = ((PyArrayObject *)self)->ob_type->tp_base->tp_as_mapping->mp_subscript((PyObject *)self, idx);
    if (r == NULL) {
        return NULL;
    }
//    DEBUGPRINTF("r is %s", DEBUGGETTYPE(r));
    ts_datetime obval;
    if (PyArray_IsScalar(r, Integer)) {
        int unit = _get_unit_from_descr(get_descr(self));

        obval = (ts_datetime)(PyInt_AsLong(r));
//        DEBUGPRINTF("index is integer, unit is %i, value is %i", unit, obval);
        result = (PyObject *)DatetimeObject_FromFreqAndValue(unit, PyInt_AsLong(r));
        Py_DECREF(r);
    }
    else {
//        DEBUGPRINTF("index is %s", DEBUGGETTYPE(r));
        result = r;
//        ts_timestatus S=((DatetimeArrayObject *)result)->status;
//        DEBUGPRINTF("chrono? %i - dups? %i - missing? %i", S.is_chrono, S.has_dups, S.has_missing);
    }
    Py_DECREF(idx);
    return result;
}


/*
def __getitem__(self, indx):
    reset_full = True
    keep_chrono = False
    # Determine what kind of index is used
    if isinstance(indx, Date):
        # indx = self.find_dates(indx)
        # indx = int(self.find_dates(indx)[0])
        indx = self.date_to_index(indx)
        reset_full = False
    elif isinstance(indx, slice):
        keep_chrono = True
    elif np.asarray(indx).dtype.kind == 'O':
        try:
            indx = self.find_dates(indx)
        except AttributeError:
            pass

    # Select the data
    r = ndarray.__getitem__(self, indx)
    # Case 1. A simple integer
    if isinstance(r, (generic, int)):
        return Date(self._unit, value=r)
    elif not getattr(r, 'ndim', 1):
        # need to check if it has a ndim attribute for situations
        # like when the datearray is the data for a maskedarray
        # or some other subclass of ndarray with wierd getitem
        # behaviour
        return Date(self._unit, value=r.item())
    else:
        if hasattr(r, '_cachedinfo'):
            _cache = r._cachedinfo
            # Select the appropriate cached representations
            _cache.update(dict([(k, _cache[k][indx])
                                for k in ('toobj', 'tostr', 'toord')
                                if _cache[k] is not None]))
            # Reset the ischrono flag if needed
            if not (keep_chrono and _cache['ischrono']):
                _cache['ischrono'] = None
            # Reset the sorting indices
            _cache['chronidx'] = None
            # Reset the steps
            _cache['steps'] = None
            if reset_full:
                _cache['full'] = None
                _cache['hasdups'] = None
        return r
*/

//#include "arrayobject.h"

//PyMappingMethods *PyArray_as_mapping = (PyArray_Type).tp_as_mapping;
NPY_NO_EXPORT PyMappingMethods DatetimeArray_as_mapping = {
    NULL,              /*mp_length*/
    &DatetimeArray_getitem,        /*mp_subscript*/
    NULL, /*mp_ass_subscript*/
};

//NPY_NO_EXPORT PyMappingMethods array_as_mapping = {
//#if PY_VERSION_HEX >= 0x02050000
//    (lenfunc)array_length,              /*mp_length*/
//#else
//    (inquiry)array_length,              /*mp_length*/
//#endif
//    (binaryfunc)array_subscript_nice,       /*mp_subscript*/
//    (objobjargproc)array_ass_sub,       /*mp_ass_subscript*/
//};



/*
 * PROPERTIES
 */

static int
DatetimeArray_ReadOnlyErr(DatetimeArrayObject *self, PyObject *value, void *closure) {
   PyErr_SetString(PyExc_AttributeError, "Cannot set read-only property");
   return -1;
};

static PyGetSetDef DatetimeArray_getseters[] = {
    {"unit", (getter)DatetimeArray_unit, (setter)DatetimeArray_ReadOnlyErr,
     "Returns the frequency.", NULL},
    {"timestep", (getter)DatetimeArray_timestep, (setter)DatetimeArray_ReadOnlyErr,
     "", NULL},
    {"freqstr", (getter)DatetimeArray_freqstr, (setter)DatetimeArray_ReadOnlyErr,
     "Returns the string representation of frequency.", NULL},
    {"steps", (getter)DatetimeArray_steps, (setter)DatetimeArray_ReadOnlyErr,
     "time steps", NULL},
    {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/*
 * METHODS
 */

static PyMethodDef DatetimeArray_methods[] = {
    { "__array_finalize__", (PyCFunction)DatetimeArray_finalize, METH_VARARGS,
      ""},
//    {"__getitem__", (PyCFunction)DatetimeArray_getitem, METH_VARARGS, ""},
    {"has_dups", (PyCFunction)DatetimeArray_has_dups, METH_VARARGS, ""},
    {"has_missing", (PyCFunction)DatetimeArray_has_missing, METH_VARARGS, ""},
    {"is_chrono", (PyCFunction)DatetimeArray_is_chrono, METH_VARARGS, ""},
    {"is_full", (PyCFunction)DatetimeArray_is_full, METH_VARARGS, ""},
    {"is_valid", (PyCFunction)DatetimeArray_is_valid, METH_VARARGS, ""},
    {"date_to_index", (PyCFunction)DatetimeArray_date_to_index, METH_VARARGS, ""},
    {0}
};


static PyTypeObject DatetimeArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
    "timeseries.DatetimeArray",                      /* tp_name */
    sizeof(DatetimeArrayObject),              /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)DatetimeArray_dealloc,          /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
//    0,                 /* tp_as_mapping */
    &DatetimeArray_as_mapping,                 /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "DatetimeArray",                          /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,     /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    DatetimeArray_methods,                    /* tp_methods */
    DatetimeArray_members,                    /* tp_members */
    DatetimeArray_getseters,                  /* tp_getset */
    0,                            /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    DatetimeArray_new,                        /* tp_new */
};







/*
 * */
void import_c_datearray(PyObject *m)
{
    import_array();
    PyDateTime_IMPORT;

    DatetimeArray_Type.tp_base = &PyArray_Type;
    if (PyType_Ready(&DatetimeArray_Type) < 0)
        return;
    Py_INCREF(&DatetimeArray_Type);
    PyModule_AddObject(m, "DatetimeArray", (PyObject *)(&DatetimeArray_Type));
    
//    PyArray_TS_DATETIME = PyArray_RegisterDataType(&TS_DATETIME_Descr);
//    if (PyArray_TS_DATETIME < 0) {
//        DEBUGPRINTF("Could not import the TS_DATETIME description.");
//        return;
//    };
//    TS_DATETIME_Descr.ob_type = &PyArrayDescr_Type;
//    Py_INCREF(&TS_DATETIME_Descr);

    // PyModule_AddObject(m, "Datetime", (PyObject *)(&TS_DATETIME_Descr));

}



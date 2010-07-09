#include "c_freqs.h"
#include "c_dates.h"
#include "c_datearray.h"
#include <datetime.h>
#include <time.h>

#include "c_lib.h"


static PyTypeObject DateTimeArray_Type;

int PyArray_TS_DATETIME;

typedef struct {
    int unit; // base in numpy
    int timestep; // num in numpy...
} DatetimeObject_MetaData;

typedef struct {
   PyObject_HEAD;
   ts_datetime obval;
   DatetimeObject_MetaData obmeta;
} DatetimeArrayObject_Type;

TS_METADATA_DTSTR = "timeunit"

#if PY_VERSION_HEX >= 0x02070000
#define Get_Datetime_MetaData(descr)                                 \
    ((descr->metadata == NULL) ? NULL :                                       \
        ((DatetimeObject_MetaData *)(PyCapsule_GetPointer(                   \
                PyDict_GetItemString(                                         \
                    descr->metadata, TS_METADATA_DTSTR), NULL))))
#else
#define Get_Datetime_MetaData(descr)                                 \
    ((descr->metadata == NULL) ? NULL :                                       \
        ((DatetimeObject_MetaData *)(PyCObject_AsVoidPtr(                    \
                PyDict_GetItemString(descr->metadata, TS_METADATA_DTSTR)))))
#endif

//----------------------------------------------------------------------------

#include "noprefix.h"

static char * _SEQUENCE_MESSAGE = "error setting an array element with a sequence";

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;

#define _COPY_N_SIZE(size)                      \
    for(i=0; i<N; i++) {                       \
        memcpy(tout, tin, size);                \
        tin += instrides;                       \
        tout += outstrides;                     \
    }                                           \
    return

    switch(elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE

}

NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size)
{
    char *a, *b, c = 0;
    int j, m;

    switch(size) {
    case 1: /* no byteswap necessary */
        break;
    case 4:
        for (a = (char*)p; n > 0; n--, a += stride - 1) {
            b = a + 3;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 8:
        for (a = (char*)p; n > 0; n--, a += stride - 3) {
            b = a + 7;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a++ = *b; *b-- = c;
            c = *a; *a = *b; *b   = c;
        }
        break;
    case 2:
        for (a = (char*)p; n > 0; n--, a += stride) {
            b = a + 1;
            c = *a; *a = *b; *b = c;
        }
        break;
    default:
        m = size/2;
        for (a = (char *)p; n > 0; n--, a += stride - m) {
            b = a + (size - 1);
            for (j = 0; j < m; j++) {
                c=*a; *a++ = *b; *b-- = c;
            }
        }
        break;
    }
}



//static int
//UNICODE_setitem(PyObject *op, char *ov, PyArrayObject *ap)
//{
//    PyObject *temp;
//    Py_UNICODE *ptr;
//    int datalen;
//#ifndef Py_UNICODE_WIDE
//    char *buffer;
//#endif
//
//    if (!PyBytes_Check(op) && !PyUnicode_Check(op) &&
//            PySequence_Check(op) && PySequence_Size(op) > 0) {
//        PyErr_SetString(PyExc_ValueError,
//                "setting an array element with a sequence");
//        return -1;
//    }
//    /* Sequence_Size might have returned an error */
//    if (PyErr_Occurred()) {
//        PyErr_Clear();
//    }
//#if defined(NPY_PY3K)
//    if (PyBytes_Check(op)) {
//        /* Try to decode from ASCII */
//        temp = PyUnicode_FromEncodedObject(op, "ASCII", "strict");
//        if (temp == NULL) {
//            return -1;
//        }
//    }
//    else if ((temp=PyObject_Str(op)) == NULL) {
//#else
//    if ((temp=PyObject_Unicode(op)) == NULL) {
//#endif
//        return -1;
//    }
//    ptr = PyUnicode_AS_UNICODE(temp);
//    if ((ptr == NULL) || (PyErr_Occurred())) {
//        Py_DECREF(temp);
//        return -1;
//    }
//    datalen = PyUnicode_GET_DATA_SIZE(temp);
//
//#ifdef Py_UNICODE_WIDE
//    memcpy(ov, ptr, MIN(ap->descr->elsize, datalen));
//#else
//    if (!PyArray_ISALIGNED(ap)) {
//        buffer = _pya_malloc(ap->descr->elsize);
//        if (buffer == NULL) {
//            Py_DECREF(temp);
//            PyErr_NoMemory();
//            return -1;
//        }
//    }
//    else {
//        buffer = ov;
//    }
//    datalen = PyUCS2Buffer_AsUCS4(ptr, (PyArray_UCS4 *)buffer,
//            datalen >> 1, ap->descr->elsize >> 2);
//    datalen <<= 2;
//    if (!PyArray_ISALIGNED(ap)) {
//        memcpy(ov, buffer, datalen);
//        _pya_free(buffer);
//    }
//#endif
//    /* Fill in the rest of the space with 0 */
//    if (ap->descr->elsize > datalen) {
//        memset(ov + datalen, 0, (ap->descr->elsize - datalen));
//    }
//    if (!PyArray_ISNOTSWAPPED(ap)) {
//        byte_swap_vector(ov, ap->descr->elsize >> 2, 4);
//    }
//    Py_DECREF(temp);
//    return 0;
//}


static int
STRING_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
    char *ptr;
    Py_ssize_t len;
    PyObject *temp = NULL;

    if (!PyBytes_Check(op) && !PyUnicode_Check(op)
            && PySequence_Check(op) && PySequence_Size(op) > 0) {
        PyErr_SetString(PyExc_ValueError,
                "setting an array element with a sequence");
        return -1;
    }
    /* Sequence_Size might have returned an error */
    if (PyErr_Occurred()) {
        PyErr_Clear();
    }
#if defined(NPY_PY3K)
    if (PyUnicode_Check(op)) {
        /* Assume ASCII codec -- function similarly as Python 2 */
        temp = PyUnicode_AsASCIIString(op);
        if (temp == NULL) return -1;
    }
    else if (PyBytes_Check(op) || PyMemoryView_Check(op)) {
        temp = PyObject_Bytes(op);
        if (temp == NULL) {
            return -1;
        }
    }
    else {
        /* Emulate similar casting behavior as on Python 2 */
        PyObject *str;
        str = PyObject_Str(op);
        if (str == NULL) {
            return -1;
        }
        temp = PyUnicode_AsASCIIString(str);
        Py_DECREF(str);
        if (temp == NULL) {
            return -1;
        }
    }
#else
    if ((temp = PyObject_Str(op)) == NULL) {
        return -1;
    }
#endif
    if (PyBytes_AsStringAndSize(temp, &ptr, &len) == -1) {
        Py_DECREF(temp);
        return -1;
    }
    memcpy(ov, ptr, MIN(ap->descr->elsize,len));
    /*
     * If string lenth is smaller than room in array
     * Then fill the rest of the element size with NULL
     */
    if (ap->descr->elsize > len) {
        memset(ov + len, 0, (ap->descr->elsize - len));
    }
    Py_DECREF(temp);
    return 0;
}



static int
VOID_setitem(PyObject *op, char *ip, PyArrayObject *ap)
{
    PyArray_Descr* descr;
    int itemsize=ap->descr->elsize;
    int res;

    descr = ap->descr;
    if (descr->names && PyTuple_Check(op)) {
        PyObject *key;
        PyObject *names;
        int i, n;
        PyObject *tup, *title;
        PyArray_Descr *new;
        int offset;
        int savedflags;

        res = -1;
        /* get the names from the fields dictionary*/
        names = descr->names;
        n = PyTuple_GET_SIZE(names);
        if (PyTuple_GET_SIZE(op) != n) {
            PyErr_SetString(PyExc_ValueError,
                    "size of tuple must match number of fields.");
            return -1;
        }
        savedflags = ap->flags;
        for (i = 0; i < n; i++) {
            key = PyTuple_GET_ITEM(names, i);
            tup = PyDict_GetItem(descr->fields, key);
            if (!PyArg_ParseTuple(tup, "Oi|O", &new, &offset, &title)) {
                ap->descr = descr;
                return -1;
            }
            ap->descr = new;
            /* remember to update alignment flags */
            if ((new->alignment > 1)
                    && ((((npy_intp)(ip+offset)) % new->alignment) != 0)) {
                ap->flags &= ~NPY_ALIGNED;
            }
            else {
                ap->flags |= NPY_ALIGNED;
            }
            res = new->f->setitem(PyTuple_GET_ITEM(op, i), ip+offset, ap);
            ap->flags = savedflags;
            if (res < 0) {
                break;
            }
        }
        ap->descr = descr;
        return res;
    }

    if (descr->subarray) {
        /* copy into an array of the same basic type */
        PyArray_Dims shape = {NULL, -1};
        PyObject *ret;
        if (!(PyArray_IntpConverter(descr->subarray->shape, &shape))) {
            PyDimMem_FREE(shape.ptr);
            PyErr_SetString(PyExc_ValueError,
                    "invalid shape in fixed-type tuple.");
            return -1;
        }
        Py_INCREF(descr->subarray->base);
        ret = PyArray_NewFromDescr(&PyArray_Type,
                descr->subarray->base, shape.len, shape.ptr,
                NULL, ip, ap->flags, NULL);
        PyDimMem_FREE(shape.ptr);
        if (!ret) {
            return -1;
        }
        PyArray_BASE(ret) = (PyObject *)ap;
        Py_INCREF(ap);
        PyArray_UpdateFlags((PyArrayObject *)ret, NPY_UPDATE_ALL);
        res = PyArray_CopyObject((PyArrayObject *)ret, op);
        Py_DECREF(ret);
        return res;
    }

    /* Default is to use buffer interface to set item */
    {
        const void *buffer;
        Py_ssize_t buflen;
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)
                || PyDataType_FLAGCHK(descr, NPY_ITEM_IS_POINTER)) {
            PyErr_SetString(PyExc_ValueError,
                    "Setting void-array with object members using buffer.");
            return -1;
        }
        res = PyObject_AsReadBuffer(op, &buffer, &buflen);
        if (res == -1) {
            goto fail;
        }
        memcpy(ip, buffer, NPY_MIN(buflen, itemsize));
        if (itemsize > buflen) {
            memset(ip + buflen, 0, itemsize - buflen);
        }
    }
    return 0;

fail:
    return -1;
}

//----------------------------------------------------------------------------
static int
get_unit_from_metadata(PyArray_Descr *descr)
{
    Datetime_MetaData *meta;

    meta = Get_Datetime_MetaData(descr);
    if (meta == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "metadata not set for descriptor");
        return -1;
    }
    return meta->unit;


//----------------------------------------------------------------------------

static Bool
TS_DATETIME_nonzero (char *ip, DateTimeArrayObject *ap)
{
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        ts_datetime *ptmp = (ts_datetime *)ip;
        return (Bool) (*ptmp != 0);
    }
    else {
        /*
         * don't worry about swap, since we are just testing
         * whether or not equal to 0
         */
        ts_datetime tmp;
        memcpy(&tmp, ip, sizeof(ts_datetime));
        return (Bool) (tmp != 0);
    }
}

static void
TS_DATETIME_copyswapn (void *dst, npy_intp dstride, void *src, npy_intp sstride,
                       npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    if (src != NULL) {
        if (sstride == sizeof(ts_datetime) && dstride == sizeof(ts_datetime)) {
            memcpy(dst, src, n*sizeof(ts_datetime));
        }
        else {
            _unaligned_strided_byte_copy(dst, dstride, src, sstride,
                    n, sizeof(ts_datetime));
        }
    }
    if (swap) {
        _strided_byte_swap(dst, dstride, n, sizeof(ts_datetime));
    }
}

static void
TS_DATETIME_copyswap (void *dst, void *src, int swap, void *NPY_UNUSED(arr))
{

    if (src != NULL) {
        /* copy first if needed */
        memcpy(dst, src, sizeof(ts_datetime));
    }
    if (swap) {
        char *a, *b, c;

        a = (char *)dst;
#if SIZEOF_DATETIME == 2
        b = a + 1;
        c = *a; *a++ = *b; *b = c;
#elif SIZEOF_DATETIME == 4
        b = a + 3;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b   = c;
#elif SIZEOF_DATETIME == 8
        b = a + 7;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b   = c;
#elif SIZEOF_DATETIME == 10
        b = a + 9;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b   = c;
#elif SIZEOF_DATETIME == 12
        b = a + 11;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b   = c;
#elif SIZEOF_DATETIME == 16
        b = a + 15;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b-- = c;
        c = *a; *a++ = *b; *b   = c;
#else
        {
            int i, nn;

            b = a + (SIZEOF_DATETIME-1);
            nn = SIZEOF_DATETIME / 2;
            for (i = 0; i < nn; i++) {
                c = *a;
                *a++ = *b;
                *b-- = c;
            }
        }
#endif
    }
}


static PyObject *
TS_DATETIME_getitem(char *ip, PyArrayObject *ap) {
    ts_datetime t1;
    DateTimeArrayObject *dtap = (DateTimeArrayObject *)ap;
	DEBUGPRINTF("getitem got called");

    if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
        t1 = *((ts_datetime *)ip);
        return PyDateTime_FromFreqAndValue((ts_datetime)t1, dtap->unit);
    }
    else {
        ap->descr->f->copyswap(&t1, ip, !PyArray_ISNOTSWAPPED(ap), ap);
        return PyDateTime_FromFreqAndValue((ts_datetime)t1, dtap->unit);
    }
    Py_XDECREF(dtap);
}


static int
TS_DATETIME_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    /* ensure alignment */
    ts_datetime temp;

    if (PyArray_IsScalar(op, Datetime)) {
        /* This needs to convert based on type */
        temp = ((PyDatetimeScalarObject *)op)->obval;
    }
//#if defined(NPY_PY3K)
//    else if (PyUString_Check(op)) {
//#else
//    else if (PyUString_Check(op) || PyUnicode_Check(op)) {
// #endif
    else if (PyString_Check(op)) {
        temp = string_to_tsdatetime(((DateTimeArrayObject *)ap)->unit, op);
//         /* FIXME:  Converts to DateTime first and therefore does not handle extended notation */
//         /* import _mx_datetime_parser 
//          * res = _mx_datetime_parser(name)
//          *  Convert from datetime to Int
//          */
//         PyObject *res, *module;
// 
//         module = PyImport_ImportModule("numpy.core._mx_datetime_parser");
//         if (module == NULL) { return -1; }
//         res = PyObject_CallMethod(module, "datetime_from_string", "O", op);
//         Py_DECREF(module);
//         if (res == NULL) { return -1; }
//         // temp = PyDateTime_AsInt64(res, ap->descr);
//         temp = 
//         Py_DECREF(res);
//         if (PyErr_Occurred()) return -1;
    }
    else if (PyInt_Check(op)) {
        temp = PyInt_AS_LONG(op);
    }
    else if (PyLong_Check(op)) {
        temp = PyLong_AsLongLong(op);
    }
    else {
        // temp = PyDateTime_AsInt64(op, ap->descr);
        temp = PyLong_AsLongLong(op);
    };

    if (PyErr_Occurred()) {
        if (PySequence_Check(op)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, _SEQUENCE_MESSAGE);
        }
        return -1;
    }
    if (ap == NULL || PyArray_ISBEHAVED(ap))
        *((datetime *)ov)=temp;
    else {
        ap->descr->f->copyswap(ov, &temp, !PyArray_ISNOTSWAPPED(ap), ap);
    }
    return 0;
}



//-----------------------------------------------------------------------------
static void
//TS_DATETIME_to_BYTE(ts_datetime *ip, byte *op, npy_intp n,
//               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
//{
//    while (n--) {
//        *op++ = (byte)*ip++;
//    }
//}
//static void
//TS_DATETIME_to_UBYTE(ts_datetime *ip, ubyte *op, npy_intp n,
//               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
//{
//    while (n--) {
//        *op++ = (ubyte)*ip++;
//    }
//}
TS_DATETIME_to_SHORT(ts_datetime *ip, short *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (short)*ip++;
    }
}
// static void
// TS_DATETIME_to_USHORT(ts_datetime *ip, ushort *op, npy_intp n,
//                PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
// {
//     while (n--) {
//         *op++ = (ushort)*ip++;
//     }
// }
static void
TS_DATETIME_to_INT(ts_datetime *ip, int *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (int)*ip++;
    }
}
//static void
//TS_DATETIME_to_UINT(ts_datetime *ip, uint *op, npy_intp n,
//               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
//{
//    while (n--) {
//        *op++ = (uint)*ip++;
//    }
//}
static void
TS_DATETIME_to_LONG(ts_datetime *ip, long *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (long)*ip++;
    }
}
// static void
// TS_DATETIME_to_ULONG(ts_datetime *ip, ulong *op, npy_intp n,
//                PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
// {
//     while (n--) {
//         *op++ = (ulong)*ip++;
//     }
// }
// static void
// TS_DATETIME_to_LONGLONG(ts_datetime *ip, longlong *op, npy_intp n,
//                PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
// {
//     while (n--) {
//         *op++ = (longlong)*ip++;
//     }
// }
// static void
// TS_DATETIME_to_ULONGLONG(ts_datetime *ip, ulonglong *op, npy_intp n,
//                PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
// {
//     while (n--) {
//         *op++ = (ulonglong)*ip++;
//     }
// }
static void
TS_DATETIME_to_FLOAT(ts_datetime *ip, float *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (float)*ip++;
    }
}
static void
TS_DATETIME_to_DOUBLE(ts_datetime *ip, double *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (double)*ip++;
    }
}
// static void
// TS_DATETIME_to_LONGDOUBLE(ts_datetime *ip, longdouble *op, npy_intp n,
//                PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
// {
//     while (n--) {
//         *op++ = (longdouble)*ip++;
//     }
// }
//static void
//TS_DATETIME_to_BOOL(ts_datetime *ip, Bool *op, npy_intp n,
//               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
//{
//    while (n--) {
//        *op++ = (Bool)*ip++;
//    }
//}
static void
TS_DATETIME_to_CFLOAT(ts_datetime *ip, float *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (float)*ip++;
        *op++ = 0.0;
    }
}
static void
TS_DATETIME_to_CDOUBLE(ts_datetime *ip, double *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = (double)*ip++;
        *op++ = 0.0;
    }
}
//static void
//TS_DATETIME_to_CLONGDOUBLE(ts_datetime *ip, longdouble *op, npy_intp n,
//               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
//{
//    while (n--) {
//        *op++ = (longdouble)*ip++;
//        *op++ = 0.0;
//    }
//}
static void
TS_DATETIME_to_OBJECT(ts_datetime *ip, PyObject **op, npy_intp n, PyArrayObject *aip,
                 PyArrayObject *NPY_UNUSED(aop))
{
    npy_intp i;
    int skip = 1;
    for (i = 0; i < n; i++, ip +=skip, op++) {
        Py_XDECREF(*op);
        *op = TS_DATETIME_getitem((char *)ip, aip);
    }
}
static void
TS_DATETIME_to_STRING(ts_datetime *ip, char *op, npy_intp n, PyArrayObject *aip,
             PyArrayObject *aop)
{
    npy_intp i;
    PyObject *temp = NULL;
    int skip = 1;
    int oskip = aop->descr->elsize;
    for (i = 0; i < n; i++, ip += skip, op += oskip) {
        temp = TS_DATETIME_getitem((char *)ip, aip);
        if (temp == NULL) {
            Py_INCREF(Py_False);
            temp = Py_False;
        }
        if (STRING_setitem(temp,(char *)op, aop)) {
            Py_DECREF(temp);
            return;
        }
        Py_DECREF(temp);
    }
}
//static void
//TS_DATETIME_to_UNICODE(ts_datetime *ip, char *op, npy_intp n, PyArrayObject *aip,
//             PyArrayObject *aop)
//{
//    npy_intp i;
//    PyObject *temp = NULL;
//    int skip = 1;
//    int oskip = aop->descr->elsize;
//    for (i = 0; i < n; i++, ip += skip, op += oskip) {
//        temp = TS_DATETIME_getitem((char *)ip, aip);
//        if (temp == NULL) {
//            Py_INCREF(Py_False);
//            temp = Py_False;
//        }
//        if (UNICODE_setitem(temp,(char *)op, aop)) {
//            Py_DECREF(temp);
//            return;
//        }
//        Py_DECREF(temp);
//    }
//}
static void
TS_DATETIME_to_VOID(ts_datetime *ip, char *op, npy_intp n, PyArrayObject *aip,
             PyArrayObject *aop)
{
    npy_intp i;
    PyObject *temp = NULL;
    int skip = 1;
    int oskip = aop->descr->elsize;
    for (i = 0; i < n; i++, ip += skip, op += oskip) {
        temp = TS_DATETIME_getitem((char *)ip, aip);
        if (temp == NULL) {
            Py_INCREF(Py_False);
            temp = Py_False;
        }
        if (VOID_setitem(temp,(char *)op, aop)) {
            Py_DECREF(temp);
            return;
        }
        Py_DECREF(temp);
    }
}

//-----------------------------------------------------------------------------


// typedef PyArray_Descr TS_DATETIME_Descr;
// TS_DATETIME_Descr.typeobject = *DateTimeTypeObject;
static PyArray_ArrFuncs _PyTS_Datetime_ArrFuncs = {
    (PyArray_GetItemFunc*)TS_DATETIME_getitem,
    (PyArray_SetItemFunc*)TS_DATETIME_setitem,
    (PyArray_CopySwapNFunc*)TS_DATETIME_copyswapn,
    (PyArray_CopySwapFunc*)TS_DATETIME_copyswap,
    (PyArray_CompareFunc*)NULL,//TS_DATETIME_compare,
    (PyArray_ArgFunc*)NULL,//TS_DATETIME_argmax,
    (PyArray_DotFunc*)NULL,//TS_DATETIME_dot,
    (PyArray_ScanFunc*)NULL,//TS_DATETIME_scan,
    (PyArray_FromStrFunc*)NULL,//TS_DATETIME_fromstr,
    (PyArray_NonzeroFunc*)TS_DATETIME_nonzero,
    (PyArray_FillFunc*)NULL,//TS_DATETIME_fill,
    (PyArray_FillWithScalarFunc*)NULL,//S_DATETIME_fillwithscalar,
    {
        NULL, NULL, NULL
    },
    {
        NULL, NULL, NULL
    },
    NULL,
    (PyArray_ScalarKindFunc*)NULL,
    NULL,
    NULL,
    (PyArray_FastClipFunc*)NULL,//(PyArray_FastClipFunc*)TS_DATETIME_fastclip,
    (PyArray_FastPutmaskFunc*)NULL,//(PyArray_FastPutmaskFunc*)TS_DATETIME_fastputmask,
    (PyArray_FastTakeFunc*)NULL,//(PyArray_FastTakeFunc*)TS_DATETIME_fasttake,
    NULL, NULL, NULL, NULL,
    {
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_BOOL,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_BYTE,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_UBYTE,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_SHORT,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_USHORT,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_INT,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_UINT,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_LONG,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_ULONG,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_LONGLONG,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_ULONGLONG,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_FLOAT,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_DOUBLE,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_LONGDOUBLE,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_CFLOAT,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_CDOUBLE,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_CLONGDOUBLE,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_TS_DATETIME,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_TIMEDELTA,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_OBJECT,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_STRING,
        (PyArray_VectorUnaryFunc*)NULL,//TS_DATETIME_to_UNICODE,
        (PyArray_VectorUnaryFunc*)TS_DATETIME_to_VOID
    }
};


NPY_NO_EXPORT PyArray_Descr TS_DATETIME_Descr = {
//    PyObject_HEAD_INIT(&PyArrayDescr_Type)
    PyObject_HEAD_INIT(NULL)
    &DatetimeObject_Type,
    'T',
    'T',
    '=',
    0,
    0,
    -1,
    1*sizeof(ts_datetime),
    offsetof(struct {char c; ts_datetime v;}, v),
    NULL,
    NULL,
    NULL,
    &_PyTS_Datetime_ArrFuncs,
    NULL,
};
// 
// int PyArray_TS_DATETIME = PyArray_RegisterDataType(&TS_DATETIME_Descr);



//-----------------------------------------------------------------------------

static PyObject *
DateTimeArray_new(PyTypeObject *cls, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"object", "unit", 0};

    PyObject *obj;
    PyArrayObject *arr = NULL;
    PyObject *unit = NULL;
    PyObject *timestep = NULL;
    DateTimeArrayObject *self;
    PyArray_Descr *descr;

    if(!PyArg_ParseTupleAndKeywords(args,kw,"OO",kwlist,
                                    &obj,
                                    &unit,&timestep))
        return NULL;

    arr = (PyArrayObject *)PyArray_FROM_O(obj);
    if(arr == NULL)
        return NULL;
    DEBUGPRINTF("We have an array...");

    // descr = PyArray_DescrFromType(PyArray_INT64);
//    descr = PyArray_DescrFromType(PyArray_TS_DATETIME);
    //    descr = PyArray_DescrFromType(TS_DATETIME_Descr.type_num);
    descr = &TS_DATETIME_Descr;
    if (descr == NULL)
        return NULL;
    DEBUGPRINTF("datetime initialize %i", descr->type_num);
    // DEBUGPRINTF("type_num; %i", (long)(&descr->type_num));
    // DEBUGPRINTF("flags: %i", (long)(&descr->flags));
    // descr = &TS_DATETIME_Descr;
    self = (DateTimeArrayObject *)PyArray_View(arr, descr, &DateTimeArray_Type);
    DEBUGPRINTF("Back in __new__");
    if(self == NULL)
        return NULL;

    if (unit == NULL){
        unit = PyInt_FromLong(FR_UND);
    }
//    self->cached_vals = cached_vals;

    self->unit = check_freq(unit);
    if (self->unit == -1)
        goto onError;

    DEBUGPRINTF("set unit");
    if (timestep == NULL)
        self->timestep = 1;
    else
        self->timestep = PyInt_AsLong(timestep);

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
DateTimeArray_dealloc(DateTimeArrayObject *self)
{
//    DEBUGPRINTF("Dropping cache");
//    Py_XDECREF(self->cached_vals);
    DEBUGPRINTF("Dropping object");
    self->base.ob_type->tp_free((PyObject*)self);
}

static PyObject*
DateTimeArray_finalize(DateTimeArrayObject *self, PyObject *args)
{
    DateTimeArrayObject *context;
    if(PyArg_ParseTuple(args,"O",&context))
    {
        if (DateTimeArray_Check(context)){
            DEBUGPRINTF("in context from DTA");
//            self->cached_vals = context->cached_vals;
            self->unit = context->unit;
            self->timestep = context->timestep;
        } else {
            DEBUGPRINTF("in context from scratch");
//            self->cached_vals = NULL;
            self->unit = FR_UND;
//            Py_XINCREF(self->cached_vals);
        };
        DEBUGPRINTF("Setting the status...");
		// self->status.is_chrono);
        self->status.is_chrono = -1;
        self->status.has_dups = -1;
        self->status.has_missing = -1;
        DEBUGPRINTF("... done");
//        Py_XINCREF(self->cached_vals);
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

static PyObject *
//DateTimeArray_freqstr(DateTimeArray *self, void *closure) {
DateTimeArray_freqstr(DateTimeArrayObject *self) {
    PyObject *key = PyInt_FromLong(self->unit);
    PyObject *freq_aliases = PyDict_GetItem(freq_dict, key);
    PyObject *main_alias = PyTuple_GET_ITEM(freq_aliases, 0);
    Py_DECREF(key);
    return main_alias;
}

static PyObject *
DateTimeArray_steps(DateTimeArrayObject *self){
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
    DEBUGPRINTF("iterators initialzed");
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
    DEBUGPRINTF("All good?");
    return (PyObject *)steps;

 fail:
    DEBUGPRINTF("DatetimeArray.steps: Oops...");
    Py_DECREF(steps);
    return NULL;
}


static int
DateTimeArray_check_status(DateTimeArrayObject *self)
{
    PyArrayIterObject *self_iter;
    npy_int64 timestep, diff;
    int is_chrono = 1, has_dups=0, has_missing=0;

    timestep = self->timestep;
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
DateTimeArray_has_dups(DateTimeArrayObject *self)
{
    if (self->status.has_dups == -1)
        if (DateTimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.has_dups == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DateTimeArray_has_missing(DateTimeArrayObject *self)
{
    if (self->status.has_missing == -1)
        if (DateTimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.has_missing == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *
DateTimeArray_is_chrono(DateTimeArrayObject *self)
{
    if (self->status.is_chrono == -1)
        if (DateTimeArray_check_status(self) < 0)
            return NULL;
    if (self->status.is_chrono == 0)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyMemberDef DateTimeArray_members[] = {
    {"unit", T_INT, offsetof(DateTimeArrayObject, unit), 0,
     "frequency"},
    {"timestep", T_INT, offsetof(DateTimeArrayObject, timestep), 0,
     "frequency"},
//     {"cached_vals", T_OBJECT_EX, offsetof(DateTimeArray, cached_vals), 0,
//      "cached_values"},
//    {"freq", T_INT, offsetof(DateObject, freq), 0,
//     "frequency"},
//    {"value", T_INT, offsetof(DateObject, value), 0,
//     "integer representation of the Date"},
    {NULL}  /* Sentinel */
};

//static PyObject *
//DateTimeArray_date_to_index(DateTimeArrayObject *self, DateObject *date){
//    return NULL;
//}
//
//
//static PyObject *
//DateTimeArray_getitem(DateTimeArray *self, PyObject *indx) {
//    int reset_full=1, keep_chrono=0;
//    DEBUGPRINTF("in __getitem__");
//
//    if (DateObject_Check(indx)) {
//        DEBUGPRINTF("index is Date");
//        indx = DateTimeArray_date_to_index(self, (DateObject *)indx);
//    }
//
//    PyObject *r, *result;
//    r = PyArray_GETITEM(self, indx);
//    if (PyInt_Check(r)) {
//        DEBUGPRINTF("index is integer");
//        result = (PyObject *)DateObject_FromFreqAndValue(self->unit, PyInt_AsLong(r));
//    }
//    else {
//        DEBUGPRINTF("index is whatever else");
//        result = r;
////        DEBUGPRINTF("chrono? %i - dups? %i - missing? %i", ((DateObject *)result)->status.is_chrono, ((DateObject *)result)->status.has_dups, ((DateObject *)result->status).has_missing);
//    }
//    Py_DECREF(r);
//    return result;
//}


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



/*
 * PROPERTIES
 */

static int
DateTimeArray_ReadOnlyErr(DateTimeArrayObject *self, PyObject *value, void *closure) {
   PyErr_SetString(PyExc_AttributeError, "Cannot set read-only property");
   return -1;
};

static PyGetSetDef DateTimeArray_getseters[] = {
    {"freqstr", (getter)DateTimeArray_freqstr, (setter)DateTimeArray_ReadOnlyErr,
     "Returns the string representation of frequency.", NULL},
    {"steps", (getter)DateTimeArray_steps, (setter)DateTimeArray_ReadOnlyErr,
     "time steps", NULL},
    {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/*
 * METHODS
 */

static PyMethodDef DateTimeArray_methods[] = {
    { "__array_finalize__", (PyCFunction)DateTimeArray_finalize, METH_VARARGS,
      ""},
//    {"__getitem__", (PyCFunction)DateTimeArray_getitem, METH_VARARGS, ""},
    {"has_dups", (PyCFunction)DateTimeArray_has_dups, METH_VARARGS, ""},
    {"has_missing", (PyCFunction)DateTimeArray_has_missing, METH_VARARGS, ""},
    {"is_chrono", (PyCFunction)DateTimeArray_is_chrono, METH_VARARGS, ""},
    {0}
};


static PyTypeObject DateTimeArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
    "timeseries.DateTimeArray",                      /* tp_name */
    sizeof(DateTimeArrayObject),              /* tp_basicsize */
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
    
    DEBUGPRINTF("numtypes:%i", NPY_NUMUSERTYPES);
    DEBUGPRINTF("udesctypes:%i", PyArray_USERDEF);
    PyArray_TS_DATETIME = PyArray_RegisterDataType(&TS_DATETIME_Descr);
    if (PyArray_TS_DATETIME < 0) {
        DEBUGPRINTF("Could not import the TS_DATETIME description.");
        return;
    };
    DEBUGPRINTF("datetime initialize %i", PyArray_TS_DATETIME);
//    Py_INCREF(&TS_DATETIME_Descr);

    DEBUGPRINTF("type    : %s", &TS_DATETIME_Descr.type);
    DEBUGPRINTF("type_num: %i", (&TS_DATETIME_Descr)->type_num);
    DEBUGPRINTF("flags   : %i", (&TS_DATETIME_Descr)->flags);
    // PyModule_AddObject(m, "DateTime", (PyObject *)(&TS_DATETIME_Descr));
    DEBUGPRINTF("c_datearray imported fine...");

}



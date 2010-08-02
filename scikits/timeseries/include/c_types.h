#ifndef _DTYPES_H
#define _DTYPES_H

#include "arrayobject.h"
#include "arrayscalars.h"

// #ifndef NPY_DATETIMEUNIT
// #define typedef enum {
//             NPY_FR_Y,
//             NPY_FR_M,
//             NPY_FR_W,
//             NPY_FR_B,
//             NPY_FR_D,
//             NPY_FR_h,
//             NPY_FR_m,
//             NPY_FR_s,
//             NPY_FR_ms,
//             NPY_FR_us,
//             NPY_FR_ns,
//             NPY_FR_ps,
//             NPY_FR_fs,
//             NPY_FR_as
//         } NPY_DATETIMEUNIT;
// #endif

/* From ndarraytypes.h & numpy.datetime.c*/
typedef struct {
        npy_int64 year;
        int month, day, day_of_year, hour, min, sec, us, ps, as;
} ts_datetimestruct;
typedef struct {
        npy_int64 months, days;
        npy_int64 secs, us, ps, as;
} ts_timedeltastruct;


typedef struct {
    int year, month, day, day_of_year;
} ymdstruct;

typedef struct {
    int hour, min, sec;
} hmsstruct;


typedef npy_int64 ts_datetime;


typedef struct {
    int unit; // base in numpy
    int timestep; // num in numpy...
    int period_end_at; //
    npy_int64 periods_per_day;
    npy_int64 secs_per_period;
    int convert_to_start;
} Datetime_Metadata;

typedef Datetime_Metadata ts_metadata;



typedef struct {
    PyObject_HEAD;
    ts_datetime obval; /* integer representation of date */
    ts_metadata obmeta;
} DatetimeObject;

#endif

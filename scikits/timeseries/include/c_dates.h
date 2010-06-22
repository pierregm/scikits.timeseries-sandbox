#ifndef C_DATES_H
#define C_DATES_H

#include "c_lib.h"

#define HIGHFREQ_ORIG 719163

////////////////////////////////////////////////////


typedef struct {
    int from_week_end; //day the week ends on in the "from" frequency
    int to_week_end; //day the week ends on in the "to" frequency

    int from_a_year_end; //month the year ends on in the "from" frequency
    int to_a_year_end; //month the year ends on in the "to" frequency

    int from_q_year_end; //month the year ends on in the "from" frequency
    int to_q_year_end; //month the year ends on in the "to" frequency
} asfreq_info;

void get_asfreq_info(int, int, asfreq_info*);
long (*get_asfreq_func(int, int, int))(long, char, asfreq_info*);

PyObject *DateArray_asfreq(PyObject *, PyObject *);
PyObject *DateArray_getDateInfo(PyObject *, PyObject *);


PyObject *c_dates_now(PyObject *, PyObject *);

PyObject *set_callback_DateFromString(PyObject *, PyObject *);
PyObject *set_callback_DateTimeFromString(PyObject *, PyObject *);

void import_c_dates(PyObject *);

#endif

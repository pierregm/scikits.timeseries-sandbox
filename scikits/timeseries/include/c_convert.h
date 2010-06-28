#ifndef C_CONVERT_H
#define C_CONVERT_H

#include "c_lib.h"
#include "dtypes.h"
#include <datetime.h>
#include <time.h>


#define GREGORIAN_CALENDAR 0
#define JULIAN_CALENDAR 1

#define HIGHFREQ_ORIG 719163
#define DAYS_TO_NPYDAYS(days) ((days) - HIGHFREQ_ORIG)
#define NPYDAYS_TO_DAYS(npy_days) ((npy_days) + HIGHFREQ_ORIG)


ymdstruct days_to_ymdstruct(npy_longlong, int);
hmsstruct seconds_to_hmsstruct(npy_longlong);


npy_int64 highunits_per_day(int);
int day_of_week(npy_longlong);
int is_leapyear(long, int);
npy_longlong year_offset(npy_longlong, int);

npy_longlong days_from_ymdc(int, int, int, int);
#define days_from_ymd(year, month, day) (days_from_ymdc((year), (month), (day), GREGORIAN_CALENDAR))

int ending_day(int);
int ending_month(int);

typedef struct {
	int ending_day; //day the week ends
    int ending_month; //month the year ends
	npy_int64 periods_per_day; // nb of periods (hh/mm/ss...) per day
	npy_int64 secs_per_period; // nb of seconds in a period
	int result_starts;
} conversion_info;

typedef npy_longlong (*conversion_function)(npy_longlong, conversion_info*);
conversion_function get_converter_from(int, int, int);
conversion_function get_converter_to(int, int, int);

npy_longlong _days_from_highfreq(npy_longlong, conversion_info*);
npy_longlong _secs_to_highfreq(npy_longlong, conversion_info*);
npy_longlong _secs_from_midnight(npy_longlong, int);
//npy_longlong (*_convert_from_days(int, int))(npy_longlong, int, char);
//npy_longlong (*_convert_to_days(int, int))(npy_longlong, int, char);
conversion_function get_converter_from_days(int, int);
conversion_function get_converter_to_days(int, int);
conversion_function convert_to_mediator(int, int, int);
conversion_function convert_from_mediator(int, int, int);

void set_conversion_info(int, char, conversion_info*);

//DERIVED FROM mx.DateTime
/*
    Functions in the following section are borrowed from mx.DateTime version
    2.0.6, and hence this code is subject to the terms of the egenix public
    license version 1.0.0
*/



#endif

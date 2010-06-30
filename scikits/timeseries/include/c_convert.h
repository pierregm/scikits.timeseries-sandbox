#ifndef C_CONVERT_H
#define C_CONVERT_H

#include "c_lib.h"
#include "dtypes.h"


#define GREGORIAN_CALENDAR 0
#define JULIAN_CALENDAR 1

#define HIGHFREQ_ORIG 719163
#define DAYS_TO_NPYDAYS(days) ((days) - HIGHFREQ_ORIG)
#define NPYDAYS_TO_DAYS(npy_days) ((npy_days) + HIGHFREQ_ORIG)

/* Returns absolute seconds from an hour, minute, and second [in np.datetime.c] */
#define secs_from_hms(hour, min, sec, multiplier) (\
  ((hour)*3600 + (min)*60 + (sec)) * (npy_int64)(multiplier)\
)

ymdstruct days_to_ymdstruct(npy_int64, int);
hmsstruct seconds_to_hmsstruct(npy_int64);

void set_datetimestruct_from_days(ts_datetimestruct*, npy_int64);
void set_datetimestruct_from_secs(ts_datetimestruct*, npy_int64);
void set_datetimestruct_from_days_and_secs(ts_datetimestruct*, npy_int64, npy_int64);


npy_int64 highunits_per_day(int);
int day_of_week(npy_int64);
int is_leapyear(long, int);
npy_int64 year_offset(npy_int64, int);
int isoweek_from_datetimestruct(ts_datetimestruct*);

npy_int64 days_from_ymdc(int, int, int, int);
#define days_from_ymd(year, month, day) (days_from_ymdc((year), (month), (day), GREGORIAN_CALENDAR))
double secs_from_ranged_hms(int, int, double);

int ending_day(int);
int ending_month(int);

typedef struct {
	int ending_day; //day the week ends
    int ending_month; //month the year ends
	npy_int64 periods_per_day; // nb of periods (hh/mm/ss...) per day
	npy_int64 secs_per_period; // nb of seconds in a period
	int result_starts;
} conversion_info;

typedef npy_int64 (*conversion_function)(npy_int64, conversion_info*);

conversion_function get_converter_from_days(int, int);
conversion_function get_converter_to_days(int, int);
conversion_function convert_to_mediator(int, int, int);
conversion_function convert_from_mediator(int, int, int);

npy_int64 _days_from_highfreq(npy_int64, conversion_info*);
npy_int64 _secs_from_highfreq(npy_int64, conversion_info*);
npy_int64 _secs_to_highfreq(npy_int64, conversion_info*);
npy_int64 _secs_from_midnight(npy_int64, int);

void set_conversion_info(int, char, conversion_info*);

void normalize_ts_timedeltastruct(ts_timedeltastruct*);
void normalize_days_secs(npy_int64*, npy_int64*);
void normalize_years_months(npy_int64*, npy_int64*);

#endif

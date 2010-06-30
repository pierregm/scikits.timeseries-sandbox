#include "c_lib.h"
#include "dtypes.h"
#include "c_freqs.h"
#include "c_convert.h"

#include <Python.h>
#include <datetime.h>
#include <time.h>


/* ---------------------------------------------------------------------------
 * Normalization utilities (from Python/datetime.c).
 */

/* Compute Python divmod(x, y), returning the quotient and storing the
 * remainder into *r.  The quotient is the floor of x/y, and that's
 * the real point of this.  C will probably truncate instead (C99
 * requires truncation; C89 left it implementation-defined).
 * Simplification:  we *require* that y > 0 here.  That's appropriate
 * for all the uses made of it.  This simplifies the code and makes
 * the overflow case impossible (divmod(LONG_MIN, -1) is the only
 * overflow case).
 */
static npy_longlong
divmod(npy_longlong x, npy_longlong y, npy_longlong *r)
{
    npy_longlong quo;

    assert(y > 0);
    quo = x / y;
    *r = x - quo * y;
    if (*r < 0) {
        --quo;
        *r += y;
    }
    assert(0 <= *r && *r < y);
    return quo;
}
/* Modified in order to deal with negative seconds higher than -day
 */
static void
normalize_pair(npy_longlong *hi, npy_longlong *lo, int factor)
{
    assert(factor > 0);
    assert(lo != hi);
    if (*lo <= -factor || *lo >= factor) {
        const npy_longlong num_hi = divmod(*lo, factor, lo);
        const npy_longlong new_hi = *hi + num_hi;
        assert(! SIGNED_ADD_OVERFLOWED(new_hi, *hi, num_hi));
        *hi = new_hi;
    }
    assert(-factor < *lo && *lo < factor);
}




/* Return the number of high unit periods per day*/
npy_int64 highunits_per_day(int freq){
    switch(freq)
    {
        case FR_DAY:
            return 1;
        case FR_HR:
            return 24;
        case FR_MIN:
            return 24*60;
        case FR_SEC:
            return 24*60*60;
        default:
            return 24*60*60 - 1;
    };
}

npy_int64 secs_per_highunits(int freq, npy_int64 multiplier)
{
    switch(freq)
    {
    case FR_SEC:
        return multiplier;
    case FR_MIN:
        return 60 * multiplier;
    case FR_HR:
        return 3600 * multiplier;
    case FR_DAY:
        return 86400 * multiplier;
    default:
        return 0;
    }
}

npy_int64 seconds_per_period(int freq, npy_int64 multiplier)
{
    switch(freq)
    {
        case FR_DAY:
            return multiplier * 86400;
        case FR_HR:
             return multiplier * 3600;
        case FR_MIN:
            return multiplier * 60;
        case FR_SEC:
             return multiplier;
    };
	return -1;
}



/* Returns the quarter */
#define month_to_quarter(month) (((month)-1)/3 + 1)
#define quarter_to_month(quarter) (((quarter)-1)*3 + 1)



/*
    Functions in the following section are borrowed from mx.DateTime version
    2.0.6, and hence this code is subject to the terms of the egenix public
    license version 1.0.0
*/

#define SECONDS_PER_DAY ((double) 86400.0)


/* Table with day offsets for each month (0-based, without and with leap) */
static int month_offset[2][13] = {
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
    { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
};

/* Table of number of days in a month (0-based, without and with leap) */
static int days_in_month[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};



/* Return the day of the week for the given absolute date. */
int day_of_week(npy_longlong absdate) {
    int day_of_week;
    if (absdate >= 1) {
        day_of_week = (absdate - 1) % 7;
    }
    else {
        day_of_week = 6 - ((-absdate) % 7);
    }
    return day_of_week;
}

/* Return 1/0 iff year points to a leap year in calendar. */
// static
int is_leapyear(register long year, int calendar) {
    if (calendar == GREGORIAN_CALENDAR) {
        return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
    } else {
        return (year % 4 == 0);
    }
}

/* Return the year offset, that is the absolute date of the day
   31.12.(year-1) in the given calendar.

   Note:
   For the Julian calendar we shift the absdate (which is measured
   using the Gregorian Epoch) value by two days because the Epoch
   (0001-01-01) in the Julian calendar lies 2 days before the Epoch in
   the Gregorian calendar. */
npy_longlong
year_offset(npy_longlong year, int calendar)
{
    year--;
    if (calendar == GREGORIAN_CALENDAR) {
        if (year >= 0 || -1/4 == -1)
            return year*365 + year/4 - year/100 + year/400;
        else
            return year*365 + (year-3)/4 - (year-99)/100 + (year-399)/400;
    }
    else if (calendar == JULIAN_CALENDAR) {
        if (year >= 0 || -1/4 == -1)
            return year*365 + year/4 - 2;
        else
            return year*365 + (year-3)/4 - 2;
    }
//    Py_Error(DateCalc_Error, "unknown calendar");
// onError:
    return -1;
}



/* Set the instance's value using the given date and time. calendar
   may be set to the flags: GREGORIAN_CALENDAR,
   JULIAN_CALENDAR to indicate the calendar to be used. */

npy_longlong
days_from_ymdc(int year, int month, int day, int calendar)
{
    int leap;
    npy_longlong yearoffset, absdate;

    /* Range check */
    Py_AssertWithArg(year > -(INT_MAX / 366) && year < (INT_MAX / 366),
                     DateCalc_RangeError,
                     "year out of range: %i",
                     year);

    /* Is it a leap year ? */
    leap = is_leapyear(year, calendar);

    /* Negative month values indicate months relative to the years end */
    if (month < 0) month += 13;
    Py_AssertWithArg(month >= 1 && month <= 12,
                     DateCalc_RangeError,
                     "month out of range (1-12): %i",
                     month);

    /* Negative values indicate days relative to the months end */
    if (day < 0) day += days_in_month[leap][month - 1] + 1;
    Py_AssertWithArg(day >= 1 && day <= days_in_month[leap][month - 1],
                     DateCalc_RangeError,
                     "day out of range: %i",
                     day);

    /* Nb of days between Dec. 31 (YYYY - 1) and Dec. 31 1969 */
    yearoffset = year_offset(year,calendar);
    if (PyErr_Occurred()) goto onError;

    absdate = day + month_offset[leap][month - 1] + yearoffset;
    return absdate;

 onError:
    return -1;
    /* return 0; That's what numpy uses */
}

#define days_from_ymd(year, month, day) (days_from_ymdc((year), (month), (day), GREGORIAN_CALENDAR))


double
secs_from_ranged_hms(int hour, int minute, double second)
{
    Py_AssertWithArg(hour >= 0 && hour <= 23,
                     DateCalc_RangeError,
                     "hour out of range (0-23): %i",
                     hour);
    Py_AssertWithArg(minute >= 0 && minute <= 59,
                     DateCalc_RangeError,
                     "minute out of range (0-59): %i",
                     minute);
    Py_AssertWithArg(second >= (double)0.0 &&
                     (second < (double)60.0 ||
                     (hour == 23 && minute == 59 && second < (double)61.0)),
                     DateCalc_RangeError,
                     "second out of range (0.0 - <60.0; <61.0 for 23:59): %f",
                     second);
    return secs_from_hms(hour, minute, second, 1);

 onError:
    return -1;
    /* return 0; (the numpy way) */
}


/* from numpy/datetime.c (reference: 1CE) */
//static
ymdstruct
days_to_ymdstruct(npy_longlong absdate, int calendar)
{
    ymdstruct ymd;
    long year;
    npy_longlong yearoffset;
    int leap, dayoffset;
    int month = 1, day = 1;
    int *monthoffset;

    /* Approximate year */
    if (calendar == JULIAN_CALENDAR) {
        year = absdate / 365.25;
    } else {
        year = absdate / 365.2425;
    };
    if (absdate > 0) year++;

    /* Apply corrections to reach the correct year */
    while (1) {
        /* Calculate the year offset */
        yearoffset = year_offset(year, calendar);
        /*
         * Backward correction: absdate must be greater than the
         * yearoffset
         */
        if (yearoffset >= absdate) {
            year--;
            continue;
        }
        leap = is_leapyear(year, calendar);

        dayoffset = absdate - yearoffset;
        /* Forward correction: non leap years only have 365 days */
        if (dayoffset > 365 && !leap) {
            year++;
            continue;
        }
        break;
    }

    /* Now iterate to find the month */
    monthoffset = month_offset[leap];
    for (month = 1; month < 13; month++) {
        if (monthoffset[month] >= dayoffset)
            break;
    }
    day = dayoffset - month_offset[leap][month-1];

    ymd.year  = year;
    ymd.month = month;
    ymd.day   = day;
    ymd.day_of_year = dayoffset;

    return ymd;
};


static int
isoweek_from_ymdc(int year, int month, int day, int calendar)
{
    int week;
    npy_longlong yearoffset = year_offset(year, calendar);
    npy_longlong absdate = days_from_ymdc(year, month, day, calendar);
    npy_longlong dayofweek = day_of_week(absdate);

    /* Estimate*/
    week = (absdate - yearoffset - 1) - dayofweek + 3;
    if (week >= 0)
        week = week / 7 + 1;

    /* Verify */
    if (week < 0){
        /* The day lies in last week of the previous year */
        if ((week > -2) ||
            (week == -2 && is_leapyear(year-1, calendar)))
            week = 53;
        else
            week = 52;
    }
    else if (week == 53) {
        /* Check if the week belongs to year or year+1 */
        if (31 - day + dayofweek < 3)
            week = 1;
    }
    return week;
};

int isoweek_from_datetimestruct(ts_datetimestruct *dinfo)
{
	return isoweek_from_ymdc(dinfo->year,
                             dinfo->month,
                             dinfo->day,
                             GREGORIAN_CALENDAR);
}



hmsstruct
seconds_to_hmsstruct(npy_longlong abstime)
{
    int hour, minute, second;
    hmsstruct hms;

    hour   = abstime / 3600;
    minute = (abstime % 3600) / 60;
    second = abstime - (hour*3600 + minute*60);

    hms.hour   = hour;
    hms.min = minute;
    hms.sec = second;

    return hms;
};


void set_datetimestruct_from_days(ts_datetimestruct *info, npy_longlong days)
{
    ymdstruct ymd = days_to_ymdstruct(days, GREGORIAN_CALENDAR);
    info->year = ymd.year;
    info->month = ymd.month;
    info->day = ymd.day;
    info->day_of_year = ymd.day_of_year;
}

void set_datetimestruct_from_secs(ts_datetimestruct *info, npy_longlong secs)
{
    hmsstruct hms = seconds_to_hmsstruct(secs);
    info->hour = hms.hour;
    info->min = hms.min;
    info->sec = hms.sec;
}

void set_datetimestruct_from_days_and_secs(ts_datetimestruct *info,
                                 npy_longlong days,
                                 npy_longlong secs)
{
    set_datetimestruct_from_days(info, days);
    set_datetimestruct_from_secs(info, secs);
}


//NPY_NO_EXPORT
npy_longlong
DatetimeStructToDatetime(int unit, ts_datetimestruct *d)
{
    npy_longlong val;
    npy_longlong days=0;
    int ugroup = get_base_unit(unit);

    if ((unit > FR_MTH) || (unit == FR_UND)) {
        days = days_from_ymd(d->year, d->month, d->day);
        if (unit >= FR_HR){
            days -= HIGHFREQ_ORIG;
        };
    };
    if (ugroup == FR_ANN) {
        val = d->year;
    }
    else if (ugroup == FR_QTR) {
        npy_longlong quarter = ((d->month -1 )/3) + 1;
        if ((unit - ugroup) > 12) {
            // quarterly frequency with year determined by ending period
            val = d->year*4 + quarter;
        }
        else {
            /* quarterly frequency with year determined by ending period
                           or has December year end*/
            val = (d->year - 1)*4 + quarter;
        };
    }
    else if (unit == FR_MTH) {
        val = (d->year-1)*12 + d->month;
    }
    else if (ugroup == FR_WK) {
        npy_longlong end_week_day, adj_day;
        end_week_day = (7 - (unit-FR_WK)) % 7;
        adj_day = days + ((7 - end_week_day) - days % 7) % 7;
        val = adj_day / 7;
    }
    else if (unit == FR_BUS) {
        npy_longlong weeks = days/7;
        val = days - weeks * 2;
        /*
        int dotw = day_of_week(days);
        if (dotw > 4){
            // Invalid business day
            val = 0;
        }
        else {
            npy_longlong x = days -2;
            val = 2 + (x/7)*5 + x%7;
        }
        */
    }
    else if ((unit == FR_DAY) || (unit==FR_UND)){
        val = days;
    }
    else if (unit == FR_HR) {
        val = days * 24 + d->hour;
    }
    else if (unit == FR_MIN){
        val = days * 1440 + d->hour * 60 + d->min;
    }
    else if (unit == FR_SEC){
        val = days *  (npy_int64)(86400)
            + secs_from_hms(d->hour, d->min, d->sec, 1);
    }
    else {
        /* Shouldn't get here */
        PyErr_SetString(PyExc_ValueError, "invalid internal frequency");
        val = -1;
    }
    return val;
}




//NPY_NO_EXPORT
npy_longlong
PyDatetime_ToDatetime(long unit, PyObject *datetime)
{
    ts_datetimestruct dinfo;
    npy_longlong val;

    if (!PyDateTime_Check(datetime) && !PyDate_Check(datetime)){
        PyObject *err_msg, *_type;
        _type = PyObject_Type(datetime);
        err_msg = PyString_FromString("Expected a datetime.date(time) object, received: ");
        PyString_ConcatAndDel(&err_msg, PyObject_Str(_type));
        PyErr_SetString(PyExc_TypeError, PyString_AsString(err_msg));
        Py_DECREF(_type);
        Py_DECREF(err_msg);
        val = -1;
    }
    else {
        dinfo.year = (npy_longlong)PyDateTime_GET_YEAR(datetime);
        dinfo.month = PyDateTime_GET_MONTH(datetime);
//        quarter=((month-1)/3)+1;
        dinfo.day = PyDateTime_GET_DAY(datetime);
        dinfo.hour = PyDateTime_DATE_GET_HOUR(datetime);
        dinfo.min = PyDateTime_DATE_GET_MINUTE(datetime);
        dinfo.sec = PyDateTime_DATE_GET_SECOND(datetime);
        //
        val = DatetimeStructToDatetime(unit, &dinfo);
    }
    return val;
}


/* Helpers for frequency conversion routines */
#define _days_to_bus_weekday(days) ((((days)/ 7) * 5) + (absdate) % 7)

static long _days_to_bus_weekend_to_monday(long absdate, int day_of_week) 
{
    if (day_of_week > 4) {
        //change to Monday after weekend
        absdate += (7 - day_of_week);
    };
    return _days_to_bus_weekday(absdate);
};

static long _days_to_bus_weekend_to_friday(long absdate, int day_of_week) 
{
    if (day_of_week > 4) {
        //change to friday before weekend
        absdate -= (day_of_week - 4);
    };
    return _days_to_bus_weekday(absdate);
};


/* --- Conversion routines                                                  */

static npy_longlong
missing_convert(npy_longlong indate, conversion_info *info) { return -1;}

static npy_longlong
no_convert(npy_longlong indate, conversion_info *info) { return indate;}


/* From days to other units ................................................*/

/* Returns the month ending the current annual/quarterly freq */
int ending_month(int unit)
{
    int end = (unit % 1000) % 12;
    return (end == 0 ? 12: end);
}
/* Returns the day ending the current weekly freq */
int ending_day(int unit) { return unit % 1000; }

static npy_longlong
_days_to_years(npy_longlong indate, conversion_info *info)
{
    ymdstruct ymd = days_to_ymdstruct(indate, GREGORIAN_CALENDAR);
    int end_month = info->ending_month;
    return (ymd.month > end_month ? ymd.year + 1: ymd.year);
}

static npy_longlong
_days_to_quarters(npy_longlong indate, conversion_info *info)
{
    ymdstruct ymd = days_to_ymdstruct(indate, GREGORIAN_CALENDAR);
    int end_month = info->ending_month;
    int year = ymd.year;
    int month = ymd.month;
    if (end_month != 12){
        month -= end_month;
        if (month <= 0)
            month += 12;
        else
            year += 1;
    }
    int quarter = month_to_quarter(month);
    return (year - 1) * 4 + quarter;
}

static npy_longlong
_days_to_months(npy_longlong indate, conversion_info *info)
{
    ymdstruct ymd = days_to_ymdstruct(indate, GREGORIAN_CALENDAR);
    return (ymd.year - 1) * 12 + ymd.month;
}

static npy_longlong
_days_to_weeks(npy_longlong indate, conversion_info *info)
{
//    ymdstruct ymd = days_to_ymdstruct(indate, GREGORIAN_CALENDAR);
    int weekend = info->ending_day;
    return (indate - (1 + weekend))/7 + 1;
}

static npy_longlong
_days_to_bus(npy_longlong indate, conversion_info *info)
{
    int dayofweek = day_of_week(indate);
    if (info->result_starts)
        return _days_to_bus_weekend_to_friday(indate, dayofweek);
    else
        return _days_to_bus_weekend_to_monday(indate, dayofweek);
}

static npy_longlong 
_days_to_bus_batch(npy_longlong indate, conversion_info *info)
{
    int dayofweek = day_of_week(indate);
    if (dayofweek > 4)
        return -1;
    else if (info->result_starts)
        return _days_to_bus_weekend_to_friday(indate, dayofweek);
    else
        return _days_to_bus_weekend_to_monday(indate, dayofweek);
};


static npy_longlong
_days_to_days(npy_longlong indate, conversion_info *info)
{
    return indate;
}

static npy_longlong
_days_to_highfreq(npy_longlong indate, conversion_info *info)
{
npy_int64 periods_per_day = info->periods_per_day;
    if (info->result_starts)
        return (indate - HIGHFREQ_ORIG) * periods_per_day;
    else
        return (indate - HIGHFREQ_ORIG + 1) * periods_per_day - 1;
}

conversion_function get_converter_from_days(int fromunit, int inbatch)
{
    int ubase = get_base_unit(fromunit);
    
    if (ubase == FR_ANN)
        return &_days_to_years;
    else if (ubase == FR_QTR)
        return &_days_to_quarters;
    else if (ubase == FR_MTH)
        return &_days_to_months;
    else if (ubase == FR_WK)
        return &_days_to_weeks;
    else if (ubase == FR_BUS)
        if (inbatch)
            return &_days_to_bus_batch;
        else
            return &_days_to_bus;
    else if ((ubase == FR_DAY) || (ubase == FR_UND))
        return &_days_to_days;
    else if (ubase > FR_DAY)
        return &_days_to_highfreq;
    return &missing_convert;
}




static npy_longlong
_days_from_years(npy_longlong indate, conversion_info *info)
{
    npy_longlong absdate, year;
    int final_adj;
    int endmonth = info->ending_month;
    int month = endmonth % 12;
    month = (month == 0 ? 1 : month+1);

    if (info->result_starts){
        year = (endmonth == 12 ? indate: indate-1);
        final_adj = 0;
    }
    else {
        year = (endmonth == 12 ? indate+1: indate);
        final_adj = -1;
    }
    absdate = days_from_ymd(year, month, 1);
    if (absdate  == INT_ERR_CODE) return INT_ERR_CODE;
    return absdate + final_adj;
}

static npy_longlong
_days_from_quarters(npy_longlong indate, conversion_info *info)
{
    npy_longlong absdate;
    int year, month, final_adj;
    int end_month = info->ending_month;

    if (info->result_starts) {
        year = (indate - 1)/4 + 1;
        month = (indate + 4)*3 - 12*year -2;
        final_adj = 0;
    }
    else {
        year = indate/4 + 1;
        month = (indate + 5)*3 - 12*year -2;
        final_adj = -1;
    };
    if (end_month != 12){
        month += end_month;
        if (month > 12)
            month -= 12;
        else
            year -= 1;
    }
    absdate = days_from_ymd(year, month, 1);
    if (absdate  == INT_ERR_CODE) return INT_ERR_CODE;
    return absdate + final_adj;
}

static npy_longlong
_days_from_months(npy_longlong indate, conversion_info *info)
{
    npy_longlong absdate;
    int year, month, final_adj;

    if (info->result_starts){
        year = (indate - 1)/12 + 1;
        month = indate - 12*year - 1;
        final_adj = 0;
    }
    else {
        year = indate/12 + 1;
        month = indate - 12*year;
        final_adj = -1;
    }
    absdate = days_from_ymd(year, month, 1);
    if (absdate  == INT_ERR_CODE) return INT_ERR_CODE;
    return absdate + final_adj;
}


static npy_longlong
_days_from_weeks(npy_longlong indate, conversion_info *info)
{
    int weekend = info->ending_day;
    if (info->result_starts)
        return indate*7 - 6 + weekend;
    else
        return indate*7 + weekend;
}

static npy_longlong
_days_from_busdays(npy_longlong indate, conversion_info *info)
{
    return ((indate-1)/5)*7 + (indate-1)%5 + 1;
}

npy_longlong
_days_from_highfreq(npy_longlong indate, conversion_info *info)
{
    npy_int64 periods_per_day = info->periods_per_day;
    if (indate < 0)
        return (indate + 1)/periods_per_day + HIGHFREQ_ORIG - 1;
    else
        return indate/periods_per_day + HIGHFREQ_ORIG;
}

conversion_function get_converter_to_days(int fromunit, int inbatch)
{
    int ubase = get_base_unit(fromunit);
    
    if (ubase == FR_ANN)
        return &_days_from_years;
    else if (ubase == FR_QTR)
        return &_days_from_quarters;
    else if (ubase == FR_MTH)
        return &_days_from_months;
    else if (ubase == FR_WK)
        return &_days_from_weeks;
    else if (ubase == FR_BUS)
        return &_days_from_busdays;
    else if ((ubase == FR_DAY) || (ubase == FR_UND))
        return &no_convert;
    else if (ubase > FR_DAY)
        return &_days_from_highfreq;
    return &missing_convert;
}




/* From seconds */

npy_longlong
_secs_from_highfreq(npy_longlong indate, conversion_info *info)
{
    npy_longlong secs_per_period = info->secs_per_period;
    if (info->result_starts)
        return indate*secs_per_period;
    else
        return (indate + 1)*secs_per_period - 1;
}

npy_longlong
_secs_from_midnight(npy_longlong indate, int unit)
{
    conversion_info info;
    set_conversion_info(unit, 'S', &info);
    npy_longlong secs=_secs_from_highfreq(indate, &info) % 86400;
    if (secs < 0)
        secs += 86400;
    return secs;
}

npy_longlong
_secs_to_highfreq(npy_longlong indate, conversion_info *info)
{
    npy_longlong secs_per_period = info->secs_per_period;
    if (indate < 0)
        return (indate + 1)/secs_per_period - 1;
    else
        return indate/secs_per_period;
}



conversion_function convert_to_mediator(int fromunit, int tounit, int inbatch)
{
    if ((fromunit > FR_DAY) && (tounit > FR_DAY))
        return &_secs_from_highfreq;
    else
        return *get_converter_to_days(fromunit, inbatch);
}

conversion_function convert_from_mediator(int fromunit, int tounit, int inbatch)
{
    if ((tounit == FR_DAY) || (tounit == FR_UND))
        return &no_convert;
    else if (tounit > FR_DAY)
        if (fromunit <= FR_DAY)
            return *get_converter_from_days(tounit, 0);
        else
            return &_secs_to_highfreq;
    else
        return *get_converter_from_days(tounit, 0);
}

void
set_conversion_info(int unit, char relation, conversion_info *info){
    int base = get_base_unit(unit);
    if ((base == FR_ANN) || (base == FR_QTR))
        info->ending_month = ending_month(unit);
    else if (base == FR_WK)
        info->ending_day = ending_day(unit);
    else if (base > FR_DAY) {
        info->periods_per_day = highunits_per_day(unit);
        info->secs_per_period = secs_per_highunits(unit, 1);
    };
    if (relation == 'S')
        info->result_starts = 1;
    else
        info->result_starts = 0;
}


void normalize_days_secs(npy_longlong *d, npy_longlong *s)
{
    if (*s <= -86400 || *s >= 86400)
        normalize_pair(d, s, 86400);
}
void normalize_years_months(npy_longlong *y, npy_longlong *m)
{
    normalize_pair(y, m, 12);
    m += 1;
}


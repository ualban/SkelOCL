/*
 * myTime.h
 *
 *  Created on: 14/apr/2014
 *      Author: alban
 */

#ifndef MYTIME_H_
#define MYTIME_H_


#include <time.h>

/** end - start */
void diff(struct timespec *start, struct timespec *end, struct timespec *diff);
/** end - start to long*/
long diff(struct timespec *start, struct timespec *end);

inline long nsec2usec(long nsec) {return nsec/1000L;}


#define GET_TIME(COUNTER) clock_gettime(CLOCK_MONOTONIC , &(COUNTER))
#define DIFF(START, END) nsec2usec(diff( &(START), &(END)))

#endif /* MYTIME_H_ */


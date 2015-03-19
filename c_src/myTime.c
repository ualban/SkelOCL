#include "myTime.h"

/** end - start */
void diff(struct timespec *start, struct timespec *end, struct timespec *diff)
{
	if ((end->tv_nsec - start->tv_nsec) < 0) {
		diff->tv_sec = (end->tv_sec - start->tv_sec) -1;
		diff->tv_nsec = 1000000000L + (end->tv_nsec - start->tv_nsec);
	} else {
		diff->tv_sec = end->tv_sec - start->tv_sec;
		diff->tv_nsec = end->tv_nsec - start->tv_nsec;
	}
}
/** end - start to long*/
long diff(struct timespec *start, struct timespec *end) {

	struct timespec tmp;
	diff(start, end, &tmp);
	return tmp.tv_nsec + (tmp.tv_sec*1000000000L);
}




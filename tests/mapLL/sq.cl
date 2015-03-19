#define DELAY_LOOPS 40

void delay() {
	double v = 2.0;
	for(unsigned int i_delay = 0; i_delay < DELAY_LOOPS; i_delay++)
		v = sin(v);
}

double sq(double x) {
	delay();
	return x*x;
}

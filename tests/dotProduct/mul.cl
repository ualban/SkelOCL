#define DELAY_LOOPS 10

void delay() {
	double v = 2.0;
	for(unsigned int i_delay = 0; i_delay < DELAY_LOOPS; i_delay++)
		v = sin(v);
}

double mul(double x, double y){
	delay();
	return x*y;
}

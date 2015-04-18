#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int main()
{
	long double i = 0;
	double upper_bound = 1000000000;
	long double approximation = 0.0;
	for (; i <= upper_bound;)
	{
		long double a = pow(-1.0, i);
		approximation += a/(2.0 * i + 1.0);

		i += 1.0;
	}
	unsigned int j = 0;
	long double approximation_copy = approximation;
	for (;j < 3; j++)
	{
		approximation += approximation_copy;
	}
	printf("approximation: %Lf\n", approximation);
}
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<stdio.h>
void sumArray(float* a,float* b,float* c,const int N)
{
	for(int i=0;i<N;i++)
	{
		c[i] = a[i]+b[i];
	//	printf("C[%d]=%.0f\n",i,c[i]);
	}
}
void initData(float* a,int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for(int i=0;i<size;i++)
	{
		a[i] = (float)(rand()&0xFF)/10.0f;
	}
}
int main(int argc,char **argv)
{
	int nElement = 1024;
	size_t nBytes = nElement*sizeof(float);
	float *h_A,*h_B,*h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);
	initData(h_A,nElement);
	initData(h_B,nElement);
	initData(h_C,nElement);
	sumArray(h_A,h_B,h_C,nElement);
	free(h_A);
	free(h_B);
	free(h_C);
}

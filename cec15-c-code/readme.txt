/*
  CEC15 Test Function Suite for Single Objective Optimization
  Jane Jing Liang 
  email: liangjing@zzu.edu.cn; liangjing@pmail.ntu.edu.cn
  Nov. 20th 2015
*/

cec15_test_func.cpp is the test function
Example:
cec15_test_func(x, f, dimension,population_size,func_num);

main.cpp is an example function about how to use cec15_test_func.cpp


#include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>
void cec15_test_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound,*bias;
int ini_flag=0,n_flag,func_flag,*SS;
void main()
{
...
}

For Linux Users:
Please  change %xx in fscanf and fprintf and do use "WINDOWS.H". 

/* 

   sample functions to be used in map and reduce skeletons

*/

//UNARY_FUNC(sq,  double, x, return(x*x);)
double sq(double x){return x*x;}

//UNARY_FUNC(inc, double, x, return(x+1.0);)
//UNARY_FUNC(sins,double, x, for(int i=0;i<800000;i++) x = sin(x); return(x); )
//UNARY_FUNC(sqi, int, x, return(x*x);)
//BINARY_FUNC(pairmul, double, x, y, return(x*y);)
//BINARY_FUNC(sum, double, x, y, return(x+y);)


/*!
 *
 *  OpenCL Map kernel for \em unary user functions.
 */
//static std::string squareKernelSourceCode(
//"__kernel void UnaryMapKernel_sq(__global double* input, __global double* output, unsigned int numElements) \n"
//"{	int i = get_global_id(0); \n"
//"	unsigned int gridSize = get_local_size(0)*get_num_groups(0);\n"
//"	while(i < numElements)"
//"	{\n"
//"		output[i] = sq(input[i]);\n"
//"		i += gridSize;\n"
//"	}\n"
//"}\n"
//);

static std::string squareKernelSourceCode(
"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n"
"double sq(double x){return x*x;}\n\n"
"__kernel void UnaryMapKernel_sq(__global double* input, __global double* output) \n"
"{	int i = get_global_id(0);\n"
"	output[i] = sq(input[i]);\n"
"}\n"
);


//TODO TERZO PARAMETRO NON PREVISTO In OCL.Cpp
static std::string squareKernelSourceCodeStride(
"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n"
"double sq(double x){return x*x;}\n\n"
"__kernel void UnaryMapKernel_sq(__global double* input, __global double* output, unsigned int numElements) \n"
"{	int i = get_global_id(0); \n"
"	unsigned int gridSize = get_local_size(0)*get_num_groups(0);\n"
"	while(i < numElements)"
"	{\n"
"		output[i] = sq(input[i]);\n"
"		i += gridSize;\n"
"	}\n"
"}\n"
);


//ERL_GPU_MAP(sq,mapsq)
//ERL_GPU_MAP(inc,mapinc)
//ERL_GPU_REDUCE(sum,reducesum)
//ERL_GPU_MAPZIP(sum,mapzipsum)
//ERL_GPU_MAPREDUCE(sq,sum,mapreducesqsum)

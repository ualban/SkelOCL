static const std::string MapKernelStr(
"__kernel void MapKernel(__global TYPE* input, __global TYPE* output, unsigned int outputOffset, unsigned int uiNumElements)\n"
"{	__private size_t i = get_global_id(0);\n"
"	if(i >= uiNumElements) return;\n"
"	output[i + outputOffset] = FUN_NAME(input[i]);\n"
"}\n"
);

static const std::string Map2KernelStr(
"__kernel void MapKernel2(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, unsigned int outputOffset, unsigned int uiNumElements)\n"
"{	__private size_t i = get_global_id(0);\n"
"	if(i >= uiNumElements) return;\n"
"	output[i + outputOffset] = FUN_NAME(input1[i], input2[i]);\n"
"}\n"
);



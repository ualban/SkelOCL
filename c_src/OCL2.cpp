#include "OCL2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>


/*OpenCL static objects*/
static cl_context context = NULL;
static cl_command_queue commandQueues[2];

static cl_device_id device = NULL;

uint device_minBaseAddrAlignByte;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;

	cl_platform_id* platformIds;

	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.

	clGetPlatformIDs(0, NULL, &numPlatforms);

	platformIds = new cl_platform_id[numPlatforms];

	errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "No OpenCL platform found." << std::endl;
		return NULL;
	}

	char** platVend = new char*[numPlatforms];

	for(uint i = 0 ; i < numPlatforms; i++) {
		size_t size = 0;

		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
		platVend[i] = new char[size];
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, size, platVend[i], NULL);

#ifdef DEBUG
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &size);
		char* platName = new char[size];
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size, platName, NULL);

		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 0, NULL, &size);
		char* drvVers = new char[size];
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, size, drvVers, NULL);

		std::cout << "P" << i << " CL_PLATFORM_NAME: "  << platName    << std::endl;
		std::cout << "P" << i << " CL_PLATFORM_VENDOR: "<< platVend[i] << std::endl;
		std::cout << "P" << i << " CL_PLATFORM_VERSION: " << drvVers     << std::endl;

		delete[] drvVers;
		delete[] platName;
#endif
	}

	const char* AMD_VENDOR_STR = "Advanced Micro Devices, Inc.";
	const char* NVIDIA_VENDOR_STR = "NVIDIA Corporation";
	const char* INTEL_VENDOR_STR = "Intel(R) Corporation";

	cl_platform_id cpu, nvidia;
	bool cpu_found = false, nvidia_found = false;

#ifdef NVIDIA

	for(uint i = 0 ; !nvidia_found && i < numPlatforms; i++){
		if( strcmp(NVIDIA_VENDOR_STR, platVend[i]) == 0) {
			nvidia = platformIds[i];
			nvidia_found = true;
		}
	}
	if(!nvidia_found) {
		std::cout << "NVIDIA Platform NOT found!\n";
		return NULL;
	}
	else {std::cout << "NVIDIA Platform FOUND!\n";}

#else
	uint i;
	for(i = 0 ; !cpu_found && i < numPlatforms  ; i++){

		if (strcmp(AMD_VENDOR_STR, platVend[i]) == 0 || strcmp(INTEL_VENDOR_STR, platVend[i]) == 0){
			cpu = platformIds[i];
			cpu_found = true;
		}

	}
	i--;
	if(!cpu_found) {
		std::cout << "CPU Platform NOT found!\n";
		return NULL;
	}
	else {std::cout << platVend[i] << " Platform FOUND!\n";}

#endif

	for(uint i = 0 ; i < numPlatforms; i++)
		delete [] (platVend[i]);


	delete [] platVend;

	delete [] platformIds;

	// Create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
			CL_CONTEXT_PLATFORM,
#ifdef NVIDIA
			(cl_context_properties) nvidia,
#else
			(cl_context_properties) cpu,
#endif
			0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
			NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "No GPU is available on the platform, falling back to CPU" << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
				NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//	commandQueues must be already allocated and NULL initialized
//
bool CreateCommandQueues(cl_context context, cl_device_id *device, cl_command_queue* cmdQueueV, int cmdQueueC)
{
	cl_int errNum;
	cl_device_id *devices;

	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return false;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return false;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS) {
		delete [] devices;
		std::cerr << "Failed to get device IDs";
		return false;
	}

	//Memory regions must be alligned to CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE (it limits the minimum buffers' size)
	clGetDeviceInfo(devices[0],CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,sizeof(cl_uint), &device_minBaseAddrAlignByte, NULL);


	cmdQueueV[0] = clCreateCommandQueue(context, devices[0], 0/*CL_QUEUE_PROFILING_ENABLE*/, NULL);
	if (cmdQueueV[0] == NULL)
	{
		delete [] devices;
		std::cerr << "Failed to create commandQueue 0";
		return false;
	}

	cmdQueueV[1] = clCreateCommandQueue(context, devices[0], 0/*CL_QUEUE_PROFILING_ENABLE*/, NULL);
	if (cmdQueueV[1] == NULL)
	{
		delete [] devices;
		std::cerr << "Failed to create commandQueue 1";
		return false;
	}


	*device = devices[0];
	delete [] devices;

	return true;
}


cl_int CreateProgramFromSrcString(cl_context context, cl_device_id device, const char *srcStr, cl_program* pProgram)
{
	cl_int errNum = 0;

	*pProgram = clCreateProgramWithSource(context,
			1, &srcStr,
			NULL,
			&errNum);
	if (pProgram == NULL) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return errNum;
	}

	errNum = clBuildProgram(*pProgram, 1, &device, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(*pProgram, device, CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);

		std::cerr <<
			"Error in kernel: " << std::endl <<
			buildLog << std::endl;

		std::cerr <<
			"Kernel source:" << std::endl
			<< srcStr << std::endl;

		clReleaseProgram(*pProgram);
		*pProgram = NULL;

		return errNum;
	}

	return errNum;
}


/*
  Create an OpenCL program from the kernel source file,
  RETURNS CL_INVALID_PROGRAM if fileName file can't be opened
*/
cl_int CreateProgramFromFileName(cl_context context, cl_device_id device, const char* fileName, cl_program* program)
{
	cl_int errNum;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return CL_INVALID_PROGRAM;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();

	return CreateProgramFromSrcString(context, device, srcStdStr.c_str(), program);

}

cl_command_queue getCommandQueue(uint index) {

	return commandQueues[index];
}

cl_context getContext() {

	return context;
}


///
//  Cleanup any created OpenCL resources
//
// just static objects
void Cleanup_static(/*cl_context _context, cl_command_queue _commandQueue*/)
{
#ifdef DEBUG
	std::cout << "Cleanup_static: Called." << std::endl;
#endif

	if(commandQueues[0])
		clReleaseCommandQueue(commandQueues[0]);
	if(commandQueues[1])
		clReleaseCommandQueue(commandQueues[1]);


	if(context) {
		clReleaseContext(context);
	}

	commandQueues[0] = commandQueues[1] = NULL;
	device = NULL;
	context = NULL;
}


bool init()
{
	// Create an OpenCL context on first available platform
	context = CreateContext();
	if (context == NULL)
	{
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return false;
	}
#ifdef DEBUG
	std::cerr << "OpenCL context created." << std::endl;
#endif

	// Create 2 command-queues on the first device available
	// on the created context
	if (!CreateCommandQueues(context, &device, commandQueues, 2))
	{
		std::cerr << "Failed to create OpenCL commandQueue." << std::endl;
		Cleanup_static(/*context, commandQueue*/);
		return false;
	}
#ifdef DEBUG
	std::cerr << "OpenCL CommandQueue created." << std::endl;
#endif

	return true;
}


cl_int createBuffer(size_t szBuffBytes, cl_mem_flags flags, cl_mem* pBuffer) {
	cl_int ciErrNum = 0;

	*pBuffer =  clCreateBuffer(context, flags, szBuffBytes, NULL, &ciErrNum);

	return ciErrNum;

}


void releaseBuffers(std::vector<cl_mem> buffers) {

	while (!buffers.empty()){
		clReleaseMemObject(buffers.back());
		buffers.pop_back();
	}

}

void releaseEvents(std::vector<cl_event> events) {

	while (!events.empty()){
		clReleaseEvent(events.back());
		events.pop_back();
	}

}

void unmapBuffers(std::vector< std::pair<cl_mem,void*> > mapPointers) {

	while (!mapPointers.empty()) {
		std::pair<cl_mem,void*>& curr = mapPointers.back();

		clEnqueueUnmapMemObject(commandQueues[0],curr.first, curr.second, 0, NULL, NULL);

		mapPointers.pop_back();
	}
}


//NON BLOCKING MAP
cl_int mapBuffer(cl_mem buffer, size_t szBufferBytes, cl_map_flags map_flags, double** pBuffer) {

	cl_int ciErrNum = 0;

	*pBuffer = (cl_double*) clEnqueueMapBuffer(commandQueues[0], buffer, CL_FALSE, map_flags, 0, szBufferBytes, 0, NULL, NULL, &ciErrNum);

	return ciErrNum;
}

cl_int mapBufferBlocking(cl_mem buffer, size_t szOffset, size_t szBufferBytes, cl_map_flags map_flags, double** pBuffer) {

	cl_int ciErrNum = 0;

	*pBuffer = (cl_double*) clEnqueueMapBuffer(commandQueues[0], buffer, CL_TRUE, map_flags, szOffset, szBufferBytes, 0, NULL, NULL, &ciErrNum);

	return ciErrNum;
}

cl_int unMapBuffer(cl_mem buffer, double* pBuffer) {

	return clEnqueueUnmapMemObject(commandQueues[0], buffer, (void*)pBuffer, 0, NULL, NULL);
}

cl_int getBufferSizeByte(cl_mem buffer, size_t* szBufferByte) {

	return clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), szBufferByte, NULL);
}


cl_int copyBuffer(cl_mem from_buff, cl_mem to_buff, size_t bytesToCopy) {

	return clEnqueueCopyBuffer(commandQueues[0], from_buff, to_buff, 0, 0, bytesToCopy, 0, NULL, NULL);
}


cl_int createBuildProgramFromFileName(const char* fileName, cl_program* program) {

	return CreateProgramFromFileName(context, device, fileName, program);
}

cl_int createBuildProgramFromString(const char* srcString, cl_program* program) {

	//use static context and device
	return CreateProgramFromSrcString(context, device, srcString, program);
}





cl_int createKernel(cl_program program, const char* kernelName, cl_kernel* kernel) {

	cl_int ciErrNum = 0;

	*kernel = clCreateKernel(program, kernelName, &ciErrNum);

	return ciErrNum;
}

cl_int cloneKernel(cl_kernel ker, cl_kernel* clonedKer) {

	cl_int ciErrNum = 0;

	size_t retSize;
	char kerName[256];

	ciErrNum = clGetKernelInfo(ker, CL_KERNEL_FUNCTION_NAME, 256, kerName, &retSize);

	if(ciErrNum != CL_SUCCESS)
		return ciErrNum;

//	if(retSize >= 256) {grow kerName and retry}

//	std::cout << kerName << std::endl; // TODO DEBUG

	cl_program prog;

	ciErrNum = clGetKernelInfo(ker, CL_KERNEL_PROGRAM, sizeof(cl_program), &prog, &retSize);

	if(ciErrNum != CL_SUCCESS )
		return ciErrNum;

	*clonedKer = clCreateKernel(prog, kerName, &ciErrNum);

	if(ciErrNum != CL_SUCCESS )
		return ciErrNum;


	return ciErrNum;
}


cl_int computeKernelWithEvents(cl_kernel kernel, size_t* szGlobalWorkSize, size_t* szLocalWorkSize,
		cl_uint num_events_in_wait_list,
		const cl_event * event_wait_list,
		cl_event * event) {

	cl_int ciErrNum = 0;

	ciErrNum = clEnqueueNDRangeKernel(commandQueues[0], kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, num_events_in_wait_list, event_wait_list, event);

	return ciErrNum;
}

/*Blocking call, clFinish before and after clEnqueueNDRangeKernel*/
cl_int computeKernel(cl_kernel kernel, size_t* szGlobalWorkSize, size_t* szLocalWorkSize) {

	cl_int ciErrNum = 0;

	ciErrNum = computeKernelWithEvents(kernel, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);

	return ciErrNum;
}


cl_int createUserEvent(cl_event *event) {

	cl_int ciErrNum = 0;

	*event = clCreateUserEvent(context, &ciErrNum);

	return ciErrNum;
}


cl_int waitForEvents(cl_uint num_events, const cl_event *evt_list) {

	//deprecated in 1.2 use clEnqueue( Marker | Barrier )WithWaitList
	return clEnqueueWaitForEvents(commandQueues[0], num_events, evt_list);
}

cl_int enqueueBarrier() {

	return clEnqueueBarrier(commandQueues[0]);
}

void releaseOCL() {
	// dispose static objects
	Cleanup_static(/*context, commandQueue*/);
}

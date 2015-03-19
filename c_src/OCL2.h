#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>

#include <CL/cl.h>

using namespace std;

bool init();

void releaseOCL();

cl_int createBuffer(size_t szBuffBytes, cl_mem_flags flags, cl_mem* pBuffer);

cl_int mapBuffer(cl_mem buffer, size_t szBufferBytes, cl_map_flags map_flags, double** pBuffer);
cl_int mapBufferBlocking(cl_mem buffer, size_t szOffset, size_t szBufferBytes, cl_map_flags map_flags, double** pBuffer);

cl_int unMapBuffer(cl_mem buffer, double* pBuffer);

cl_int getBufferSizeByte(cl_mem buffer, size_t* szBufferByte);

cl_int copyBuffer(cl_mem from_buff, cl_mem to_buff, size_t bytesToCopy);

void releaseBuffers(std::vector<cl_mem> );

void releaseEvents(std::vector<cl_event> events);

void unmapBuffers(std::vector< std::pair<cl_mem,void*> > mapPointers);



cl_int createBuildProgramFromFileName(const char* fileName, cl_program* program);

cl_int createBuildProgramFromString(const char* srcString, cl_program* program);

cl_int createKernel(cl_program program, const char* kernelName, cl_kernel* kernel);
cl_int cloneKernel(cl_kernel ker, cl_kernel* clonedKer);

cl_int computeKernel(cl_kernel kernel, size_t* szGlobalWorkSize, size_t* szLocalWorkSize);

cl_int computeKernelWithEvents(cl_kernel kernel, size_t* szGlobalWorkSize, size_t* szLocalWorkSize,
		cl_uint num_events_in_wait_list,
		const cl_event * event_wait_list,
		cl_event * event);

cl_int createUserEvent(cl_event *event);

cl_int waitForEvents(cl_uint num_events, const cl_event *evt_list);

cl_int enqueueBarrier();

cl_command_queue getCommandQueue(uint index);
cl_context getContext();

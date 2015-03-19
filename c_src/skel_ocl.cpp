#include "erl_nif.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include <cerrno>

#include <vector>
#include <unordered_map>

#include "OCL2.h"

using namespace std;

#include "errors.cpp"

#include "utils.cpp"

#include "myTime.h"


//kernels implementations
#include "kernels/map_kernel.h"
#include "kernels/reduce_kernel.h"


#define NUM_SEGM 8

#define ONE 1
#define TWO 2

const uint NUM_QUEUES = TWO; //ONE or TWO

const uint MAX_QUEUES = TWO;


static bool ocl_initialised = false;

#define OCL_INIT_CHECK() if(!ocl_initialised) return make_error(env, ATOM(skel_ocl_not_initialized) );

#define NIF_ARITY_CHECK(ARITY) if(argc != ARITY) return enif_make_badarg(env);

 //DEBUG
//#define PRINT_ARR_DEBUG(X,SIZE) \
//		std::cerr << #X << ".length = " << SIZE << ". "<< #X << " = ["; \
//		for(unsigned i = 0; i < SIZE - 1; i++) \
//		std::cerr << (X)[i] << ","; \
//		std::cerr << (X)[ SIZE - 1] << "]\n" << std::endl;
//
//static void print_buffer_debug(const char* bName, cl_mem buffer, size_t szOffset, size_t szBufferSegment) {
//
//	double * pBuffer = NULL;
//
//	//obtain a pointer to work on the buffer
//	mapBufferBlocking(buffer, szOffset, szBufferSegment, CL_MAP_READ, &pBuffer);
//
//	cerr << bName << "_";
//	PRINT_ARR_DEBUG(pBuffer, szBufferSegment / sizeof(double))
//
//	unMapBuffer(buffer, pBuffer);
//}

//defined in OCL2.cpp
extern uint device_minBaseAddrAlignByte;



/****************  RESOURCE OBJECTS ***************/


//Define a destructor for OBJTYPE using BODY
#define DEF_DTOR(OBJTYPE, BODY) \
		static void dtor_##OBJTYPE(ErlNifEnv* env, void* obj) { \
			OBJTYPE* _obj = (OBJTYPE*) obj; \
			if(*_obj) { \
				/*cerr << "DEBUG :: " << #OBJTYPE << "_dtor"<< endl;*/\
				BODY \
			} \
		}

//Host and device buffers
static ErlNifResourceType* hostBuffer_rt = NULL;
static ErlNifResourceType* deviceBuffer_rt = NULL;

DEF_DTOR(cl_mem, (NULL) ;)

//CL Program
static ErlNifResourceType* program_rt = NULL;
DEF_DTOR(cl_program, clReleaseProgram(*_obj);)


//CL Kernel with synchronized access
static ErlNifResourceType* kernel_sync_rt = NULL;

typedef struct _kernel_sync {
	cl_kernel kernel;
	ErlNifMutex *mtx;
} kernel_sync;

static kernel_sync* ctor_kernel_sync(cl_kernel kernel) {

	kernel_sync* pKernel_s = (kernel_sync*) enif_alloc_resource( kernel_sync_rt, sizeof(kernel_sync) );

	pKernel_s->mtx = enif_mutex_create((char*)"setkernelArgMutex");

	pKernel_s->kernel = kernel;

	return pKernel_s;
}

static void dtor_kernel_sync(ErlNifEnv* env, void* obj)
{
	kernel_sync* ker_s = (kernel_sync*) obj;
	if(ker_s) {

#ifdef DEBUG
		cerr << "DEBUG :: kernel_sync_dtor" << endl;
#endif
		enif_mutex_destroy(ker_s->mtx);

		clReleaseKernel(ker_s->kernel);
	}
}

//CL event
static ErlNifResourceType* event_rt = NULL;
DEF_DTOR(cl_event, clReleaseEvent(*_obj);)

/*
*	Worker thread handler, exploits NIF garbage collection mechanism
*	so to be sure to join a thread before module unload (as required by NIF spec)
*/
static ErlNifResourceType* thread_handler_rt = NULL;
/*thread handler: it just joins a thread*/
DEF_DTOR(ErlNifTid, enif_thread_join(*_obj,NULL);)


//Resource object declaration macro, used in the load function
#define DECL_RESOURCE_OBJ(RTYPE_VAR, NAME, DTOR, FLAGS) \
{	ErlNifResourceType* rt = enif_open_resource_type(env, NULL, NAME, DTOR, FLAGS, NULL); \
	if (rt == NULL) return -1; \
	RTYPE_VAR = rt;\
}

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{

	/*BUFFERS RO */
	DECL_RESOURCE_OBJ(hostBuffer_rt, "host_buffer", dtor_cl_mem, ERL_NIF_RT_CREATE)
	DECL_RESOURCE_OBJ(deviceBuffer_rt, "device_buffer", dtor_cl_mem, ERL_NIF_RT_CREATE)

	/*Program and Kernel RO*/
	DECL_RESOURCE_OBJ(program_rt, "program", dtor_cl_program, ERL_NIF_RT_CREATE)
	DECL_RESOURCE_OBJ(kernel_sync_rt, "kernel", dtor_kernel_sync, ERL_NIF_RT_CREATE)

	/*Event RO*/
	DECL_RESOURCE_OBJ(event_rt, "event", dtor_cl_event, ERL_NIF_RT_CREATE)

	/*worker thread RO*/
	DECL_RESOURCE_OBJ(thread_handler_rt, "thread_handler", dtor_ErlNifTid, ERL_NIF_RT_CREATE)

	return 0;
}

/********************    BUFFER MANAGEMENT NIFs   ************************************/

static const char* MEM_READ_FLAG = "read";
static const char* MEM_WRITE_FLAG = "write";
static const char* MEM_READ_WRITE_FLAG = "read_write";

static const size_t MEM_FLAG_ATM_MAX_LEN  = strlen(MEM_READ_WRITE_FLAG)+1;

static bool parseMemFlags(const char* flag, cl_mem_flags* parsedFlag) {

	//parse flag
	if(strcmp(MEM_READ_FLAG, flag) == 0)
		*parsedFlag |= CL_MEM_READ_ONLY;
	else
		if(strcmp(MEM_WRITE_FLAG, flag) == 0)
			*parsedFlag |= CL_MEM_WRITE_ONLY;
		else
			if(strcmp(MEM_READ_WRITE_FLAG, flag) == 0)
				*parsedFlag |= CL_MEM_READ_WRITE;
			else
				return false;

	return true;
}


static ERL_NIF_TERM getBufferSize(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	OCL_INIT_CHECK()

	/*get the parameter (Buffer::buffer())***************/
	NIF_ARITY_CHECK(1)

	cl_mem* pBuffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &pBuffer)) { //is it a hostBuffer?

		if (!enif_get_resource(env, argv[0], deviceBuffer_rt, (void**) &pBuffer)) { // or a deviceBuffer?
			cerr << "ERROR :: releaseBuffer: 1st parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*pBuffer == NULL)
		return enif_make_badarg(env);

	/********************************************************/


	size_t szBufferByte = 0;

	CHK_SUCCESS(getBufferSizeByte(*pBuffer, &szBufferByte);)


	return enif_make_ulong(env, szBufferByte);
}


static ERL_NIF_TERM allocDeviceBuffer(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//-type rw_flags() :: read | write | read_write
	//allocBuffer(SizeByte::non_neg_integer(), Flags::rw_flag()) -> {ok, Buffer} | {error, Why}

	OCL_INIT_CHECK()

	/*get the parameter (SizeByte::non_neg_integer(), Flags::rw_flag())***************/
	NIF_ARITY_CHECK(2)

	//SizeByte::non_neg_integer()
	size_t szBufferByte;
	if (!enif_get_ulong(env, argv[0], &szBufferByte)) {

		cerr << "ERROR :: allocDeviceBuffer: 1st parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}

	//Flags::rw_flag()
	char flag[MEM_FLAG_ATM_MAX_LEN];
	if(! enif_get_atom(env, argv[1], flag, MEM_FLAG_ATM_MAX_LEN, ERL_NIF_LATIN1)) {

		cerr << "ERROR :: allocDeviceBuffer: 2nd parameter is not an rw_flag()" << endl ;
		return enif_make_badarg(env);
	}
	/**********************************************************************************/


	//parse flag
	cl_mem_flags mem_flags = 0;

	if(!parseMemFlags(flag, &mem_flags)) {
		cerr << "ERROR :: allocDeviceBuffer: 2nd parameter is not an rw_flag()" << endl ;
		return enif_make_badarg(env);
	}

	/*allocate the memory for the resource (a cl_mem) */
	cl_mem* pBuffer = NULL;

	pBuffer = (cl_mem*) enif_alloc_resource(deviceBuffer_rt, sizeof(cl_mem));

	//allocate the buffer
	CHK_SUCCESS_CLEANUP(createBuffer(szBufferByte, mem_flags, pBuffer); , enif_release_resource(pBuffer);)

	/*grant co-ownership to erlang*/
	ERL_NIF_TERM bufferRO = enif_make_resource(env, pBuffer);

	/*DON'T transfer the ownership to erlang because a releaseBuffer NIF is provided*/

	return enif_make_tuple2(env, ATOM(ok), bufferRO);
}

static ERL_NIF_TERM allocHostBuffer(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//allocHostBuffer(SizeByte::non_neg_integer()) -> {ok, Buffer} | {error, Why}

	OCL_INIT_CHECK()

	/*get the parameter (SizeByte::non_neg_integer())***************/
	NIF_ARITY_CHECK(1)

	//SizeByte::non_neg_integer()
	size_t szBufferByte;
	if (!enif_get_ulong(env, argv[0], &szBufferByte)) {
		cerr << "ERROR :: allocHostBuffer: first parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}

	/**********************************************************************************/

	/*allocate the memory for the resource (a cl_mem) */
	cl_mem* pBuffer = NULL;

	pBuffer = (cl_mem*) enif_alloc_resource( hostBuffer_rt, sizeof(cl_mem));

	//request buffer in (pinned) host memory
	CHK_SUCCESS_CLEANUP(createBuffer(szBufferByte, CL_MEM_ALLOC_HOST_PTR, pBuffer); , enif_release_resource(pBuffer);)

	/*grant co-ownership to erlang*/
	ERL_NIF_TERM bufferRO = enif_make_resource(env, pBuffer);

	/*DON'T transfer the ownership to erlang because a releaseBuffer NIF is provided*/

	return enif_make_tuple2(env, ATOM(ok), bufferRO);
}


static ERL_NIF_TERM releaseBuffer(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//releaseBuffer(Buffer::buffer())

	OCL_INIT_CHECK()

	/*get the parameter (Buffer::buffer())***********************************/
	NIF_ARITY_CHECK(1)

	cl_mem* pBuffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &pBuffer)) { //is a hostBuffer?

		if (!enif_get_resource(env, argv[0], deviceBuffer_rt, (void**) &pBuffer)) { // or a deviceBuffer?
			cerr << "ERROR :: releaseBuffer: 1st parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*pBuffer == NULL)
		return enif_make_badarg(env);

	/**********************************************************************************/


	cl_int err =
			clReleaseMemObject(*pBuffer);

	*pBuffer = NULL;
	enif_release_resource(pBuffer);

	if(err != CL_SUCCESS)
		return make_error_cl(env,err);

	return ATOM(ok);

}


static ERL_NIF_TERM copyBufferToBufferSameSize(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//copyBufferToBuffer(From::buffer(), To::buffer()) -> ok

	OCL_INIT_CHECK()

	/*get the parameters (From::buffer(), To::buffer())***************/
	NIF_ARITY_CHECK(2)

	cl_mem* src_buffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &src_buffer)) { //is a hostBuffer?

		if (!enif_get_resource(env, argv[0], deviceBuffer_rt, (void**) &src_buffer)) { // or a deviceBuffer?
			cerr << "ERROR :: copyBufferToBufferSameSize: first parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*src_buffer == NULL)
		return enif_make_badarg(env);

	cl_mem* dst_buffer = NULL;
	if (!enif_get_resource(env, argv[1], hostBuffer_rt, (void**) &dst_buffer)) { //is a hostBuffer?

		if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &dst_buffer)) { // or a deviceBuffer?
			cerr << "ERROR :: copyBufferToBufferSameSize: 2nd parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*dst_buffer == NULL)
			return enif_make_badarg(env);

	/**********************************************************************************/


#ifdef TIME
	timespec fun_start, fun_end;
	GET_TIME((fun_start));
#endif

	size_t szSrcBufferByte = 0, szDestBufferByte = 0;

	CHK_SUCCESS(getBufferSizeByte(*src_buffer, &szSrcBufferByte);)
	CHK_SUCCESS(getBufferSizeByte(*dst_buffer, &szDestBufferByte);)

	// Input Buffers must have the same size
	if(szSrcBufferByte != szDestBufferByte)
		return make_error(env, ATOM(skel_ocl_buffers_different_size));

	CHK_SUCCESS(copyBuffer(*src_buffer, *dst_buffer, szSrcBufferByte);)


#ifdef SEQ
	clFinish(getCommandQueue(0));
#endif
#ifdef TIME
	GET_TIME((fun_end));

	cerr << "copy: "<< nsec2usec(diff(&(fun_start), &(fun_end))) << endl;
#endif


	return ATOM(ok);
}



static ERL_NIF_TERM copyBufferToBufferSize(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//copyBufferToBuffer(From::buffer(), To::buffer(), CopySizeByte::non_neg_integer()) -> ok

	OCL_INIT_CHECK()

	/*get the parameters (From::buffer(), To::buffer(), CopySizeByte::non_neg_integer())***************/
	NIF_ARITY_CHECK(3)

	cl_mem* src_buffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &src_buffer)) { //is a hostBuffer?

		if (!enif_get_resource(env, argv[0], deviceBuffer_rt, (void**) &src_buffer)) { // or a deviceBuffer?
			cerr << "ERROR :: copyBufferToBufferSize: first parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*src_buffer == NULL)
		return enif_make_badarg(env);

	cl_mem* dst_buffer = NULL;
	if (!enif_get_resource(env, argv[1], hostBuffer_rt, (void**) &dst_buffer)) { //is a hostBuffer?

		if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &dst_buffer)) { // or a deviceBuffer?
			cerr << "ERROR :: copyBufferToBufferSize: 2nd parameter is not a buffer" << endl ;
			return enif_make_badarg(env);
		}
	}
	if(*dst_buffer == NULL)
			return enif_make_badarg(env);

	//CopySizeByte::non_neg_integer()
	size_t szCopyByte;
	if (!enif_get_ulong(env, argv[2], &szCopyByte)) {

		cerr << "ERROR :: copyBufferToBuffer: 3rd parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}

	/**********************************************************************************/

	CHK_SUCCESS(copyBuffer(*src_buffer, *dst_buffer, szCopyByte);)

	return ATOM(ok);
}



/** Copies the content of a list into a buffer*/
static ERL_NIF_TERM listToBuffer(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//listToBuffer(From::[float()], To::hostBuffer()) -> ok | {error, Why}
	OCL_INIT_CHECK()

	/*get the parameters (To::list())***************/
	NIF_ARITY_CHECK(2)

	ERL_NIF_TERM list = argv[0];
	if(!enif_is_list(env, list)) {
		cerr << "ERROR :: listToBuffer: first parameter is not a list" << endl;
		return enif_make_badarg(env);
	}

	cl_mem* pBuffer = NULL;
	if (!enif_get_resource(env, argv[1], hostBuffer_rt, (void**) &pBuffer) || *pBuffer == NULL) {
		cerr << "ERROR :: listToBuffer: second parameter is not a buffer" << endl ;
		return enif_make_badarg(env);
	}
	/************************************************************/

#ifdef TIME
	timespec fun_start, fun_end;
	GET_TIME((fun_start));
#endif

	//copy
	unsigned int list_len = 0, szListBytes = 0;

	enif_get_list_length(env, list, &list_len);
	szListBytes = list_len * sizeof(double);


	double* pMappedBuffer = NULL;
	//obtain a pointer to work on the buffer (out of bounds check done by OpenCL, returns CL_INVALID_VALUE)
	CHK_SUCCESS( mapBufferBlocking(*pBuffer, 0, szListBytes, CL_MAP_WRITE, &pMappedBuffer); )

	if(pMappedBuffer == NULL) {
		cerr << "pMappedBuffer is NULL."<< endl;
		return make_error(env, ATOM(cl_error));
	}

	// copies the content of input list into pBuffer
	list_to_double_array(env, list, list_len, pMappedBuffer);

	//unMap when done copying
	CHK_SUCCESS( unMapBuffer(*pBuffer, pMappedBuffer); )


#ifdef TIME
	GET_TIME((fun_end));

	cerr << "Unmarshalling: "<< nsec2usec(diff(&(fun_start), &(fun_end))) << endl;
#endif

	return ATOM(ok);
}

/** Create a list containing the elements of a buffer*/
static ERL_NIF_TERM _bufferToListLength(ErlNifEnv * env, cl_mem* pBuffer, size_t listLen) {

	size_t listLenByte = listLen * sizeof(double);

	double* pDoubleArray = NULL;

	//obtain a pointer to work on the buffer
	CHK_SUCCESS( mapBufferBlocking(*pBuffer, 0, listLenByte, CL_MAP_READ, &pDoubleArray);)

	ERL_NIF_TERM list = double_array_to_list(env, pDoubleArray, listLen);

	//unMap when done
	CHK_SUCCESS( unMapBuffer(*pBuffer, pDoubleArray); )

	return enif_make_tuple2(env, ATOM(ok), list);

}


/** Create a list containing the elements of a buffer*/
static ERL_NIF_TERM bufferToListLength(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//%%bufferToList(From::buffer(), Length::pos_integer()) -> {ok, [float()]}

	OCL_INIT_CHECK()

	/*get the parameters (From::Buffer, Length::pos_integer())***************/
	NIF_ARITY_CHECK(2)

	cl_mem* pBuffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &pBuffer) || *pBuffer == NULL) {

		cerr << "ERROR :: bufferToList: 1st parameter is not a host buffer" << endl ;
		return enif_make_badarg(env);
	}

	//Length::pos_integer()
	size_t listLen;
	if (!enif_get_ulong(env, argv[1], &listLen)) {

		cerr << "ERROR :: bufferToList: 2nd parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}

	/************************************************/

	// check that buffer contains enough elements

	size_t szBufferByte = 0;
	CHK_SUCCESS( getBufferSizeByte(*pBuffer, &szBufferByte); )

	size_t listLenByte = listLen * sizeof(double);

	if(listLenByte > szBufferByte) {
		cerr << "ERROR :: bufferToList: buffer doesn't contain Length elements" << endl ;
		return enif_make_badarg(env);
	}

	return _bufferToListLength(env, pBuffer, listLen);
}

/** Create a list containing the elements of a buffer*/
static ERL_NIF_TERM bufferToList(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//bufferToList(From::Buffer) -> {ok, []} | {error, Why}
	OCL_INIT_CHECK()
	/*get the parameter (From::Buffer)***************/

	NIF_ARITY_CHECK(1)

	cl_mem* pBuffer = NULL;
	if (!enif_get_resource(env, argv[0], hostBuffer_rt, (void**) &pBuffer) || *pBuffer == NULL) {

		cerr << "ERROR :: bufferToList: first parameter is not a host buffer" << endl ;
		return enif_make_badarg(env);
	}
	/************************************************/


#ifdef TIME
	timespec fun_start, fun_end;
	GET_TIME((fun_start));
#endif

	size_t szBufferByte;

	CHK_SUCCESS( getBufferSizeByte(*pBuffer, &szBufferByte); )

	size_t numBufferElements = szBufferByte / sizeof(double);

	ERL_NIF_TERM toReturn = _bufferToListLength(env, pBuffer, numBufferElements);


#ifdef TIME
	GET_TIME((fun_end));

	cerr << "Marshalling: "<< nsec2usec(diff(&(fun_start), &(fun_end))) << endl;
#endif

	return toReturn;
}




/********************    PROGRAM AND KERNEL NIFS    **************************/

static ERL_NIF_TERM buildProgramFromCharP(ErlNifEnv * env, const char* progSrc){

	/*allocate the memory for the resource (a cl_program) */
	cl_program* pProgram = NULL;

	pProgram = (cl_program*) enif_alloc_resource(
							program_rt,
							sizeof(cl_program)
						);

	//create and build the program releasing the resource in case of errors
	CHK_SUCCESS_CLEANUP(createBuildProgramFromString(progSrc, pProgram); , enif_release_resource(pProgram);)

//	enif_free(progSrc); should be freed by the caller

	/*grant co-ownership to erlang*/
	ERL_NIF_TERM programRO = enif_make_resource(env, pProgram);

	/*transfer the ownership to erlang*/
	enif_release_resource(pProgram);

	return enif_make_tuple2(env, ATOM(ok), programRO);

}

//Unmarshal an Erlang list allocating the needed space
#define ENIF_GET_STRING(CHAR_P, ARG) \
{ \
		unsigned int strLen = 0; /*without \0 */ \
		if(!enif_get_list_length(env, ARG, &strLen)) \
			return enif_make_badarg(env); \
		CHAR_P = (char *) enif_alloc(strLen++); /*including null terminator*/\
		enif_get_string(env, ARG, CHAR_P, strLen, ERL_NIF_LATIN1); \
}

#define ENIF_GET_ATOM(CHAR_P, ARG) \
{ \
	uint atom_len = 0; \
	if (!enif_get_atom_length(env, ARG,&atom_len, ERL_NIF_LATIN1)) \
		return enif_make_badarg(env); \
	CHAR_P = (char *) enif_alloc(atom_len++); /*including null terminator*/ \
	enif_get_atom(env, ARG, CHAR_P, atom_len,ERL_NIF_LATIN1); \
}

static ERL_NIF_TERM buildProgramFromString(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//buildProgramFromString(ProgramSrcString::iolist())
	OCL_INIT_CHECK()
	/*get the parameter (ProgramSrcString::iolist())***********************************/
	NIF_ARITY_CHECK(1)

	//ProgramSrcString may be a binary
	char *progSrc = NULL;
	ErlNifBinary bin;

	if(enif_is_binary(env, argv[0])) {

		enif_inspect_iolist_as_binary(env, argv[0], &bin);

		progSrc = (char*) enif_alloc(bin.size+1);

		strncpy(progSrc, (char*) bin.data, bin.size);

	}
	else //ProgramSrcString must then be a list
		ENIF_GET_STRING(progSrc, argv[0])

	/**********************************************************************************/

	ERL_NIF_TERM toReturnTerm = buildProgramFromCharP(env, progSrc);

	if(!progSrc)
		enif_free(progSrc);

	return toReturnTerm;

}

static ERL_NIF_TERM buildProgramFromFile(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//buildProgram(ProgramSrcPath::nonempty_string())
	OCL_INIT_CHECK()
	/*get the parameter (ProgramSrcPath::nonempty_string())*****************************/
	NIF_ARITY_CHECK(1)

	char *progPath = NULL;

	ENIF_GET_STRING(progPath, argv[0]) // progPath must be enif_free'd

	/**********************************************************************************/

	//Read from the file
	char* progSrc = NULL;

	if(! (progSrc = readFromFile(progPath)) )
		return enif_make_tuple2(env, ATOM(error), enif_make_atom(env,"opening_file"));


	ERL_NIF_TERM toReturn = buildProgramFromCharP(env, progSrc);

	enif_free(progPath);
	enif_free(progSrc);

	return toReturn;

}

static ERL_NIF_TERM _createKernel(ErlNifEnv * env, cl_program* pProgram, const char* kerName) {

	cl_kernel kernel = NULL;
	CHK_SUCCESS(createKernel(*pProgram, kerName, &kernel); )

	/*crate a kernel_sync*/
	kernel_sync* pKernel_s = ctor_kernel_sync(kernel);

	/*grant co-ownership to erlang*/
	ERL_NIF_TERM kernelRO = enif_make_resource(env, pKernel_s);

	/*transfer the ownership to erlang*/
	enif_release_resource(pKernel_s);


	return enif_make_tuple2(env, ATOM(ok), kernelRO);
}

static ERL_NIF_TERM createKernel(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//%%createKernel(Prog::program(), KerName::nonempty_string())
	OCL_INIT_CHECK()
	/*get the parameters (Prog::program(), KerName::nonempty_string())***************/
	NIF_ARITY_CHECK(2)

	//get Program

	cl_program* pProgram = NULL;
	if (!enif_get_resource(env, argv[0], program_rt, (void**) &pProgram)) {
			cerr << "ERROR :: createKernel: 1st parameter is not a program" << endl ;
			return enif_make_badarg(env);
	}

	char* kerName = NULL;
	ENIF_GET_STRING(kerName, argv[1])

	/**********************************************************************************/

	ERL_NIF_TERM toReturn = _createKernel(env, pProgram, kerName);

	enif_free(kerName);

	return toReturn;

}


static ERL_NIF_TERM generateProgramSrcFromFile(ErlNifEnv * env, char* funSrcPath, string fun_name, string type, string mapKernelStr, string& progSrcStr) {

		string funSrcStr = readFromFileStr(funSrcPath);

		if( funSrcStr.compare("NULL") == 0)
			return enif_make_tuple2(env, ATOM(error), ATOM(opening_file));

		string kernelSrcStr(mapKernelStr);

		replaceTextInString(kernelSrcStr, std::string("TYPE"), type);
		replaceTextInString(kernelSrcStr, std::string("FUN_NAME"), fun_name);

		//generate the final program from the src from file and kernel definition

		if(type.compare("double") == 0)
			progSrcStr.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");

		progSrcStr.append(funSrcStr);
		progSrcStr.append(kernelSrcStr);

		return ATOM(ok);

}


typedef unordered_map<std::string, cl_program> CLPrograms_Cache;

//Store built programs so to avoid building them again when used more than once
static CLPrograms_Cache* clPrograms_Cache;


static ERL_NIF_TERM createSkeletonKernelFromFile(ErlNifEnv * env, char* skeletonFunSrcFilePath, const string& skeletonFunName, const string& skeletonSrcStr, const char* kernelName, const string& type, bool cache_program)
{

	cl_program program;

	string skeletonFunSrcPath_str = string(skeletonFunSrcFilePath);

	if(cache_program && (clPrograms_Cache->count(skeletonFunSrcPath_str) > 0)) { //1 found, 0 not found

		#ifdef DEBUG
		cerr<< "DEBUG - looking for cached program.\n";
		#endif
		program = (*clPrograms_Cache)[skeletonFunSrcPath_str]; //use cached program

	}
	else {

		#ifdef DEBUG
		cerr<< "DEBUG - generating program.\n";
		#endif
		//generate the source for the final program from the source code in the file and skeleton kernel definition
		string progSrcStr;
		ERL_NIF_TERM result_atom =
				generateProgramSrcFromFile(env, skeletonFunSrcFilePath, skeletonFunName, type, skeletonSrcStr, progSrcStr);

		if(result_atom != ATOM(ok))
			return result_atom;

		const char* progSrc =  progSrcStr.c_str();

		CHK_SUCCESS(createBuildProgramFromString(progSrc, &program);) //TODO program never released

		if(cache_program) {
			#ifdef DEBUG
			cerr<< "DEBUG - adding new program to cache.\n";
			#endif
			(*clPrograms_Cache)[skeletonFunSrcPath_str] = program; //save built program into cache
		}

	}

	return _createKernel(env, &program, kernelName);


}

static ERL_NIF_TERM createMapKernelFromFile(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//createMapKernel(MapSrcFunFile::nonempty_string(), FunName::nonempty_string(), FunArity::pos_integer(), ProgCaching::atom())
	OCL_INIT_CHECK()
	/*get the parameter (MapFunSrcPath::nonempty_string())*****************************/
	NIF_ARITY_CHECK(4)

	char *mapFunSrcPath = NULL;
	ENIF_GET_STRING(mapFunSrcPath, argv[0]) // mapFunSrcPath must be enif_free'd

	char *mapFunName = NULL;
	ENIF_GET_STRING(mapFunName, argv[1]) // mapFunName must be enif_free'd

	uint funArity = 0;
	enif_get_uint(env,argv[2], &funArity);

	if(! enif_is_atom(env,argv[3]))
		return enif_make_badarg(env);

	ERL_NIF_TERM prog_caching_atom = argv[3];

	/**********************************************************************************/


	const string* pSkeletonSrcStr;
	const char* mapKernelName;

	switch (funArity)
	{	case 1 : 	mapKernelName = "MapKernel";
					pSkeletonSrcStr = &MapKernelStr;
					break;

		case 2 : 	mapKernelName = "MapKernel2";
					pSkeletonSrcStr = &Map2KernelStr;
					break;

		default: 	enif_free(mapFunSrcPath); //FunArity not supported
					enif_free(mapFunName);
					return make_error(env, ATOM(skel_ocl_fun_arity_not_supported));
	}

	const char* type = "double";

	bool cache_program = enif_is_identical(ATOM(cache), prog_caching_atom);

	ERL_NIF_TERM toReturn =
			createSkeletonKernelFromFile(env, mapFunSrcPath, string(mapFunName), *pSkeletonSrcStr, mapKernelName, type, cache_program);

	enif_free(mapFunSrcPath);
	enif_free(mapFunName);

	return toReturn;

}

static ERL_NIF_TERM createReduceKernelFromFile(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//buildProgram(ReduceFunSrcPath::nonempty_string(), ReduceFunName::nonempty_string())
	OCL_INIT_CHECK()
	/*get the parameter (ReduceFunSrcPath::nonempty_string())**************************/
	NIF_ARITY_CHECK(3)

	char *reduceFunSrcPath = NULL;

	ENIF_GET_STRING(reduceFunSrcPath, argv[0]) // reduceFunSrcPath must be enif_free'd

	char *reduceFunName = NULL;

	ENIF_GET_STRING(reduceFunName, argv[1]) // reduceFunName must be enif_free'd


	if(! enif_is_atom(env,argv[2]))
		return enif_make_badarg(env);

	ERL_NIF_TERM prog_caching_atom = argv[2];

	/**********************************************************************************/


	const char* reduceKernelName = "ReduceKernel";
	const char* type = "double";

	bool cache_program = enif_is_identical(ATOM(cache), prog_caching_atom);

	ERL_NIF_TERM toReturn =
			createSkeletonKernelFromFile(env, reduceFunSrcPath, string(reduceFunName), ReduceKernelStr, reduceKernelName, type, cache_program);

	enif_free(reduceFunSrcPath);
	enif_free(reduceFunName);

	return toReturn;

}


/*****************   SKELETONS   ***********************************************/




/*****************   MAP   ***********************************************/


/***************Map on device buffers***********************/

/*Implementation of n-ary map using device buffers */
/* Requirements:
 * - All buffers in pInputBuffrV must have the same size,
 * - pInputBufferC > 0
 * */
static ERL_NIF_TERM mapDD_Impl(ErlNifEnv * env, kernel_sync* pKernel_s, void** _pInputBufferV, uint pInputBufferC, void* _pOutputBuffer) {

#ifdef TIME
	timespec
		fun_start,
		fun_prologue_end,
		run_end,
		fun_end;

	GET_TIME((fun_start));
#endif

	cl_mem** pInputBufferV  = (cl_mem**)_pInputBufferV;
	cl_mem* pOutputBuffer = (cl_mem*) _pOutputBuffer;

	const uint OFFSET = 0;

	//All buffers have the same size, then check just the first one
	size_t szInputBufferByte = 0;
	CHK_SUCCESS(getBufferSizeByte(*(pInputBufferV[0]), &szInputBufferByte);)

	//*********set map kernel parameters**********************
	uint uiNumElem = szInputBufferByte / sizeof(double);

	cl_int ciErrNum = 0;

	uint i = 0;

	// clSetKernelArg is not thread-safe
	enif_mutex_lock(pKernel_s->mtx);

	for(i=0; i < pInputBufferC; i++)
		ciErrNum |= clSetKernelArg(pKernel_s->kernel, i, sizeof(cl_mem), (void*) pInputBufferV[i]);

	i--;
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, i+1, sizeof(cl_mem), (void*) pOutputBuffer);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, i+2, sizeof(unsigned int), (void*) &OFFSET); //Offset
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, i+3, sizeof(unsigned int), (void*) &uiNumElem);


	enif_mutex_unlock(pKernel_s->mtx);

	CHK_SUCCESS(ciErrNum;)


#ifdef TIME
		GET_TIME((fun_prologue_end));
#endif

	//**********execute the kernel*********************

	size_t szGlobalWorkSize =  uiNumElem;
//	size_t szLocalWorkSize =  1;

	//esegui calcolo
	CHK_SUCCESS(computeKernel(pKernel_s->kernel, &szGlobalWorkSize, NULL/*&szLocalWorkSize*/);) //let OpenCL decide the local work-size

#ifdef TIME
		GET_TIME((run_end));

		fun_end = run_end;
#endif

#ifdef TIME
	cerr <<
			"prologue + kernel time: " << nsec2usec(diff(&(fun_start), &(fun_end))) <<
			endl <<
			"\tFunction prologue time: " << nsec2usec(diff(&(fun_start), &fun_prologue_end)) <<
			endl <<
			"\tKERNEL computation time: " << nsec2usec(diff(&(fun_prologue_end), &run_end)) <<
			endl
			;
#endif

	return ATOM(ok);

}

/* unary map adapter (in/out on device)
 **/
static ERL_NIF_TERM mapDD(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//mapDD(Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(3)

	/*get the parameters (Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())************/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: mapDD: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	cl_mem* pInputBuffer = NULL;
	if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &pInputBuffer) || *pInputBuffer == NULL) {
		cerr << "ERROR :: mapDD: 2nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}

	cl_mem* pOutputBuffer = NULL;
	if (!enif_get_resource(env, argv[2], deviceBuffer_rt, (void**) &pOutputBuffer) || *pOutputBuffer == NULL) {
		cerr << "ERROR :: mapDD: 3nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/

	uint pInputBufferC = 1;
	cl_mem* pInputBufferV[1] = { pInputBuffer };

	return mapDD_Impl(env, pKernel_s,(void**) pInputBufferV, pInputBufferC, pOutputBuffer);
}



/*Binary map adapter (in/out on device)
 **/
static ERL_NIF_TERM map2DD(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//mapDD(Kernel::kernel(), InputBuffer1::deviceBuffer(), InputBuffer2::deviceBuffer(), OutputBuffer::deviceBuffer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(4)

	/*get the parameters (Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())************/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: map2DD: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	cl_mem* pInputBuffer1 = NULL;
	if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &pInputBuffer1) || *pInputBuffer1 == NULL) {
		cerr << "ERROR :: map2DD: 2nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}

	cl_mem* pInputBuffer2 = NULL;
	if (!enif_get_resource(env, argv[2], deviceBuffer_rt, (void**) &pInputBuffer2) || *pInputBuffer2 == NULL) {
		cerr << "ERROR :: map2DD: 3nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}

	cl_mem* pOutputBuffer = NULL;
	if (!enif_get_resource(env, argv[3], deviceBuffer_rt, (void**) &pOutputBuffer) || *pOutputBuffer == NULL) {
		cerr << "ERROR :: map2DD: 4th parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/

	size_t szInputBuffer1Byte = 0, szInputBuffer2Byte = 0;

	CHK_SUCCESS(getBufferSizeByte(*pInputBuffer1, &szInputBuffer1Byte);)
	CHK_SUCCESS(getBufferSizeByte(*pInputBuffer2, &szInputBuffer2Byte);)

	// Input Buffers must have the same size
	if(szInputBuffer1Byte != szInputBuffer2Byte)
		return make_error(env, ATOM(skel_ocl_buffers_different_size));

	uint pInputBufferC = 2; //Map2
	cl_mem * pInputBufferV[2] = { pInputBuffer1, pInputBuffer2 };

	return mapDD_Impl(env,pKernel_s,(void**) pInputBufferV, pInputBufferC, pOutputBuffer);

}


struct Map_fun_counters {

	struct Map_fun_round_counters  {
		uint numInput;
		timespec
			start[2], //TODO NUM_QUEUES
			load_start[2],
				*load_input_start[2],
				*load_unmarsh_end[2],
				*load_copyHD_end[2],
			load_end[2],

			run_start[2],
			run_end[2],

			unload_start,
				unload_copy_start[2],
				unload_copy_end[2],
				unload_marsh_end[2],
			unload_end,
			end[2];


		Map_fun_round_counters(uint _numInput) {

			numInput = _numInput;

			for(uint i = 0; i< 2; i++) {
				load_input_start[i] = new timespec[numInput];
				load_unmarsh_end[i] = new timespec[numInput];
				load_copyHD_end[i] = new timespec[numInput];
			}


		};

		~Map_fun_round_counters(){

			for(uint i = 0; i< 2; i++) {
				delete [] load_input_start[i];
				delete [] load_unmarsh_end[i];
				delete [] load_copyHD_end[i];
			}

		};


	}; //end Map_fun_round_counters

	timespec
		fun_start, fun_prologue, fun_end,
		mapInput_start, *mapInput_end,
		unmapInput_start, unmapInput_end,
		mapOutput_start, mapOutput_end,
		unmapOutput_end,

		releaseInBuffersH_T, releaseOutBufferH_T,
		releaseOutBufferD_T, releaseInBuffersD_T_start, releaseInBuffersD_T_end
	;

	uint numRounds;

	Map_fun_round_counters** round;

	Map_fun_counters(const uint& _numRounds, const uint& numInput) {

		mapInput_end = new timespec[numInput];

		numRounds = _numRounds;

		round = new Map_fun_round_counters*[numRounds];

		for(uint i = 0; i< numRounds; i++)
			round[i] = new Map_fun_round_counters(numInput);
	}

	~Map_fun_counters() {


		for(uint i = 0; i < numRounds; i++)
			delete round[i];

		delete [] round;

		delete [] mapInput_end;

	}

};







class MapLL_thread_params{

public:
	bool errorSignal;
	bool requestedExit;

	ERL_NIF_TERM errorTerm;

	cl_event mapOutput_evt;
	cl_event unmapInput_evt;
	cl_event lastMarshalling_evt;

	cl_event**  run_start_evt; //NUM_QUEUES for each round
	cl_event**  run_done_evt; //NUM_QUEUES for each round

	ErlNifEnv* env;
	ErlNifMutex* env_mtx;

	uint pInputC;
	cl_command_queue cmdQs[NUM_QUEUES];
	uint cmdQsC;
	cl_kernel* kernels;
	cl_mem** inputBuffersD;
	cl_mem outputBufferD;


	//Shared InputH/outputH. set by load, used by unload in marshalling.
	//Unload never writes a segment that Load hasn't copied yet on the device; ensured by stage sync: Load < run < unload
	double* pMappedOutputBufferH;
	cl_mem* inputBuffersH;
	cl_mem outputBufferH;

	ERL_NIF_TERM* currList;


	Map_fun_counters* counters;

	uint numQueues;
	uint numRounds;
	uint DELAY_LOOPS;

	uint uiNumElem;
	uint uiNumElemSegment;
	size_t szSegment;

	uint numSegments;

	ERL_NIF_TERM* outputTerms;


	explicit MapLL_thread_params(

			cl_event _mapOutput_evt,
			cl_event _unmapInput_evt,
			cl_event _lastMarshalling_evt,
			cl_event**  _run_start_evt,
			cl_event**  _run_done_evt,

			ErlNifEnv* _env,
			ErlNifMutex* _env_mtx,
			uint _pInputC,
			cl_command_queue* _cmdQs,
			cl_kernel* _kernels,
			cl_mem** _inputBuffersD,
			cl_mem _outputBufferD,

			cl_mem* _inputBuffersH,
			cl_mem _outputBufferH,

			ERL_NIF_TERM* _currList,

			Map_fun_counters* _counters,

			uint _numRounds,
			uint _numQueues,
			uint _DELAY_LOOPS,

			uint _uiNumElem,
			uint _uiNumElemSegment,
			size_t _szSegment,

			ERL_NIF_TERM* _outputTerms
	)
	{

		errorSignal = false;
		requestedExit = false;

		errorTerm = 0;

		mapOutput_evt = _mapOutput_evt;
		unmapInput_evt = _unmapInput_evt;

		lastMarshalling_evt = _lastMarshalling_evt;

		run_start_evt = _run_start_evt;
		run_done_evt = _run_done_evt;

		env = _env;
		env_mtx = _env_mtx;

		pInputC = _pInputC;

		for(int i = 0; i < NUM_QUEUES; i++)
			cmdQs[i] = _cmdQs[i];


		kernels = _kernels;
		inputBuffersD = _inputBuffersD;
		outputBufferD = _outputBufferD;


		inputBuffersH = _inputBuffersH;
		outputBufferH = _outputBufferH;

		currList = _currList;

		counters = _counters;

		numRounds = _numRounds;
		numQueues = _numQueues;
		DELAY_LOOPS = _DELAY_LOOPS;

		uiNumElem = _uiNumElem;
		uiNumElemSegment = _uiNumElemSegment;
		szSegment = _szSegment;

		outputTerms = _outputTerms;

		numSegments = uiNumElem/uiNumElemSegment;


	}


	void signalError() {

#ifdef DEBUG
		cerr << "DEBUG: MapLL - signalError().\n";
#endif

		errorSignal = true;
	}

	void requestExit() {
		requestedExit = true;
	}

};

void destroyMutexes(ErlNifMutex* mtxs[], uint len) {

	for (uint i = 0; i < len; ++i) {
		enif_mutex_destroy(mtxs[i]);
		mtxs[i] = NULL;
	}
}



struct MapLL_loadStage_unmarshalling_thread_params {

	Barrier* barrier;
	MapLL_thread_params* map_params;

	uint* segmentOffset;
	double** pMappedInputBuffersH;

	uint* idx;

	cl_command_queue curr_cmdQ;

	bool* seqMode;

	cl_event** round_start;

	bool errorSignal;
	bool* global_errorSignal;
	ERL_NIF_TERM errorTerm;
};

static void* mapLL_loadStage_unmarshalling_Thread(void* obj) {


#ifdef DEBUG
	char msg[128];
#endif

	std::pair<uint, MapLL_loadStage_unmarshalling_thread_params*> *id_load_params =
			(std::pair<uint, MapLL_loadStage_unmarshalling_thread_params*>*) obj;

	uint& i_input = id_load_params->first; //thread's ID
	MapLL_loadStage_unmarshalling_thread_params*& load_params = id_load_params->second;


	//local variables
	MapLL_thread_params*& params = load_params->map_params;

	Barrier*& barrier = load_params->barrier;

	uint*& segmentOffset = (load_params->segmentOffset);

	ErlNifEnv*& env = params->env;
	ErlNifMutex*& env_mtx = params->env_mtx;
	Map_fun_counters* &counters = params->counters;

	cl_command_queue& curr_cmdQ = load_params->curr_cmdQ;
	bool*& seqMode = (load_params->seqMode);
	cl_event**& round_start_evts = (load_params->round_start);

	uint*& idx = (load_params->idx);



	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load[%u] - starting\n", i_input); cerr << msg;
	#endif

	//unmarshalling:  L -> H
	//copy data:	H -> D
	for (uint i_round = 0 ; i_round < params->numRounds; ++i_round) {


		for(uint i_queue = 0 ; i_queue <  params->numQueues; i_queue++) {

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - awaiting signal [%u][%u] \n", i_input, i_round,i_queue); cerr << msg;
			#endif

			//wait for signal from load master thread
			clWaitForEvents(1, &(round_start_evts[i_round][i_queue]));

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - starting [%u][%u] \n", i_input, i_round,i_queue); cerr << msg;
			#endif


			//some other thread (unmarshalling or one of map's threads) has signaled an error, exit
			if(load_params->errorSignal || *(load_params->global_errorSignal)) {

				#ifdef DEBUG
				sprintf(msg, "DEBUG: Load[%u] - Someone signaled an error, exit.\n", i_input); cerr << msg;
				#endif
				goto cleanup;
			}


#ifdef TIME
			GET_TIME((counters->round[i_round]->load_input_start[i_queue][i_input]));
#endif

			//************list -> H ************

			double* pMappedInputBufferSegmentH = load_params->pMappedInputBuffersH[i_input] + (*segmentOffset);

			// copy current input list segment into the corresponding part of bufferH
			uint numUnmarshElems=
					list_to_double_arrayN(
							params->env,
							params->currList[i_input],
							pMappedInputBufferSegmentH, params->uiNumElemSegment,
							&(params->currList[i_input])
					);
			//currList is updated to the first element of the next segment (of the list)

			if(numUnmarshElems != params->uiNumElemSegment)  {//some problem happened during unmarshalling
				#ifdef DEBUG
				sprintf(msg, "DEBUG: Load[%u] - [%u][%u] - Error unmarshalling\n", i_input, i_round, i_queue); cerr << msg;
				#endif

				load_params->errorSignal = true;
				load_params->errorTerm = sync_make_error(env, params->env_mtx, ATOM(unmarshalling_error));

				return NULL;
			}

			//print_buffer_debug("inputBuffersH", inputBuffersH[i_input], oneIdx * params->szSegment, params->szSegment); // DEBUG
			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - [%u][%u] - L -> H\n", i_input,  i_round, i_queue );
			cerr << msg;
			#endif


#ifdef TIME
			GET_TIME((counters->round[i_round]->load_unmarsh_end[i_queue][i_input]));
#endif


#ifdef SEQ
			*seqMode = true;
#endif

			cl_event copy_Evt;
			//************  H -> D ************
			THREAD_CHK_SUCCESS_CLEANUP_GOTO(
					clEnqueueWriteBuffer(curr_cmdQ, params->inputBuffersD[i_input][*idx], CL_FALSE, 0, params->szSegment,
							(void*)pMappedInputBufferSegmentH, 0, NULL, *seqMode ? &copy_Evt : NULL
					); ,
					load_params->errorTerm,
					{
							sprintf(msg, "DEBUG: Load[%u] - [%u][%u] - ERROR H -> D \n", i_input, i_round, i_queue); cerr << msg;
							load_params->errorSignal = true;
					}
			)
			clFlush(params->cmdQs[i_queue]);


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - [%u][%u] - H -> D\n", i_input, i_round, i_queue);
			cerr << msg;
			#endif

#ifdef SEQ
			clWaitForEvents(1, &copy_Evt);
			clReleaseEvent(copy_Evt);
#endif

#ifdef TIME
			GET_TIME((counters->round[i_round]->load_copyHD_end[i_queue][i_input]));
#endif

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - WAITING at barrier [%u][%u]\n", i_input, i_queue, i_round); cerr << msg;
			#endif
			//synch with others unmarshalling threads
			barrier->await();

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load[%u] - CROSSED barrier [%u][%u]\n", i_input, i_queue, i_round); cerr << msg;
			#endif

		}
	}

	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load[%u] - DONE\n", i_input); cerr << msg;
	#endif

	cleanup:

	//in case of error must clear the skipped barrier, so to avoid deadlocking other load threads
	if(load_params->errorSignal || *(load_params->global_errorSignal)) {
		#ifdef DEBUG
		sprintf(msg, "DEBUG: Load[%u] - CLEARING skipped barrier\n", i_input); cerr << msg;
		#endif
		barrier->await();

	}

	return NULL;

}



static void* mapLL_loadStage_Thread(void* obj) {

	#ifdef DEBUG
	char msg[256];
	#endif

	#ifdef DEBUG
	cerr << "DEBUG: Load - Starting.\n";
	#endif


	MapLL_thread_params* params = (MapLL_thread_params *) obj;

	//parametri

	//env, env_mtx needed by CLEANUP macros
	ErlNifEnv* env = params->env;
	ErlNifMutex* env_mtx = params->env_mtx;

	Map_fun_counters*& counters = params->counters;

	//local variables
	size_t szInput_local = (params->szSegment / params->uiNumElemSegment) * params->uiNumElem;

	cl_int ciErrNum;

	double* pMappedInputBuffersH[params->pInputC];//pointers mapped into input buffers
	bool inputBufferIsMapped = false;

	bool seqMode = false;//is sequential execution activated?

	//initilize synchronization barrier for coordinating #pInputC load threads + master load thread
	Barrier round_end_barrier(params->pInputC + 1);


	//check if load stage is configured (in mapLL)
	bool unload_stage_isPresent = params->outputTerms != NULL ? true : false;


#ifdef TIME
		GET_TIME(counters->mapInput_start);
#endif

	//map InputBuffer
	for(uint i_list = 0; i_list < params->pInputC; ++i_list) {

		pMappedInputBuffersH[i_list] = (cl_double*)
						clEnqueueMapBuffer(params->cmdQs[0], params->inputBuffersH[i_list], CL_TRUE, CL_MAP_WRITE, 0, szInput_local, 0, NULL, NULL, &ciErrNum);

		THREAD_CHK_SUCCESS_CLEANUP_GOTO(
				ciErrNum; ,
				params->errorTerm,
				{
					cerr << "DEBUG: Load - clEnqueueMapBuffer\n";
					params->signalError();
				}
		)
#ifdef TIME
		GET_TIME(counters->mapInput_end[i_list]);
#endif
	}

	inputBufferIsMapped = true;

	//set output buffer's pointer to mapped buffer memory, since inputH[0] is outputH
	params->pMappedOutputBufferH = pMappedInputBuffersH[0];


	//get OpenCL context needed by clCreateUserEvent
	cl_context context;
	clGetCommandQueueInfo(params->cmdQs[0], CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);

	cl_event* round_start_evts[params->numRounds];

	for (uint i_round = 0 ; i_round < params->numRounds; i_round++) {

		round_start_evts[i_round] = new cl_event[params->numQueues];

		for(uint i_queue = 0 ; i_queue <  params->numQueues; i_queue++)
			round_start_evts[i_round][i_queue] = clCreateUserEvent(context, NULL);
	}



	MapLL_loadStage_unmarshalling_thread_params unmarsh_params;

	//initialize params' loop-independent variables
	unmarsh_params.barrier = &round_end_barrier;
	unmarsh_params.map_params = params;
	unmarsh_params.pMappedInputBuffersH = pMappedInputBuffersH;

	unmarsh_params.seqMode = &seqMode;
	unmarsh_params.round_start = round_start_evts;

	unmarsh_params.errorSignal = false;//unmarshalling threads signal (local)
	unmarsh_params.global_errorSignal = &params->errorSignal; //map threads signal (global)

	//loop-dependent variables
	//unmarsh_params.curr_cmdQ;
	//unmarsh_params.idx;
	//unmarsh_params.segmentOffset;




	//create unmarshalling threads
	ErlNifTid unmarshalling_threads[params->pInputC];

	std::pair<uint, MapLL_loadStage_unmarshalling_thread_params*> *id_params[params->pInputC];//(Thread-ID, thread params)

	for(uint i_input = 0; i_input < params->pInputC; i_input++) {

		id_params[i_input] = new std::pair<uint, MapLL_loadStage_unmarshalling_thread_params*>(i_input, &unmarsh_params);

		enif_thread_create((char*)"Unmarshalling", &unmarshalling_threads[i_input], mapLL_loadStage_unmarshalling_Thread, (void*) id_params[i_input], NULL);
	}





	/******LOAD LOOP*****/
	uint zeroIdx;//first segment index (the one that will be computed by the first queue)
	uint i_round;
	for (i_round = 0 ; i_round < params->numRounds; ++i_round) {

		zeroIdx = (params->numQueues * i_round);

		uint idx;
		uint segmentOffset;


		#ifdef DEBUG
		sprintf(msg, "DEBUG: Load - Starting Round[%d]\n", i_round);
		cerr << msg;
		#endif

		for(uint i_queue = 0 ; i_queue <  params->numQueues; i_queue++) {

#ifdef TIME
			GET_TIME((params->counters->round[i_round]->start[i_queue]));

			params->counters->round[i_round]->load_start[i_queue] =
						params->counters->round[i_round]->start[i_queue];

#endif

			idx = zeroIdx + i_queue;

			segmentOffset = (idx * params->uiNumElemSegment);

			cl_command_queue& curr_cmdQ = params->cmdQs[i_queue];

			cl_event& curr_run_start_evt = params->run_start_evt[i_queue][i_round];


			if(i_round > 0) { //from the first round onwards
				uint round_to_wait = i_round - 1;
				cl_event& curr_run_done_evt = params->run_done_evt[i_queue][round_to_wait];

				#ifdef DEBUG
				sprintf(msg, "DEBUG: Load - %d - Waiting for unload[%d][%d] to start.\n", i_queue, i_queue, round_to_wait);
				cerr << msg;
				#endif
				//WAIT for RUN stage, so to synch with unload stage
				clWaitForEvents(1, &(curr_run_done_evt));

			}


			if(params->errorSignal) { //someone has signaled an error, cleanup and exit
				#ifdef DEBUG
				sprintf(msg, "DEBUG: Load - %d - Someone has signaled an error, cleanup and exit\n", i_queue);
				cerr << msg;
				#endif
				goto cleanup;
			}

			/*
			 * list -> H
			 * H -> D
			 */

			//update unmarshalling threads' loop-dependent variables
			unmarsh_params.curr_cmdQ = curr_cmdQ;
			unmarsh_params.idx = &idx;
			unmarsh_params.segmentOffset = &segmentOffset;


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load - signaling unmarshalling threads [%d][%d]\n",i_round,i_queue); cerr << msg;
			#endif
			//signal unmarshalling threads
			clSetUserEventStatus(round_start_evts[i_round][i_queue], CL_SUCCESS);


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load - WAITING at barrier [%d][%d]\n",i_round,i_queue); cerr << msg;
			#endif
			//await unmarshalling completion
			round_end_barrier.await();

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load - barrier CROSSED [%d][%d]\n",i_round,i_queue); cerr << msg;
			#endif

			//check if everything went fine, otherwise set error term (i.e. propagate the error) and cleanup
			if(unmarsh_params.errorSignal){

			#ifdef DEBUG
			sprintf(msg, "DEBUG: Load - some unmarshalling thread signaled an error, cleanup.\n"); cerr << msg;
			#endif

				params->errorTerm = unmarsh_params.errorTerm;
				params->signalError();

				goto cleanup;
			}

			if(params->errorTerm){

				goto cleanup;
			}

			//******Segment is now loaded on device******

			#ifdef TIME
				GET_TIME((counters->round[i_round]->load_end[i_queue]));
			#endif

			//signal run stage, so it can process the segment i just loaded
			clSetUserEventStatus(curr_run_start_evt, CL_SUCCESS);


		}//queue loop end

	}//round loop end

	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - WAITING for unload, must UnmapInput\n"); cerr << msg;
	#endif


	if(unload_stage_isPresent) {
		//overlap unmapping with last marshalling (which is done by unload thread)
		clWaitForEvents(1, &params->unmapInput_evt);
	}


#ifdef TIME
		GET_TIME((counters->unmapInput_start));
#endif

	if(inputBufferIsMapped) {

		#ifdef DEBUG
		sprintf(msg, "DEBUG: Load - UNMAPPING Input\n"); cerr << msg;
		#endif

		//unmap input buffers
		for(uint i_list = 1/*0*/; i_list < params->pInputC; ++i_list)//starts from 1, 0 is reused as output buffer so is umapped by unload thread
			clEnqueueUnmapMemObject(params->cmdQs[0], params->inputBuffersH[i_list], pMappedInputBuffersH[i_list], 0, NULL, NULL);

#ifdef TIME
		GET_TIME((counters->unmapInput_end));
#endif
	}
	inputBufferIsMapped = false;



	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - CLEANING UP\n"); cerr << msg;
	#endif




	cleanup:

	//in case of error
	if(params->errorSignal || unmarsh_params.errorSignal)  {//release every lock before exiting to avoid deadlocks

		#ifdef DEBUG
		sprintf(msg, "DEBUG: Load - Error detected, set all events\n"); cerr << msg;
		#endif

		for (uint i_round = 0 ; i_round < params->numRounds; ++i_round)
			for (uint i_queue = 0 ; i_queue < params->numQueues; ++i_queue) {

				clSetUserEventStatus(params->run_start_evt[i_queue][i_round], CL_SUCCESS);

				clSetUserEventStatus(round_start_evts[i_round][i_queue], CL_SUCCESS);
			}

	}

	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - joining unmarshalling threads\n"); cerr << msg;
	#endif
	//join unmarshalling threads
	for(uint i_input = 0; i_input < params->pInputC; i_input++)
		enif_thread_join(unmarshalling_threads[i_input], NULL);


	if(inputBufferIsMapped) {
		//unmap input buffers
		for(uint i_list = 1/*0*/; i_list < params->pInputC; ++i_list) //starts from 1, 0 is reused as output buffer so is umapped by unload thread
			clEnqueueUnmapMemObject(params->cmdQs[0], params->inputBuffersH[i_list], pMappedInputBuffersH[i_list], 0, NULL, NULL);
	}


	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - RELEASING Input buffers [1,N_Input]\n"); cerr << msg;
	#endif
	//release inputBuffersH[] szInput
	for (uint i_list = 0; i_list < params->pInputC; ++i_list) //starts from 0 (unlike unmapping) because inputBuffersH[0] has been retained upon creation to allow it (by Master)
		clReleaseMemObject(params->inputBuffersH[i_list]);

#ifdef TIME
		GET_TIME((counters->releaseInBuffersH_T));
#endif


	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - DELETING id-params pairs\n"); cerr << msg;
	#endif

	//delete id-params pairs
	for (uint i_input = 0; i_input < params->pInputC; ++i_input)
		delete id_params[i_input];


	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - RELEASING round_start_events\n"); cerr << msg;
	#endif

	//release events
	for (uint i_round = 0 ; i_round < params->numRounds; i_round++) {

		for(uint i_queue = 0 ; i_queue <  params->numQueues; i_queue++)
			clReleaseEvent(round_start_evts[i_round][i_queue]);

		delete[] round_start_evts[i_round];
	}



	#ifdef DEBUG
	sprintf(msg, "DEBUG: Load - DONE\n");
	cerr << msg;
	#endif

	return NULL;
}




static void* mapLL_runStage_Thread(void* obj) {
	#ifdef DEBUG
	char msg[256];
	#endif

	MapLL_thread_params* params = (MapLL_thread_params *) obj;

	//local

	//needed by CLEANUP macros
	ErlNifEnv*& env = params->env;
	ErlNifMutex*& env_mtx = params->env_mtx;

	Map_fun_counters*& counters = params->counters;

	size_t szGlobalWorkSize =  params->uiNumElemSegment;

	bool isLastStage = params->outputTerms == NULL ? true : false;//Am I the last stage of the pipeline?

	uint zeroIdx = 0;
	for (uint i_round = 0 ; i_round < params->numRounds; ++i_round) {

		zeroIdx = (params->numQueues * i_round);


		#ifdef DEBUG
		sprintf(msg, "DEBUG: Run - Starting round[%d]\n", i_round);
		cerr << msg;
		#endif

		uint idx = 0;
		for (uint i_queue = 0 ; i_queue < params->numQueues; ++i_queue) {


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Run - %d - waiting to start\n", i_queue);
			cerr << msg;
			#endif

			idx = zeroIdx + i_queue;

			//current loop values
			cl_kernel& curr_kernel =  params->kernels[i_queue];
			cl_command_queue& curr_cmdQ = params->cmdQs[i_queue];

			cl_event& curr_run_start_evt = params->run_start_evt[i_queue][i_round];
			cl_event& curr_run_done_evt = params->run_done_evt[i_queue][i_round];

			uint segmentOffset = idx * params->uiNumElemSegment;


			//wait for load stage
			clWaitForEvents(1, &(curr_run_start_evt));


			if(params->errorSignal) { //Someone signaled an error, exit.
				#ifdef DEBUG
				sprintf(msg, "DEBUG: Run - %d - Someone signaled an error, exit.\n", i_queue);
				cerr << msg;
				#endif
				goto cleanup;
			}


			//*********set map kernels parameters**********************


			uint i_input = 0;

			uint numKerParams = params->pInputC + 3;


			cl_int ciErrNum[numKerParams];



			for(i_input=0; i_input < params->pInputC; i_input++)
				ciErrNum[i_input] = clSetKernelArg(curr_kernel, i_input, sizeof(cl_mem), (void*) &(params->inputBuffersD[i_input][idx]));

			i_input--;

			ciErrNum[i_input+1] = clSetKernelArg(curr_kernel, i_input+1, sizeof(cl_mem), (void*) &(params->outputBufferD));//round independent
			ciErrNum[i_input+2] = clSetKernelArg(curr_kernel, i_input+2, sizeof(unsigned int), (void*) &(segmentOffset));

			ciErrNum[i_input+3] = clSetKernelArg(curr_kernel, i_input+3, sizeof(unsigned int), (void*) &(params->uiNumElemSegment));//round independent


			for(uint i = 0; i < numKerParams; i++) {
				if(ciErrNum[i] != CL_SUCCESS) {
#ifdef DEBUG
					sprintf(msg, "DEBUG: Run - %d - Error SetKernel #%d\n", i_queue, i); cerr << msg;
#endif
					params->signalError();
					params->errorTerm = sync_make_error_cl(env, env_mtx, ciErrNum[i]);

					goto cleanup;
				}
			}


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Run - %d - Kernel Set\n", i_queue);
			cerr << msg;
			#endif


#ifdef TIME
			GET_TIME((counters->round[i_round]->run_start[i_queue]));
#endif


#ifdef SEQ
				seqMode = true;
#endif

			//************ Run ************
			cl_event exec_Evt;
			THREAD_CHK_SUCCESS_CLEANUP_GOTO(
					clEnqueueNDRangeKernel(curr_cmdQ, curr_kernel, 1, NULL, &szGlobalWorkSize, NULL/*&szLocalWorkSize*/, 0, NULL, &exec_Evt); ,
					params->errorTerm,
					{
						sprintf(msg, "DEBUG: Run - %d - Error NDRangeKernel\n", i_queue); cerr << msg;
						params->signalError();
					}
			)
			clFlush(curr_cmdQ);


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Run - %d - run\n", i_queue);
			cerr << msg;
			#endif


			clWaitForEvents(1, &exec_Evt);
			clReleaseEvent(exec_Evt);



#ifdef TIME
			GET_TIME((counters->round[i_round]->run_end[i_queue]));
			if(isLastStage)
				counters->round[i_round]->end[i_queue] = counters->round[i_round]->run_end[i_queue];
#endif

			//clWaitForEvents(1, &exec1_Evt);



			#ifdef DEBUG
			sprintf(msg, "DEBUG: Run - %d - wake up unload\n", i_queue);
			cerr << msg;
			#endif
			//signal unload stage
			clSetUserEventStatus(curr_run_done_evt, CL_SUCCESS);

		}//end queue

	}//end round

	cleanup:

	//in case of error
	if(params->errorSignal)  {//to avoid deadlocks, release every lock before exiting

		#ifdef DEBUG
		sprintf(msg, "DEBUG: Run - Error detected, set all events\n");
		cerr << msg;
		#endif

		for (uint i_round = 0 ; i_round < params->numRounds; ++i_round)
			for (uint i_queue = 0 ; i_queue < params->numQueues; ++i_queue)
				clSetUserEventStatus(params->run_done_evt[i_queue][i_round], CL_SUCCESS);
	}

	#ifdef DEBUG
	sprintf(msg, "DEBUG: Run - Done\n");
	cerr << msg;
	#endif

	return NULL;
}


static ERL_NIF_TERM mapLD_Impl_3Thread(ErlNifEnv * env, kernel_sync* pKernel_s, void** _pInputV, uint pInputC, void* _pOutputBuffer, uint listLength) {

#ifdef DEBUG
	cerr << "DEBUG: M - mapLD_Impl_3Thread" << endl;
#endif

	ERL_NIF_TERM toReturn;

	const uint NUM_SEGMENTS = NUM_SEGM;

	const uint NUM_QUEUES_LOCAL = NUM_QUEUES;

	const uint numRounds = NUM_SEGMENTS/NUM_QUEUES_LOCAL;


#if defined(TIME) || defined(TIME_FUN)

#define DIFF_T0(COUNTER) nsec2usec(diff( &(counters->fun_start), &(COUNTER)))

	Map_fun_counters* counters = new Map_fun_counters(numRounds, pInputC);
#endif

#ifdef TIME

	timespec createEventsT_start, createEventsT_end,
	createInputBuffersH[pInputC],
	createInputBuffersD[pInputC],

	createOutputBufferD;

	timespec joinT[2],

	releaseEventsT;


#endif

#if defined(TIME) || defined(TIME_FUN)
	GET_TIME(counters->fun_start);
#endif


	std::vector<cl_event> events;

	//cast input/output to type relevant here
	cl_mem outputBufferD = *((cl_mem*) _pOutputBuffer);

	//pInputV is an array of ERL_NIF_TERM* representing, each one, an erlang list
	ERL_NIF_TERM** inputLists = (ERL_NIF_TERM**) _pInputV;

	//input lists must have at least minListLen elements,
	//otherwise they can't be split in NUM_SEGMENTS segments
	uint listLen = listLength;


	uint minListLen = device_minBaseAddrAlignByte / sizeof(double);
	if(listLen/NUM_SEGMENTS < minListLen)
		return make_error(env, ATOM(skel_ocl_input_list_too_short));


	uint uiNumElem = listLen;

	uint uiNumElemSegment = uiNumElem / NUM_SEGMENTS;

	size_t szInput = uiNumElem * sizeof(double);

	size_t szSegment = szInput / NUM_SEGMENTS;


#ifdef DEBUG
	cerr << "DEBUG: M - mapLD_impl:" << "#Elem: "<< uiNumElem << " #ElemSegment: "<< uiNumElemSegment <<" szInput: "<< szInput << " szSegment: "<< szSegment << endl;
#endif

	cl_int ciErrNum = 0;

	//working queues
	cl_command_queue cmdQs[2] = {getCommandQueue(0), getCommandQueue(1)};

	cl_context context = getContext();

	//mutex protecting env from concurrent modifications (enif_make_xxx)
	ErlNifMutex* env_mtx = enif_mutex_create((char*)"env_mtx");


#ifdef TIME
	GET_TIME(createEventsT_start);
#endif

	//********events setup [NUM_QUEUES][numRounds])
	cl_event* run_start_evt[NUM_QUEUES_LOCAL];
	cl_event* run_done_evt[NUM_QUEUES_LOCAL];

	for (int i = 0; i < NUM_QUEUES_LOCAL; ++i) {
		run_start_evt[i] = new cl_event[numRounds];
		run_done_evt[i] = new cl_event[numRounds];

		for (int i_round = 0; i_round < numRounds; ++i_round) {
			run_start_evt[i][i_round] = clCreateUserEvent(context, &ciErrNum);
			run_done_evt[i][i_round] = clCreateUserEvent(context, &ciErrNum);

			events.push_back(run_start_evt[i][i_round]);
			events.push_back(run_done_evt[i][i_round]);
		}
	}

#ifdef TIME
	GET_TIME(createEventsT_end);
#endif

	//*********threads setup
	//Load, run. Each stage works on 2 command queues
	const uint NUM_STAGES = 2;
	ErlNifTid tid[NUM_STAGES];

	ErlNifTid& loadStage = tid[0];
	ErlNifTid& runStage = tid[1];

	//object holding everything needed by the threads
	MapLL_thread_params* conf = NULL;


	//***********for each input list, create:
	// one inputBufferH
	// NUM_SEGMENTS inputBufferD

	cl_mem inputBuffersH[pInputC];

	//[pInputC][NUM_SEGMENTS]
	cl_mem* inputBuffersD[pInputC];
	for (int i = 0; i < pInputC; ++i)
		inputBuffersD[i] = new cl_mem[NUM_SEGMENTS];



	for(uint i_list = 0; i_list < pInputC; ++i_list) {
		// create one input host buffer of size szInput, released by Unload
		CHK_SUCCESS_CLEANUP_GOTO(
				createBuffer(szInput, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &(inputBuffersH[i_list])); ,
				;
		)

#ifdef TIME
		GET_TIME(createInputBuffersH[i_list]);
#endif


#ifdef DEBUG
		cerr << "DEBUG: M - " << "inputBuffersH[" << i_list << "] CREATED: size: " << szInput << endl;
#endif


		//*********create NUM_SEGMENTS input device buffers having size szSegment*********
		//released by Unload
		for (int i_segm = 0; i_segm < NUM_SEGMENTS; ++i_segm) {

			CHK_SUCCESS_CLEANUP_GOTO(
					createBuffer(szSegment, CL_MEM_READ_ONLY, &(inputBuffersD[i_list][i_segm]) ); ,
					;
			)


#ifdef TIME
			GET_TIME(createInputBuffersD[i_list]);
#endif

#ifdef DEBUG
			cerr << "DEBUG: M - " << "inputBuffersD["<<i_list<<"][" <<i_segm<<"] CREATED: size: " << szSegment<< endl;
#endif
		}
#ifdef DEBUG
		cerr << "DEBUG: M - " << "inputBuffersD CREATED"<< endl;
#endif

	}

#ifdef TIME
	GET_TIME(createOutputBufferD); //2-3 usecs, lazy allocation
#endif


	//Clone kernel, 'cause i need two of them, one per queue
	cl_kernel kernels[NUM_QUEUES_LOCAL];

	kernels[0] = pKernel_s->kernel;

	if(NUM_QUEUES_LOCAL == TWO)
		cloneKernel(pKernel_s->kernel, &kernels[1]);

	if(NUM_QUEUES_LOCAL > MAX_QUEUES) {
		toReturn = make_error(env, ATOM(ocl_num_queue_not_supported));
		goto cleanup;
	}



	//****************************Main LOOP************************


	ERL_NIF_TERM currList[pInputC]; //iterator on the list, points to the head of the current segment of the list
	for (uint i = 0; i < pInputC; ++i)
		currList[i] = *(inputLists[i]);


	conf = new MapLL_thread_params (
			NULL,//mapOutput_evt,
			NULL,//unmapInput_evt,
			NULL,//lastMarshalling_evt,
			run_start_evt,
			run_done_evt,
			env, env_mtx,
			pInputC, cmdQs, kernels,
			inputBuffersD, outputBufferD,
			inputBuffersH,
			NULL, //outputBufferH,
			currList,
#ifdef TIME
			counters,
#else
			NULL,
#endif
			numRounds,
			NUM_QUEUES_LOCAL,
			0, //DELAY_LOOPS,
			uiNumElem, uiNumElemSegment, szSegment,
			NULL//outputTerms
			);


#ifdef DEBUG
	cerr << "DEBUG: M - Starting threads." << endl;
#endif

	enif_thread_create((char*) "mapLL loadStage", &loadStage, mapLL_loadStage_Thread, conf, NULL);

	enif_thread_create((char*) "mapLL runStage", &runStage, mapLL_runStage_Thread, conf, NULL);


#ifdef TIME
	GET_TIME((counters->fun_prologue));
#endif


	if(conf->errorSignal) {
		//something's gone wrong, cleanup and return the error term (the last one generated)
		#ifdef DEBUG
		cerr << "DEBUG: M - Got an error: cleanup and exit"<< endl;
		#endif

		toReturn = conf->errorTerm;
	}

	//wait for termination
	for(uint i = 0; i < NUM_STAGES; ++i) {

		enif_thread_join(tid[i], NULL);

#ifdef TIME
		GET_TIME((joinT[i]));
#endif

#ifdef DEBUG
		char msg[64];
		sprintf(msg, "DEBUG: M - Thread %d joined\n",i);
		cerr << msg;
#endif
	}

	//check again after joining, shouldn't be useful since list creation starts after the last marshalling, just before outputBuffer's unmapping,
	//at the very end of the computation when nobody can signal an error anymore.
	if(conf->errorSignal) {
		//something's gone wrong, cleanup and return the error term (the last one generated)
#ifdef DEBUG
		cerr << "DEBUG: M - Got an error: cleanup and exit"<< endl;
#endif
		toReturn = conf->errorTerm;
	}

	//******************CLEANUP label******************
	cleanup:


	//******************EVENTS******************
	if(!events.empty())
		releaseEvents(events);

	for(uint i = 0; i < NUM_QUEUES_LOCAL; i++){
		delete [] run_start_evt[i];
		delete [] run_done_evt[i];
	}
#ifdef TIME
	GET_TIME((releaseEventsT));
#endif

	//******************KERNELS******************
	if(NUM_QUEUES_LOCAL == 2){
		if(kernels[1])
			clReleaseKernel(kernels[1]); //2 usecs
		kernels[1] = NULL;
	}

	//******************ENV MUTEX******************
	if(env_mtx)
		destroyMutexes(&env_mtx, 1);//1 usec

	//******************CONF******************
	if(conf)
		delete conf;


#ifdef DEBUG
	cerr << "DEBUG: M - Cleanup done. Return"<< endl;
#endif


#if defined(TIME) || defined(TIME_FUN)
	GET_TIME((counters->fun_end));
#endif



	//************************FOLLOWS PROFILING INFOs********************


#if defined(TIME) || defined(TIME_FUN)

	cerr << endl <<
			"--------------------------------- map"<< pInputC <<"LD: (usec)---------------------------------" <<
			endl <<
			"Processing " << pInputC << " X " << uiNumElem <<" elements in "<< NUM_SEGMENTS <<" segments (" << uiNumElemSegment <<" per segment)." <<
			endl <<
			"Total run time: " << DIFF_T0(counters->fun_end) <<
			endl;

#endif

	/*
	 createEventsT,
	createInputBuffersH[pInputC], mapInputBuffers[pInputC],
	createInputBuffersD[pInputC],
 	mapOutputBufferH,
	createOutputBufferD;
	 */
#ifdef TIME

	long prologueTime = DIFF_T0(counters-> fun_prologue);


	cerr << endl <<
			"Function prologue time: " <<  prologueTime << "\t\t\t\t\tT: "<< prologueTime <<
			endl;

	char str[128];
	sprintf(str, "\tcreateEvents: %lu \t\t\t\t\tT: %lu\n",  DIFF(createEventsT_start, createEventsT_end), DIFF_T0( createEventsT_end)); cerr << str;

	for (uint i_input = 0; i_input < pInputC; i_input++) {
		sprintf(str, "\tcreateInputBuffersH[%d]: \t\t\t\tT: %lu\n", i_input, DIFF_T0( createInputBuffersH[i_input])); cerr << str;

		sprintf(str,
				"Load - mapInputBuffers[%d]: %lu \t\t\t\tT: %lu\n", i_input,
				i_input == 0 ?
						DIFF( counters->mapInput_start, counters->mapInput_end[i_input]) : DIFF( counters->mapInput_end[i_input-1], counters->mapInput_end[i_input]),
				DIFF_T0( counters->mapInput_end[i_input])
		);
		cerr << str;


		sprintf(str, "\tcreateInputBuffersD[%d]: \t\t\t\tT: %lu\n", i_input, DIFF_T0( createInputBuffersD[i_input])); cerr << str;
	}


	uint kernelTotal = 0, inputTotal = 0, outputTotal = 0, marshTotal = 0, unmarshTotal = 0;
	uint i_round;
	//round statistics
	for (i_round = 0; i_round < numRounds; i_round++) {
		uint tKernel[2] = {0,0};
		uint tInput = 0, tOutput = 0;
		uint tStartRound[2];

		cerr <<
				"\nRound["<< i_round << "] total time: " << DIFF(( counters->round[i_round]->start[0]), ( counters->round[i_round]->end[NUM_QUEUES_LOCAL-1]))<<
				endl;


		for (int i_queue = 0; i_queue < NUM_QUEUES_LOCAL; i_queue++) {

			inputTotal +=
					tInput = DIFF(( counters->round[i_round]->load_start[i_queue]), (counters->round[i_round]->load_end[i_queue]));


			tStartRound[i_queue] = DIFF_T0(counters->round[i_round]-> start[i_queue]);
			cerr <<
					"\n\t"<< i_queue << " - R["<< i_round << "] Start round\t\t\t\t\tT: " << tStartRound[i_queue]<<
					endl <<
					"\t"<< i_queue<< " - R["<< i_round << "] Load total time: " << tInput <<
					endl;

			for (uint i_input = 0; i_input < pInputC; i_input++) {
				long unmarsh = 0, t_Unmarsh;
				long cpDH_0 = 0, t_cpDH_0;

				unmarshTotal +=
						unmarsh = DIFF((counters->round[i_round]->load_input_start[i_queue][i_input]), (counters->round[i_round]->load_unmarsh_end[i_queue][i_input]));

				t_Unmarsh = DIFF_T0(counters->round[i_round]-> load_unmarsh_end[i_queue][i_input]);


				cpDH_0 = DIFF((counters->round[i_round]->load_unmarsh_end[i_queue][i_input]), (counters->round[i_round]->load_copyHD_end[i_queue][i_input]));
				t_cpDH_0 = DIFF_T0(counters->round[i_round]-> load_copyHD_end[i_queue][i_input]);

				cerr <<
						"\t\t"<< i_queue << " - R["<< i_round << "] Input["<< i_input<<"] unmarshall: " << unmarsh << "\t\tT: "<<  t_Unmarsh <<
						endl <<
						"\t\t"<< i_queue << " - R["<< i_round << "] Input["<< i_input<<"] H -> D: " << cpDH_0 << "\t\t\tT: "<<  t_cpDH_0 <<
						endl;
			}



			long t_Kernel = DIFF_T0(counters->round[i_round]-> run_end[i_queue]);
			kernelTotal +=
					tKernel[i_queue] = DIFF((counters->round[i_round]->run_start[i_queue]), (counters->round[i_round]->run_end[i_queue]));

			cerr <<
					"\t" << i_queue<< " - R["<< i_round << "] KERNEL computation time: " << tKernel[i_queue] << "\t\t\tT: "<<  t_Kernel <<
					endl;
				endl;

		}// end queues
	}//end rounds

	cerr << "Load - unmapInputBuffers: "<< DIFF(counters->unmapInput_start, counters->unmapInput_end) <<
			"\t\t\t\t\tT: " << DIFF_T0( counters->unmapInput_end)
	<< endl;

	cerr << "Load - releaseInBuffersH: " << DIFF(counters->unmapInput_end, counters->releaseInBuffersH_T) <<"\t\t\t\t\tT: " << DIFF_T0( counters->releaseInBuffersH_T) << "\n";


	for(int i = 0; i < NUM_STAGES; i++)
		cerr << "M - join "<< i << "\t\t\t\t\t\t\tT: " << DIFF_T0( joinT[i]) << "\n";


	cerr << "M - releaseEvents: "<< DIFF(joinT[NUM_STAGES-1], releaseEventsT ) << "\t\t\t\t\t\tT: " << DIFF_T0( releaseEventsT) << "\n";

	cerr << endl <<
			"KERNEL total time: " << kernelTotal << ". avg on "<< numRounds <<" rounds: " << kernelTotal/ NUM_SEGMENTS <<
			endl <<
			"Load time: " << inputTotal << ". avg: " << inputTotal / NUM_SEGMENTS <<
			endl <<
			"\tunmarshTime: " << unmarshTotal << ". avg: " << unmarshTotal / (NUM_SEGMENTS * pInputC) <<
			endl <<
			"\tcopyHDTime: " <<  inputTotal - unmarshTotal << ". avg: " << (inputTotal - unmarshTotal) / (NUM_SEGMENTS * pInputC) <<
			endl <<
			endl;

#endif

#if defined(TIME) || defined(TIME_FUN)

#undef DIFF_T0


	delete counters;
	counters = NULL;
#endif


	return ATOM(ok);

}

static ERL_NIF_TERM mapLD(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//mapLD(Kernel::kernel(), InputList::[double()], OutputBuffer::deviceBuffer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(4)

	/*get the parameters (Kernel::kernel(), InputList::[double()], OutputBuffer::deviceBuffer(), InputLength::non_neg_integer())****/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: mapLD: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	ERL_NIF_TERM inputList = argv[1];
	if (!enif_is_list(env, argv[1])) {
				cerr << "ERROR :: mapLD: 2nd parameter is not a list" << endl ;
				return enif_make_badarg(env);
	}

	cl_mem* pOutputBufferD = NULL;
	if (!enif_get_resource(env, argv[2], deviceBuffer_rt, (void**) &pOutputBufferD) || *pOutputBufferD == NULL) {
		cerr << "ERROR :: mapLD: 3nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}

	uint inputListLength;
	if (!enif_get_uint(env, argv[3], &inputListLength)) {

		cerr << "ERROR :: mapLD: 4th parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/


	uint pInputC = 1;
	ERL_NIF_TERM* pInputV[1] = { &inputList };

#ifdef DEBUG
	cerr << "DEBUG: mapLD" << endl;
#endif

	return mapLD_Impl_3Thread(env, pKernel_s,(void**) pInputV, pInputC, pOutputBufferD, inputListLength); //ok  | {error, Why}


}



static ERL_NIF_TERM map2LD(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//map2LD(Kernel::kernel(), InputList1::[double()], InputList2::[double()], OutputBuffer::deviceBuffer(), InputLength::non_neg_integer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(5)

	/*get the parameters (Kernel::kernel(), InputList1::[double()], InputList2::[double()], OutputBuffer::deviceBuffer())************/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: map2LD: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	ERL_NIF_TERM inputLists[2];

	inputLists[0] = argv[1];
	if (!enif_is_list(env, argv[1])) {
		cerr << "ERROR :: map2LD: 2nd parameter is not a list" << endl ;
		return enif_make_badarg(env);
	}

	inputLists[1] = argv[2];
	if (!enif_is_list(env, argv[2])) {
		cerr << "ERROR :: map2LD: 2nd parameter is not a list" << endl ;
		return enif_make_badarg(env);
	}

	cl_mem* pOutputBuffer = NULL;
	if (!enif_get_resource(env, argv[3], deviceBuffer_rt, (void**) &pOutputBuffer) || *pOutputBuffer == NULL) {
		cerr << "ERROR :: map2LD: 4th parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}


	uint inputListLength;
	if (!enif_get_uint(env, argv[4], &inputListLength)) {

		cerr << "ERROR :: mapLL: 5th parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/

	uint pInputC = 2; //Map2
	ERL_NIF_TERM* pInputV[2] = { &(inputLists[0]), &(inputLists[1]) };

#ifdef DEBUG
	cerr << "DEBUG: map2LD"<< endl;
#endif


	return mapLD_Impl_3Thread(env, pKernel_s,(void**) pInputV, pInputC, pOutputBuffer, inputListLength); //ok  | {error, Why}
}


static void* mapLL_unloadStage_Thread(void* obj) {

	#ifdef DEBUG
	char msg[256];
	#endif
	MapLL_thread_params* params = (MapLL_thread_params *) obj;

	//parameters

	ErlNifEnv*& env = params->env; //needed by CLEANUP macros
	ErlNifMutex*& env_mtx = params->env_mtx;

	Map_fun_counters*& counters = params->counters;

	//locals
	size_t szInput_local = (params->szSegment / params->uiNumElemSegment) * params->uiNumElem;

	cl_int ciErrNum;



	uint zeroIdx;
	uint segmentOffset;
	size_t szSegmentOffset;

	for (uint i_round = 0 ; i_round < params->numRounds; ++i_round) {

		zeroIdx = (params->numQueues * i_round);

		#ifdef DEBUG
		sprintf(msg, "DEBUG: Unload - Starting round[%d]\n", i_round);
		cerr << msg;
		#endif


		uint idx = 0;
		for (uint i_queue = 0 ; i_queue < params->numQueues; ++i_queue) {


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Unload - %d - waiting to start round[%d]\n", i_queue, i_round);
			cerr << msg;
			#endif

			idx = zeroIdx + i_queue;

			segmentOffset = idx * params->uiNumElemSegment;
			szSegmentOffset = idx * params->szSegment;


			cl_command_queue& curr_cmdQ = params->cmdQs[i_queue];
			cl_event& curr_run_done_evt = params->run_done_evt[i_queue][i_round];


			//WAIT for RUN stage
			clWaitForEvents(1, &(curr_run_done_evt));

			if(params->errorSignal) { //Someone signaled an error, exit.
				#ifdef DEBUG
				sprintf(msg, "DEBUG: Unload - %d - Someone signaled an error, exit.\n", i_queue);
				cerr << msg;
				#endif
				goto cleanup;
			}

#ifdef TIME
			GET_TIME(counters->round[i_round]->unload_copy_start[i_queue]);
#endif

			//*********** D -> H ************
			double* pMappedOutputBufferSegmentH = params->pMappedOutputBufferH + segmentOffset;


			//blocking read because i need the data immediately for the marshalling
			THREAD_CHK_SUCCESS_CLEANUP_GOTO(
					clEnqueueReadBuffer(curr_cmdQ, params->outputBufferD, CL_TRUE, szSegmentOffset, params->szSegment, (void*)pMappedOutputBufferSegmentH, 0, NULL , NULL); ,

					params->errorTerm ,
					{
						sprintf(msg, "DEBUG: Unload - %d - Error D -> H\n", i_queue); cerr << msg;
						params->signalError();
					}
			)
			clFlush(curr_cmdQ);


			#ifdef DEBUG
			sprintf(msg, "DEBUG: Unload - %d - D -> H\n", i_queue);
			cerr << msg;
			#endif


#ifdef TIME
			GET_TIME(counters->round[i_round]->unload_copy_end[i_queue]);
#endif


			//overlap last marshalling with input buffer unmapping (last segment of the last round)
			if(i_round == params->numRounds - 1 && i_queue == NUM_QUEUES-1)
				clSetUserEventStatus(params->unmapInput_evt, CL_SUCCESS);


			//************ H -> L ************

			ERL_NIF_TERM* outputTermsSegment = params->outputTerms + segmentOffset;

			enif_mutex_lock(env_mtx);

			for(uint i = 0; i < params->uiNumElemSegment; i++)
				outputTermsSegment[i] = enif_make_double(env, pMappedOutputBufferSegmentH[i]);

			enif_mutex_unlock(env_mtx);

#ifdef TIME
			GET_TIME((counters->round[i_round]->unload_marsh_end[i_queue]));
#endif



#ifdef TIME
			GET_TIME(counters->round[i_round]->end[i_queue]);
#endif

		#ifdef DEBUG
			sprintf(msg, "DEBUG: Unload - %d - H -> L\nDEBUG: %d - Round END\n", i_queue, i_queue);
			cerr << msg;
		#endif

		}//end queue

	}//end round

	//signal master to start list creation, overlaps with the following unmap
	clSetUserEventStatus(params->lastMarshalling_evt, CL_SUCCESS);

	cleanup:

	if(params->errorSignal)  {//release every lock before exiting to avoid deadlocks

		#ifdef DEBUG
		sprintf(msg, "DEBUG: Unload - Error detected, set all events\n");
		cerr << msg;
		#endif

		clSetUserEventStatus(params->unmapInput_evt, CL_SUCCESS);
		clSetUserEventStatus(params->lastMarshalling_evt, CL_SUCCESS);
	}

	//UNMAP using last queue
	clEnqueueUnmapMemObject(params->cmdQs[NUM_QUEUES-1], params->outputBufferH, params->pMappedOutputBufferH, 0, NULL, NULL);

#ifdef TIME
	GET_TIME(counters->unmapOutput_end);
#endif

	//**********************RELEASE BUFFERS********************
	// outputBufferH szInput
	clReleaseMemObject(params->outputBufferH);

#ifdef TIME
	GET_TIME(counters->releaseOutBufferH_T);
#endif

	//outputBufferD szInput
	clReleaseMemObject(params->outputBufferD);
	#ifdef TIME
		GET_TIME(counters->releaseOutBufferD_T);
	#endif


#ifdef TIME
	GET_TIME((counters->releaseInBuffersD_T_start));
#endif
	//inputBuffersD[pInputC][NUM_SEGMENTS] * szSegment = szInput
	for (uint i_input = 0; i_input <  params->pInputC; ++i_input) {

		for (int i_segm = 0; i_segm <  params->numSegments; ++i_segm) {
			clReleaseMemObject(params->inputBuffersD[i_input][i_segm]);
		}

		delete [] params->inputBuffersD[i_input];
	}
#ifdef TIME
	GET_TIME((counters->releaseInBuffersD_T_end));
#endif


	#ifdef DEBUG
	sprintf(msg, "DEBUG: Unload - Done\n");
	cerr << msg;
	#endif

	return NULL;
}



static ERL_NIF_TERM mapLL_Impl_3Thread(ErlNifEnv * env, kernel_sync* pKernel_s, void** _pInputV, uint pInputC, void* _pOutputBuffer, uint listLength) {

#ifdef DEBUG
	cerr << "DEBUG: M - mapLL_Impl_3Thread" << endl;
#endif

	ERL_NIF_TERM toReturn;

	const uint NUM_SEGMENTS = NUM_SEGM;

	const uint NUM_QUEUES_LOCAL = NUM_QUEUES;

	const uint numRounds = NUM_SEGMENTS/NUM_QUEUES_LOCAL;


#if defined(TIME) || defined(TIME_FUN)

#define DIFF_T0(COUNTER) nsec2usec(diff( &(counters->fun_start), &(COUNTER)))

	Map_fun_counters* counters = new Map_fun_counters(numRounds, pInputC);
#endif

#ifdef TIME

	timespec createEventsT_start, createEventsT_end,
	createInputBuffersH[pInputC],
	createInputBuffersD[pInputC],

	createOutputBufferD;

	timespec joinT[3],
	list_createdT_start, list_createdT_end,
	releaseEventsT;

#endif

#if defined(TIME) || defined(TIME_FUN)
	GET_TIME(counters->fun_start);
#endif


	std::vector<cl_event> events;

	//cast input/output to type relevant here
	//pInputV is an array of ERL_NIF_TERM* representing, each one, an erlang list
	ERL_NIF_TERM** inputLists = (ERL_NIF_TERM**) _pInputV;

	//input lists must have at least minListLen elements,
	//otherwise they can't be split in NUM_SEGMENTS segments
	uint listLen = listLength;


	uint minListLen = device_minBaseAddrAlignByte / sizeof(double);
	if(listLen/NUM_SEGMENTS < minListLen)
		return make_error(env, ATOM(skel_ocl_input_list_too_short));


	uint uiNumElem = listLen;

	uint uiNumElemSegment = uiNumElem / NUM_SEGMENTS;

	size_t szInput = uiNumElem * sizeof(double);

	size_t szSegment = szInput / NUM_SEGMENTS;


#ifdef DEBUG
	cerr << "DEBUG: M - mapLL_impl:" << "#Elem: "<< uiNumElem << " #ElemSegment: "<< uiNumElemSegment <<" szInput: "<< szInput << " szSegment: "<< szSegment << endl;
#endif

	cl_int ciErrNum = 0;

	//working queues
	cl_command_queue cmdQs[2] = {getCommandQueue(0), getCommandQueue(1)};

	cl_context context = getContext();

	//mutex protecting env from concurrent modifications (enif_make_xxx)
	ErlNifMutex* env_mtx = enif_mutex_create((char*)"env_mtx");

	//terms array from which to create the output list using enif_make_list_from_array, filled by unload stage
	ERL_NIF_TERM* outputTerms = NULL;


#ifdef TIME
	GET_TIME(createEventsT_start);
#endif

	//********events setup [NUM_QUEUES][numRounds])
	cl_event* run_start_evt[NUM_QUEUES_LOCAL];
	cl_event* run_done_evt[NUM_QUEUES_LOCAL];

	for (int i = 0; i < NUM_QUEUES_LOCAL; ++i) {
		run_start_evt[i] = new cl_event[numRounds];
		run_done_evt[i] = new cl_event[numRounds];

		for (int i_round = 0; i_round < numRounds; ++i_round) {
			run_start_evt[i][i_round] = clCreateUserEvent(context, &ciErrNum);
			run_done_evt[i][i_round] = clCreateUserEvent(context, &ciErrNum);

			events.push_back(run_start_evt[i][i_round]);
			events.push_back(run_done_evt[i][i_round]);

		}
	}

	//to allow overlapping between unmarshalling and mapping the outputBuffer (can't map more in parallel)
	// fired by load stage after finishing the mapping of inputBuffer, wait by unload
	cl_event mapOutput_evt = clCreateUserEvent(context, &ciErrNum);
	events.push_back(mapOutput_evt);

	//to allow overlapping between input buffer unmapping and last marshalling (set by unload, wait by load).
	cl_event unmapInput_evt = clCreateUserEvent(context, &ciErrNum);
	events.push_back(unmapInput_evt);

	//to overlap output list creation to unmapOutputBuffer (set by unload, wait by master)
	cl_event lastMarshalling_evt = clCreateUserEvent(context, &ciErrNum);
	events.push_back(lastMarshalling_evt);

#ifdef TIME
	GET_TIME(createEventsT_end);
#endif

	//*********threads setup: for now, one thread per stage.
	//Load, run, unload. Each stage works on 2 command queues
	const uint NUM_STAGES = 3;
	ErlNifTid tid[NUM_STAGES];

	ErlNifTid& loadStage = tid[0];
	ErlNifTid& runStage = tid[1];
	ErlNifTid& unloadStage = tid[2];

	//object holding everything needed by the threads
	MapLL_thread_params* conf = NULL;


	//***********for each input list, create:
	// one inputBufferH
	// NUM_SEGMENTS inputBufferD

	cl_mem inputBuffersH[pInputC];

	//[pInputC][NUM_SEGMENTS]
	cl_mem* inputBuffersD[pInputC];
	for (int i = 0; i < pInputC; ++i)
		inputBuffersD[i] = new cl_mem[NUM_SEGMENTS];


	//reuse first inputBuffersH as output buffer
	cl_mem outputBufferH;

	// device output buffer (same size as input)
	cl_mem outputBufferD;

	for(uint i_list = 0; i_list < pInputC; ++i_list) {
		// create one input host buffer of size szInput, released by Unload
		CHK_SUCCESS_CLEANUP_GOTO(
				createBuffer(szInput, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &(inputBuffersH[i_list])); ,
				;
		)

#ifdef TIME
		GET_TIME(createInputBuffersH[i_list]);
#endif


#ifdef DEBUG
		cerr << "DEBUG: M - " << "inputBuffersH[" << i_list << "] CREATED: size: " << szInput << endl;
#endif


		//*********create NUM_SEGMENTS input device buffers having size szSegment*********
		//released by Unload
		for (int i_segm = 0; i_segm < NUM_SEGMENTS; ++i_segm) {

			CHK_SUCCESS_CLEANUP_GOTO(
					createBuffer(szSegment, CL_MEM_READ_ONLY, &(inputBuffersD[i_list][i_segm]) ); ,
					;
			)


#ifdef TIME
			GET_TIME(createInputBuffersD[i_list]);
#endif

#ifdef DEBUG
			cerr << "DEBUG: M - " << "inputBuffersD["<<i_list<<"][" <<i_segm<<"] CREATED: size: " << szSegment<< endl;
#endif
		}
#ifdef DEBUG
		cerr << "DEBUG: M - " << "inputBuffersD CREATED"<< endl;
#endif

	}

	outputBufferH = inputBuffersH[0];//reuse first inputBufferH as outputBufferH
	clRetainMemObject(outputBufferH);//inputBuffer e' rilasciato da Load, ma outputBuffer da Unload, cosi' posso lasciare la release in Load


	//*********Create one OUTPUT buffer on DEVICE*********
	//released by Unload
	CHK_SUCCESS_CLEANUP_GOTO(
			createBuffer(szInput, CL_MEM_WRITE_ONLY, &(outputBufferD) ); ,
			;
	)

#ifdef TIME
	GET_TIME(createOutputBufferD); //2-3 usecs, lazy allocation
#endif


	//****Create the array that will store the ERL_NIF_TERM representing the output elements*********

	outputTerms =
			(ERL_NIF_TERM*) enif_alloc(sizeof(ERL_NIF_TERM) * uiNumElem);

	if(outputTerms == NULL) {
		cerr << "DEBUG: M - " << "enif_alloc returned NULL."<< ciErrNum << endl;

		toReturn = make_error(env, ATOM(erl_enomem));
		goto cleanup;
	}



	//Clone kernel, 'cause i need two of them, one per queue
	cl_kernel kernels[NUM_QUEUES_LOCAL];

	kernels[0] = pKernel_s->kernel;

	if(NUM_QUEUES_LOCAL == TWO)
		cloneKernel(pKernel_s->kernel, &kernels[1]);

	if(NUM_QUEUES_LOCAL > MAX_QUEUES) {
		toReturn = make_error(env, ATOM(ocl_num_queue_not_supported));
		goto cleanup;
	}



	//****************************Main LOOP************************


	ERL_NIF_TERM currList[pInputC]; //iterator on the list, points to the head of the current segment of the list
	for (uint i = 0; i < pInputC; ++i)
		currList[i] = *(inputLists[i]);


	conf = new MapLL_thread_params (
			mapOutput_evt,
			unmapInput_evt,
			lastMarshalling_evt,
			run_start_evt, run_done_evt,
			env, env_mtx,
			pInputC, cmdQs, kernels,
			inputBuffersD, outputBufferD,
			inputBuffersH,
			outputBufferH,
			currList,
#ifdef TIME
			counters,
#else
			NULL,
#endif
			numRounds,
			NUM_QUEUES_LOCAL,
			0, //DELAY_LOOPS,
			uiNumElem, uiNumElemSegment, szSegment,
			outputTerms
			);



#ifdef DEBUG
	cerr << "DEBUG: M - Starting threads." << endl;
#endif

	

	enif_thread_create((char*) "mapLL loadStage", &loadStage, mapLL_loadStage_Thread, conf, NULL);

	enif_thread_create((char*) "mapLL runStage", &runStage, mapLL_runStage_Thread, conf, NULL);

	enif_thread_create((char*) "mapLL unloadStage", &unloadStage, mapLL_unloadStage_Thread, conf, NULL);


#ifdef TIME
	GET_TIME((counters->fun_prologue));
#endif


	//wait last marshalling then start creating the list
	clWaitForEvents(1, &conf->lastMarshalling_evt);


	if(conf->errorSignal) {
		//something's gone wrong, cleanup and return the error term (the last one generated)
		#ifdef DEBUG
		cerr << "DEBUG: M - Got an error: cleanup and exit"<< endl;
		#endif

		toReturn = conf->errorTerm;
	}
	else {
		//everything went fine
		#ifdef DEBUG
		cerr << "DEBUG: M - Creating output list\n";
		#endif


#ifdef TIME
		GET_TIME(list_createdT_start);
#endif

		toReturn = enif_make_tuple2(env,ATOM(ok),enif_make_list_from_array(env, outputTerms, uiNumElem));

#ifdef TIME
		GET_TIME(list_createdT_end);
#endif
	}


	//wait for termination
	for(uint i = 0; i < NUM_STAGES; ++i) {

		enif_thread_join(tid[i], NULL);

#ifdef TIME
		GET_TIME((joinT[i]));
#endif

#ifdef DEBUG
		char msg[64];
		sprintf(msg, "DEBUG: M - Thread %d joined\n",i);
		cerr << msg;
#endif
	}

	//check again after joining, shouldn't be useful since list creation starts after the last marshalling, just before outputBuffer's unmapping,
	//at the very end of the computation when nobody can signal an error anymore.
	if(conf->errorSignal) {
		//something's gone wrong, cleanup and return the error term (the last one generated)
#ifdef DEBUG
		cerr << "DEBUG: M - Got an error: cleanup and exit"<< endl;
#endif
		toReturn = conf->errorTerm;
	}

	//******************CLEANUP label******************
	cleanup:

	if(outputTerms) {
		enif_free(outputTerms);//2-3 usecs
		outputTerms = NULL;
	}


	//******************EVENTS******************
	if(!events.empty())
		releaseEvents(events);

	for(uint i = 0; i < NUM_QUEUES_LOCAL; i++){
		delete [] run_start_evt[i];
		delete [] run_done_evt[i];
	}
#ifdef TIME
	GET_TIME((releaseEventsT));
#endif

	//******************KERNELS******************
	if(NUM_QUEUES_LOCAL == 2){
		if(kernels[1])
			clReleaseKernel(kernels[1]); //2 usecs
		kernels[1] = NULL;
	}

	//******************ENV MUTEX******************
	if(env_mtx)
		destroyMutexes(&env_mtx, 1);//1 usec

	//******************CONF******************
	if(conf)
		delete conf;


#ifdef DEBUG
	cerr << "DEBUG: M - Cleanup done. Return"<< endl;
#endif


#if defined(TIME) || defined(TIME_FUN)
	GET_TIME((counters->fun_end));
#endif



	//************************PROFILING INFOs********************

	//	map time counters
	//	start[2],
	//	load_start[2],
	//		*load_input_start[2],
	//		*load_unmarsh_end[2],
	//		*load_copyHD_end[2],
	//	load_end[2],
	//
	//	run_end[2],
	//
	//	unload_start,
	//		unload_copy_end[2],
	//		unload_marsh_end[2],
	//	unload_end,
	//	end[2];




#if defined(TIME) || defined(TIME_FUN)

	cerr << endl <<
			"--------------------------------- map"<< pInputC <<"LL: (usec)---------------------------------" <<
			endl <<
			"Processing " << pInputC << " X " << uiNumElem <<" elements in "<< NUM_SEGMENTS <<" segments (" << uiNumElemSegment <<" per segment)."<<
			endl <<
			"Total run time: " << DIFF_T0(counters->fun_end) <<
			endl;

#endif

#ifdef TIME

	long prologueTime = DIFF_T0(counters-> fun_prologue);


	cerr << endl <<
			"Function prologue time: " <<  prologueTime << "\t\t\t\t\tT: "<< prologueTime <<
			endl;

	char str[128];
	sprintf(str, "\tcreateEvents: %lu \t\t\t\t\tT: %lu\n",  DIFF(createEventsT_start, createEventsT_end), DIFF_T0( createEventsT_end)); cerr << str;

	for (uint i_input = 0; i_input < pInputC; i_input++) {
		sprintf(str, "\tcreateInputBuffersH[%d]: \t\t\t\tT: %lu\n", i_input, DIFF_T0( createInputBuffersH[i_input])); cerr << str;

		sprintf(str,
				"Load - mapInputBuffers[%d]: %lu \t\t\t\tT: %lu\n", i_input,
				i_input == 0 ?
						DIFF( counters->mapInput_start, counters->mapInput_end[i_input]) : DIFF( counters->mapInput_end[i_input-1], counters->mapInput_end[i_input]),
				DIFF_T0( counters->mapInput_end[i_input])
		);
		cerr << str;

		sprintf(str, "\tcreateInputBuffersD[%d]: \t\t\t\tT: %lu\n", i_input, DIFF_T0( createInputBuffersD[i_input])); cerr << str;
	}


	sprintf(str, "\tcreateOutputBufferD: \t\t\t\t\tT: %lu\n",  DIFF_T0( createOutputBufferD)); cerr << str;


	uint kernelTotal = 0, inputTotal = 0, outputTotal = 0, marshTotal = 0, unmarshTotal = 0;
	uint i_round;
	//round statistics
	for (i_round = 0; i_round < numRounds; i_round++) {
		uint tKernel[2] = {0,0};
		uint tInput = 0, tOutput = 0;
		uint tStartRound[2];

		cerr <<
				"\nRound["<< i_round << "] total time: " << DIFF(( counters->round[i_round]->start[0]), ( counters->round[i_round]->end[NUM_QUEUES_LOCAL-1]))<<
				endl;


		for (int i_queue = 0; i_queue < NUM_QUEUES_LOCAL; i_queue++) {

			inputTotal +=
					tInput = DIFF(( counters->round[i_round]->load_start[i_queue]), (counters->round[i_round]->load_end[i_queue]));


			tStartRound[i_queue] = DIFF_T0(counters->round[i_round]-> start[i_queue]);
			cerr <<
					"\n\t"<< i_queue << " - R["<< i_round << "] Start round\t\t\t\t\tT: " << tStartRound[i_queue]<<
					endl <<
					"\t"<< i_queue<< " - R["<< i_round << "] Load total time: " << tInput <<
					endl;

			for (uint i_input = 0; i_input < pInputC; i_input++) {
				long unmarsh = 0, t_Unmarsh;
				long cpDH_0 = 0, t_cpDH_0;

				unmarshTotal +=
						unmarsh = DIFF((counters->round[i_round]->load_input_start[i_queue][i_input]), (counters->round[i_round]->load_unmarsh_end[i_queue][i_input]));

				t_Unmarsh = DIFF_T0(counters->round[i_round]-> load_unmarsh_end[i_queue][i_input]);


				cpDH_0 = DIFF((counters->round[i_round]->load_unmarsh_end[i_queue][i_input]), (counters->round[i_round]->load_copyHD_end[i_queue][i_input]));
				t_cpDH_0 = DIFF_T0(counters->round[i_round]-> load_copyHD_end[i_queue][i_input]);

				cerr <<
						"\t\t"<< i_queue << " - R["<< i_round << "] Input["<< i_input<<"] unmarshall: " << unmarsh << "\t\tT: "<<  t_Unmarsh <<
						endl <<
						"\t\t"<< i_queue << " - R["<< i_round << "] Input["<< i_input<<"] H -> D: " << cpDH_0 << "\t\t\tT: "<<  t_cpDH_0 <<
						endl;
			}



			long t_Kernel = DIFF_T0(counters->round[i_round]-> run_end[i_queue]);
			kernelTotal +=
					tKernel[i_queue] = DIFF((counters->round[i_round]->run_start[i_queue]), (counters->round[i_round]->run_end[i_queue]));

			cerr <<
					"\t" << i_queue<< " - R["<< i_round << "] KERNEL computation time: " << tKernel[i_queue] << "\t\t\tT: "<<  t_Kernel <<
					endl;



			long marsh, cpDH;
			long t_marsh, t_cpDH;

			cpDH = DIFF((counters->round[i_round]->unload_copy_start[i_queue]), (counters->round[i_round]->unload_copy_end[i_queue]));
			t_cpDH = DIFF_T0(counters->round[i_round]->unload_copy_end[i_queue]);

			marshTotal +=
					marsh = DIFF((counters->round[i_round]->unload_copy_end[i_queue]), (counters->round[i_round]->unload_marsh_end[i_queue]));
			t_marsh = DIFF_T0(counters->round[i_round]->unload_marsh_end[i_queue]);


			cerr <<
					"\t"<< i_queue << " - R["<< i_round << "] Unload time: " << cpDH + marsh<<
					endl <<
					"\t\t"<< i_queue << " - R[" << i_round << "] D -> H: " << cpDH << "\t\t\t\tT: "<<  t_cpDH <<
					endl <<
					"\t\t"<< i_queue << " - R["<< i_round << "] marshall: " << marsh << "\t\t\tT: "<<  t_marsh <<
					endl;

		}// end queues
	}//end rounds

	cerr << "Load - unmapInputBuffers: "<< DIFF(counters->unmapInput_start, counters->unmapInput_end) <<
			"\t\t\t\t\tT: " << DIFF_T0( counters->unmapInput_end)
	<< endl;

	cerr << "Load - releaseInBuffersH: " << DIFF(counters->unmapInput_end, counters->releaseInBuffersH_T) <<"\t\t\t\t\tT: " << DIFF_T0( counters->releaseInBuffersH_T) << "\n";


	cerr << "Unload - unmapOutputBuffers: "<< DIFF(counters->round[numRounds-1]->end[NUM_QUEUES_LOCAL-1], counters->unmapOutput_end) <<
			"\t\t\t\tT: " << DIFF_T0( counters->unmapOutput_end)
	<< endl;

	cerr << "Unload - releaseOutBufferH: " << DIFF(counters->unmapOutput_end, counters->releaseOutBufferH_T) <<"\t\t\t\tT: " << DIFF_T0(counters->releaseOutBufferH_T) << "\n";

	cerr << "Unload - releaseOutBufferD: " << DIFF( counters->releaseOutBufferH_T, counters->releaseOutBufferD_T) <<"\t\t\t\tT: " << DIFF_T0(counters->releaseOutBufferD_T) << "\n";

	cerr << "Unload - releaseInBuffersD: " << DIFF(counters->releaseInBuffersD_T_start, counters->releaseInBuffersD_T_end) <<"\t\t\t\tT: " << DIFF_T0(counters->releaseInBuffersD_T_end) << "\n";


	cerr << "M - make_list: "<<  DIFF(list_createdT_start, list_createdT_end)<< "\t\t\t\t\t\tT: " << DIFF_T0( list_createdT_end) << "\n";


	for(int i = 0; i < NUM_STAGES; i++)
		cerr << "M - join "<< i << "\t\t\t\t\t\t\tT: " << DIFF_T0( joinT[i]) << "\n";


	cerr << "M - releaseEvents: "<< DIFF(joinT[NUM_STAGES-1], releaseEventsT ) << "\t\t\t\t\t\tT: " << DIFF_T0( releaseEventsT) << "\n";

	cerr << endl <<
			"KERNEL total time: " << kernelTotal << ". avg on "<< numRounds <<" rounds: " << kernelTotal/ NUM_SEGMENTS <<
			endl <<
			"Load time: " << inputTotal << ". avg: " << inputTotal / NUM_SEGMENTS <<
			endl <<
			"\tunmarshTime: " << unmarshTotal << ". avg: " << unmarshTotal / (NUM_SEGMENTS * pInputC) <<
			endl <<
			"\tcopyHDTime: " <<  inputTotal - unmarshTotal << ". avg: " << (inputTotal - unmarshTotal) / (NUM_SEGMENTS * pInputC) <<
			endl <<
			"\tmarshTime: " << marshTotal << ". avg: " << (marshTotal) / NUM_SEGMENTS <<
			endl;

#endif

#if defined(TIME) || defined(TIME_FUN)

#undef DIFF_T0

	delete counters;
	counters = NULL;
#endif


	return toReturn; //return list
}


static ERL_NIF_TERM mapLL(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	OCL_INIT_CHECK()
	const uint NUM_ARGS = 3;
	//mapLL(Kernel::kernel(), InputList::[double()], InputLength::non_neg_integer())

	NIF_ARITY_CHECK(NUM_ARGS)

	/*get the parameters (Kernel::kernel(), InputList::[double()], InputLength::non_neg_integer()****/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
		cerr << "ERROR :: mapLL: 1st parameter is not a kernel_sync" << endl ;
		return enif_make_badarg(env);
	}

	ERL_NIF_TERM inputList = argv[1];
	if (!enif_is_list(env, argv[1])) {
		cerr << "ERROR :: mapLL: 2nd parameter is not a list" << endl ;
		return enif_make_badarg(env);
	}


	uint inputListLength;
	if (!enif_get_uint(env, argv[2], &inputListLength)) {

		cerr << "ERROR :: mapLL: 3nd parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}

	/********************************************************************************/


	uint pInputC = 1;
	ERL_NIF_TERM* pInputV[1] = { &inputList };

#ifdef DEBUG
	cerr << "DEBUG: mapLL" << endl;
#endif

	return mapLL_Impl_3Thread(env,pKernel_s, (void**) pInputV, pInputC, NULL, inputListLength);

}


static ERL_NIF_TERM map2LL(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	//map2LL(Kernel::kernel(), InputList1::[double()], InputList2::[double()])
	OCL_INIT_CHECK()
	const int ARGS_NUM = 4;

	NIF_ARITY_CHECK(ARGS_NUM)

	/*get the parameters (Kernel::kernel(), InputList::[double()]****/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
		cerr << "ERROR :: map2LL: 1st parameter is not a kernel_sync" << endl ;
		return enif_make_badarg(env);
	}

	ERL_NIF_TERM inputList1 = argv[1];
	if (!enif_is_list(env, argv[1])) {
		cerr << "ERROR :: map2LL: 2nd parameter is not a list" << endl ;
		return enif_make_badarg(env);
	}

	ERL_NIF_TERM inputList2 = argv[2];
	if (!enif_is_list(env, argv[2])) {
		cerr << "ERROR :: map2LL: 3rd parameter is not a list" << endl ;
		return enif_make_badarg(env);
	}

	uint inputListLength;
	if (!enif_get_uint(env, argv[3], &inputListLength)) {

		cerr << "ERROR :: map2LL: 4th parameter is not a non_neg_integer()" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/


	uint pInputC = 2;
	ERL_NIF_TERM* pInputV[2] = { &inputList1, &inputList2 };

#ifdef DEBUG
	cerr << "DEBUG: map2LL" << endl;
#endif

	return mapLL_Impl_3Thread(env, pKernel_s, (void**) pInputV, pInputC, NULL, inputListLength);


}




/***********************   REDUCE   ********************************************/


static ERL_NIF_TERM reduceDD_Impl(ErlNifEnv * env, kernel_sync* pKernel_s, cl_mem* pInputBuffer, cl_mem* pOutputBuffer) {

	// %%reduceDD(Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())

	//Works with modified reduce6 kernel

	//Max values from NVIDIA reduce example
    const size_t maxThreads = 256;
    const size_t maxBlocks = 64;

	size_t uiNumBlocks = 0;
	size_t uiNumThreads = 0;

    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    //********Check buffers dimensions: Input must be power of 2
	size_t szInputBufferByte = 0;

	CHK_SUCCESS(getBufferSizeByte(*pInputBuffer, &szInputBufferByte);)

	uint uiN = szInputBufferByte / sizeof(double); //problem size is a power of 2

	getNumBlocksAndThreads(uiN, maxBlocks,maxThreads, uiNumBlocks, uiNumThreads);

	//allocate device output buffer having szOutputBufferByte size
	size_t szOutputBufferByte = sizeof(double) * uiNumBlocks ; //one value per block

	cl_mem outputBufferD = NULL;

	CHK_SUCCESS(createBuffer(szOutputBufferByte, CL_MEM_READ_WRITE, &outputBufferD);)

	// Decide size of shared memory (one value per thread but kernel needs at least 64*sizeof(double) bytes)
	size_t szLocalMemByte = (uiNumThreads <= 32) ? 2 * uiNumThreads * sizeof(double) : uiNumThreads * sizeof(double);

//	cerr << "DEBUG :: reduceDD: " << "problem size is:" << uiN << " maxThreads:" << maxThreads << " uiNumBlocks:" <<
//				uiNumBlocks << " uiNumThreads:" << uiNumThreads << " localMemSize:"<< szLocalMemByte << endl ;

	//*********set reduce kernel parameters (first reduction)**********************
	cl_int ciErrNum = 0;

	// clSetKernelArg is not thread-safe
	enif_mutex_lock(pKernel_s->mtx);

	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 0, sizeof(cl_mem), (void*) pInputBuffer);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 1, sizeof(cl_mem), (void*) &outputBufferD);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 2, sizeof(unsigned int), (void*) &uiN);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 3, szLocalMemByte, NULL);

	enif_mutex_unlock(pKernel_s->mtx);

	CHK_SUCCESS_CLEANUP(ciErrNum; , clReleaseMemObject(outputBufferD);)


	//****First reduce all elements so that each block produces one element**********
	globalWorkSize[0] = uiNumThreads * uiNumBlocks;
	localWorkSize[0]  =	uiNumThreads;

	cl_command_queue cmdQ_0 = getCommandQueue(0);

	//event which the second round awaits, needed to express the dependency that exists between the two rounds.
	//It's mandatory only when using an out-of-order queue.
	cl_event firstRoundEvt = NULL;

	CHK_SUCCESS_CLEANUP(
		clEnqueueNDRangeKernel(cmdQ_0, pKernel_s->kernel, 1, NULL, &(globalWorkSize[0]), &(localWorkSize[0]), 0, NULL, &firstRoundEvt); ,
		clReleaseMemObject(outputBufferD);

	)

	//*********Second reduction**********************
	uiN = uiNumBlocks;

	//since problem size (uiN) has changed, recompute uiNumThreads and szLocalMemByte
	getNumBlocksAndThreads(uiN, maxBlocks, maxThreads, uiNumBlocks, uiNumThreads);
	szLocalMemByte = (uiNumThreads <= 32) ? 2 * uiNumThreads * sizeof(double) : uiNumThreads * sizeof(double);

//	cerr << "DEBUG :: reduceDD: " << "problem size is:" << uiN << " maxThreads:" << maxThreads << " uiNumBlocks: 1" <<
//			" uiNumThreads:" << uiNumThreads << " localMemSize:"<< szLocalMemByte << endl ;

	enif_mutex_lock(pKernel_s->mtx);

	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 0, sizeof(cl_mem), (void*) &outputBufferD);  //use as input previous reduction's output buffer
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 1, sizeof(cl_mem), (void*) &outputBufferD);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 2, sizeof(unsigned int), (void*) &uiN);
	ciErrNum |= clSetKernelArg(pKernel_s->kernel, 3, szLocalMemByte, NULL);

	enif_mutex_unlock(pKernel_s->mtx);

	CHK_SUCCESS_CLEANUP(ciErrNum; , clReleaseMemObject(outputBufferD);)


	//**********compute the result scalar from numBlock vector*********************
	globalWorkSize[0] = 1 * uiNumThreads; //only one block left
	localWorkSize[0]  =	uiNumThreads;

	CHK_SUCCESS_CLEANUP(
			clEnqueueNDRangeKernel(cmdQ_0, pKernel_s->kernel, 1, NULL, &(globalWorkSize[0]), &(localWorkSize[0]), 1, &firstRoundEvt, NULL); ,
			clReleaseMemObject(outputBufferD);
			)

	//******copy the resulting scalar into the user-provided output Buffer
	//the value to be copied is outputBufferD[0]
	CHK_SUCCESS_CLEANUP(copyBuffer(outputBufferD, *pOutputBuffer, sizeof(double)); , clReleaseMemObject(outputBufferD);)


//	print_buffer_debug("Reduce_Output",*pOutputBuffer, sizeof(double)); //DEBUG


	clReleaseMemObject(outputBufferD);

	return ATOM(ok);

}



static ERL_NIF_TERM reduceDD(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	// %%reduceDD(Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(3)
	/*get the parameters (Kernel::kernel(), InputBuffer::deviceBuffer(), OutputBuffer::deviceBuffer())**********************/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: reduceDD: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	cl_mem* pInputBuffer = NULL;
	if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &pInputBuffer) || *pInputBuffer == NULL) {
		cerr << "ERROR :: reduceDD: 2nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}

	cl_mem* pOutputBuffer = NULL;
	if (!enif_get_resource(env, argv[2], deviceBuffer_rt, (void**) &pOutputBuffer) || *pOutputBuffer == NULL) {
		cerr << "ERROR :: reduceDD: 3nd parameter is not a device buffer" << endl ;
		return enif_make_badarg(env);
	}
	/********************************************************************************/

	size_t szInputBufferByte = 0;

	CHK_SUCCESS(getBufferSizeByte(*pInputBuffer, &szInputBufferByte);)

	uint uiN = szInputBufferByte / sizeof(double);

	if(!isPow2(uiN)) { //problem size MUST BE A POWER OF 2
		cerr << "ERROR :: reduceDD: problem size is not a power of two." << endl ;
		return enif_make_badarg(env);
	}

	return reduceDD_Impl(env,pKernel_s, pInputBuffer, pOutputBuffer);

}


static ERL_NIF_TERM reduceDL_Impl(ErlNifEnv * env, kernel_sync* pKernel_s, cl_mem* pInputBufferD, cl_mem* pOutputBuffer) {

#if defined(TIME) || defined(TIME_FUN)
	timespec
		fun_start,
		fun_end;

	GET_TIME((fun_start));
#endif

#ifdef TIME
	timespec
		fun_prologue_end,
		run_end,
		marsh_end;
#endif

	(void) pOutputBuffer; //unused

	std::vector<cl_mem> buffers;

	size_t szInputBufferDByte = 0;

	CHK_SUCCESS(getBufferSizeByte(*pInputBufferD, &szInputBufferDByte);)

	//*******allocate device input and output buffers having the size of the input and output ones*************
	cl_mem outputBufferD = NULL;

	size_t szOutputBufferDByte = sizeof(double);

	CHK_SUCCESS_CLEANUP(createBuffer(szOutputBufferDByte, CL_MEM_READ_WRITE, &outputBufferD); , releaseBuffers(buffers); )
	buffers.push_back(outputBufferD);

#ifdef TIME
		GET_TIME((fun_prologue_end));
#endif

	ERL_NIF_TERM returnTerm =
			reduceDD_Impl( env, pKernel_s, pInputBufferD, &outputBufferD);

#ifdef TIME
		clFinish(getCommandQueue(0));
		GET_TIME((run_end));
#endif

	if(returnTerm == ATOM(ok)) { // no errors occurred, return the resulting double

		double *pBuffer = NULL;
		CHK_SUCCESS_CLEANUP(mapBufferBlocking(outputBufferD, 0, szOutputBufferDByte, CL_MAP_READ, &pBuffer); , releaseBuffers(buffers);)

		returnTerm = enif_make_list1(env, enif_make_double(env, pBuffer[0]));

		unMapBuffer(outputBufferD, pBuffer);
	}

#ifdef TIME
		GET_TIME((marsh_end));
#endif


	releaseBuffers(buffers);

	ERL_NIF_TERM toReturn =  enif_make_tuple2(env, ATOM(ok), returnTerm);

#if defined(TIME) || defined(TIME_FUN)
	GET_TIME((fun_end));
#endif

#if defined(TIME) || defined(TIME_FUN)

	cerr <<
			endl <<
			"--------------------------------- ReduceDL (usec)---------------------------------" <<
			endl <<
			"Total run time: " << DIFF(fun_start, fun_end) << endl;
#endif

#ifdef TIME
	cerr <<
			"Function prologue time: " << DIFF(fun_start, fun_prologue_end) <<
			endl <<
			"KERNEL computation time: " << DIFF(fun_prologue_end, run_end) <<
			endl <<
			"Marshalling time: " << DIFF(run_end, marsh_end) <<
			endl
			;
#endif

	return toReturn;

}

static ERL_NIF_TERM reduceDL(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	// %%reduceHH(Kernel::kernel(), InputBuffer::deviceBuffer())
	OCL_INIT_CHECK()
	NIF_ARITY_CHECK(2)
	/*get the parameters (Kernel::kernel(), InputBuffer::deviceBuffer())**********************/

	kernel_sync* pKernel_s = NULL;
	if (!enif_get_resource(env, argv[0], kernel_sync_rt, (void**) &pKernel_s)) {
			cerr << "ERROR :: reduceDL: 1st parameter is not a kernel_sync" << endl ;
			return enif_make_badarg(env);
	}

	cl_mem* pInputBufferD = NULL;
	if (!enif_get_resource(env, argv[1], deviceBuffer_rt, (void**) &pInputBufferD)  || *pInputBufferD == NULL) {
		cerr << "ERROR :: reduceDL: 2nd parameter is not a host buffer" << endl ;
		return enif_make_badarg(env);
	}

	/********************************************************************************/

	ERL_NIF_TERM result =
			reduceDL_Impl(env, pKernel_s, pInputBufferD, NULL);

	if( enif_is_list(env, result))
		return enif_make_tuple2(env, ATOM(ok), result);
	else return result;

}




static ERL_NIF_TERM skeletonlib(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
	return enif_make_string(env,"Skel OCL ready",ERL_NIF_LATIN1);
}



/**
 * Initialize OpenCL.
 * Returns
 * - ok
 * - {error, "OpenCL already initialized."} when called more than once
 * - badarg in the case of any error during the initialization
 */
static ERL_NIF_TERM initOCL(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	if(!ocl_initialised) {

		if(!init())
			return make_error(env, ATOM(openCL_init_failed) );
		else {
			ocl_initialised = true;

			//init cl_program cache
			clPrograms_Cache = new CLPrograms_Cache();

			return ATOM(ok);
		}
	}
	else return
			make_error(env, ATOM(openCL_already_init) );
}

static ERL_NIF_TERM releaseOCL(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {

	if(ocl_initialised) {

#ifdef DEBUG
		std::cerr << "calling releaseOCL()" << std::endl;
#endif
		releaseOCL();
#ifdef DEBUG
		std::cerr << "releaseOCL() ok" << std::endl;
#endif

		delete clPrograms_Cache;

		ocl_initialised = false;

		return ATOM(ok);
	}
	else return enif_make_tuple2(
			env,
			ATOM(error),
			ATOM(OpenCL_not_initialized)
	);

}




/*

  NIF interface bindings 

 */
static ErlNifFunc nif_funcs[] = {

		{"skeletonlib"			, 1, 	skeletonlib				},
		{"cl_init"				, 0, 	initOCL					},
		{"cl_release"			, 0,	releaseOCL				},

		//Buffer
		{"listToBuffer"			, 2,	listToBuffer			},
		{"bufferToList"			, 1,	bufferToList			},
		{"bufferToList"			, 2,	bufferToListLength		},
		{"getBufferSize"		, 1,	getBufferSize			},
		{"allocDeviceBuffer"	, 2,	allocDeviceBuffer		},
		{"allocHostBuffer"		, 1,	allocHostBuffer			},
		{"releaseBuffer"		, 1,	releaseBuffer			},
		{"copyBufferToBuffer"	, 2,	copyBufferToBufferSameSize},
		{"copyBufferToBuffer"	, 3,	copyBufferToBufferSize	},
		//Program and kernel
		{"buildProgramFromFile"	, 1,	buildProgramFromFile	},
		{"buildProgramFromString",1,	buildProgramFromString	},
		{"createKernel"			, 2,	createKernel			},
		//MAP
		{"createMapKernel"		, 4,	createMapKernelFromFile	},

		{"mapDD"				, 3,	mapDD					},

		{"mapLD"				, 4,	mapLD					},
		{"mapLL"				, 3,	mapLL					},

		{"map2DD"				, 4,	map2DD					},
		{"map2LD"				, 5,	map2LD					},
		{"map2LL"				, 4,	map2LL					},

		//REDUCE
		{"createReduceKernel"	, 3,	createReduceKernelFromFile},

		{"reduceDD"				, 3,	reduceDD				},
		{"reduceDL"				, 2,	reduceDL				}

};

ERL_NIF_INIT(skel_ocl,nif_funcs,load,NULL,NULL,NULL)



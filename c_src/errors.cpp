/*********Atoms and errors functions**********/

//REQUIRES in scope:
//ErlNifEnv* env;
#define CHK_SUCCESS_CLEANUP(FUN_CALL, CLEANUP) \
		{ 	cl_int err;\
			err = FUN_CALL \
			if(err != CL_SUCCESS) { \
				CLEANUP \
				return make_error_cl(env,err); \
			} \
		}

//REQUIRES in scope:
//ERL_NIF_TERM toReturn;
//ErlNifEnv* env;
#define CHK_SUCCESS_CLEANUP_GOTO(FUN_CALL, CLEANUP) \
		{ 	cl_int err;\
			err = FUN_CALL \
			if(err != CL_SUCCESS) { \
				CLEANUP \
				toReturn = make_error_cl(env,err); \
				goto cleanup;\
			} \
		}


//#define THREAD_CHK_SUCCESS_CLEANUP(FUN_CALL, RET_VAR, CLEANUP) \
//		{ 	cl_int err;\
//			err = FUN_CALL \
//			if(err != CL_SUCCESS) { \
//				CLEANUP \
//				RET_VAR = sync_make_error_cl(env, env_mtx, err); \
//				return NULL;\
//			} \
//		}

//REQUIRES in scope:
//ERL_NIF_TERM toReturn;
//ErlNifEnv* env;
//ErlNifMutex* env_mtx;
#define THREAD_CHK_SUCCESS_CLEANUP_GOTO(FUN_CALL, RET_VAR, CLEANUP) \
		{ 	cl_int err; char msg[128];\
			err = FUN_CALL \
			if(err != CL_SUCCESS) { \
				CLEANUP \
				RET_VAR = sync_make_error_cl(env, env_mtx, err); \
				goto cleanup;\
			} \
		}

//REQUIRES in scope:
//ErlNifEnv* env;
#define CHK_SUCCESS(FUN_CALL) \
		{ 	cl_int err;\
			err = FUN_CALL \
			if(err != CL_SUCCESS) \
				return make_error_cl(env,err); \
		}


#define ATOM(name) get_atom(env, #name)

/* Get an atom by name, before creating one checks if it already exists.*/
ERL_NIF_TERM get_atom(ErlNifEnv* _env, const char* atom_name) {

	ERL_NIF_TERM atom;

	if(!enif_make_existing_atom(_env, atom_name, &atom, ERL_NIF_LATIN1))
		atom = enif_make_atom(_env, atom_name);

	return atom;
}


ERL_NIF_TERM parse_error_env(ErlNifEnv* env, cl_int err)
{
	#ifdef DEBUG
	char msg[64];
	sprintf(msg, "CL_ERROR_CODE: #%d\n", err);
	std::cerr << msg;
	#endif
	switch(err) {
	case CL_DEVICE_NOT_FOUND:
		return ATOM(cl_device_not_found);
	case CL_DEVICE_NOT_AVAILABLE:
		return ATOM(cl_device_not_available);
	case CL_COMPILER_NOT_AVAILABLE:
		return ATOM(cl_compiler_not_available);
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return ATOM(cl_mem_object_allocation_failure);
	case CL_OUT_OF_RESOURCES:
		return ATOM(cl_out_of_resources);
	case CL_OUT_OF_HOST_MEMORY:
		return ATOM(cl_out_of_host_memory);
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return ATOM(cl_profiling_info_not_available);
	case CL_MEM_COPY_OVERLAP:
		return ATOM(cl_mem_copy_overlap);
	case CL_IMAGE_FORMAT_MISMATCH:
		return ATOM(cl_image_format_mismatch);
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return ATOM(cl_image_format_not_supported);
	case CL_BUILD_PROGRAM_FAILURE:
		return ATOM(cl_build_program_failure);
	case CL_MAP_FAILURE:
		return ATOM(cl_map_failure);
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return ATOM(cl_misaligned_sub__buffer_offset);
	case CL_INVALID_VALUE:
		return ATOM(cl_invalid_value);
	case CL_INVALID_DEVICE_TYPE:
		return ATOM(cl_invalid_device_type);
	case CL_INVALID_PLATFORM:
		return ATOM(cl_invalid_platform);
	case CL_INVALID_DEVICE:
		return ATOM(cl_invalid_device);
	case CL_INVALID_CONTEXT:
		return ATOM(cl_invalid_context);
	case CL_INVALID_QUEUE_PROPERTIES:
		return ATOM(cl_invalid_queue_properties);
	case CL_INVALID_COMMAND_QUEUE:
		return ATOM(cl_invalid_command_queue);
	case CL_INVALID_HOST_PTR:
		return ATOM(cl_invalid_host_ptr);
	case CL_INVALID_MEM_OBJECT:
		return ATOM(cl_invalid_mem_object);
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return ATOM(cl_invalid_image_format_descriptor);
	case CL_INVALID_IMAGE_SIZE:
		return ATOM(cl_invalid_image_size);
	case CL_INVALID_SAMPLER:
		return ATOM(cl_invalid_sampler);
	case CL_INVALID_BINARY:
		return ATOM(cl_invalid_binary);
	case CL_INVALID_BUILD_OPTIONS:
		return ATOM(cl_invalid_build_options);
	case CL_INVALID_PROGRAM:
		return ATOM(cl_invalid_program);
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return ATOM(cl_invalid_program_executable);
	case CL_INVALID_KERNEL_NAME:
		return ATOM(cl_invalid_kernel_name);
	case CL_INVALID_KERNEL_DEFINITION:
		return ATOM(cl_invalid_kernel_definition);
	case CL_INVALID_KERNEL:
		return ATOM(cl_invalid_kernel);
	case CL_INVALID_ARG_INDEX:
		return ATOM(cl_invalid_arg_index);
	case CL_INVALID_ARG_VALUE:
		return ATOM(cl_invalid_arg_value);
	case CL_INVALID_ARG_SIZE:
		return ATOM(cl_invalid_arg_size);
	case CL_INVALID_KERNEL_ARGS:
		return ATOM(cl_invalid_kernel_args);
	case CL_INVALID_WORK_DIMENSION:
		return ATOM(cl_invalid_work_dimension);
	case CL_INVALID_WORK_GROUP_SIZE:
		return ATOM(cl_invalid_work_group_size);
	case CL_INVALID_WORK_ITEM_SIZE:
		return ATOM(cl_invalid_work_item_size);
	case CL_INVALID_GLOBAL_OFFSET:
		return ATOM(cl_invalid_global_offset);
	case CL_INVALID_EVENT_WAIT_LIST:
		return ATOM(cl_invalid_event_wait_list);
	case CL_INVALID_EVENT:
		return ATOM(cl_invalid_event);
	case CL_INVALID_OPERATION:
		return ATOM(cl_invalid_operation);
	case CL_INVALID_GL_OBJECT:
		return ATOM(cl_invalid_gl_object);
	case CL_INVALID_BUFFER_SIZE:
		return ATOM(cl_invalid_buffer_size);
	case CL_INVALID_MIP_LEVEL:
		return ATOM(cl_invalid_mip_level);
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return ATOM(cl_invalid_global_work_size);
	default:
		return ATOM(cl_unknown);
	}
}


// {error, Why}
ERL_NIF_TERM make_error(ErlNifEnv* env, ERL_NIF_TERM why_atom)
{
	return enif_make_tuple2(env, ATOM(error), why_atom);
}

ERL_NIF_TERM sync_make_error(ErlNifEnv* env, ErlNifMutex* env_mtx, ERL_NIF_TERM why_atom)
{
	ERL_NIF_TERM toReturn;

	enif_mutex_lock(env_mtx);

	toReturn = make_error(env, why_atom);

	enif_mutex_unlock(env_mtx);

	return toReturn;
}


// {error, Why}
ERL_NIF_TERM make_error_cl(ErlNifEnv* env, cl_int err)
{
	return enif_make_tuple2(env, ATOM(error), parse_error_env(env,err));
}

ERL_NIF_TERM sync_make_error_cl(ErlNifEnv* env, ErlNifMutex* env_mtx, cl_int err)
{
	ERL_NIF_TERM toReturn;

	enif_mutex_lock(env_mtx);

	toReturn = enif_make_tuple2(env, ATOM(error), parse_error_env(env, err));

	enif_mutex_unlock(env_mtx);

	return toReturn;
}




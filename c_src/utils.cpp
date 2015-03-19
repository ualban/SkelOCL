/**
 * A thread synchronization barrier implemented using erl_nif synch facilities
 */
class Barrier {

private:
	uint N_THREADS;

	uint waiting_threads_counts[2];
	uint current_counter;

	ErlNifMutex* mtx;
	ErlNifCond* cond;

public:
	explicit Barrier(uint n_Threads) {
		N_THREADS = n_Threads;
		waiting_threads_counts[0] = waiting_threads_counts[1] = 0;
		current_counter = 0;

		mtx = enif_mutex_create((char*) "map_load_thread_cond_mtx");
		cond = enif_cond_create((char*) "map_load_thread_cond");

		//std::cerr << "Barrier("<< N_THREADS <<") CREATED"<<endl;
	}
	~Barrier() {
		enif_cond_destroy(cond);
		enif_mutex_destroy(mtx);
	}

	void await(){

		enif_mutex_lock(mtx);

		uint local_counter = current_counter;

		waiting_threads_counts[local_counter]++;
		//std::cerr << "await(): waiting_threads_counts[local_counter]++: "<< waiting_threads_counts[local_counter] <<endl;


		if(waiting_threads_counts[local_counter] < N_THREADS) {
			//std::cerr << "await(): Barrier NOT full" <<endl;

			while(waiting_threads_counts[local_counter] < N_THREADS) {
			//	std::cerr << "await(): cond_wait waiting_threads_counts[local_counter]: "<< waiting_threads_counts[local_counter] <<endl;
				enif_cond_wait(cond, mtx);
			}
		}
		else {
			//std::cerr << "await(): Barrier FULL. waiting_threads:" <<waiting_threads_counts[local_counter] << " broadcast!" <<endl;

			current_counter ^= 1;
			waiting_threads_counts[current_counter] = 0;

			enif_cond_broadcast(cond);
		}

	//	std::cerr << "await(): END" <<endl;
		enif_mutex_unlock(mtx);
	}

};

/*array must be big enough to contain list.length elements*/
uint list_to_double_arrayN(ErlNifEnv *env, ERL_NIF_TERM list, double* array, uint arrayLen, ERL_NIF_TERM* lastListCell) {

	if(array == NULL)
		return 0;

	uint len = 0;

	if(enif_is_list(env, list)) {

		ERL_NIF_TERM curr_cell = list;

		uint i = 0;
		for(i = 0; enif_is_list(env, curr_cell) && i < arrayLen; i++) {

			ERL_NIF_TERM hd, tl;
			if(!enif_get_list_cell(env, curr_cell, &hd, &tl))
				break;//list is empty

			if(!(enif_get_double(env, hd, &(array[i]) ))) {
				std::cerr << "DEBUG: list_to_double_arrayN - Error: attempt to read float from something else!" << std::endl;
			}
			curr_cell = tl;

		}

		if(lastListCell != NULL) //user requested the term denoting the last cell
			*lastListCell = curr_cell;

		return i;

	} else { //term "list" is not a list

		#ifdef DEBUG
		std::cerr << "DEBUG: list_to_double_arrayN - Error: trying to convert a non list as a list"<< std::endl ;
		#endif

		return 0;
	}
}


uint sync_list_to_double_arrayN(ErlNifEnv *env, ErlNifMutex *mtx, ERL_NIF_TERM list, double* array, uint arrayLen, ERL_NIF_TERM* lastListCell) {

	enif_mutex_lock(mtx);

	uint result = list_to_double_arrayN(env, list, array,  arrayLen, lastListCell);

	enif_mutex_unlock(mtx);
	
	return result;

}

void list_to_double_array(ErlNifEnv *env, ERL_NIF_TERM list, uint listLen, double* array) {

	if(array == NULL)
		return;

	uint len = 0;

	if(enif_is_list(env, list)) {

		list_to_double_arrayN(env, list, array, listLen, NULL);
	}

}


ERL_NIF_TERM double_array_to_list(ErlNifEnv *env, double* array, size_t array_size) {

	if(array == NULL)
		return ATOM(error);

	ERL_NIF_TERM* floatTermArray =
			(ERL_NIF_TERM*) enif_alloc(sizeof(ERL_NIF_TERM) * array_size);

	for(uint i = 0; i < array_size; i++)
		floatTermArray[i] = enif_make_double(env, array[i]);

	ERL_NIF_TERM res = enif_make_list_from_array(env, floatTermArray, array_size);

	enif_free(floatTermArray);

	return res;
}

/*returns a string containing the content of the file at filePath
 * otherwise "NULL".*/
string readFromFileStr(char* filePath) {

	ifstream file(filePath, ios::in);
	if (!file.is_open()) {
		cerr << "Failed to open file for reading: " << filePath << endl;
		return "NULL";
	}

	ostringstream oss;
	oss << file.rdbuf();

	return oss.str();
}

/*returns a dinamically allocated string containing the content of the file at filePath
 * otherwise NULL.
 * The returned sting must be enif_free'd when not needed.
 * */
char* readFromFile(char* filePath) {

	string srcStdStr = readFromFileStr(filePath);

	char* toReturn = (char*) enif_alloc(srcStdStr.length()+1);

	return strcpy(toReturn, srcStdStr.c_str());

}


#define MIN(a, b) ((a < b) ? a : b)

int inline isPow2(unsigned int v) { return v && !(v & (v - 1));}

/*
 * Return a value that is nearest value that is power of 2.
 */
unsigned int nextPow2( unsigned int x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

// Round Up Division function
size_t roundUp(uint group_size, uint global_size)
{
    uint r = global_size % group_size;
    if(r == 0)
        return global_size;
    else
        return global_size + group_size - r;
}

/*!
 * Compute the number of threads and blocks to use for the REDUCTION kernel.
 * We set threads / block to the minimum of maxThreads and n/2 where n is
 * problem size. We observe the maximum specified number of blocks, because
 * each kernel thread can process more than 1 elements.
 *
 * \param n Problem size.
 * \param maxBlocks Maximum number of blocks that can be used.
 * \param maxThreads Maximum number of threads that can be used.
 * \param blocks An output parameter passed by reference. Specify number of blocks to be used.
 * \param threads An output parameter passed by reference. Specify number of threads to be used.
 */
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, long unsigned int &blocks, long unsigned int &threads)
{
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);

        if(maxBlocks > 0)
        	blocks = MIN(maxBlocks, blocks);
}

/*
 *  It finds all instances of a string in another string and replaces it with
 *  a third string.
 *
 */
void replaceTextInString(std::string& text, std::string find, std::string replace)
{
        std::string::size_type pos=0;
        while((pos = text.find(find, pos)) != std::string::npos)
        {
            text.erase(pos, find.length());
            text.insert(pos, replace);
            pos+=replace.length();
        }
}



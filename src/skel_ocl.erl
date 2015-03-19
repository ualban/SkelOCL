-module(skel_ocl).
-on_load(init/0).

-export([
	%%initializations NIFs
	init/0,
	cl_init/0,
	cl_release/0,
	skeletonlib/1,
	checkResult/1,

	%%Buffer NIFs
	listToBuffer/2,
	bufferToList/1,
	bufferToList/2,
	getBufferSize/1,
	allocDeviceBuffer/2,
	allocHostBuffer/1,
	releaseBuffer/1,
	copyBufferToBuffer/2,
	copyBufferToBuffer/3,
	
	%%Program and Kernel NIFs
	buildProgramFromString/1,
	buildProgramFromFile/1,
	createKernel/2,
	
	%Map NIF
	createMapKernel/2,
	createMapKernel/3,
	createMapKernel/4,
	
	mapDD/3,
	mapLL/3,
	mapLD/4,
	
	map2DD/4,
	map2LD/5,
	map2LL/4,
	
	%MapReduce
	mapReduceLL/3,
		
	%Reduce NIF
	createReduceKernel/2,
	createReduceKernel/3,
	
	reduceDD/3,
	reduceDL/2,
	reduceLL/3

	]).

init()->  %%TODO path is relative, can be used just by modules in the same directory
	erlang:load_nif("skel_ocl",0).

%% init() ->
%% 	PrivDir = case code:priv_dir(?MODULE) of
%% 				  {error, bad_name} ->
%% 					  EbinDir = filename:dirname(code:which(?MODULE)),
%% 					  AppPath = filename:dirname(EbinDir),
%% 					  filename:join(AppPath, "priv");
%% 				  Path ->
%% 					  Path
%% 			  end,
%% 	erlang:load_nif("." ++ filename:join(PrivDir, "skel_ocl"), 0).


-spec checkResult(Result) -> Result | Why when
		Result		:: 	{ok, Result} | {error, Why},
		Result		::	any(),
		Why			::	atom().

checkResult({ok, Ref}) when is_reference(Ref)-> 
	Ref;
checkResult({ok, Result}) -> 
	Result;
checkResult({error, Why}) -> 
	erlang:error(Why);
checkResult(ok) ->
	ok;
checkResult(ecl_lib_not_loaded) -> 
	erlang:error(ecl_lib_not_loaded)
.




%%Buffer Types and functions ----------------------------------

-type hostBuffer()		::	binary().
-type deviceBuffer()	::	binary().
-type buffer()			::	hostBuffer() | deviceBuffer().

-type rw_flag() 		::	read | write | read_write.

%%Get the size of Buffer in Bytes
-spec getBufferSize(Buffer) -> {ok, SizeByte} | {error, Why} when
		SizeByte	::	non_neg_integer(),
		Buffer		::	buffer(),
		Why			::	atom().

getBufferSize(_) -> {error, ecl_lib_not_loaded}.


%%Allocate a buffer on the host of size SizeByte bytes
-spec allocHostBuffer(SizeByte) -> {ok, Buffer} | {error, Why} when
		SizeByte	::	non_neg_integer(),
		Buffer		::	hostBuffer(),
		Why			::	atom().

allocHostBuffer(_) -> {error, ecl_lib_not_loaded}.

%%Allocate a buffer on the device of size SizeByte bytes, specifying kernel read/write permissions
-spec allocDeviceBuffer(SizeByte, Flags) -> {ok, Buffer} | {error, Why} when
		SizeByte	::	non_neg_integer(),
		Flags		:: 	rw_flag(),
		Buffer		::	deviceBuffer(),
		Why			::	atom().

allocDeviceBuffer(_,_) -> {error, ecl_lib_not_loaded}.

%%release a previously allocated buffer
-spec releaseBuffer(Buffer) -> ok when
		Buffer		::	buffer()
.

releaseBuffer(_) -> {error, ecl_lib_not_loaded}.

%%Copy data from a buffer to another having the same size
-spec copyBufferToBuffer(From, To) -> ok | {error, Why} when
		From		::	buffer(),
		To			::	buffer(),
		Why			::	atom().

copyBufferToBuffer(_,_) -> {error, ecl_lib_not_loaded}.

%%Copy CopySizeByte data from a buffer to another
-spec copyBufferToBuffer(From, To, CopySizeByte) -> ok | {error, Why} when
		From		::	buffer(),
		To			::	buffer(),
		CopySizeByte::	non_neg_integer(),
		Why			::	atom().

copyBufferToBuffer(_,_,_) -> {error, ecl_lib_not_loaded}.

%%Copy the content of an Erlang list into a host buffer.
-spec listToBuffer(From, To) -> ok | {error, Why} when
		From		::	[float()],
		To			::	hostBuffer(),
		Why			::	atom().

listToBuffer(_,_) -> {error, ecl_lib_not_loaded}.


%%Create a list having as elements the data in the host buffer 
-spec bufferToList(From) -> {ok, [float()]} | {error, Why} when
		From		::	hostBuffer(),
		Why			::	atom().

bufferToList(_) -> {error, ecl_lib_not_loaded}.


%%Create a list containing the first ListLength elements in the buffer 
-spec bufferToList(From, ListLength) -> {ok, [float()]} | {error, Why} when
		From		::	hostBuffer(),
		ListLength	::	pos_integer(),
		Why			::	atom().

bufferToList(_,_) -> {error, ecl_lib_not_loaded}.



%%Program and Kernel Types and functions ----------------------------------
-type program()		::	binary().
-type kernel()		::	binary().

%%Build an OpenCL program from the source code in the ProgSrcString iolist()
-spec  buildProgramFromString(ProgSrcString) -> {ok, program()} | {error, Why} when
		ProgSrcString::	iolist(),
		Why			::	atom().

buildProgramFromString(_) -> {error, ecl_lib_not_loaded}.

%%Build an OpenCL program object from the source code in the ProgSrcFile file
-spec  buildProgramFromFile(ProgSrcFile) -> {ok, program()} | {error, Why} when
		ProgSrcFile	::	nonempty_string(),
		Why			::	atom().

buildProgramFromFile(_) -> {error, ecl_lib_not_loaded}.

%%Create an OpenCL kernel object from a program object.
-spec  createKernel(Prog, KerName) -> {ok, kernel()} | {error, Why} when
		Prog		::	program(),
		KerName		::	nonempty_string(),
		Why			::	atom().

createKernel(_,_) -> {error, ecl_lib_not_loaded}.


%%Map functions -----------------------------

%%Create a map compatible kernel object from the source in MapSrcFunFile. The kernel name is FunName.
-spec   createMapKernel(MapSrcFunFile, FunName) -> {ok, kernel()} | {error, Why} when
		MapSrcFunFile	::	nonempty_string(),
		FunName			::	nonempty_string(),
		Why				::	atom().

createMapKernel(MapSrcFunFile, FunName) ->
	createMapKernel(MapSrcFunFile, FunName, 1, cache).

%%Create a map compatible kernel object from the source in MapSrcFunFile. The kernel name is FunName, the arity is FunArity
-spec   createMapKernel(MapSrcFunFile, FunName, FunArity) -> {ok, kernel()} | {error, Why} when
		MapSrcFunFile	::	nonempty_string(),
		FunName			::	nonempty_string(),
		FunArity		:: 	pos_integer(),
		Why				::	atom().

createMapKernel(MapSrcFunFile, FunName, FunArity) ->
	createMapKernel(MapSrcFunFile, FunName,FunArity, cache).

%%Create a map compatible kernel object from the source in MapSrcFunFile.
%%The kernel name is FunName, the arity is FunArity.
%%Program caching policy must also be specified. 
-spec   createMapKernel(MapSrcFunFile, FunName, FunArity, CacheKernel) -> {ok, kernel()} | {error, Why} when
		MapSrcFunFile	::	nonempty_string(),
		FunName			::	nonempty_string(),
		FunArity		:: 	pos_integer(),
		CacheKernel		::	cache | no_cache,
		Why				::	atom().

createMapKernel(_,_,_,_) -> {error, ecl_lib_not_loaded}.



%% -spec  mapHH(Kernel, Input, Output) -> ok | {error, Why} when
%% 		Kernel		::	kernel(),
%% 		Input		::	hostBuffer(),
%% 		Output		::	hostBuffer(),
%% 		Why			::	atom().
%% 
%% mapHH(_,_,_)-> {error, ecl_lib_not_loaded}.

%%Map skeleton working on device buffers. User function specified as kernel objects.
-spec  mapDD(Kernel, Input, Output) -> ok | {error, Why} when
		Kernel		::	kernel(),
		Input		::	deviceBuffer(),
		Output		::	deviceBuffer(),
		Why			::	atom().

mapDD(_,_,_)-> {error, ecl_lib_not_loaded}.

%%Similar to lists:zipWith(Combine, L1, L2) -> L3 but restricted to double
%%Binary Map skeleton working on device buffers. User function specified as kernel objects.
%%Similar to lists:zipWith
-spec  map2DD(Kernel, Input1, Input2, Output) -> ok | {error, Why} when
		Kernel		::	kernel(),
		Input1		::	deviceBuffer(),
		Input2		::	deviceBuffer(),
		Output		::	deviceBuffer(),
		Why			::	atom().

map2DD(_,_,_,_)-> {error, ecl_lib_not_loaded}.

%%Map skeleton: input is a list, output is a device buffer. User function specified as kernel objects.
-spec  mapLD(Kernel, Input, Output, InputLength) -> ok | {error, Why} when
		Kernel		::	kernel(),
		Input		::	[float()],
		Output		::	deviceBuffer(),
		InputLength ::	non_neg_integer(),
		Why			::	atom().

mapLD(_,_,_,_)-> {error, ecl_lib_not_loaded}.

%%Binary Map skeleton: input is a list, output is a device buffer. User function specified as kernel objects.
-spec  map2LD(Kernel, Input1, Input2, Output, InputLength) -> ok | {error, Why} when
		Kernel		::	kernel(),
		Input1		::	[float()],
		Input2		::	[float()],
		Output		::	deviceBuffer(),
		InputLength ::	non_neg_integer(),
		Why			::	atom().

map2LD(_,_,_,_,_)-> {error, ecl_lib_not_loaded}.

%% Map skeleton: input is a list, output too. User function specified as kernel objects.
-spec  mapLL(Kernel, Input, InputLength) -> {ok, Result} when
		Kernel		::	kernel(),
		Input		::	[float()],
		InputLength ::	non_neg_integer(),
		Result		::	[float()].

mapLL(_,_,_)-> {error, ecl_lib_not_loaded}.

%%Binary Map skeleton: input is a list, output too. User function specified as kernel objects.
-spec  map2LL(Kernel, Input1, Input2, InputLength) -> {ok, Result} when
		Kernel		::	kernel(),
		Input1		::	[float()],
		Input2		::	[float()],
		InputLength ::	non_neg_integer(),
		Result		::	[float()].

map2LL(_,_,_,_)-> {error, ecl_lib_not_loaded}.


%%Reduce functions -----------------------------

%%Create a reduce compatible kernel object from the source in ReduceSrcFunFile.
%%The kernel name is FunName.
-spec   createReduceKernel(ReduceSrcFunFile, FunName ) -> {ok, kernel()} | {error, Why} when
		ReduceSrcFunFile	::	nonempty_string(),
		FunName				::	nonempty_string(),
		Why					::	atom().

createReduceKernel(ReduceSrcFunFile, FunName) -> 
	createReduceKernel(ReduceSrcFunFile, FunName, cache).


%%Create a reduce compatible kernel object from the source in ReduceSrcFunFile.
%%The kernel name is FunName
%%Program caching policy must also be specified.
-spec   createReduceKernel(ReduceSrcFunFile, FunName, CacheKernel) -> {ok, kernel()} | {error, Why} when
		ReduceSrcFunFile	::	nonempty_string(),
		FunName				::	nonempty_string(),
		CacheKernel			::	cache | no_cache,
		Why					::	atom().

createReduceKernel(_,_,_) -> {error, ecl_lib_not_loaded}.


%% -spec  reduceHH(Kernel, Input) -> ok | {error, Why} when
%% 		Kernel		::	kernel(),
%% 		Input		::	hostBuffer(),
%% 		Why			::	atom().
%% 
%% reduceHH(_,_) -> {error, ecl_lib_not_loaded}.

 %%Reduce skeleton working on device buffers. User function specified as kernel objects.
-spec  reduceDD(Kernel, Input, Output) -> ok | {error, Why} when
		Kernel		::	kernel(),
		Input		::	deviceBuffer(),
		Output		::	deviceBuffer(),
		Why			::	atom().

reduceDD(_,_,_) -> {error, ecl_lib_not_loaded}.

%%Reduce skeleton: input is a device buffer, output is a list. User function specified as kernel objects.
-spec  reduceDL(Kernel, Input) -> {ok, Result} | {error, Why} when
		Kernel		::	kernel(),
		Input		::	deviceBuffer(),
		Result		::	[float()],
		Why			::	atom().

reduceDL(_,_) -> {error, ecl_lib_not_loaded}.

%%Reduce skeleton: both input and output are lists. User function specified as kernel objects.
-spec  reduceLL(Kernel, Input, InputLength) -> {ok, Result} | {error, Why} when
		Kernel		::	kernel(),
		Input		::	[float()],
		InputLength ::	non_neg_integer(),
		Result		::	[float()],
		Why			::	atom().

reduceLL(ReduceKernel, InputList, InputListLength) -> 
	
	SzDouble = 8,
	SzBufferByte = InputListLength * SzDouble,

	%%allocate host buffer and initialize it using InputList
	InputH = checkResult( allocHostBuffer(SzBufferByte) ),
	checkResult(listToBuffer(InputList, InputH)),
	
	%%allocate device buffer and copy the input data from the host buffer.
	InputD = checkResult( allocDeviceBuffer(SzBufferByte, read_write) ),
	copyBufferToBuffer(InputH, InputD),
	
	%%don't need host buffer anymore
	releaseBuffer(InputH),
	
	%%compute Reduce using the DL version
	ResultList = reduceDL(ReduceKernel, InputD),
			
	releaseBuffer(InputD),
	
	ResultList
.



skeletonlib(_)->
	 {error, ecl_lib_not_loaded}.

cl_init()->
	 {error, ecl_lib_not_loaded}.

cl_release()->
	{error, ecl_lib_not_loaded}.


%%------- Skeleton Composition example -----------------
%%
-spec  mapReduceLL(MapKernel, ReduceKernel, {InputList, InputListLength}) -> {ok, Result} when
		MapKernel	::	kernel(),
		ReduceKernel::	kernel(),
		InputList	::	[float()],
		InputListLength ::	non_neg_integer(),
		Result		::	[float()]
.

mapReduceLL(MapKernel, ReduceKernel, {InputList, InputListLength}) ->
	
	SzDouble = 8,
	SzBufferByte = InputListLength * SzDouble,

	MapOutputD = checkResult( allocDeviceBuffer(SzBufferByte, read_write) ),
	
	checkResult( mapLD(MapKernel, InputList, MapOutputD, InputListLength) ),
	
	Result = reduceDL(ReduceKernel, MapOutputD),
			
	releaseBuffer(MapOutputD),
		
	Result
.

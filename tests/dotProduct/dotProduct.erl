-module(dotProduct).

-export([
	 	ocl_dotProduct_test/1,
		ocl_dotProduct/3,
		
		erl_dotProduct_test/1,
		erl_dotProduct/2,
		
		main/2,
		test/1
		]).

-import(erl_utils, 
		[isPow2/1,
		 dummyLoadLoop/2
		]).

-import(test_avg, [test_avg/4]).

-define(ERL_DELAY, 10).

%%Example: Dot Product
%%
%%Compute dot product of two vectors (as erlang lists) using map2 and reduce.
%%Both pure erlang (erl_) and skel_ocl (ocl_) implementation are provided.
%%
%%The skel_ocl implementation show how to pipeline different skeletons.



%%  TEST on 2^10 elements vectors
%%
%% c(skel_ocl),
%% skel_ocl:cl_init(),
%% c(erl_utils),
%% cd("tests/dotProduct"),
%% c(dotProduct),
%% dotProduct:ocl_dotProduct_test(erl_utils:pow2(10)).


main(ocl, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	ocl_dotProduct_test(erl_utils:pow2(N_ELEM_EXP_POW2))
	;
main(erl, N_ELEM_EXP_POW2) ->
	erl_dotProduct_test(erl_utils:pow2(N_ELEM_EXP_POW2))
	;
main(test, N_ELEM_EXP_POW2) ->
	test(erl_utils:pow2(N_ELEM_EXP_POW2))
.

generate_input(NumVal,Val) ->
	V1 = [ X+0.0 || X <- lists:seq(0,NumVal-1) ],
	V2 = [ X+0.0 || X <- lists:duplicate(NumVal,Val) ],
	
	[V1,V2]
	.


%% computation correctness test

test(NumVal) ->
	
	%% must be a power of 2
	case isPow2(NumVal) of 
		false -> erlang:error(input_not_pow2);
		true -> ok 
	end,
	
	io:format("~n-------dotProduct TEST-------~n"),
	
	
	io:format("Generating input...~n"),
 	Val = 2,
	[V1,V2] = generate_input(NumVal,Val),
	
	
	io:format("~nComputing: OCL...~n"),
	[OCL_Result] = ocl_dotProduct(NumVal, V1, V2),
	
	io:format("~nComputing: ERL...~n"),
	ERL_Result = erl_dotProduct(V1, V2),

	io:format("~nTest..."),
	
	if
		OCL_Result == ERL_Result -> io:format("PASSED!~n"); 
		true -> io:format("~nERROR. FAILED!!~n")
	end
.



ocl_dotProduct(NumVal, V1, V2) ->

	SzDouble = 8,
	SzBufferByte = NumVal * SzDouble,
		
	MapFunSrcFile = "mul.cl", MapFunName = "mul",
	ReduceFunSrcFile = "sum.cl", ReduceFunName = "sum",
	
 	PWD = element(2,file:get_cwd()) ++ "/",
	
	MapFunSrcFileAbsPath =   PWD ++ MapFunSrcFile, 
	ReduceFunSrcFileAbsPath = PWD ++ ReduceFunSrcFile,
		

	%%Create and compile kernels
	MapMulKernel = skel_ocl:checkResult(skel_ocl:createMapKernel(MapFunSrcFileAbsPath, MapFunName, 2)),
	ReduceSumKernel = skel_ocl:checkResult(skel_ocl:createReduceKernel(ReduceFunSrcFileAbsPath, ReduceFunName)),
	
	%%allocate work buffer on device
	MapOutputD = skel_ocl:checkResult(skel_ocl:allocDeviceBuffer(SzBufferByte, read_write)),

	skel_ocl:checkResult( skel_ocl:map2LD(MapMulKernel, V1, V2, MapOutputD, NumVal) ),
	
 	Result = skel_ocl:checkResult( skel_ocl:reduceDL(ReduceSumKernel, MapOutputD) ),

	skel_ocl:releaseBuffer(MapOutputD),
		
	Result
.

ocl_dotProduct_test(NumVal) ->
	
	%% must be a power of 2
	case isPow2(NumVal) of 
		false -> erlang:error(input_not_pow2);
		true -> ok 
	end,
 	Val = 2,

	io:format("~n-------ocl_dotProduct TEST-------~n"),
	io:format("Vectors' dimension: ~w~nV1 = seq(0,~w)~nV2 = duplicate(~w,~w): ~n~n", [NumVal, NumVal-1, NumVal, Val]),
	
	[V1,V2] = generate_input(NumVal,Val),
	
%% 	F_start = now(),
%% 
%% 	Result = dotProduct(NumVal, V1, V2),
%% 	
%% 	F_end = now(),
%% 
%% 	io:format("ocl_dotProduct_test total time: ~w usec~n", [timer:now_diff(F_end, F_start)]),
%% 	
%% 	Result

	Runs = 10,
	
	Median = test_avg:test_avg(dotProduct, ocl_dotProduct, [NumVal, V1, V2], Runs),
	io:format("ocl_dotProduct_test median total time (~w rounds): ~w usec~n", [Runs, Median])
.



%%%----------pure Erlang version----------------
zipFun(X,Y) -> 
	dummyLoadLoop(2.0,?ERL_DELAY), X*Y.

sum(X,Y) -> X+Y.

erl_dotProduct(V1, V2) ->

	Sums = lists:zipwith(fun zipFun/2, V1, V2),
	lists:foldl(fun sum/2, 0.0, Sums)
.


erl_dotProduct_test(NumVal) ->

	Val = 2,

	io:format("~n-------erl_dotProduct TEST-------~n"),
	io:format("Vectors' dimensions: ~w~nV1:seq(0,~w)~nV2:duplicate(~w,~w): ~n", [NumVal, NumVal-1, NumVal, Val]),
	io:format("Delay: ~w~n~n", [?ERL_DELAY]),
	
	[V1,V2] = generate_input(NumVal,Val),
	
%% 	F_start = now(),
%% 
%% 	Result = erl_dotProduct( V1, V2),
%% 	
%% 	F_end = now(),
%% 
%% 	io:format("erl_dotProduct_test total time: ~w usec~n", [timer:now_diff(F_end, F_start)]),
%% 	
%% 	Result

	Runs = 10,
	
	Median = test_avg:test_avg(dotProduct, erl_dotProduct, [V1, V2], Runs),
	io:format("erl_dotProduct_test median total time (~w rounds): ~w usec~n", [Runs, Median])
.




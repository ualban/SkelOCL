-module(mapLL_test).

-export([		 
		 ocl_mapLL/1,
		 erl_mapLL/1,
		 main/2

		]).

-import(test_avg,[test_avg/4]).

-import(erl_utils,
		[
		 isPow2/1,
		 pow2/1,
 		 dummyLoadLoop/2
		]).

-define(ERL_DELAY, 10).

%%-------Example: Map from list to list-------
%%
%%Compute a dummy kernel on erlang list so to highlight its performance vs the pure erlang implementation.
%%
%%MapLL is implemented as a three stage (unmarshalling, computation, marshalling) pipeline.
%%Unmarshalling is the stage that takes the data from the erlang list and copies them into a cl_buffer; marshalling does the opposite.
%%The list is divided in segments so to overlap the 3 stages and maximize throughput.
%%
%%List handling cost is hidden by computation if the kernel is complex enough ( it takes more time than (un-)marshaling).

%% c(skel_ocl),
%% skel_ocl:cl_init(),
%% c(erl_utils),
%% cd(tests),
%% c(mapLL_test),
%% mapLL_test:ocl_mapLL(erl_utils:pow2(10)).


main(ocl, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	ocl_mapLL(erl_utils:pow2(N_ELEM_EXP_POW2))
	;
main(erl, N_ELEM_EXP_POW2) ->
	erl_mapLL(erl_utils:pow2(N_ELEM_EXP_POW2))
	;
main(compare, N_ELEM_EXP_POW2) ->
	
	NumVal = erl_utils:pow2(N_ELEM_EXP_POW2),
	
	skel_ocl:cl_init(),
	
	L1 = [ X+0.0 || X <- lists:seq(1, NumVal) ],
	
	ERL = lists:map(fun sq/1, L1),
	
	MapSqKernel = 
		skel_ocl:checkResult(
		  skel_ocl:createMapKernel("sq.cl", "sq")
		),
	
	OCL = 
		skel_ocl:checkResult(skel_ocl:mapLL(MapSqKernel, L1, NumVal)),
	
	erl_utils:equalsLists(ERL, OCL)
.


%% mapExample(List) ->
%% 	
%% 	%%Square function
%% 	Sq = fun X -> X*X end,
%% 
%% 	lists:map(Sq, List)
%% .
%% 
%% mapLLExample(List)->
%% 	{ok, SqKernel} = createMapKernel("sq.cl","sq"),
%% 	skel_ocl:mapLL(SqKernel, List, length(List))
%% .

sq(X) -> 
	dummyLoadLoop(2.0, ?ERL_DELAY), 
	X*X
.

erl_mapLL(NumVal) ->
				
	io:format("~n-------erl_mapLL TEST-------~n"),
	io:format("Number of values: ~w, Bytes: ~w, Input: seq(0, ~w).~n", [NumVal,NumVal*8,NumVal-1]),
	io:format("Map function: fun sq/1 ~n"),
	io:format("Delay: ~w~n~n", [?ERL_DELAY]),
	
	L1 = [ X+0.0 || X <- lists:seq(0, NumVal-1) ],

%% 	F_start = now(),
%% %% 	Result = 
%% 		lists:map(fun sq/1, L1k),
%% 	F_end = now(),
%% 	
%% 	io:format("erl_mapLL_test total time: ~w usec~n", [timer:now_diff(F_end, F_start)])
%% 	, Result

	NumRounds = 10,
	Median = test_avg:test_avg(lists, map, [fun sq/1, L1], NumRounds),

	io:format("erl_mapLL_test median total time over ~w  executions: ~w usec~n", [NumRounds, Median])
.

ocl_mapLL(NumVal) ->
	
	case isPow2(NumVal) of 
		false -> erlang:error(input_not_pow2);
		true -> ok 
	end,
	
	PWD = element(2,file:get_cwd()) ++ "/",
	
	FunSrcFile = "sq.cl", FunName = "sq",
	
	FunSrcFileAbsPath = PWD ++ FunSrcFile,
	
	io:format("~n-------ocl_mapLL TEST-------~n"),
	io:format("Number of values: ~w, Bytes: ~w, Input: seq(0, ~w).~n", [NumVal, NumVal*8, NumVal-1]),
	io:format("Map function file: ~p~n", [FunSrcFileAbsPath]),
	io:format("Map function name: ~p~n~n", [FunName]),
	
	L1 = [ X+0.0 || X <- lists:seq(0, NumVal-1) ],

%% 	F_start = now(),
	
	MapSqKernel = 
		skel_ocl:checkResult(
		  skel_ocl:createMapKernel(FunSrcFileAbsPath, FunName)
		),
%% 	Kernel_creation = now(),
%% 	io:format("L1: ~w~n", [L1]),
	

%% 	{ok, Result} = skel_ocl:ocl_mapLL(MapSqKernel, L1),

	NumRounds = 10,

	Median = test_avg:test_avg(skel_ocl, mapLL, [MapSqKernel, L1, NumVal], NumRounds),

	io:format("ocl_mapLL_test median total time over ~w  executions: ~w usec~n", [NumRounds, Median])
	
%% 	,io:format("Kernel Compilation: ~w~n", [timer:now_diff(Kernel_creation, F_start)]),
	
%% 	,io:format("Kernel exec: ~w~n", [timer:now_diff(K_end, K_start)])		
.

-module(mapReduce_test).

-export([		 
		 naive_mapReduceLL/3,
		 test/2,
		 main/2
		]).


-import(erl_utils,
		[
		 isPow2/1,
		 pow2/1
		]).

-import(skel_ocl, [checkResult/1,
				   createMapKernel/2,
				   createReduceKernel/2,
				   mapLL/3,
				   reduceLL/3
				  ]).



main(optimized, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	test(erl_utils:pow2(N_ELEM_EXP_POW2), optimized)
;
main(naive, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	test(erl_utils:pow2(N_ELEM_EXP_POW2), naive)
.



naive_mapReduceLL(MapKernel, ReduceKernel, {InputList, InputListLength}) ->
	
	%%Map Skeleton: list -> list
 	MapOutputList = checkResult(mapLL(MapKernel, InputList, InputListLength)),
	
	%%Reduce skeleton: list -> list
	reduceLL(ReduceKernel, MapOutputList, InputListLength)
.

test(NumVal, Type) -> 

	case isPow2(NumVal) of 
		false -> erlang:error(input_not_pow2);
		true -> ok 
	end,
			
	MapFunSrcFile = "sq.cl", MapFunName = "sq",
	ReduceFunSrcFile = "sum.cl", ReduceFunName = "sum",
	
 	PWD = element(2,file:get_cwd()) ++ "/",
	
	MapFunSrcFileAbsPath =   PWD ++ MapFunSrcFile, 
	ReduceFunSrcFileAbsPath = PWD ++ ReduceFunSrcFile,

	InputList = [ X+0.0 || X <- lists:seq(0, NumVal-1) ],


	%%create Kernels objects
	MapSqKernel = checkResult(createMapKernel(MapFunSrcFileAbsPath, MapFunName)),
	ReduceSumKernel = checkResult(createReduceKernel(ReduceFunSrcFileAbsPath, ReduceFunName)),


	NumRounds = 10,
	
	case Type of
		naive ->
			Median = test_avg:test_avg(?MODULE, naive_mapReduceLL, [MapSqKernel, ReduceSumKernel,{InputList, NumVal}], NumRounds),
			io:format("naive_mapReduceLL median total time over ~w  executions: ~w usec~n", [NumRounds, Median])
			%%,

			%%naive_mapReduceLL(MapSqKernel, ReduceSumKernel,{InputList, NumVal})
		;
		optimized -> 
			Median = test_avg:test_avg(skel_ocl, mapReduceLL, [MapSqKernel, ReduceSumKernel,{InputList, NumVal}], NumRounds),
			io:format("optimized_mapReduceLL median total time over ~w  executions: ~w usec~n", [NumRounds, Median])
			%%,
			%%skel_ocl:mapReduceLL(MapSqKernel, ReduceSumKernel,{InputList, NumVal})
	end
.



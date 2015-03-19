-module(usageExample).

-export([exampleFun/1]).


-import(erl_utils,
		[
		 isPow2/1,
		 pow2/1
		]).

-import(skel_ocl,
		[checkResult/1,
		 createMapKernel/2,
		 mapLL/3
		]).

%Usage example showing the mapLL skeleton mapping the kenrnel defined in exampleKernel.cl onto a list of NumVal Erlang float().
%The kernel must be compiled using createMapKernel(PATH), the resulting object is passed to the skeleton.
%NumVal must be a power of 2.

exampleFun(NumVal) ->
	
	case isPow2(NumVal) of 
		false -> erlang:error(input_not_pow2);
		true -> ok 
	end,
			
	MapFunSrcFile = "exampleKernel.cl", MapFunName = "foo",

 	PWD = element(2,file:get_cwd()) ++ "/",
	
	MapFunSrcFileAbsPath =   PWD ++ MapFunSrcFile,

	InputList = [ X+0.0 || X <- lists:seq(0, NumVal-1) ],

	%%create Kernels objects
	MapSqKernel = checkResult(createMapKernel(MapFunSrcFileAbsPath, MapFunName)),

	%%Map Skeleton: list -> list
 	MapOutputList = checkResult(mapLL(MapSqKernel, InputList, NumVal)),

	MapOutputList
.

-module(integration).

-export([
		 erl_integr_test/3,
		 ocl_integr_test/3,
		 
		 integrate/5,
		 main/2
		]).

-import(erl_utils, [for/3,dummyLoadLoop/2]).

-define(ERL_DELAY, 40).
%%-------Example: Integration-------
%%
%%Compute numerical integral (using the midpoint method) using map and reduce.
%%Both pure erlang (erl_) and skel_ocl (ocl_) implementation are provided.

  
%% c(skel_ocl),
%% skel_ocl:cl_init(),
%% c(erl_utils),
%% cd(tests),
%% c(integration),
%% integration:ocl_integr_test(10,20,erl_utils:pow2(10)).

	
	
main(ocl, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	ocl_integr_test(10, 20, erl_utils:pow2(N_ELEM_EXP_POW2))
	;
main(erl, N_ELEM_EXP_POW2) ->
	erl_integr_test(10, 20, erl_utils:pow2(N_ELEM_EXP_POW2))
;
main(compare, N_ELEM_EXP_POW2) ->
	skel_ocl:cl_init(),
	io:format("~n-------Integration COMPARE mode-------~n"),
	
	A = 10, B = 20,
	
	N_nodes = erl_utils:pow2(N_ELEM_EXP_POW2),
	
	Nodes_list = generate_nodes(A, B, N_nodes),
	
	%ERLANG
	io:format("~nComputing: ERL...~n"),
	ERL = integrate(erl, fun integrandFun/1, A, B, {Nodes_list, N_nodes}),
	
	io:format("~nComputing: OCL...~n"),
	OCL = integrate(ocl, "f_integrand", A, B, {Nodes_list, N_nodes}),
	
%% 	io:format("ERL:~w, OCL:~w", [ERL,OCL])
	
	io:format("~nTest..."),
	
	if
		trunc(ERL) == trunc(OCL) -> io:format("PASSED!~n"); 
		true -> io:format("~nERROR. FAILED!!~n")
	end
.

	
%%skel_ocl implementation
ocl_integr_test(A, B, N_nodes) ->

 	Integrand = "f_integrand",%name of the function to be integrated, must be the same of .cl kernel file
	Nodes_list = generate_nodes(A, B, N_nodes),
	

	io:format("~n-------ocl_integr_test-------~n"),
	io:format("Integrand OCL Kernel file: ~p~n",[Integrand++".cl"]),
	io:format("Bounds: [~w,~w]~n",[A,B]),
	io:format("Number of Nodes: ~w~n~n",[N_nodes]),

	Runs = 10,
	
	Median = test_avg:test_avg(integration, integrate, [ocl, Integrand, A, B, {Nodes_list, N_nodes}], Runs),
	io:format("ocl_integr_test median total time (~w rounds): ~w usec~n", [Runs, Median])
.


integrandFun(X) -> 
	dummyLoadLoop(2.0,?ERL_DELAY),
	X+2
.

%%pure erlang implementation
erl_integr_test(A,B,N_nodes) ->

 	Integrand = fun integrandFun/1,
	Nodes_list = generate_nodes(A, B, N_nodes),

	io:format("~n-------erl_integr_test-------~n"),
	io:format("Integrand fun: \"fun(X)-> X+2 end\"~n"),
	io:format("Delay: ~w~n", [?ERL_DELAY]),
	io:format("Bounds: [~w,~w]~n",[A,B]),
	io:format("Number of Nodes: ~w~n~n",[N_nodes]),
	
	Runs = 10,
	
	Median = test_avg:test_avg(integration, integrate, [erl, Integrand, A, B, {Nodes_list, N_nodes}], Runs),
	io:format("erl_integr_test median total time (~w rounds): ~w usec~n", [Runs, Median])

%% 	integrate(erl, Integrand, A, B, N_nodes)
.

integrate(Type, Integrand, A, B, {Nodes_list, N_nodes}) -> 
	
	H = (B-A)/N_nodes,
	
	[Sum] = 
		case Type of
			erl -> erl_integr_mapReduce(Integrand, {Nodes_list, N_nodes});
			ocl -> ocl_integr_mapReduce(Integrand, {Nodes_list, N_nodes});
			_ -> erlang:error(wrong_type_error)
		end,

	Sum*H
.

generate_nodes(A, B, N_nodes) -> 
	
	H = (B-A)/N_nodes,
	
	MidPoint = fun(I) -> A + ( (I+1.0) - 0.5 ) * H end,

	Points_method = MidPoint,

	for(0, N_nodes-1, Points_method)
.



%%Numerical Integration implemented whith skel_ocl map and reduce.
ocl_integr_mapReduce(IntegrandFunName, { Nodes_List, N_nodes }) ->
		
	ReduceFunSrcFile = "sum.cl", ReduceFunName = "sum",
	
 	PWD = element(2,file:get_cwd()) ++ "/",
	
	%%The integrand's src file has the same name of the integrand function
	Integrand_SrcFileAbsPath =  PWD ++ IntegrandFunName ++ ".cl", 
		
	ReduceFunSrcFileAbsPath = PWD ++ ReduceFunSrcFile,
	
%%  F_start = now(),
		
	IntegrandMapKernel = skel_ocl:checkResult(skel_ocl:createMapKernel(Integrand_SrcFileAbsPath, IntegrandFunName)),
	ReduceSumKernel = skel_ocl:checkResult(skel_ocl:createReduceKernel(ReduceFunSrcFileAbsPath, ReduceFunName)),
	
%%  Kernel_creation_end = now(),

%%  MapReduce_start = Kernel_creation_end,

	%%use mapReduceLL, implemented in erlang using skel_ocl functions
	Result = skel_ocl:checkResult(skel_ocl:mapReduceLL(IntegrandMapKernel, ReduceSumKernel, {Nodes_List, N_nodes})),
	
%%  MapReduce_end = now(),
%% 	io:format("ocl_mapReduce time:~nCreateKernels: ~w~ncomputation: ~w~n", [timer:now_diff(Kernel_creation_end, F_start), timer:now_diff(MapReduce_end, MapReduce_start)]),

	Result	
.



erl_integr_mapReduce(F, {Nodes_List, _N_nodes}) ->
	
	Sum_fun = fun(X,Y)-> X+Y end,
	
%% 	MapReduce_start = now(),
	
	Values_list = lists:map(F, Nodes_List),
	Sum = lists:foldl(Sum_fun, 0.0, Values_list),

%%  MapReduce_end = now(),

%% 	io:format("erl_mapReduce_time: ~w~n", [timer:now_diff(MapReduce_end, MapReduce_start)]),

	[Sum]
.
	




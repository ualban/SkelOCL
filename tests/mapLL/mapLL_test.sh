#!/usr/bin/escript -c
%%! -smp enable -sname mapLL_test  -pa ../../ebin

main([TYPE, N_ELEM_EXP_POW2 | []]) ->
%%	io:format("~p",[code:get_path()])
	
	mapLL_test:main(list_to_atom(TYPE), list_to_integer(N_ELEM_EXP_POW2))
;
main(_Args) -> 
	io:format("Usage: mapLL_test [ocl|erl|compare] pow2_exp~n")
.


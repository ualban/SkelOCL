#!/usr/bin/escript -c
%%! -smp enable -sname dotProduct_test -pa ../../ebin

main([TYPE, N_ELEM_EXP_POW2 | []]) ->
%%	io:format("~p",[code:get_path()])
	
	dotProduct:main(list_to_atom(TYPE), list_to_integer(N_ELEM_EXP_POW2))
;
main(_Args) -> 
	io:format("Usage: dotProduct_test [ocl|erl|test] pow2_exp~n")
.

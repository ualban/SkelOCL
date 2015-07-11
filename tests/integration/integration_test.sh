#!/usr/bin/escript -c
%%! -smp enable -sname integration_test -pa ../../ebin

main([TYPE, N_ELEM_EXP_POW2 | []]) ->
	integration:main(list_to_atom(TYPE), list_to_integer(N_ELEM_EXP_POW2))
;
main(_Args) -> 
	io:format("Usage: integration_test [ocl|erl|compare] pow2_exp~n")
.

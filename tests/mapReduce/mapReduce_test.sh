#!/usr/bin/escript -c
%%! -smp enable -sname mapReduce_test -pa ../../ebin


main([TYPE, N_ELEM_EXP_POW2 | []]) ->
%%	io:format("~p",[code:get_path()]),
	

	mapReduce_test:main(list_to_atom(TYPE), list_to_integer(N_ELEM_EXP_POW2))
;
main(_) ->
	io:format("Usage: mapReduce_test [naive|optimized] N_ELEM_EXP_POW2~n")
.

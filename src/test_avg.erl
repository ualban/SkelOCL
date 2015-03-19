-module(test_avg).

-export([test_avg/4]).


test_avg(M, F, A, N) when N > 0 ->
	
	io:format("~w Runs (usecs):~n", [N] ),

	io:format("[ "),
    L = test_loop(M, F, A, N, []),
	io:format("]~n"),
	
	Length = length(L),
	L_sorted = lists:sort(L),
    Min = lists:min(L_sorted),
    Max = lists:max(L_sorted),
    Med = lists:nth(round((Length / 2)), L_sorted),
    Avg = round(lists:foldl(fun(X, Sum) -> X + Sum end, 0, L) / Length),
    io:format("Range: ~b - ~b usec~n"
          "Median: ~b usec~n"
          "Average: ~b usec~n",
          [Min, Max, Med, Avg]),
    Med.
 
test_loop(_M, _F, _A, 0, List) ->
    List;
test_loop(M, F, A, N, List) ->
    {T, _Result} = timer:tc(M, F, A),
	
	io:format("~w ",[T]),
	
	erlang:garbage_collect(),
    test_loop(M, F, A, N - 1, [T|List]).
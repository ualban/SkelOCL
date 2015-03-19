-module(erl_utils).

-export([
		 isPow2/1,
		 pow2/1,
		 ceil2/1,
		 loop/3,
		 for/3,
		
		 equalsLists/2, 
		 
		 dummyLoadLoop/2
		
		]).


isPow2(2) ->
		true;
isPow2(X) when is_integer(X), X > 2 ->
	case X rem 2 of
		0 -> isPow2(X div 2);
		_ -> false
	end
.

pow2(Exp) ->
	trunc(math:pow(2, Exp)).


ceil2_i(X, Exp)  when is_float(X), X == 2 ->
	pow2(Exp+1);
ceil2_i(X, Exp)  when is_float(X), X < 2 ->
	pow2(Exp+1);
ceil2_i(X, Exp) when is_float(X), X > 2 ->
		ceil2_i(X/2, Exp+1)
.

%%rounds up to the next power of two
ceil2(X) ->
	ceil2_i(X+0.0 ,0).


dummyLoadLoop(Val, 0) ->
	Val;
dummyLoadLoop(Val, Loops) ->
	dummyLoadLoop(math:sin(Val), Loops - 1).


loop(_F, 0, X) -> 
	X;
loop(F, Idx, X) ->
	loop(F, Idx-1, F(X))
.


%% [F(I) | I from 0 to Max]
for(Max, Max, F) ->
	[F(Max)];
for(I, Max, F) ->
	[F(I) | for(I+1, Max, F)]
.

-spec  equalsLists(L1, L2) -> {true, Length} | {false, MismatchIdx} when
		L1				::	[any()],
  		L2				::	[any()],
		Length			::	non_neg_integer(),
		MismatchIdx		::	non_neg_integer()
.


equalsLists(L1,L2) when is_list(L1) andalso is_list(L2) ->

	equalsListsA(L1,L2,0)
.

equalsListsA([H1 | T1], [H2 | T2], I)-> 
	
	if
		H1 =:= H2 -> equalsListsA(T1, T2, I+1);
		H1 =/= H2 -> {false, I}
	end
;
equalsListsA([], [], I) -> 
	{true, I}
;
equalsListsA([], [_H2 | _T2], I) -> 
	{false, I}
;
equalsListsA([_H1 |_T1], [], I) -> 
	{false, I}
.




#	Process a series of newline-seperated games in PGN format into single-line
#	games that are easy to read and process by the engine.
#
#	50,201 games are present in the original data, but some will be filtered
#	out, depending on the explanation for the game result. The possible
#	explanations include:
#
#	For wins/losses:
#		1. [Color] checkmated
#		2. [Color] forfeits by disconnection
#		3. [Color] forfeits on time
#		4. [Color] resigns
#		5. [Color] wins by adjudication
#
#	2, 3, and 5 will be filtered out.
#
#	For draws:
#		1. [Color1] ran out of time and [color2] has no material to mate
#		2. Game drawn because both players ran out of time
#		3. Game drawn by mutual agreement
#		4. Game drawn by repetition
#		5. Game drawn by stalemate
#		6. Game drawn by the 50 move rule
#		7. Neither player has mating material
#
#	1 and 2 will be filtered out.
file_in=2019_games_raw.pgn
file_out_t=2019_games_processed_t.txt
file_out_v=2019_games_processed_v.txt
num_t=43700
num_v=212

#	Take only the move sequences from the pgn;
#	remove notation which my engine does not use (i.e. "#" or "+" for checks;
#	comments about the result of the game);
#	remove move numbers, which are redundant information
grep "^1." $file_in \
	| grep -Ev 'time|disconnection|adjudication' \
	| sed -r 's/[+#]| \{.*\}|[0-9]*\. //g' \
	| head -n $num_t > $file_out_t
	
grep "^1." $file_in \
	| grep -Ev 'time|disconnection|adjudication' \
	| sed -r 's/[+#]| \{.*\}|[0-9]*\. //g' \
	| tail -n $num_v > $file_out_v
	
#	The notation in the resulting file, $file_out, now may differ from the
#	accepted notation within the engine (defined in Game.py) by:
#		1. en passant: "e.p." is used within the engine only
#		2. ambiguous moves: a bishop, knight, or queen move could be
#			disambiguated by providing file or rank-- my engine uses file in
#			this case, but I don't think there is a standard for which to use
#		3. currently, a bug in the engine misnames some ambiguous moves

#   First, tactical positions were downloaded from:
#   https://database.lichess.org/lichess_db_puzzle.csv.zst
#   and decompressed. This script was run interactively to filter out candidate
#   games (that don't have enough moves or involve en passant) to greatly
#   narrow the set of positions to generate examples from in Python later
#   (since the Python processing is computationally expensive)

library(tidyverse)
library(here)

tactics_path = here('external', 'lichess_db_puzzle.csv')
out_path = here('external', 'preprocessed_tactics', 'all_tactics.csv.gz')

dir.create(dirname(out_path), showWarnings = FALSE)

#   Minimum number of moves after the starting tactical position to require. The
#   idea is to provide a meaningful reward label in addition to the policy label
min_num_extra_moves = 2

read_csv(tactics_path) |>
    select(FEN, Moves, Rating) |>
    #   Don't consider tactical positions where en passant might be relevant
    filter(str_split_i(FEN, ' ', 4) == '-') |>
    mutate(
        num_moves = sapply(Moves, function(x) length(strsplit(x, ' ')[[1]]))
    ) |>
    #   Filter too-short tactical sequences
    filter(num_moves > min_num_extra_moves + 2) |>
    write_csv(out_path)

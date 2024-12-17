#   First, 3 zip files of PGN data were downloaded from the URLs:
#   https://theweekinchess.com/zips/twic1569g.zip
#   https://theweekinchess.com/zips/twic1570g.zip
#   https://theweekinchess.com/zips/twic1571g.zip
#   (unclear if the URLs are permanent), and extracted as a PGN file in this
#   directory. This script was run interactively to preprocess and filter the
#   messy original files into an easy-to-read format for the engine

library(tidyverse)
library(here)

pgn_paths = here(
    'external',
    c('twic1569.pgn', 'twic1570.pgn', 'twic1571.pgn')
)
out_test_path = here('external', 'preprocessed_games', 'test_games.txt.gz')
out_train_path = here('external', 'preprocessed_games', 'train_games%s.txt.gz')
elo_cutoff = 2000
test_size = 1000        # games in test set
train_batch_size = 1000 # games per training output file
min_num_moves = 10

set.seed(0)

clean_pgn = function(pgn_path) {
    game = readLines(pgn_path)

    #   Remove empty lines and result strings in the game headers
    game = game[game != ""]
    game = game[!grepl('Result', game)]

    #   Add a separator at the end of the game result string, and repartition
    #   lines based on that separator instead of '\n'. This way we have one game
    #   per line
    game = gsub('([01]-[10]|1/2-1/2)', '\\1|', paste(game, collapse = ' '))
    game = strsplit(game, '\\|')[[1]]

    #   Only keep game where both white and black's elo is defined
    game = game[grepl('WhiteElo', game) & grepl('BlackElo', game)]

    #   Filter to games with minimum ELO cutoff for white and black
    white_elo = str_extract(game, 'WhiteElo "([0-9]+)"', group = 1) |>
        as.numeric()
    black_elo = str_extract(game, 'BlackElo "([0-9]+)"', group = 1) |>
        as.numeric()
    game = game[(white_elo > elo_cutoff) & (black_elo > elo_cutoff)]

    #   Some games are not standard chess but chess960, which we don't care
    #   about
    game = game[!grepl('Chess960', game)]

    #   Clean up into a space-separated series of moves ending in a result
    #   string
    game = game |>
        #   Delete the header of each game
        str_replace('.*\\] *', '') |>
        #   Remove numbers before each turn
        str_replace_all('[0-9]+\\. ', '') |>
        #   The engine doesn't use '+' or '#' for check or checkmate
        str_replace_all('[+#]', '')

    #   Some games are extrememly short. Drop them
    game = game[
        sapply(game, function(x) length(strsplit(x, ' ')[[1]]) >= min_num_moves)
    ]

    return(game)
}

#   Preprocess games from all PGNs and combine into one vector
games = do.call(c, lapply(pgn_paths, clean_pgn)) |>
    unique() #  Surprisingly, there are a couple duplicate games

#   Out of paranoia, randomly sort the games in case there is some pattern
#   associated with the order of the games
games = sample(games)

#   Write test games to file
out_con = gzfile(out_test_path, 'w')
writeLines(games[1:test_size], con = out_con)
close(out_con)

#   Write training games in batches of predetermined size
for (i in seq_len((length(games) - test_size) %/% train_batch_size)) {
    out_con = gzfile(sprintf(out_train_path, i))
    these_games = games[
        (test_size + (i - 1) * train_batch_size + 1):
        (test_size + i * train_batch_size)
    ]
    writeLines(these_games, con = out_con)
    close(out_con)
}

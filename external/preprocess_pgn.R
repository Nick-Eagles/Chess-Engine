#   First, a zip file of PGN data was downloaded from https://theweekinchess.com/zips/twic1569g.zip
#   (unclear if the URL is permanent), and extracted as a PGN file in this
#   directory. This script was run interactively to preprocess and filter the
#   messy original file into an easy-to-read format for the engine

library(tidyverse)
library(here)

pgn_path = here('external', 'twic1569.pgn')
out_path = here('external', '%s_games.txt')
elo_cutoff = 2000

game = readLines(pgn_path)

#   Remove empty lines and result strings in the game headers
game = game[game != ""]
game = game[!grepl('Result', game)]

#   Add a separator at the end of the game result string, and repartition lines
#   based on that separator instead of '\n'. This way we have one game per line
game = gsub('([01]-[10]|1/2-1/2)', '\\1|', paste(game, collapse = ' '))
game = strsplit(game, '\\|')[[1]]

#   Only keep game where both white and black's elo is defined
game = game[grepl('WhiteElo', game) & grepl('BlackElo', game)]

#   Filter to games with minimum ELO cutoff for white and black
white_elo = str_extract(game, 'WhiteElo "([0-9]+)"', group = 1) |> as.numeric()
black_elo = str_extract(game, 'BlackElo "([0-9]+)"', group = 1) |> as.numeric()
game = game[(white_elo > elo_cutoff) & (black_elo > elo_cutoff)]

#   Clean up into a space-separated series of moves ending in a result string
game = game |>
    #   Delete the header of each game
    str_replace('.*\\] *', '') |>
    #   Remove numbers before each turn
    str_replace_all('[0-9]+\\. ', '') |>
    #   The engine doesn't use '+' or '#' for check or checkmate
    str_replace_all('[+#]', '')

#   For some reason, there are games with no moves. Drop them
game = game[!grepl('^(1-0|0-1|1/2-1/2)$', game)]

out_path = sprintf(out_path, length(game))
writeLines(game, con = out_path)

library(ggplot2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")
costData = read.csv(file="costs.csv")
costData$isStart = as.factor(costData$isStart)

#costData = tail(costData, 100)
ggplot(costData) +
  geom_point(mapping = aes(x=epochNum, y=cost, color=costType, shape=isStart)) +
  labs(title="Cost on data vs. epoch") +
  geom_smooth(data = costData[costData$costType == 'v_cost' & costData$isStart == 0,],
              mapping = aes(x=epochNum, y=cost),
              method='loess')

# Read config to see how many epochs are in one "episode"
config = readLines('../config.txt')
correctLine = sapply(config, function(x) nchar(x) > 5 && substr(x, 1, 6) == 'epochs')
numEpochs = as.numeric(substr(config[correctLine], 8, 8))

# Print mean drop in validation cost within each episode
startCosts = costData[(costData$isStart == 1) & (costData$costType == 'v_cost'), 'cost']
endCosts = costData[(costData$epochNum %% numEpochs == 0) & (costData$costType == 'v_cost') & 
                    (costData$epochNum > 0) & (costData$isStart == 0), 'cost']
mean(startCosts - endCosts)

library(ggplot2)
library(reshape2)

net = 'tf_ex'

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine")
costData = read.csv(file=file.path('nets', net, 'costs.csv'))

costData$is_start = costData$epoch == 0
costData$epoch = 1:nrow(costData)

costData = melt(costData, c('epoch', 'is_start'))
costData$is_validation = sapply(costData$variable, function(x) substr(x, 1, 3) == 'val')
#costData = costData[costData$costType == 'v_cost',]
#costData = costData[5:nrow(costData),]
ggplot(costData) +
  geom_point(mapping = aes(x=epoch, y=value, color=is_validation, shape=variable)) +
  labs(title="Loss on data vs. epoch", 
       color="Is validation?", 
       shape="Loss type", 
       x="epoch number") +
  scale_shape_manual(values=3 * c(1:5, 1:5))

#############################################################
#   Some stats you can perform, optionally, on the data
#############################################################

# Check how many epochs are in one "episode" via distance between first 2 starting costs
temp = as.numeric(costData$epochNum[costData$isStart == 1][c(1, 3)])
numEpochs = temp[2] - temp[1]

# Print mean drop in cost within each episode
for (cType in c('t_cost', 'v_cost')) {
    tempData = costData[costData$costType == cType,]
  
    startEpochs = tempData$epochNum == tempData[tempData$isStart, 'epochNum']
  
    startCosts = tempData[tempData$isStart, 'cost']
    endCosts = tempData[! tempData$isStart & startEpochs & tempData$epochNum > 0, 'cost']
    
    print(paste0("Mean drop in ", cType, ": ", mean(startCosts - endCosts)))
}

# A linear model of the starting costs vs. epoch number
lm(startCosts ~ costData$epochNum[match(startCosts, costData$cost)])

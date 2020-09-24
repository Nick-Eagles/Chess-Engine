library(ggplot2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")
#setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/MNIST_chess/visualization")
costData = read.csv(file="costs.csv")
costData$isStart = as.logical(costData$isStart)

costData = tail(costData, 400)
#costData = costData[costData$costType == 'v_cost',]
#costData = costData[5:nrow(costData),]
ggplot(costData) +
  geom_point(mapping = aes(x=epochNum, y=cost, color=costType, shape=isStart)) +
  labs(title="Cost on data vs. epoch", color="loss type", shape="starting loss?", x="epoch number") +
  scale_color_manual(labels=c("training", "validation"), values=c("#EE1111","#11CCEE")) +
  scale_shape_manual(labels=c("False", "True"), values=c(16, 2)) +
  geom_smooth(data = costData[costData$costType == 'v_cost' & costData$isStart == 0,],
              mapping = aes(x=epochNum, y=cost),
              method='lm')

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

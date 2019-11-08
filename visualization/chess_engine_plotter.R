library(ggplot2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")
costData = read.csv(file="costs.csv")
costData$isStart = as.factor(costData$isStart)

#costData = tail(costData, 200)
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

# Print mean drop in validation cost within each episode
startCosts = costData[(costData$isStart == 1) & (costData$costType == 'v_cost'), 'cost']
endCosts = costData[(costData$epochNum %% numEpochs == 0) & (costData$costType == 'v_cost') & 
                    (costData$epochNum > 0) & (costData$isStart == 0), 'cost']
mean(startCosts - endCosts)

# A linear model of the starting costs vs. epoch number
lm(startCosts ~ costData$epochNum[match(startCosts, costData$cost)])

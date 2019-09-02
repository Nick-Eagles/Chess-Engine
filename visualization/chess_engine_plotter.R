library(ggplot2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")
costData = read.csv(file="costs.csv")

#costData = tail(costData, 40)
ggplot(costData) +
  geom_point(mapping = aes(x=epochNum, y=cost, color=costType)) +
  labs(title="Cost on data vs. epoch")
 

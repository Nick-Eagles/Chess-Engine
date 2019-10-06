library('ggplot2')

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")
rewards = as.numeric(read.csv('rewards.csv', header=FALSE)[1,])

# localDensity measures the fraction of observed rewards close to the value of each reward
tol = 0.05
localDensity = sapply(rewards, function(r) sum(abs(rewards - r) < tol))/length(rewards)
rDf = data.frame('fractionOfRewards' = localDensity, 'rewardValues' = rewards)

# Thus this plot is a sort of "continuous" variant of a barplot
ggplot(rDf) +
  geom_point(aes(x = rewardValues, y = fractionOfRewards))

library(ggplot2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine/visualization")

lossData = read.csv('test_losses.csv')
ggplot(lossData) +
  geom_point(mapping = aes(x=learnRate, y=loss)) +
  labs(title="Loss on data vs. learning rate",  x="learn rate") +
  scale_x_continuous(trans='log10') +
  ylim(c(0.65,0.8))

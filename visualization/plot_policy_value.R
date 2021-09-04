library(ggplot2)
library(reshape2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine")
loss_path = 'visualization/external_generator_costs.csv'

plot_metric = function(metric) {
  if (metric == 'accuracy') {
    colnames_of_interest = c('epoch',
                             'policy_end_square_categorical_accuracy',
                             'val_policy_end_square_categorical_accuracy')
    title = "Accuracy on data vs. epoch"
    color_var = "Accuracy type"
  } else if (metric == 'loss') {
    colnames_of_interest = c('epoch',
                             'policy_end_square_loss',
                             'val_policy_end_square_loss')
    title = "Loss on data vs. epoch"
    color_var = "Loss type"
  } else {
    stop('Acceptable metrics are "loss" or "accuracy"')
  }
  
  loss_data = read.csv(loss_path)[, colnames_of_interest]
  loss_data$epoch = 1:nrow(loss_data)
  
  loss_data = melt(loss_data, 'epoch')
  
  ggplot(loss_data) +
    geom_point(mapping = aes(x=epoch, y=value, color=variable)) +
    labs(title=title, 
         color=color_var, 
         x="epoch number")
}

plot_metric('loss')
plot_metric('accuracy')

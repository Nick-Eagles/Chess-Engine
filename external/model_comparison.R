library(ggplot2)
library(reshape2)

setwd("C:/Users/Nick/Appdata/Local/Programs/Python/Python37/Scripts/Chess_Engine")

compare_optimizers = function() {
    colnames_of_interest = c('epoch', 'loss', 'val_loss')
    
    # Read losses in, and only extract overall/average training and validation losses
    sgd_losses = read.csv(file=file.path('visualization', 'external_sgd_costs.csv'))[, colnames_of_interest]
    sgd_losses$optim = 'sgd'
    
    adam_losses = read.csv(file=file.path('visualization', 'external_adam_costs.csv'))[, colnames_of_interest]
    adam_losses$optim = 'adam'
    
    rmsprop_losses = read.csv(file=file.path('visualization', 'external_rmsprop_costs.csv'))[, colnames_of_interest]
    rmsprop_losses$optim = 'rmsprop'
    
    # Format data for plotting (comparing losses, splitting by training/test and optimizer)
    loss_data = rbind(sgd_losses, adam_losses, rmsprop_losses)
    loss_data  = melt(loss_data, c('epoch', 'optim'))
    
    # Plot
    ggplot(loss_data) +
        geom_point(mapping = aes(x=epoch, y=value, color=optim, shape=variable)) +
        labs(title="Loss on data vs. epoch", 
             color="Optimizer", 
             shape="Loss type", 
             x="epoch number")
    
    loss_data[loss_data$epoch == max(loss_data$epoch) & loss_data$optim == 'adam',]
    min(loss_data$value[loss_data$optim == 'adam' & loss_data$variable == 'val_loss'])
}

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
    
    loss_data = read.csv(file=file.path('visualization', 'external_adam_costs.csv'))[, colnames_of_interest]
    
    loss_data  = melt(loss_data, 'epoch')
    
    ggplot(loss_data) +
        geom_point(mapping = aes(x=epoch, y=value, color=variable)) +
        labs(title=title, 
             color=color_var, 
             x="epoch number")
}

plot_metric('loss')
plot_metric('accuracy')

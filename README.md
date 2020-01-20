# Google-FluidNet-Reproduce-Python
This is a trial of reproduction of the 'FluidNet' that has been developed by Jonathan Tompson et al. in Lua. The original article and code can be seen in 'https://arxiv.org/abs/1607.03597'

Here the 'data' only provide data from the fluid simulation of 5 items (for both training set and testing set), while the original code uses the data from 320 items. These data are further converted from .bin to .txt. 

The text files with suffix 'flags' indicate the geometry of the item, suffix 'UDiv' indicate the distribution of velocity divergence at that moment, while the suffix 'p' indicate the distribution of pressure at that moment. This model with LeNet structure is used to predict pressure based on the geometry and the velocity divergence.

However, some problems occurred in the overall process. The description is that: In the training process, the weights and biases of the overall model does not varies in only 2-3 iterations while one epoch has 80 iterations (with batch size = 4, number of items is 5 while 64 group of files in each item). 

Therefore the overall loss of the training and testing become consistent in every epoch. (In different running the values are totally the same)

I have further checked and it showed that the gradient of the model in back propagation becomes zero in only 2-3 iterations. I have tried various hyperparameters and this problem still exist. I will add more description if available.

Initial parameters:
batch size = 4
learning rate = 0.001
criterion = MSELoss
optimizer = Adam

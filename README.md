This repository is an implementation for a contrastive unsupervised learning technique that uses self classification.
The main idea is to identify each image in the dataset with an index. Then pass modified versions of each image to the network and train it so that it correctly associates each augmented version to the correct image in the dataset. This idea was first introduced in "Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks", Dosovitskiy, Alexey, et al.


During training the network consists of an encoder + a classifier head with as many outputs as items in the dataset. Each output represents the probability the network assigns an input image to correspond to that datapoint. For evaluation the head is removed and a new head is placed on top of the encoder. This new head has as many outputs as the number of classes in the task of interest and is finetuned on this task.

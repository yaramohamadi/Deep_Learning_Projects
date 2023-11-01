# Adverserial Training vs. Angular Loss for Robust Classification

The goal of this project is to train a model that achieves better generalization and robustness at the same time compared to the baseline. To achieve this goal, we use two methods as defense for our model, Adversarial Training, and using the Angular Loss. We evaluate the performance of a base classification model under various scenarios before and after robustification using these methods and compare the results. 

The dataset is CIFAR10 and the base model is ResNet18. 20% of the train dataset is used for training and the rest for validation with stratification based on the labels. The test dataset also is used for testing. The model's last layer's size is reduced from 512 to 128. The rest of the hyperparameters are based on the following figure:

<img width="421" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/c8f39a02-ff7e-4c37-b055-88f40b189765">


## Baseline training

The following figure shows train and validation accuracy and loss during training:

<img width="437" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/1a4d0a69-61fc-4785-a919-6442f24a7429">


With 98.08% accuracy, the model almost perfectly fits to the train data and achieves a 72% validation accuracy. Test accuracy in this situation is 70.93%.

Let's visualize how robust the learnt embeddings are. For this, using the 128 dimension representations from the last residual layer, we train a UMAP model in an unsupervised manner on the training dataset. The datapoints are then mapped to the following 2D space: 

<img width="380" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/f0836e05-ff20-4b24-80ad-df999eeb8668">


The model is successful in distinguishing the classes in the training dataset and also seperates the validation and test datapoints from different classes reasonably well. 

## Attacking the model


The adverserial attacks used are the Fast Gradient Method, Color Jitter, and Gaussian Noise. The hyperparameters are based on the following table:

<img width="426" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/c2c6d856-d246-45d8-a432-ef937162ac7a">


Each transformation is applied with a 30% probability on the dataset. The following figure shows an example of 4 original images on the left and their perturbed counterparts are on the right:

<img width="436" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/27735e33-327d-4b2c-a715-0249faa9a3e4">


After applying the perturbations on the test dataset, the accuracy of the trained model drops significantly from 70.93% down to 46.56%.

The 2D representations of the datapoints in each perturbed dataset now look like this and they are a lot less seperable comparing to the clean datapoints:

<img width="349" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/42b18586-a43d-4f5b-afff-2aac0d9c98a1">


Next, we will explore our 2 methods for making our model more robust to the perturbations.

## Adversarial Training

In this method, we train the model on the same types of perturbation applied during the test to robustify our model to them.

The following figure shows CLEAN train and validation dataset accuracies and losses during training:

<img width="434" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/b9836766-0adc-415f-8e51-43f1ca9ca406">



The model now achieves 78.3% training accuracy, illustrating the difficulity of the new task compared to the baseline training. Also, we see a 68.66% validation and 68.46% test accuracy on clean data. This is a case of losing generalization when adding robustness. Note that the accuracy on the perturbed test dataset is now 51.92% which is better than the baseline's which was 46.56%.

The 2D representations of the datapoints in each **perturbed** dataset now look like this and they are a a bit more seperable in some cases comparing to the baseline, while still failing to seperate many of the datapoints:

<img width="401" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/99e1638c-bb7b-4cf1-a384-8052863c13f8">


## Angular Loss

Angular Loss is similar to TripletLoss. However, instead of modifying distances, it modifies the triangular angles between anchor, positive, and negative points. By limitting the angle of the negative head, the positive and anchor heads get closer to each other. 

**Problems of TripletLoss that AngularLoss addresses:**
- Distance is sensitive to scale and variance of the data. Therefore, a single distance metric cannot always correctly capture the interclass differences. Furthermore, it fails to capture the intraclass differences since each class may posess a unique variance. Instead, AngularLoss is robust to the variance and scale of the data.
  
- In the TripletLoss, only (Anchor, Positive) and (Anchor, Negative) pairs are taken into account and the (Positive, Negative) relationship is neglected. This results in the need for a lot of data for correct training with TripletLoss. Instead, according to the 
trigonometric relationship, by controlling two of the relations in AngularLoss, the third parameter is indirectly also controlled (A triangle's angles sum up to 180 degrees).

The AngularLoss function looks like this: 

<img width="312" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/c25cb662-281b-4acd-8d69-312957550a84">


where x_a and x_p are the anchor and positive datapoints.

### Training

To make sure the batches are class balanced, I have created a custom batch sampler that takes 260 samples includinng 26 from each class. The input to the loss function is the **normalized** 128 dimensional representation before the softmax layer. By doing this normalization, the outputs are mapped to a sphere. Also, alpha parameter for the angular loss is set to 45 degrees.

The following figure shows CLEAN train and validation dataset losses during training:

<img width="276" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/53042da7-440e-4c69-b11e-eef92295eb8a">


The model has converged afte about 40 epochs and the train and validation losses are 1.5 and 0.345 respectively.

Since AngularLoss is a metric learning approach, we cannot directly obtain classification outputs from the trained model. The model outputs a representation in which the datapoints from the same class are closer to each other. For obtaining the accuracy, I trained a K-Nearest Neighbor with K=3 on these representations to do classification. Using this classifier model, the accuracy on the clean train, validation and test datasets are 99.8%, 72.67% and 73.23%. This model already surpasses the base model in terms of accuracy on the clean data! This is good news so far, because Adversarial Training could not retain the accuracy of the base model on the clean data. The accuracy of this model on the perturbed train, validation, and test data is also 72.79%, 49.63%, and 52.07% which is again good news! The model is also more robust comparing to the baseline model. However, the robustness is not as strong as Adversarial Training. 

We should note that in order to make a more correct comparison, we should have classified the first two methods also using a KNN classification head. I think if we would have done that, AngularLoss may have beaten Adversarial Robustness in the cases of robustness as well.

The 2D representations of the datapoints in each **perturbed** dataset look like this and they are alot more seperable comparing to the baseline. The reason is the use of metric learning, which directly works on seperability and distance of the classes.

<img width="394" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/533b6467-4734-4935-9a42-8da45a4143db">


## Summary

The following table summarizes the performance of the 3 models:

<img width="348" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/7b2283bb-047c-4cf5-8b20-efeb767892a8">


We saw that Angular Loss and Adversarial Training could robustify the model to certain adversarial attacks and perturbations. The Angular Loss enforces better seperability among the classes and in practice may work better than Adversarial Training. Also, more robustness does not always imply less generalization on clean data, as Angular Loss not only robustified the model, but also improved generalization.

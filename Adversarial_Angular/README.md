# Adverserial Training vs. Angular Loss for Robust Classification

The goal of this project is to train a model that achieves better generalization and robustness at the same time compared to the baseline. To achieve this goal, we use two methods as defense for our model, Adversarial Training, and using the Angular Loss. We evaluate the performance of a base classification model under various scenarios before and after robustification using these methods and compare the results. 

The dataset is CIFAR10 and the base model is ResNet18. 20% of the train dataset is used for training and the rest for validation with stratification based on the labels. The test dataset also is used for testing. The model's last layer's size is reduced from 512 to 128. The rest of the hyperparameters are based on the following figure:

<img width="470" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/40c03ec0-ad50-4dbf-bfb1-bd09c181d6ad">

## Baseline training

The following figure shows train and validation accuracy and loss during training:

<img width="487" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/ec080860-d15c-441d-b980-c1cb46f1e74c">

With 98.08% accuracy, the model almost perfectly fits to the train data and achieves a 72% validation accuracy. Test accuracy in this situation is 70.93%.

Let's visualize how robust the learnt embeddings are. For this, using the 128 dimension representations from the last residual layer, we train a UMAP model in an unsupervised manner on the training dataset. The datapoints are then mapped to the following 2D space: 

<img width="424" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/207c491c-fecd-41b8-95af-794a654434b8">

The model is successful in distinguishing the classes in the training dataset and also seperates the validation and test datapoints from different classes reasonably well. 

## Attacking the model


The adverserial attacks used are the Fast Gradient Method, Color Jitter, and Gaussian Noise. The hyperparameters are based on the following table:

<img width="473" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/ad071aae-63c2-453f-a3f6-c1a368b232a9">

Each transformation is applied with a 30% probability on the dataset. The following figure shows an example of 4 original images on the left and their perturbed counterparts are on the right:

<img width="492" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/9cd973da-3334-4384-9517-1d043cb686ff">

After applying the perturbations on the test dataset, the accuracy of the trained model drops significantly from 70.93% down to 46.56%.

The 2D representations of the datapoints in each perturbed dataset now look like this and they are a lot less seperable comparing to the clean datapoints:

t<img width="390" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/f61a9bf9-2017-4b7f-9d15-e4c0f3ba6901">

Next, we will explore our 2 methods for making our model more robust to the perturbations.

## Adversarial Training

In this method, we train the model on the same types of perturbation applied during the test to robustify our model to them.

The following figure shows CLEAN train and validation dataset accuracies and losses during training:

<img width="477" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/b8c6cc39-2746-4204-ab55-14811e7db672">

The model now achieves 78.3% training accuracy, illustrating the difficulity of the new task compared to the baseline training. Also, we see a 68.66% validation and 68.46% test accuracy on clean data. This is a case of losing generalization when adding robustness. Note that the accuracy on the perturbed test dataset is now 51.92% which is better than the baseline's which was 46.56%.

The 2D representations of the datapoints in each **perturbed** dataset now look like this and they are a a bit more seperable in some cases comparing to the baseline, while still failing to seperate many of the datapoints:

<img width="444" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/1dc4c6d4-16c8-462d-9731-093b5bc6acb3">

## Angular Loss

Angular Loss is similar to TripletLoss. However, instead of modifying distances, it modifies the triangular angles between anchor, positive, and negative points. By limitting the angle of the negative head, the positive and anchor heads get closer to each other. 

**Problems of TripletLoss that AngularLoss addresses:**
- Distance is sensitive to scale and variance of the data. Therefore, a single distance metric cannot always correctly capture the interclass differences. Furthermore, it fails to capture the intraclass differences since each class may posess a unique variance. Instead, AngularLoss is robust to the variance and scale of the data.
  
- In the TripletLoss, only (Anchor, Positive) and (Anchor, Negative) pairs are taken into account and the (Positive, Negative) relationship is neglected. This results in the need for a lot of data for correct training with TripletLoss. Instead, according to the 
trigonometric relationship, by controlling two of the relations in AngularLoss, the third parameter is indirectly also controlled (A triangle's angles sum up to 180 degrees).

The AngularLoss function looks like this: 

<img width="396" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/73ea1797-e294-4c9a-82da-7fbca03d718e">

where x_a and x_p are the anchor and positive datapoints and x_c = (x_a + x_p)/2.

### Training

To make sure the batches are class balanced, I have created a custom batch sampler that takes 260 samples includinng 26 from each class. The input to the loss function is the **normalized** 128 dimensional representation before the softmax layer. By doing this normalization, the outputs are mapped to a sphere. Also, alpha parameter for the angular loss is set to 45 degrees.

The following figure shows CLEAN train and validation dataset losses during training:

<img width="286" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/44b58408-cafb-4104-9d45-c208c21d325a">

The model has converged afte about 40 epochs and the train and validation losses are 1.5 and 0.345 respectively.

Since AngularLoss is a metric learning approach, we cannot directly obtain classification outputs from the trained model. The model outputs a representation in which the datapoints from the same class are closer to each other. For obtaining the accuracy, I trained a K-Nearest Neighbor with K=3 on these representations to do classification. Using this classifier model, the accuracy on the clean train, validation and test datasets are 99.8%, 72.67% and 73.23%. This model already surpasses the base model in terms of accuracy on the clean data! This is good news so far, because Adversarial Training could not retain the accuracy of the base model on the clean data. The accuracy of this model on the perturbed train, validation, and test data is also 72.79%, 49.63%, and 52.07% which is again good news! The model is also more robust comparing to the baseline model. However, the robustness is not as strong as Adversarial Training. 

We should note that in order to make a more correct comparison, we should have classified the first two methods also using a KNN classification head. I think if we would have done that, AngularLoss may have beaten Adversarial Robustness in the cases of robustness as well.

The 2D representations of the datapoints in each **perturbed** dataset look like this and they are alot more seperable comparing to the baseline. The reason is the use of metric learning, which directly works on seperability and distance of the classes.

<img width="444" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/1dc4c6d4-16c8-462d-9731-093b5bc6acb3">

## Summary

The following table summarizes the performance of the 3 models:

<img width="364" alt="image" src="https://github.com/yaramohamadi/Adversarial_Angular/assets/20110907/61007772-2842-480e-95b4-f0d26e4a88df">

We saw that Angular Loss and Adversarial Training could robustify the model to certain adversarial attacks and perturbations. The Angular Loss enforces better seperability among the classes and in practice may work better than Adversarial Training. Also, more robustness does not always imply less generalization on clean data, as Angular Loss not only robustified the model, but also improved generalization.

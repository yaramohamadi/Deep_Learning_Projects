This project includes analysis of three explainabiliy methods in different applications as follows:

- SHAP for Life Expectancy Prediction
- LIME for Image Classificattion
- D-RISE Saliency Maps for Object Detection


# SHAP for Life Expectancy Prediction

In this section, we analyze the performance of a linear regression model on the [Life Expectancy dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) using Kernel SHAP and Deep SHAP.

**Summary Plots:**

<img width="527" alt="Screenshot 2023-11-01 145046" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/209b9b22-6623-44ab-a83a-fcf4598c40ed">

Force Plots of a **Belgian person with high life expectancy**:

<img width="539" alt="Screenshot 2023-11-01 145046" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/4933d131-0412-4a8d-8f33-c6204985cfdd">

Force Plots of a **Russian person with low life expectancy**:

<img width="519" alt="Screenshot 2023-11-01 145046" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/56503949-8286-4225-9c87-295118df4f91">

# LIME for Image Classificattion

Using the [LIME python library](https://lime-ml.readthedocs.io/en/latest/index.html), and the LIME_image module, we now analyze the predictions of MobileNet V2 trained on ImageNet for a few arbitrary images (The image classes should already be present in the ImageNet).

#### First input image (Creepy one!):

<img width="427" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/8335116e-c2d0-4ce5-a788-47649b92cfcd">

Model top-5 outputs: **Hammerhead 90%**, Tiger shark 6.8%, Great white shark 2%, Dugong 2%.

The following figure from left to right indicates the boundaries, Pros and Cons reigons, All of the superpixels, and their heatmap using the LIME package. The shark's head contributes the most positively to the network's prediction.

<img width="547" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/a06aa971-448d-4417-a194-ed126c80c5ff">

#### Second input image (Confusing one with two possible classes!):

<img width="381" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/bfbae5aa-4d4f-4106-8e8f-5c8cd59cdca2">

Model top-5 outputs: **Pembroke 62.8%**, Cardigan 35.8%, Chihuahua 5%, Papillon 2%, Norwich terrier 1%.

The following figure shows the LIME explanation. The dog's head contributes the most positively to the prediction and the presence of the cat has little contribution to the model's output.

<img width="542" alt="image" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/c9be1bdf-485c-447c-afe4-d1936f0e3d51">

# D-RISE Saliency Maps for Object Detection

Using D-RISE Saliency Maps, let's analyze the predictions of an object detector.

**First image results:**

<img width="542" alt="Screenshot 2023-11-01 145046" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/5b87dc21-8e5a-4de3-9749-3ed1cc0a6fc8">

**Second image results (Fail mode):**

<img width="548" alt="Screenshot 2023-11-01 145046" src="https://github.com/yaramohamadi/Deep_Learning_Projects/assets/20110907/77f2e54f-4c88-40ab-a6da-e3d2f714e01c">

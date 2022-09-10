In the recent years, and after introducing the first Generative Adversarial Network, there has been many attempts at making the training of generative models more stable and their outputs more realistic. However, eventhough GAN produces better looking images than VAEs, it lags behind VAE in that it cannot encode real data (This is important because even when using Generative Models, our goal is often classification)
In [BiGAN](https://arxiv.org/abs/1605.09782) paper, an extra encoder was trained alongside the generator and the discriminator to allow data encoding. However, BiGAN is an old paper, and it does not have the innovations of recent GAN works. Later, in [BigBiGAN](https://arxiv.org/abs/1907.02544) paper, the architecture of [BigGAN](https://arxiv.org/abs/1809.11096) and the idea of BiGAN were combined so the results looked great in addition to the possiblity to encode data. 

In the first part of this project, I performed an in depth analysis of the elements present in BigBiGAN's loss function (See the figure below). 

<img src="imgs/bigbiganloss.png" data-canonical-src="imgs/bigbiganloss.png" width="200" />

I used the MNIST dataset and tried turning off each of the loss elements Sx, Sz, and Sxz, hence gaining an intuition on the effect of each one in the final performance.



# Sources

These sources support the corrected technical content in the two study weeks. The notes synthesize them rather than reproducing them.

## Your study material

| Material | Used for |
| --- | --- |
| Transcripts 21-30 | Optimizer sequence, exploding-gradient motivation, initialization, and dropout scope. |
| Transcripts 31-39 | CNN topic order, image formats, convolution/padding/pooling, dense head, and RGB pipeline. |
| `Complete Deep Learning/20-24 Optimizers.pdf` and `Practicals/Optimizers.ipynb` | Existing optimizer explanations and practical handoff. |
| `Complete Deep Learning/27-8 Weight initialization Techniques.pdf` and `29-Dropout Layer.pdf` | Existing visual explanations for stability and regularization. |
| `Complete Deep Learning/30-38 CNN.pdf` | Existing CNN diagrams and shape-trace motivation. |
| `Optimizers/*.png` and `Weight init & Gradient overloading/*.png` | Preserved as temporary reference images under `tmp/source-assets/`. |
| 15 supplied CNN-basics screenshots | Reconciled for architecture orientation, task map, image tensors, convolution, pooling, and classifier flow; retained only under `tmp/reference-images/cnn-basics/`. |

## External sources

- [Glorot and Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html) - Xavier/Glorot initialization and variance propagation.
- [He et al. (2015), *Delving Deep into Rectifiers*](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) - He initialization for rectifier networks.
- [Sutskever et al. (2013), *On the importance of initialization and momentum in deep learning*](https://proceedings.mlr.press/v28/sutskever13.html) - interaction of initialization and momentum.
- [Duchi, Hazan, and Singer (2011), *Adaptive Subgradient Methods*](https://jmlr.org/beta/papers/v12/duchi11a.html) - AdaGrad.
- [Hinton’s RMSProp course materials](https://www.cs.toronto.edu/~hinton/coursera_slides.html) - RMSProp provenance and intuition.
- [Kingma and Ba (2014), *Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) - Adam’s moment estimates and bias correction.
- [Srivastava et al. (2014), *Dropout*](https://jmlr.csail.mit.edu/beta/papers/v15/srivastava14a.html) - dropout’s regularization mechanism.
- [Keras Dropout documentation](https://keras.io/api/layers/regularization_layers/dropout/) - current `rate` and inference behavior.
- [LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition*](https://leon.bottou.org/papers/lecun-98h) - foundational convolutional-network reference.
- [Keras Conv2D documentation](https://keras.io/api/layers/convolution_layers/convolution2d/) - current tensor layout, filters, padding, and convolution-layer behavior.
- [Keras MaxPooling2D documentation](https://keras.io/2/api/layers/pooling_layers/max_pooling2d/) - pooling behavior and output-shape formulas.
- [Keras AveragePooling2D documentation](https://keras.io/api/layers/pooling_layers/average_pooling2d/) - average-pooling semantics and shapes.
- [Keras computer-vision examples](https://keras.io/examples/vision/) - task and model-family handoff examples.

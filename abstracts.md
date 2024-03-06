Taki, M. (2017). Deep Residual Networks and Weight Initialization. arXiv, 1709.02956. Retrieved from https://arxiv.org/abs/1709.02956v1

    - ResNets are relatively insensitive to choice of initial weights
    - how batch normalization improves backpropagation in ResNet
    - we propose new weight initialization distribution to prevent exploding gradients

Wang, W., Sun, Y., Eriksson, B., Wang, W., & Aggarwal, V. (2018). Wide Compression: Tensor Ring Nets. arXiv, 1802.09052. Retrieved from https://arxiv.org/abs/1802.09052v1

    - Tensor Ring (TR) factorizations to compress existing MLPs and CNNs
    - with little or no quality degredation on image classification

Oseledets, I. V. (2011). Tensor-Train Decomposition. SIAM J. Sci. Comput. Retrieved from https://epubs.siam.org/doi/abs/10.1137/090752286?journalCode=sjoce3

    - A new way to approximate tensor with a product of simpler tensors (eq. 1.2)

Yin, M., Sui, Y., Liao, S., & Yuan, B. (2021). Towards Efficient Tensor Decomposition-Based DNN Model Compression with Optimization Framework. arXiv, 2107.12422. Retrieved from https://arxiv.org/abs/2107.12422v1

    - Problem: compressing CNNs with Tensor train (TT) and Tensor ring (TR) suffers significant accuracy loss
    - A new approach requires a specific training procedure
    - very high compression performance with high accuracy
    - Also works for RNNs

Hanin, B., & Rolnick, D. (2018). How to Start Training: The Effect of Initialization and Architecture. arXiv, 1803.01719. Retrieved from https://arxiv.org/abs/1803.01719v3

    - Study failure modes for early training in deep ReLU nets:
    - 1) exploding or vanishing mean activation length
    - 2) exponentially large variance of activation length
    - For FCN, the cure of 1) require a specific init, and the cure of 2) require a specific constraint
    - For ResNets, the cure of 1) require a specific scaling, then 2) also gets cured

Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. arXiv, 1602.07868. Retrieved from https://arxiv.org/abs/1602.07868v3

    - a weight that decouples the length of vectors from their direction
    - we improve the conditioning of the optimization problem and we speed up convergence of SGD
    - is inspired by batch normalization but is more widely applicable
    - useful in supervised image recognition, generative modelling, and deep RL

Arpit, D., Campos, V., & Bengio, Y. (2019). How to Initialize your Network? Robust Initialization for WeightNorm & ResNets. arXiv, 1906.02341. Retrieved from https://arxiv.org/abs/1906.02341v2

    - a novel initialization strategy for weight normalized networks with and without residual connections
    - is based on mean field approximation
    - outperforms existing methods in generalization, robustness to hyper-parameters and variance between seeds

Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv, 1312.6120. Retrieved from https://arxiv.org/abs/1312.6120v3

    - Consider a DNN without activations
    - Despite the linearity, they have nonlinear GD dynamics, long plateaus followed by rapid transitions
    - Discussing some initial conditions on the weights that emerge during unsupervised pretraining
    - Propose a dynamical isometry (all singular values of the Jacobian concentrate near 1)
    - Propose orthonormal initialization
    - Faithful gradient propagation occurs in a special regime known as the edge of chaos

Pennington, J., Schoenholz, S. S., & Ganguli, S. (2017). Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. arXiv, 1711.04735. Retrieved from https://arxiv.org/abs/1711.04735v1

    - Authors compute analytically the entire singular value distribution of a DNN’s input-output Jacobian
    - ReLU networks are incapable of dynamical isometry (see https://arxiv.org/abs/1312.6120)
    - Sigmoidal networks with orthogonal weight initialization can achieve isometry and outperform ReLU nets
    - DNNs achieving dynamical isometry learn orders of magnitude faster than networks that do not

Tarnowski, W., Warchoł, P., Jastrzębski, S., Tabor, J., & Nowak, M. A. (2018). Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function. arXiv, 1809.08848. Retrieved from https://arxiv.org/abs/1809.08848v3

    - In ResNets dynamical isometry (https://arxiv.org/abs/1312.6120) is achievable for any activation function
    - Use Free Probability and Random Matrix Theories (FPT & RMT)
    - Study initial and late phases of the learning processes

Mishkin, D., & Matas, J. (2015). All you need is a good init. arXiv, 1511.06422. Retrieved from https://arxiv.org/abs/1511.06422v7

    - Layer-sequential unit-variance (LSUV) initialization:
    - 1) use orthonormal matrices
    - 2) normalize the variance of the output of each layer to be equal to one

Hu, W., Xiao, L., & Pennington, J. (2020). Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks. arXiv, 2001.05992. Retrieved from https://arxiv.org/abs/2001.05992v1

    - Proof that orthogonal initialization speeds up convergence
    - With it, the width for efficient convergence is independent of the depth (without it does not)
    - Is related to the principle of dynamical isometry (https://arxiv.org/abs/1312.6120)

Xiao, L., Bahri, Y., Sohl-Dickstein, J., Schoenholz, S. S., & Pennington, J. (2018). Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks. arXiv, 1806.05393. Retrieved from https://arxiv.org/abs/1806.05393v2

    - Are residual connections and batch normalization necessary for very deep nets?
    - No, just use a Delta-Orthogonal initialization and appropriate (in this case, tanh) nonlinearity.
    - This research is based on a mean field theory and dynamical isometry (https://arxiv.org/abs/1312.6120)

Xie, D., Xiong, J., & Pu, S. (2017). All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation. arXiv, 1703.01827. Retrieved from https://arxiv.org/abs/1703.01827v3

    - Problem: how to train deep nets without any shortcuts/identity mappings?
    - Solution: regularizer which utilizes orthonormality and a backward error modulation mechanism.

Wang, J., Chen, Y., Chakraborty, R., & Yu, S. X. (2019). Orthogonal Convolutional Neural Networks. arXiv, 1911.12207. Retrieved from https://arxiv.org/abs/1911.12207v3

    - Orthogonal convolution: filter orthogonality with doubly block-Toeplitz matrix representation
    - Outperforms the kernel orthogonality, learns more diverse and expressive features

Balestriero, R., & Baraniuk, R. (2018). Mad Max: Affine Spline Insights into Deep Learning. arXiv, 1805.06576. Retrieved from https://arxiv.org/abs/1805.06576v5

    - A large class of DNs can be written as a composition of maxaffine spline operators (MASOs)
    - This links DNs to the theory of vector quantization (VQ) and K-means clustering
    - Propose a simple penalty term to loss function to significantly improve performance

Lezcano-Casado, M., & Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group. arXiv, 1901.08428. Retrieved from https://arxiv.org/abs/1901.08428v3

    - A reparametrization to perform unconstrained optimizaion with orthogonal and unitary constraints
    - We apply our results to RNNs with orthogonal recurrent weights, yielding a new architecture called EXPRNN
    - Faster, accurate, and more stable convergence
    - https://github.com/pytorch/pytorch/issues/48144

Bingham, G., & Miikkulainen, R. (2021). AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks. arXiv, 2109.08958. Retrieved from https://arxiv.org/abs/2109.08958v2

    - A weight initialization algorithm that automatically adapts to different architectures
    - Scales the weights by tracking the mean and variance of signals as they propagate through the network
    - Improves performance of convolutional, residual, and transformer networks

Sander, M. E., Ablin, P., & Peyré, G. (2022). Do Residual Neural Networks discretize Neural Ordinary Differential Equations? arXiv, 2205.14612. Retrieved from https://arxiv.org/abs/2205.14612v2

    - Are discrete dynamics defined by a ResNet close to the continuous one of a Neural ODE?
    - Several theoretical results
    - A simple technique to train ResNets without storing activations
    - Recover the approximated activations during the backward pass by using a reverse-time Euler scheme
    - Fine-tuning very deep ResNets without memory consumption in the residual layers

Marion, P., Wu, Y.-H., Sander, M. E., & Biau, G. (2023). Implicit regularization of deep residual networks towards neural ODEs. arXiv, 2309.01213. Retrieved from https://arxiv.org/abs/2309.01213v2

    - Proof that if ResNet is initialized as a discretization of a neural ODE, then it holds throughout training

Peluchetti, S., & Favaro, S. (2019). Infinitely deep neural networks as diffusion processes. arXiv, 1905.11065. Retrieved from https://arxiv.org/abs/1905.11065v3

    - For deep nets with iid weight init, the dependency on the input vanishes as depth increases to infinity.
    - Under some assumptions, infinitely deep ResNets converge to SDEs (diffusion processes)
    - They do not suffer from the above property

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. arXiv, 1806.07366. Retrieved from https://arxiv.org/abs/1806.07366v5

    - Continuous-depth residual networks and continuous-time latent variable models, continuous normalizing flows
    - The derivative of the hidden state is parameterized
    - The output of the network is computed using a blackbox differential equation solver
    - Have constant memory cost
    - Adapt evaluation strategy to each input
    - Can explicitly trade numerical precision for speed
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

Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. (2015). Tensorizing Neural Networks. arXiv, 1509.06569. Retrieved from https://arxiv.org/abs/1509.06569v2

    - Authors convert FCN weight matrices to Tensor Train format (TT-layer, TensorNet)
    - Number of parameters is reduced by a huge factor (up to 200000 times for VGG dense layers)
    - The expressive power is preserved

Cichocki, A., Lee, N., Oseledets, I. V., Phan, A.-H., Zhao, Q., & Mandic, D. (2016). Low-Rank Tensor Networks for Dimensionality Reduction and Large-Scale Optimization Problems: Perspectives and Challenges PART 1. arXiv, 1609.00893. Retrieved from https://arxiv.org/abs/1609.00893v3

    - A book (pt. 1) about Tucker and Tensor Train (TT) decompositions and their extensions or generalizations
    - This can be used to convert intractable huge-scale optimization problems into a set of smaller problems
    - Chapter 1: Introduction and Motivation
    - Chapter 2: Tensor Operations and Tensor Network Diagrams
    - Chapter 3: Constrained Tensor Decompositions: From Two-way to Multiway Component Analysis
    - Chapter 4: Tensor Train Decompositions: Graphical Interpretations and Algorithms
    - Chapter 5: Discussion and Conclusions

Cichocki, A., Phan, A.-H., Zhao, Q., Lee, N., Oseledets, I. V., Sugiyama, M., & Mandic, D. (2017). Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations. Part 2 Applications and Future Perspectives. arXiv, 1708.09165. Retrieved from https://arxiv.org/abs/1708.09165v1

    -  A book (pt. 2) about tensor network models for super-compressed representation of data/parameters
    -  Emphasis is on the tensor train (TT) and Hierarchical Tucker (HT) decompositions
    -  Applied areas: regression and classification, eigenvalue decomposition, Riemannian optimization, DNNs
    -  Part 1 and Part 2 of this work can be used either as stand-alone separate texts

Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2019). Hyperbolic Image Embeddings. arXiv, 1904.02239. Retrieved from https://arxiv.org/abs/1904.02239v2

    - Hyperbolic embeddings as an alternative to Euclidean and spherical embeddings
    - Hyperbolic spaces are more suitable for embedding data with such hierarchical structure
    - Experiments with few-shot learning and person re-identification demonstrate these embeddings are beneficial
    - Propose an approach to evaluate the hyperbolicity of a dataset using Gromov δ-hyperbolicity

Novikov, A., Trofimov, M., & Oseledets, I. (2016). Exponential Machines. arXiv, 1605.03795. Retrieved from https://arxiv.org/abs/1605.03795v3

    - Exponential Machines (ExM), a predictor that models all interactions of every order
    - The Tensor Train format regularizes an exponentially large tensor of parameters
    - SOTA performance on synthetic data with high-order interactions

Khrulkov, V., & Oseledets, I. (2017). Art of singular vectors and universal adversarial perturbations. arXiv, 1709.03582. Retrieved from https://arxiv.org/abs/1709.03582v2

    -  A new algorithm for constructing adversarial perturbations
    -  Computing (p, q)-singular vectors of the Jacobian matrices of hidden layers of a network

Khrulkov, V., Novikov, A., & Oseledets, I. (2017). Expressive power of recurrent neural networks. arXiv, 1711.00811. Retrieved from https://arxiv.org/abs/1711.00811v2

    - As known, deep Hierarchical Tucker CNNs have exponentially higher expressive power than shallow networks
    - We prove the same for RNNs with Tensor Train (TT) decomposition
    - We compare expressive powers of the HT- and TT-Networks
    - We implement the recurrent TT-Networks and provide numerical evidence of their expressivity

Vasilescu, M. A. O. (2023). Causal Deep Learning: Causal Capsules and Tensor Transformers. arXiv, 2301.00314. Retrieved from https://arxiv.org/abs/2301.00314v1

    - NNs and tensor factorization methods may perform causal inference, or simply perform regression
    - A new deep neural network composed of a hierarchy of autoencoders
    - This results in a hierarchy of kernel tensor factor models
    - Forward causal questions (what if?) and inverse causal questions (why?) are addressed

Wang, M., Pan, Y., Xu, Z., Yang, X., Li, G., & Cichocki, A. (2023). Tensor Networks Meet Neural Networks: A Survey and Future Perspectives. arXiv, 2302.09019. Retrieved from https://arxiv.org/abs/2302.09019v2

    - Tensor networks (TNs) were introduced to solve the curse of dimensionality in large-scale tensors
    - We refer to the combinations of NNs and TNs as tensorial neural networks (TNNs)
    - Three primary aspects: network compression, information fusion, and quantum circuit simulation
    - Methods for improving TNNs, implementing TNNs, future directions

Stoudenmire, E. M., & Schwab, D. J. (2016). Supervised Learning with Quantum-Inspired Tensor Networks. arXiv, 1605.05775. Retrieved from https://arxiv.org/abs/1605.05775v2

    - Algorithms for optimizin Tensor networks can be adapted to supervised learning tasks

Ma, X., Zhang, P., Zhang, S., Duan, N., Hou, Y., Song, D., & Zhou, M. (2019). A Tensorized Transformer for Language Modeling. arXiv, 1906.09777. Retrieved from https://arxiv.org/abs/1906.09777v3

    - We propose Multi-linear attention with Block-Term Tensor Decomposition (BTD)
    - This not only largely compress the model parameters but also obtain performance improvements

Lu, Y., Ma, C., Lu, Y., Lu, J., & Ying, L. (2020). A Mean-field Analysis of Deep ResNet and Beyond: Towards Provable Optimization Via Overparameterization From Depth. arXiv, 2003.05508. Retrieved from https://arxiv.org/abs/2003.05508v2

    - Question: why do ResNets achieve zero training loss, while optimization landscape is highly non-convex?
    - We propose a new continuum limit of deep ResNets with a good landscape where every local minimizer is global
    - We apply existing mean-field analyses of two-layer networks to deep networks
    - We propose several novel training schemes which result in strong empirical performance

Cohen, A.-S., Cont, R., Rossier, A., & Xu, R. (2021). Scaling Properties of Deep Residual Networks. arXiv, 2105.12245. Retrieved from https://arxiv.org/abs/2105.12245v2

    - We investigate the scaling behavior of trained ResNet weights as the number of layers increases
    - Found at least three different scaling regimes
    - In two of these regimes, the properties may be described in terms of a class of ODEs or SDEs

Cirone, N. M., Lemercier, M., & Salvi, C. (2023). Neural signature kernels as infinite-width-depth-limits of controlled ResNets. arXiv, 2303.17671. Retrieved from https://arxiv.org/abs/2303.17671v2

    - We consider randomly initialized controlled ResNets defined as Euler-discretizations of neural controlled differential equations, a unified architecture which enconpasses both RNNs and ResNets
    - Study convergence in the infinite-depth regime

Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. arXiv, 2005.08926. Retrieved from https://arxiv.org/abs/2005.08926v2

    - Problem in neural ODEs: no mechanism for adjusting the trajectory based on subsequent observations
    - We demonstrate how this may be resolved through the mathematics of controlled differential equations
    - This is applicable to the partially observed irregularly-sampled multivariate time series
    - SOTA performance against similar (ODE or RNN based) models in empirical studies on a range of datasets
    - Theoretical results demonstrating universal approximation

Fang, C., Lee, J. D., Yang, P., & Zhang, T. (2020). Modeling from Features: a Mean-field Framework for Over-parameterized Deep Neural Networks. arXiv, 2007.01452. Retrieved from https://arxiv.org/abs/2007.01452v1

    - A new framework to analyze neural network training
    - We capture the evolution of an over-parameterized DNN trained by Gradient Descent
    - Global convergence proof for over-parameterized DNN in the mean-field regime

Ding, Z., Chen, S., Li, Q., & Wright, S. (2021). Overparameterization of deep ResNet: zero loss and mean-field analysis. arXiv, 2105.14417. Retrieved from https://arxiv.org/abs/2105.14417v3

    - Study ResNet convergence in the infinite-depth and infinite-width regime
    - GD becomes a gradient flow for a probability distribution that is characterized by a PDE
    - Results suggest that the training of the large enough ResNet gives a near-zero loss
    - Estimates of the depth and width needed to reduce the loss below a given threshold

Thorpe, M., & van Gennip, Y. (2018). Deep Limits of Residual Neural Networks. arXiv, 1810.11741. Retrieved from https://arxiv.org/abs/1810.11741v4

    - Study ResNet as a discretisation of an ODE
    - Some convergence studies that connect the discrete setting to a continuum problem

Allen-Zhu, Z., Li, Y., & Song, Z. (2018). A Convergence Theory for Deep Learning via Over-Parameterization. arXiv, 1811.03962. Retrieved from https://arxiv.org/abs/1811.03962v5

    - Study the theory of multi-layer networks
    - Proof that SGD can find global minima on the training objective of over-parameterized DNNs
    - Key insight is that in a neighborhood of the random initialization, the opt. landscape is almost convex
    - This implies an equivalence between over-parameterized finite width NNs and neural tangent kernel
    - Our theory at least applies to FCN, CNN and ResNet

Goyle, V., Krishnaswamy, P., Ravikumar, K. G., Chattopadhyay, U., & Goyle, K. (2023). Neural Machine Translation For Low Resource Languages. arXiv, 2304.07869. Retrieved from https://arxiv.org/abs/2304.07869v2

    - In low-resouce NMT, there is no comprehensive survey done to identify what approaches work well
    - We take mBART as a baseline
    - We applt techniques like transfer learning, back translation and focal loss, and verify their effect

Raunak, V., Dalmia, S., Gupta, V., & Metze, F. (2020). On Long-Tailed Phenomena in Neural Machine Translation. arXiv, 2010.04924. Retrieved from https://arxiv.org/abs/2010.04924v1

    - Problem: NMT models struggle with generating low-frequency tokens
    - Penalizing  low-confidence predictions hurts beam search performance
    - We propose Anti-Focal loss, a generalization of Focal loss and cross-entropy
    - Anti-Focal loss allocates less relative loss to low-confidence predictions
    - It leads to significant gains over cross-entropy, especially on the generation of low-frequency words

Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. arXiv, 1804.08771. Retrieved from https://arxiv.org/abs/1804.08771v2

    - BLEU is a parameterized metric
    - These parameters are often not reported
    - The main culprit is different tokenization and normalization schemes applied to the reference
    - The author provide a new tool, SacreBLEU, to use a common BLEU scheme

Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2017). Visualizing the Loss Landscape of Neural Nets. arXiv, 1712.09913. Retrieved from https://arxiv.org/abs/1712.09913v3

    - Simple visualization strategies fail to accurately capture the local geometry
    - We present a visualization method based on "filter normalization"
    - When networks become deep, loss surface turns from convex to chaotic, but skip connections prevent this
    - We measure non-convexity by calculating eigenvalues of the Hessian around local minima
    - We show that SGD optimization trajectories lie in an extremely low dimensional space
    - This can be explained by the presence of large, nearly convex regions in the loss landscape

## 1994

Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Trans. Neural Networks, 5(2), 157–166. Retrieved from https://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf

    - It is experimentally known that RNNs poorly learn long-term dependencies (we also show this)
    - Theoretic result: either a dynamic system is resistant to noise, or efficiently trainable by GD
    - The gradient exponentially vanishes or explodes in RNNs
    - We propose alternatives to standard GD (time-weighted pseudo-Newton algorithm)

## 1997

Hochreiter, S., & Schmidhuber, J. (1997). Flat Minima. Neural Comput., 9(1), 1–42. Retrieved from https://www.bioinf.jku.at/publications/older/3304.pdf

    - Bayesian argument suggests that flat minima correspond to "simple" networks and low expected overfitting
    - Our algorithm requires the computation of second order derivatives
    - But it has backprop's order of complexity
    - In stock market prediction, it outperforms backprop, weight decay and optimal brain surgeon

## 2003

Bengio, Y., Paiement, J.-f., Vincent, P., Delalleau, O., Roux, N., & Ouimet, M. (2003). Out-of-Sample Extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering. Advances in Neural Information Processing Systems, 16. Retrieved from https://papers.nips.cc/paper_files/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html

## 2004

Grandvalet, Y., & Bengio, Y. (2004). Semi-supervised Learning by Entropy Minimization. Advances in Neural Information Processing Systems, 17. Retrieved from https://papers.nips.cc/paper_files/paper/2004/hash/96f2b50b5d3613adf9c27049b2a888c7-Abstract.html

## 2006

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504–507. Retrieved from https://www.cs.toronto.edu/~hinton/absps/science.pdf

    - High-dimensional data can be converted to low-dimensional codes by training a multilayer NN autoencoder
    - Problem: this works well only if the initial weights are close to a good solution
    - We describe an effective way of initializing the weights
    - Such autoencoder learn codes that work much better than PCA as a tool to reduce the dimensionality of data

Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. Neural Comput., 16764513. Retrieved from https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf

    - Problem: the explaining-away effects make inference difficult in Deep Belief Networks (DBN)
    - We propose an algorithm to learn DBNs one layer at a time
    - The top two layers form an undirected associative memory
    - Our 3-layer network forms a good generative model of the joint distribution of MNIST
    - The low-dimensional data manifolds are modeled by long ravines in the free-energy landscape of the top-level associative memory, and it is easy to explore these ravines

## 2007

Bengio, Y., Lamblin, P., Popovici, D., Larochelle, H., & Montreal, U. (2007). Greedy layer-wise training of deep networks. Advances in Neural Information Processing Systems, 19. Retrieved from https://proceedings.neurips.cc/paper_files/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf

    - Problem: gradient-based optimization of DNNs from random initialization often get stuck in poor solutions
    - We explore variants of Deep Belief Networks layer-wise learning algorithm to better understand its success
    - We found that layer-wise unsupervised training strategy helps the optimization by initializing weights in a region near a good local minimum, giving rise to internal distributed representations that are high-level abstractions

Bengio, Y., & LeCun, Y. (2007). Scaling learning algorithms towards AI. Retrieved from https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+lecun_chapter2007.pdf

Ranzato, M., Poultney, C., Chopra, S., & Cun, Y. (2006). Efficient Learning of Sparse Representations with an Energy-Based Model. Advances in Neural Information Processing Systems, 19. Retrieved from https://proceedings.neurips.cc/paper/2006/hash/87f4d79e36d68c3031ccf6c55e9bbd39-Abstract.html

    - We train a linear encoder-decoder with sparsifying non-linearity in unsupervised way
    - The non-linearity turns a code vector into a quasi-binary sparse code vector
    - Learning proceeds in a two-phase EM-like fashion
    - We use the proposed method to initialize the first CNNlayer to achieve SOTA on MNIST

## 2008

Lee, H., Ekanadham, C., & Ng, A. (2007). Sparse deep belief net model for visual area V2. Advances in Neural Information Processing Systems, 20. Retrieved from https://papers.nips.cc/paper_files/paper/2007/hash/4daa3db355ef2b0e64b472968cb70f0d-Abstract.html

    - We develop a sparse variant of the deep belief networks
    - The first layer, similar to prior work on sparse coding and ICA, results in edge filters
    - The second layer learns "corner" features that mimic properties of visual cortical area V2

Le Roux, N., & Bengio, Y. (2008). Representational power of restricted boltzmann machines and deep belief networks. arXiv, 18254699. Retrieved from https://pubmed.ncbi.nlm.nih.gov/18254699

Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. Proceedings of the 25th International Conference on Machine Learning, 1096–1103. Retrieved from https://www.cl.uni-heidelberg.de/courses/ws14/deepl/LeRouxBengio07.pdf

## 2009

Bengio, Y. (2009). Learning Deep Architectures for AI. Foundations, 2(1), 1–55. Retrieved from https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf

    - We discuss the motivations and principles regarding learning algorithms for deep architectures
    - We discuss multiple levels of distributed representations of the data
    - We focus on the Deep Belief Networks, and their component elements, the Restricted Boltzmann Machine

Erhan, D., Manzagol, P.-A., Bengio, Y., Bengio, S., & Vincent, P. (2009). The Difficulty of Training Deep Architectures and the Effect of Unsupervised Pre-Training. Artificial Intelligence and Statistics. PMLR. Retrieved from https://proceedings.mlr.press/v5/erhan09a.html

Larochelle, H., Bengio, Y., Louradour, J., & Lamblin, P. (2009). Exploring Strategies for Training Deep Neural Networks. Journal of Machine Learning Research, 1, 140. Retrieved from https://jmlr.org/papers/volume10/larochelle09a/larochelle09a.pdf

## 2010

Erhan, D., Bengio, Y., Courville, A., Manzagol, P.-A., Vincent, P., & Bengio, S. (2010). Why Does Unsupervised Pre-training Help Deep Learning? Journal of Machine Learning Research, 11(19), 625–660. Retrieved from https://www.jmlr.org/papers/v11/erhan10a.html

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v9/glorot10a.html

Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2010). Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. J. Mach. Learn. Res., 11, 3371–3408. Retrieved from https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

## 2011

Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v15/glorot11a.html

Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive Auto-Encoders: Explicit Invariance During Feature Extraction. International Conference on Machine Learning. Retrieved from https://icml.cc/2011/papers/455_icmlpaper.pdf

## 2012

Bengio, Y., Courville, A., & Vincent, P. (2012). Representation Learning: A Review and New Perspectives. arXiv, 1206.5538. Retrieved from https://arxiv.org/abs/1206.5538v3

    - This paper is about representation learning
    - We discuss advances in probabilistic models, auto-encoders, manifold learning, and deep networks
    - We discuss the problem of the multimodality of the posterior P(h|x)
    - We cover many high-level generic priors that we believe could improve representation learning
    - The long-term objective is to discover learning algorithms that can disentangle underlying factors

Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. arXiv, 1206.5533. Retrieved from https://arxiv.org/abs/1206.5533v2

Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv, 1207.0580. Retrieved from https://arxiv.org/abs/1207.0580v1

    - Dropout method: randomly omitting half of the feature detectors (FD) on each training case
    - This prevents co-adaptations, when a FD is only helpful in the context of several other specific FDs
    - Instead, each neuron learns to detect a feature that is generally helpful
    - Gives big improvements on many benchmark tasks

Bengio, Y., Mesnil, G., Dauphin, Y., & Rifai, S. (2012). Better Mixing via Deep Representations. arXiv, 1207.4404. Retrieved from https://arxiv.org/abs/1207.4404v1

**Pascanu**, R., Mikolov, T., & Bengio, Y. (2012). On the difficulty of training Recurrent Neural Networks. arXiv, 1211.5063. Retrieved from https://arxiv.org/abs/1211.5063v2

    - We attempt to improve the understanding of gradient vanishing and exploding in RNNs
    - We propose a gradient norm clipping strategy to deal with exploding gradients
    - We propose a soft constraint for the vanishing gradients problem

Schoelkopf, B., Janzing, D., Peters, J., Sgouritsa, E., Zhang, K., & Mooij, J. (2012). On Causal and Anticausal Learning. arXiv, 1206.6471. Retrieved from https://arxiv.org/abs/1206.6471v1

    - To predict one variable from another, it helps to know the causal structure underlying the variables
    - We discuss causal (predict Effect from Cause) and anticausal (Cause from Effect) predictions
    - Hypothesis: under an independence assumption for causal mechanism and input, semi-supervised learning works better in anticausal or confounded problems
    - This can be useful for covariate shift, concept drift, transfer learning, semi-supervised learning

## 2013

Ba, J., & Frey, B. (2013). Adaptive dropout for training deep neural networks. Advances in Neural Information Processing Systems, 26. Retrieved from https://papers.nips.cc/paper_files/paper/2013/hash/7b5b23f4aadf9513306bcd59afb6e4c9-Abstract.html

    - We propose standout method, when dropout is performed with a specific binary belief network
    - Both networks can be trained jointly

Baldi, P., & Sadowski, P. J. (2013). Understanding Dropout. Advances in Neural Information Processing Systems, 26. Retrieved from https://papers.nips.cc/paper_files/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html

    - We propose a general formalism for studying dropout
    - The averaging properties of dropout are characterized by three recursive equations
    - We also show how dropout performs SGD on a regularized error function

! Bengio, Y. (2013). Deep Learning of Representations: Looking Forward. arXiv, 1305.0445. Retrieved from https://arxiv.org/abs/1305.0445v2

    - We present a few appealing directions of research towards deep learning challenges, including:
    - 1) Scaling computations
    - 2) Reducing the difficulties in optimizing parameters
    - 3) designing (or avoiding) expensive inference and sampling
    - 4) Helping to learn representations that better disentangle the unknown underlying factors of variation

Bengio, Y., Yao, L., Alain, G., & Vincent, P. (2013). Generalized Denoising Auto-Encoders as Generative Models. arXiv, 1305.6663. Retrieved from https://arxiv.org/abs/1305.6663v4

Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv, 1308.3432. Retrieved from https://arxiv.org/abs/1308.3432v1

Bengio, Y., Thibodeau-Laufer, É., Alain, G., & Yosinski, J. (2013). Deep Generative Stochastic Networks Trainable by Backprop. arXiv, 1306.1091. Retrieved from https://arxiv.org/abs/1306.1091v5

Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). Maxout Networks. arXiv, 1302.4389. Retrieved from https://arxiv.org/abs/1302.4389v4

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ...Bengio, Y. (2013). Challenges in Representation Learning: A report on three machine learning contests. arXiv, 1307.0414. Retrieved from https://arxiv.org/abs/1307.0414v1

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv, 1312.6114. Retrieved from https://arxiv.org/abs/1312.6114v11

    - We aim to learn a directed probabilistic model with continuous latent variables
    - A reparameterization of the ELBO yields an estimator that can be optimized using SGD
    - We fit an approximate inference model to the intractable posterior using this estimator

Wager, S., Wang, S., & Liang, P. (2013). Dropout Training as Adaptive Regularization. arXiv, 1307.1493. Retrieved from https://arxiv.org/abs/1307.1493v2

Wang, S., & Manning, C. (2013). Fast dropout training. International Conference on Machine Learning. PMLR. Retrieved from https://proceedings.mlr.press/v28/wang13a.html

## 2014

! Bengio, Y. (2014). How Auto-Encoders Could Provide Credit Assignment in Deep Networks via Target Propagation. arXiv, 1407.7906. Retrieved from https://arxiv.org/abs/1407.7906v3

    - Three existing approaches are discussed to propagate training signals across the levels of representation
    - There is a fourth explored approach, to which the proposal discussed here belongs, based on target propagation
    - We propose to train DL models with reconstruction as a layer-local training signal
    - The auto-encoder may take both a representation of input and target
    - A deep auto-encoder decoding path generalizes gradient propagation
    - So, we are proposing to learn the back-propagation computation
    - The potential of this framework is to provide a biologically plausible credit assignment mechanism that would replace and not require back-propagation

Bornschein, J., & Bengio, Y. (2014). Reweighted Wake-Sleep. arXiv, 1406.2751. Retrieved from https://arxiv.org/abs/1406.2751v4

Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., & LeCun, Y. (2014). The Loss Surfaces of Multilayer Networks. arXiv, 1412.0233. Retrieved from https://arxiv.org/abs/1412.0233v3

    - We study FCN loss surface using random matrix theory
    - Assumptions of i) variable independence, ii) overparametrization, and iii) uniformity
    - FCN loss landscape is similar to the Hamiltonian of the H-spin spherical spin-glass model
    - We study locations of critical points (maxima, minima, and saddle points) of the random loss function
    - They are located in a well-defined band lower-bounded by the global minimum
    - They are of high quality measured by the test error
    - Both simulated annealing and SGD converges to them
    - On the contrary, in small-size networks poor quality local minima have nonzero probability of being recovered

Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. arXiv, 1406.2572. Retrieved from https://arxiv.org/abs/1406.2572v1

    - We discuss high-dimensional non-convex optimization
    - It is often believed that local minima are a problem for GD and quasi-Newton methods
    - We argue that saddle points, not local minima are a problem
    - They are surrounded by high error plateaus and give the illusory impression of local minima
    - We proopse the saddle-free Newton method, that can rapidly escape high dimensional saddle points
    - It has superior optimization performance for RNNs

Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation. arXiv, 1410.8516. Retrieved from https://arxiv.org/abs/1410.8516v6

    - We think that a good representation is one in which the data has a distribution that is easy to model
    - We propose Non-linear Independent Component Estimation (NICE) for modeling complex high-dimensional densities
    - It learns factorized latent distribution, i.e. independent latent variables
    - This approach yields good generative models on four image datasets and can be used for inpainting

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ...Bengio, Y. (2014). Generative Adversarial Networks. arXiv, 1406.2661. Retrieved from https://arxiv.org/abs/1406.2661v1

Kingma, D. P., & Welling, M. (2014). Efficient Gradient-Based Inference through Transformations between Bayes Nets and Neural Nets. arXiv, 1402.0480. Retrieved from https://arxiv.org/abs/1402.0480v5

Maeda, S.-i. (2014). A Bayesian encourages dropout. arXiv, 1412.7003. Retrieved from https://arxiv.org/abs/1412.7003v3

    - We provide a Bayesian interpretation to dropout
    - The inference after dropout training can be considered as an approximate inference by Bayesian model averaging
    - This view enables us to optimize the dropout rate (Bayesian dropout)

Montúfar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the Number of Linear Regions of Deep Neural Networks. arXiv, 1402.1869. Retrieved from https://arxiv.org/abs/1402.1869v2

Ozair, S., & Bengio, Y. (2014). Deep Directed Generative Autoencoders. arXiv, 1410.0630. Retrieved from https://arxiv.org/abs/1410.0630v1

    - We consider discrete data
    - The objective is to learn an encoder f that maps X to f(X) that has a much simpler distribution
    - Generating samples from the model is straightforward using ancestral sampling
    - We can pre-train and stack these encoders, gradually transforming the data distribution

Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. arXiv, 1401.4082. Retrieved from https://arxiv.org/abs/1401.4082v3

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). FitNets: Hints for Thin Deep Nets. arXiv, 1412.6550. Retrieved from https://arxiv.org/abs/1412.6550v4

Schmidhuber, J. (2014). Deep Learning in Neural Networks: An Overview. arXiv, 1404.7828. Retrieved from https://arxiv.org/abs/1404.7828v4

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(56), 1929–1958. Retrieved from https://jmlr.org/papers/v15/srivastava14a.html

    - We show that dropout improves the performance of NNss on supervised learning, tasks in vision, speech recognition, document classification and computational biology

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? arXiv, 1411.1792. Retrieved from https://arxiv.org/abs/1411.1792v1

Zhang, S., Choromanska, A., & LeCun, Y. (2014). Deep learning with Elastic Averaging SGD. arXiv, 1412.6651. Retrieved from https://arxiv.org/abs/1412.6651v8

Zhao, P., & Zhang, T. (2014). Accelerating Minibatch Stochastic Gradient Descent using Stratified Sampling. arXiv, 1405.3080. Retrieved from https://arxiv.org/abs/1405.3080v1

## 2015

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A., & Bengio, Y. (2015). A Recurrent Latent Variable Model for Sequential Data. arXiv, 1506.02216. Retrieved from https://arxiv.org/abs/1506.02216v6

Courbariaux, M., Bengio, Y., & David, J.-P. (2015). BinaryConnect: Training Deep Neural Networks with binary weights during propagations. arXiv, 1511.00363. Retrieved from https://arxiv.org/abs/1511.00363v3

Ge, R., Huang, F., Jin, C., & Yuan, Y. (2015). Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition. arXiv, 1503.02101. Retrieved from https://arxiv.org/abs/1503.02101v1

Peters, J., Bühlmann, P., & Meinshausen, N. (2015). Causal inference using invariant prediction: identification and confidence intervals. arXiv, 1501.01332. Retrieved from https://arxiv.org/abs/1501.01332v3

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. arXiv, 1507.06228. Retrieved from https://arxiv.org/abs/1507.06228v2

Wiatowski, T., & Bölcskei, H. (2015). A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction. arXiv, 1512.06293. Retrieved from https://arxiv.org/abs/1512.06293v3

Wu, H., & Gu, X. (2015). Towards Dropout Training for Convolutional Neural Networks. arXiv, 1512.00242. Retrieved from https://arxiv.org/abs/1512.00242v1

## 2016

Abdi, M., & Nahavandi, S. (2016). Multi-Residual Networks: Improving the Speed and Accuracy of Residual Networks. arXiv, 1609.05672. Retrieved from https://arxiv.org/abs/1609.05672v4

Baldassi, C., Gerace, F., Lucibello, C., Saglietti, L., & Zecchina, R. (2016). Learning may need only a few bits of synaptic precision. arXiv, 1602.04129. Retrieved from https://arxiv.org/abs/1602.04129v2

Bottou, L., Curtis, F. E., & Nocedal, J. (2016). Optimization Methods for Large-Scale Machine Learning. arXiv, 1606.04838. Retrieved from https://arxiv.org/abs/1606.04838v3

Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., ...Zecchina, R. (2016). Entropy-SGD: Biasing Gradient Descent Into Wide Valleys. arXiv, 1611.01838. Retrieved from https://arxiv.org/abs/1611.01838v5

Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv, 1602.02830. Retrieved from https://arxiv.org/abs/1602.02830v3

Freeman, C. D., & Bruna, J. (2016). Topology and Geometry of Half-Rectified Network Optimization. arXiv, 1611.01540. Retrieved from https://arxiv.org/abs/1611.01540v4

Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. (2016). Deep Networks with Stochastic Depth. arXiv, 1603.09382. Retrieved from https://arxiv.org/abs/1603.09382v3

Im, D. J., Tao, M., & Branson, K. (2016). An empirical analysis of the optimization of deep network loss surfaces. arXiv, 1612.04010. Retrieved from https://arxiv.org/abs/1612.04010v4

Kawaguchi, K. (2016). Deep Learning without Poor Local Minima. arXiv, 1605.07110. Retrieved from https://arxiv.org/abs/1605.07110v3

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. arXiv, 1609.04836. Retrieved from https://arxiv.org/abs/1609.04836v2

Xie, B., Liang, Y., & Song, L. (2016). Diverse Neural Network Learns True Target Functions. arXiv, 1611.03131. Retrieved from https://arxiv.org/abs/1611.03131v3

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. arXiv, 1611.03530. Retrieved from https://arxiv.org/abs/1611.03530v2

## 2017

Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., ...Lacoste-Julien, S. (2017). A Closer Look at Memorization in Deep Networks. arXiv, 1706.05394. Retrieved from https://arxiv.org/abs/1706.05394v2

Balan, R., Singh, M., & Zou, D. (2017). Lipschitz Properties for Deep Convolutional Networks. arXiv, 1701.05217. Retrieved from https://arxiv.org/abs/1701.05217v1

Bartlett, P., Foster, D. J., & Telgarsky, M. (2017). Spectrally-normalized margin bounds for neural networks. arXiv, 1706.08498. Retrieved from https://arxiv.org/abs/1706.08498v2

Bojarski, M., Yeres, P., Choromanska, A., Choromanski, K., Firner, B., Jackel, L., & Muller, U. (2017). Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car. arXiv, 1704.07911. Retrieved from https://arxiv.org/abs/1704.07911v1

Brutzkus, A., Globerson, A., Malach, E., & Shalev-Shwartz, S. (2017). SGD Learns Over-parameterized Networks that Provably Generalize on Linearly Separable Data. arXiv, 1710.10174. Retrieved from https://arxiv.org/abs/1710.10174v1

Chang, B., Meng, L., Haber, E., Ruthotto, L., Begert, D., & Holtham, E. (2017). Reversible Architectures for Arbitrarily Deep Residual Neural Networks. arXiv, 1709.03698. Retrieved from https://arxiv.org/abs/1709.03698v2

Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. arXiv, 1703.04933. Retrieved from https://arxiv.org/abs/1703.04933v2

Du, S. S., Lee, J. D., Tian, Y., Poczos, B., & Singh, A. (2017). Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima. arXiv, 1712.00779. Retrieved from https://arxiv.org/abs/1712.00779v2

Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. arXiv, 1707.04585. Retrieved from https://arxiv.org/abs/1707.04585v1

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. arXiv, 1706.04599. Retrieved from https://arxiv.org/abs/1706.04599v2

Hoffer, E., Hubara, I., & Soudry, D. (2017). Train longer, generalize better: closing the generalization gap in large batch training of neural networks. arXiv, 1705.08741. Retrieved from https://arxiv.org/abs/1705.08741v2

Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). Snapshot Ensembles: Train 1, get M for free. arXiv, 1704.00109. Retrieved from https://arxiv.org/abs/1704.00109v1

Huang, F., Ash, J., Langford, J., & Schapire, R. (2017). Learning Deep ResNet Blocks Sequentially using Boosting Theory. arXiv, 1706.04964. Retrieved from https://arxiv.org/abs/1706.04964v4

Jastrzębski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2017). Three Factors Influencing Minima in SGD. arXiv, 1711.04623. Retrieved from https://arxiv.org/abs/1711.04623v3

Laurent, T., & von Brecht, J. (2017). Deep linear neural networks with arbitrary loss: All local minima are global. arXiv, 1712.01473. Retrieved from https://arxiv.org/abs/1712.01473v2

Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., & Sohl-Dickstein, J. (2017). Deep Neural Networks as Gaussian Processes. arXiv, 1711.00165. Retrieved from https://arxiv.org/abs/1711.00165v3

Liu, T., Lugosi, G., Neu, G., & Tao, D. (2017). Algorithmic stability and hypothesis complexity. arXiv, 1702.08712. Retrieved from https://arxiv.org/abs/1702.08712v2

Lu, Y., Zhong, A., Li, Q., & Dong, B. (2017). Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations. arXiv, 1710.10121. Retrieved from https://arxiv.org/abs/1710.10121v3

Molchanov, D., Ashukha, A., & Vetrov, D. (2017). Variational Dropout Sparsifies Deep Neural Networks. arXiv, 1701.05369. Retrieved from https://arxiv.org/abs/1701.05369v3

Nguyen, Q., & Hein, M. (2017). The loss surface of deep and wide neural networks. arXiv, 1704.08045. Retrieved from https://arxiv.org/abs/1704.08045v2

Reddi, S. J., Zaheer, M., Sra, S., Poczos, B., Bach, F., Salakhutdinov, R., & Smola, A. J. (2017). A Generic Approach for Escaping Saddle points. arXiv, 1709.01434. Retrieved from https://arxiv.org/abs/1709.01434v1

Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., & Bottou, L. (2017). Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. arXiv, 1706.04454. Retrieved from https://arxiv.org/abs/1706.04454v3

Safran, I., & Shamir, O. (2017). Spurious Local Minima are Common in Two-Layer ReLU Neural Networks. arXiv, 1712.08968. Retrieved from https://arxiv.org/abs/1712.08968v3

Smith, L. N., & Topin, N. (2017). Exploring loss function topology with cyclical learning rates. arXiv, 1702.04283. Retrieved from https://arxiv.org/abs/1702.04283v1

Smith, L. N., & Topin, N. (2017). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv, 1708.07120. Retrieved from https://arxiv.org/abs/1708.07120v3

Smith, S. L., & Le, Q. V. (2017). A Bayesian Perspective on Generalization and Stochastic Gradient Descent. arXiv, 1710.06451. Retrieved from https://arxiv.org/abs/1710.06451v3

Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2017). Don't Decay the Learning Rate, Increase the Batch Size. arXiv, 1711.00489. Retrieved from https://arxiv.org/abs/1711.00489v2

Soudry, D., & Hoffer, E. (2017). Exponentially vanishing sub-optimal local minima in multilayer neural networks. arXiv, 1702.05777. Retrieved from https://arxiv.org/abs/1702.05777v5

Wiatowski, T., Grohs, P., & Bölcskei, H. (2017). Energy Propagation in Deep Convolutional Neural Networks. arXiv, 1704.03636. Retrieved from https://arxiv.org/abs/1704.03636v3

Wu, L., Zhu, Z., & E, W. (2017). Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes. arXiv, 1706.10239. Retrieved from https://arxiv.org/abs/1706.10239v2

You, Y., Zhang, Z., Hsieh, C.-J., Demmel, J., & Keutzer, K. (2017). ImageNet Training in Minutes. arXiv, 1709.05011. Retrieved from https://arxiv.org/abs/1709.05011v10

## 2018

Allen-Zhu, Z., Li, Y., & Song, Z. (2018). A Convergence Theory for Deep Learning via Over-Parameterization. arXiv, 1811.03962. Retrieved from https://arxiv.org/abs/1811.03962v5

Athiwaratkun, B., Finzi, M., Izmailov, P., & Wilson, A. G. (2018). There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average. arXiv, 1806.05594. Retrieved from https://arxiv.org/abs/1806.05594v3

Bartlett, P. L., Helmbold, D. P., & Long, P. M. (2018). Gradient descent with identity initialization efficiently learns positive definite linear transformations by deep residual networks. arXiv, 1802.06093. Retrieved from https://arxiv.org/abs/1802.06093v4

Belkin, M., Ma, S., & Mandal, S. (2018). To understand deep learning we need to understand kernel learning. arXiv, 1802.01396. Retrieved from https://arxiv.org/abs/1802.01396v3

Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2018). Invertible Residual Networks. arXiv, 1811.00995. Retrieved from https://arxiv.org/abs/1811.00995v3

Bengio, Y., Lodi, A., & Prouvost, A. (2018). Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. arXiv, 1811.06128. Retrieved from https://arxiv.org/abs/1811.06128v2

Cooper, Y. (2018). The loss landscape of overparameterized neural networks. arXiv, 1804.10200. Retrieved from https://arxiv.org/abs/1804.10200v1

Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially No Barriers in Neural Network Energy Landscape. arXiv, 1803.00885. Retrieved from https://arxiv.org/abs/1803.00885v5

Galloway, A., Tanay, T., & Taylor, G. W. (2018). Adversarial Training Versus Weight Decay. arXiv, 1804.03308. Retrieved from https://arxiv.org/abs/1804.03308v3

Golmant, N., Vemuri, N., Yao, Z., Feinberg, V., Gholami, A., Rothauge, K., ...Gonzalez, J. (2018). On the Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent. arXiv, 1811.12941. Retrieved from https://arxiv.org/abs/1811.12941v1

Gotmare, A., Keskar, N. S., Xiong, C., & Socher, R. (2018). A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation. arXiv, 1810.13243. Retrieved from https://arxiv.org/abs/1810.13243v1

Hahn, S., & Choi, H. (2018). Understanding Dropout as an Optimization Trick. arXiv, 1806.09783. Retrieved from https://arxiv.org/abs/1806.09783v3

Hernández-García, A., & König, P. (2018). Do deep nets really need weight decay and dropout? arXiv, 1802.07042. Retrieved from https://arxiv.org/abs/1802.07042v3

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., & Bengio, Y. (2018). Learning deep representations by mutual information estimation and maximization. arXiv, 1808.06670. Retrieved from https://arxiv.org/abs/1808.06670v5

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv, 1803.05407. Retrieved from https://arxiv.org/abs/1803.05407v3

Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. arXiv, 1806.07572. Retrieved from https://arxiv.org/abs/1806.07572v4

Karakida, R., Akaho, S., & Amari, S.-i. (2018). Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. arXiv, 1806.01316. Retrieved from https://arxiv.org/abs/1806.01316v3

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2018). Effect of Depth and Width on Local Minima in Deep Learning. arXiv, 1811.08150. Retrieved from https://arxiv.org/abs/1811.08150v4

Kawaguchi, K., & Bengio, Y. (2018). Depth with Nonlinearity Creates No Bad Local Minima in ResNets. arXiv, 1810.09038. Retrieved from https://arxiv.org/abs/1810.09038v3

Liang, S., Sun, R., Lee, J. D., & Srikant, R. (2018). Adding One Neuron Can Eliminate All Bad Local Minima. arXiv, 1805.08671. Retrieved from https://arxiv.org/abs/1805.08671v1

Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018). Don't Use Large Mini-Batches, Use Local SGD. arXiv, 1808.07217. Retrieved from https://arxiv.org/abs/1808.07217v6

Liu, J., & Xu, L. (2018). Accelerating Stochastic Gradient Descent Using Antithetic Sampling. arXiv, 1810.03124. Retrieved from https://arxiv.org/abs/1810.03124v1

Martin, C. H., & Mahoney, M. W. (2018). Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning. arXiv, 1810.01075. Retrieved from https://arxiv.org/abs/1810.01075v1

Matthews, A. G. d. G., Rowland, M., Hron, J., Turner, R. E., & Ghahramani, Z. (2018). Gaussian Process Behaviour in Wide Deep Neural Networks. arXiv, 1804.11271. Retrieved from https://arxiv.org/abs/1804.11271v2

Mei, S., Montanari, A., & Nguyen, P.-M. (2018). A Mean Field View of the Landscape of Two-Layers Neural Networks. arXiv, 1804.06561. Retrieved from https://arxiv.org/abs/1804.06561v2

Nouiehed, M., & Razaviyayn, M. (2018). Learning Deep Models: Critical Points and Local Openness. arXiv, 1803.02968. Retrieved from https://arxiv.org/abs/1803.02968v2

Ruthotto, L., & Haber, E. (2018). Deep Neural Networks Motivated by Partial Differential Equations. arXiv, 1804.04272. Retrieved from https://arxiv.org/abs/1804.04272v2

Scaman, K., & Virmaux, A. (2018). Lipschitz regularity of deep neural networks: analysis and efficient estimation. arXiv, 1805.10965. Retrieved from https://arxiv.org/abs/1805.10965v2

Shamir, O. (2018). Are ResNets Provably Better than Linear Predictors? arXiv, 1804.06739. Retrieved from https://arxiv.org/abs/1804.06739v4

Xing, C., Arpit, D., Tsirigotis, C., & Bengio, Y. (2018). A Walk with SGD. arXiv, 1802.08770. Retrieved from https://arxiv.org/abs/1802.08770v4

Yun, C., Sra, S., & Jadbabaie, A. (2018). Small nonlinearities in activation functions create bad local minima in neural networks. arXiv, 1802.03487. Retrieved from https://arxiv.org/abs/1802.03487v4

Yun, C., Sra, S., & Jadbabaie, A. (2018). Efficiently testing local optimality and escaping saddles for ReLU networks. arXiv, 1809.10858. Retrieved from https://arxiv.org/abs/1809.10858v2

Zaeemzadeh, A., Rahnavard, N., & Shah, M. (2018). Norm-Preservation: Why Residual Networks Can Become Extremely Deep? arXiv, 1805.07477. Retrieved from https://arxiv.org/abs/1805.07477v5

Zhang, L., & Schaeffer, H. (2018). Forward Stability of ResNet and Its Variants. arXiv, 1811.09885. Retrieved from https://arxiv.org/abs/1811.09885v1

Zhang, J., Liu, T., & Tao, D. (2018). An Information-Theoretic View for Deep Learning. arXiv, 1804.09060. Retrieved from https://arxiv.org/abs/1804.09060v8

Zhang, C., Öztireli, C., Mandt, S., & Salvi, G. (2018). Active Mini-Batch Sampling using Repulsive Point Processes. arXiv, 1804.02772. Retrieved from https://arxiv.org/abs/1804.02772v2

## 2019

Allen-Zhu, Z., & Li, Y. (2019). What Can ResNet Learn Efficiently, Going Beyond Kernels? arXiv, 1905.10337. Retrieved from https://arxiv.org/abs/1905.10337v3

Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv, 1907.02893. Retrieved from https://arxiv.org/abs/1907.02893v3

Arpit, D., Campos, V., & Bengio, Y. (2019). How to Initialize your Network? Robust Initialization for WeightNorm & ResNets. arXiv, 1906.02341. Retrieved from https://arxiv.org/abs/1906.02341v2

Bengio, Y., Deleu, T., Rahaman, N., Ke, R., Lachapelle, S., Bilaniuk, O., ...Pal, C. (2019). A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms. arXiv, 1901.10912. Retrieved from https://arxiv.org/abs/1901.10912v2

Ding, T., Li, D., & Sun, R. (2019). Sub-Optimal Local Minima Exist for Neural Networks with Almost All Non-Linear Activations. arXiv, 1911.01413. Retrieved from https://arxiv.org/abs/1911.01413v3

Fort, S., & Jastrzebski, S. (2019). Large Scale Structure of Neural Network Loss Landscapes. arXiv, 1906.04724. Retrieved from https://arxiv.org/abs/1906.04724v1

Ghorbani, B., Krishnan, S., & Xiao, Y. (2019). An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. arXiv, 1901.10159. Retrieved from https://arxiv.org/abs/1901.10159v1

He, F., Liu, T., & Tao, D. (2019). Why ResNet Works? Residuals Generalize. arXiv, 1904.01367. Retrieved from https://arxiv.org/abs/1904.01367v1

Jiang, A. H., Wong, D. L.-K., Zhou, G., Andersen, D. G., Dean, J., Ganger, G. R., ...Pillai, P. (2019). Accelerating Deep Learning by Focusing on the Biggest Losers. arXiv, 1910.00762. Retrieved from https://arxiv.org/abs/1910.00762v1

Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2019). Subspace Inference for Bayesian Deep Learning. arXiv, 1907.07504. Retrieved from https://arxiv.org/abs/1907.07504v1

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2019). Every Local Minimum Value is the Global Minimum Value of Induced Model in Non-convex Machine Learning. arXiv, 1904.03673. Retrieved from https://arxiv.org/abs/1904.03673v3

Labach, A., Salehinejad, H., & Valaee, S. (2019). Survey of Dropout Methods for Deep Neural Networks. arXiv, 1904.13310. Retrieved from https://arxiv.org/abs/1904.13310v2

Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. arXiv, 1902.06720. Retrieved from https://arxiv.org/abs/1902.06720v4

Liang, S., Sun, R., & Srikant, R. (2019). Revisiting Landscape Analysis in Deep Neural Networks: Eliminating Decreasing Paths to Infinity. arXiv, 1912.13472. Retrieved from https://arxiv.org/abs/1912.13472v1

Liu, T., Chen, M., Zhou, M., Du, S. S., Zhou, E., & Zhao, T. (2019). Towards Understanding the Importance of Shortcut Connections in Residual Networks. arXiv, 1909.04653. Retrieved from https://arxiv.org/abs/1909.04653v3

Nakamura, K., & Hong, B.-W. (2019). Adaptive Weight Decay for Deep Neural Networks. arXiv, 1907.08931. Retrieved from https://arxiv.org/abs/1907.08931v2

Rangamani, A., Nguyen, N. H., Kumar, A., Phan, D., Chin, S. H., & Tran, T. D. (2019). A Scale Invariant Flatness Measure for Deep Network Minima. arXiv, 1902.02434. Retrieved from https://arxiv.org/abs/1902.02434v1

Shen, X., Tian, X., Liu, T., Xu, F., & Tao, D. (2019). Continuous Dropout. arXiv, 1911.12675. Retrieved from https://arxiv.org/abs/1911.12675v1

Simsekli, U., Sagun, L., & Gurbuzbalaban, M. (2019). A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks. arXiv, 1901.06053. Retrieved from https://arxiv.org/abs/1901.06053v1

Siu, C. (2019). Residual Networks Behave Like Boosting Algorithms. arXiv, 1909.11790. Retrieved from https://arxiv.org/abs/1909.11790v1

Sohl-Dickstein, J., & Kawaguchi, K. (2019). Eliminating all bad Local Minima from Loss Landscapes without even adding an Extra Unit. arXiv, 1901.03909. Retrieved from https://arxiv.org/abs/1901.03909v1

Wen, Y., Luk, K., Gazeau, M., Zhang, G., Chan, H., & Ba, J. (2019). An Empirical Study of Large-Batch Stochastic Gradient Descent with Structured Covariance Noise. arXiv, 1902.08234. Retrieved from https://arxiv.org/abs/1902.08234v4

Yun, C., Sra, S., & Jadbabaie, A. (2019). Are deep ResNets provably better than linear predictors? arXiv, 1907.03922. Retrieved from https://arxiv.org/abs/1907.03922v2

## 2020

Agarwal, C., D'souza, D., & Hooker, S. (2020). Estimating Example Difficulty Using Variance of Gradients. arXiv, 2008.11600. Retrieved from https://arxiv.org/abs/2008.11600v4

Chan, K. H. R., Yu, Y., You, C., Qi, H., Wright, J., & Ma, Y. (2020). Deep Networks from the Principle of Rate Reduction. arXiv, 2010.14765. Retrieved from https://arxiv.org/abs/2010.14765v1

Choe, Y. J., Ham, J., & Park, K. (2020). An Empirical Study of Invariant Risk Minimization. arXiv, 2004.05007. Retrieved from https://arxiv.org/abs/2004.05007v2

Faghri, F., Duvenaud, D., Fleet, D. J., & Ba, J. (2020). A Study of Gradient Variance in Deep Learning. arXiv, 2007.04532. Retrieved from https://arxiv.org/abs/2007.04532v1

Huang, K., Wang, Y., Tao, M., & Zhao, T. (2020). Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? -- A Neural Tangent Kernel Perspective. arXiv, 2002.06262. Retrieved from https://arxiv.org/abs/2002.06262v2

Lengerich, B., Xing, E. P., & Caruana, R. (2020). Dropout as a Regularizer of Interaction Effects. arXiv, 2007.00823. Retrieved from https://arxiv.org/abs/2007.00823v2

Melkman, A. A., Guo, S., Ching, W.-K., Liu, P., & Akutsu, T. (2020). On the Compressive Power of Boolean Threshold Autoencoders. arXiv, 2004.09735. Retrieved from https://arxiv.org/abs/2004.09735v1

Mixon, D. G., Parshall, H., & Pi, J. (2020). Neural collapse with unconstrained features. arXiv, 2011.11619. Retrieved from https://arxiv.org/abs/2011.11619v1

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training. arXiv, 2008.08186. Retrieved from https://arxiv.org/abs/2008.08186v2

Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., & Lajoie, G. (2020). Gradient Starvation: A Learning Proclivity in Neural Networks. arXiv, 2011.09468. Retrieved from https://arxiv.org/abs/2011.09468v4

Queiruga, A. F., Erichson, N. B., Taylor, D., & Mahoney, M. W. (2020). Continuous-in-Depth Neural Networks. arXiv, 2008.02389. Retrieved from https://arxiv.org/abs/2008.02389v1

Sankar, A. R., Khasbage, Y., Vigneswaran, R., & Balasubramanian, V. N. (2020). A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization. arXiv, 2012.03801. Retrieved from https://arxiv.org/abs/2012.03801v2

Sun, R., Li, D., Liang, S., Ding, T., & Srikant, R. (2020). The Global Landscape of Neural Networks: An Overview. arXiv, 2007.01429. Retrieved from https://arxiv.org/abs/2007.01429v1

Wang, L., Shen, B., Zhao, N., & Zhang, Z. (2020). Is the Skip Connection Provable to Reform the Neural Network Loss Landscape? arXiv, 2006.05939. Retrieved from https://arxiv.org/abs/2006.05939v1

Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. arXiv, 2002.08791. Retrieved from https://arxiv.org/abs/2002.08791v4

Yang, Z., Yu, Y., You, C., Steinhardt, J., & Ma, Y. (2020). Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. arXiv, 2002.11328. Retrieved from https://arxiv.org/abs/2002.11328v3

Zhao, P., Chen, P.-Y., Das, P., Ramamurthy, K. N., & Lin, X. (2020). Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. arXiv, 2005.00060. Retrieved from https://arxiv.org/abs/2005.00060v2

## 2021

Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T.-Y., ...Zoph, B. (2021). Revisiting ResNets: Improved Training and Scaling Strategies. arXiv, 2103.07579. Retrieved from https://arxiv.org/abs/2103.07579v1

Benton, G. W., Maddox, W. J., Lotfi, S., & Wilson, A. G. (2021). Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling. arXiv, 2102.13042. Retrieved from https://arxiv.org/abs/2102.13042v2

Bond-Taylor, S., Leach, A., Long, Y., & Willcocks, C. G. (2021). Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. arXiv, 2103.04922. Retrieved from https://arxiv.org/abs/2103.04922v4

Cao, S. (2021). Choose a Transformer: Fourier or Galerkin. arXiv, 2105.14995. Retrieved from https://arxiv.org/abs/2105.14995v4

Federici, M., Tomioka, R., & Forré, P. (2021). An Information-theoretic Approach to Distribution Shifts. arXiv, 2106.03783. Retrieved from https://arxiv.org/abs/2106.03783v2

Izmailov, P., Nicholson, P., Lotfi, S., & Wilson, A. G. (2021). Dangers of Bayesian Model Averaging under Covariate Shift. arXiv, 2106.11905. Retrieved from https://arxiv.org/abs/2106.11905v2

Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). What Are Bayesian Neural Network Posteriors Really Like? arXiv, 2104.14421. Retrieved from https://arxiv.org/abs/2104.14421v1

Han, X. Y., Papyan, V., & Donoho, D. L. (2021). Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path. arXiv, 2106.02073. Retrieved from https://arxiv.org/abs/2106.02073v4

Liu, M., Chen, L., Du, X., Jin, L., & Shang, M. (2021). Activated Gradients for Deep Neural Networks. arXiv, 2107.04228. Retrieved from https://arxiv.org/abs/2107.04228v1

Meunier, L., Delattre, B., Araujo, A., & Allauzen, A. (2021). A Dynamical System Perspective for Lipschitz Neural Networks. arXiv, 2110.12690. Retrieved from https://arxiv.org/abs/2110.12690v2

Rame, A., Dancette, C., & Cord, M. (2021). Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization. arXiv, 2109.02934. Retrieved from https://arxiv.org/abs/2109.02934v3

Roberts, D. A., Yaida, S., & Hanin, B. (2021). The Principles of Deep Learning Theory. arXiv, 2106.10165. Retrieved from https://arxiv.org/abs/2106.10165v2

Sander, M. E., Ablin, P., Blondel, M., & Peyré, G. (2021). Momentum Residual Neural Networks. arXiv, 2102.07870. Retrieved from https://arxiv.org/abs/2102.07870v3

Schneider, F., Dangel, F., & Hennig, P. (2021). Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks. arXiv, 2102.06604. Retrieved from https://arxiv.org/abs/2102.06604v2

Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Towards Causal Representation Learning. arXiv, 2102.11107. Retrieved from https://arxiv.org/abs/2102.11107v1

Shang, Y., Duan, B., Zong, Z., Nie, L., & Yan, Y. (2021). Lipschitz Continuity Guided Knowledge Distillation. arXiv, 2108.12905. Retrieved from https://arxiv.org/abs/2108.12905v1

Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. arXiv, 2102.11174. Retrieved from https://arxiv.org/abs/2102.11174v3

Smith, S. L., Dherin, B., Barrett, D. G. T., & De, S. (2021). On the Origin of Implicit Regularization in Stochastic Gradient Descent. arXiv, 2101.12176. Retrieved from https://arxiv.org/abs/2101.12176v1

Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., & Qu, Q. (2021). A Geometric Analysis of Neural Collapse with Unconstrained Features. arXiv, 2105.02375. Retrieved from https://arxiv.org/abs/2105.02375v1

Ziyin, L., Li, B., Simon, J. B., & Ueda, M. (2021). SGD with a Constant Large Learning Rate Can Converge to Local Maxima. arXiv, 2107.11774. Retrieved from https://arxiv.org/abs/2107.11774v4

## 2022

Ahn, K., Zhang, J., & Sra, S. (2022). Understanding the unstable convergence of gradient descent. arXiv, 2204.01050. Retrieved from https://arxiv.org/abs/2204.01050v2

Arjevani, Y., & Field, M. (2022). Annihilation of Spurious Minima in Two-Layer ReLU Networks. arXiv, 2210.06088. Retrieved from https://arxiv.org/abs/2210.06088v1

Bai, Q., Rosenberg, S., & Xu, W. (2022). A Geometric Understanding of Natural Gradient. arXiv, 2202.06232. Retrieved from https://arxiv.org/abs/2202.06232v3

Christof, C., & Kowalczyk, J. (2022). On the Omnipresence of Spurious Local Minima in Certain Neural Network Training Problems. arXiv, 2202.12262. Retrieved from https://arxiv.org/abs/2202.12262v2

Fedus, W., Dean, J., & Zoph, B. (2022). A Review of Sparse Expert Models in Deep Learning. arXiv, 2209.01667. Retrieved from https://arxiv.org/abs/2209.01667v1

Juneja, J., Bansal, R., Cho, K., Sedoc, J., & Saphra, N. (2022). Linear Connectivity Reveals Generalization Strategies. arXiv, 2205.12411. Retrieved from https://arxiv.org/abs/2205.12411v5

Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi, S. J., ...Kumar, S. (2022). The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers. arXiv, 2210.06313. Retrieved from https://arxiv.org/abs/2210.06313v2

Zhou, J., You, C., Li, X., Liu, K., Liu, S., Qu, Q., & Zhu, Z. (2022). Are All Losses Created Equal: A Neural Collapse Perspective. arXiv, 2210.02192. Retrieved from https://arxiv.org/abs/2210.02192v2

Zhou, J., Li, X., Ding, T., You, C., Qu, Q., & Zhu, Z. (2022). On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features. arXiv, 2203.01238. Retrieved from https://arxiv.org/abs/2203.01238v2

## 2023

Altintas, G. S., Bachmann, G., Noci, L., & Hofmann, T. (2023). Disentangling Linear Mode-Connectivity. arXiv, 2312.09832. Retrieved from https://arxiv.org/abs/2312.09832v1

Andriushchenko, M., Croce, F., Müller, M., Hein, M., & Flammarion, N. (2023). A Modern Look at the Relationship between Sharpness and Generalization. arXiv, 2302.07011. Retrieved from https://arxiv.org/abs/2302.07011v2

Araujo, A., Havens, A., Delattre, B., Allauzen, A., & Hu, B. (2023). A Unified Algebraic Perspective on Lipschitz Neural Networks. arXiv, 2303.03169. Retrieved from https://arxiv.org/abs/2303.03169v2

Dang, H., Tran, T., Osher, S., Tran-The, H., Ho, N., & Nguyen, T. (2023). Neural Collapse in Deep Linear Networks: From Balanced to Imbalanced Data. arXiv, 2301.00437. Retrieved from https://arxiv.org/abs/2301.00437v5

Dubois, Y., Hashimoto, T., & Liang, P. (2023). Evaluating Self-Supervised Learning via Risk Decomposition. arXiv, 2302.03068. Retrieved from https://arxiv.org/abs/2302.03068v3

Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking Distribution Shift in Tabular Data with TableShift. arXiv, 2312.07577. Retrieved from https://arxiv.org/abs/2312.07577v3

Gao, P., Xu, Q., Wen, P., Shao, H., Yang, Z., & Huang, Q. (2023). A Study of Neural Collapse Phenomenon: Grassmannian Frame, Symmetry and Generalization. arXiv, 2304.08914. Retrieved from https://arxiv.org/abs/2304.08914v2

Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2023). A mathematical perspective on Transformers. arXiv, 2312.10794. Retrieved from https://arxiv.org/abs/2312.10794v3

Li, W., Peng, Y., Zhang, M., Ding, L., Hu, H., & Shen, L. (2023). Deep Model Fusion: A Survey. arXiv, 2309.15698. Retrieved from https://arxiv.org/abs/2309.15698v1

Liu, Z., Xu, Z., Jin, J., Shen, Z., & Darrell, T. (2023). Dropout Reduces Underfitting. arXiv, 2303.01500. Retrieved from https://arxiv.org/abs/2303.01500v2

Pan, L., & Cao, X. (2023). Towards Understanding Neural Collapse: The Effects of Batch Normalization and Weight Decay. arXiv, 2309.04644. Retrieved from https://arxiv.org/abs/2309.04644v2

Peng, Z., Qi, L., Shi, Y., & Gao, Y. (2023). A Theoretical Explanation of Activation Sparsity through Flat Minima and Adversarial Robustness. arXiv, 2309.03004. Retrieved from https://arxiv.org/abs/2309.03004v4

Súkeník, P., Mondelli, M., & Lampert, C. (2023). Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model. arXiv, 2305.13165. Retrieved from https://arxiv.org/abs/2305.13165v1

## 2024

Guo, L., Ross, K., Zhao, Z., Andriopoulos, G., Ling, S., Xu, Y., & Dong, Z. (2024). Cross Entropy versus Label Smoothing: A Neural Collapse Perspective. arXiv, 2402.03979. Retrieved from https://arxiv.org/abs/2402.03979v2



Dudzik, A., & Veličković, P. (2022). Graph Neural Networks are Dynamic Programmers. arXiv, 2203.15544. Retrieved from https://arxiv.org/abs/2203.15544v3

Ibarz, B., Kurin, V., Papamakarios, G., Nikiforou, K., Bennani, M., Csordás, R., ...Veličković, P. (2022). A Generalist Neural Algorithmic Learner. arXiv, 2209.11142. Retrieved from https://arxiv.org/abs/2209.11142v2

Rodionov, G., & Prokhorenkova, L. (2023). Neural Algorithmic Reasoning Without Intermediate Supervision. arXiv, 2306.13411. Retrieved from https://arxiv.org/abs/2306.13411v2

Rodionov, G., & Prokhorenkova, L. (2024). Discrete Neural Algorithmic Reasoning. arXiv, 2402.11628. Retrieved from https://arxiv.org/abs/2402.11628v1

Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., ...Chen, W. (2024). ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. arXiv, 2403.03853. Retrieved from https://arxiv.org/abs/2403.03853v2
## 1984

Hinton, G. (1984, January 01). Boltzmann machines: Constraint satisfaction networks that learn. Retrieved from https://www.cs.utoronto.ca/~hinton/absps/bmtr.pdf

    - We present a parallel constrain satisfactory network called “Boltzmann Machine”
    - We describe a method based on statistical mechanics to learn from examples
    - It creates internal representations

## 1985

Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for boltzmann machines. Cognitive Science, 9(1), 147–169. Retrieved from https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf

    - We present a Boltzmann Machine
    - The machine is composed of 0-1 units connected by bidirectional links that can take on real values
    - We describe a learning method to create internal representations based on statistical mechanics
    - When shown a partial example, the network can complete it: the system will then find the minimum energy configuration that is compatible with that input
    - However, the current version of the learning algorithm is very slow

## 1986

Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. Parallel Distributed Processing, 1. Retrieved from https://www.cs.toronto.edu/~hinton/absps/pdp7.pdf

    - We need to find a way for a Boltzmann Machine to escape from local minima during a relaxation search
    - We found that this can be done by using a stochastic decision rule
    - An information needed to do credit assignment is propagated, and the network reaches thermal equilibrium
    - The network constructs distributed representations which are resistant to minor damage
    - They exhibit repid relearning after major damage

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536. Retrieved from https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf

    - We describe a back-propagation learning procedure for neural networks
    - Internal hidden units are able to learn important features of the task domain

## 1992

Neal, R. M. (1992). Connectionist Learning of Belief Networks. Artif. Intell. Retrieved from https://www.semanticscholar.org/paper/Connectionist-Learning-of-Belief-Networks-Neal/a120c05ad7cd4ce2eb8fb9697e16c7c4877208a5

    - Belief networks usually represent knowledge derived from experts
    - We describe how to learn "sigmoid" and "noisy-OR" belief networks using Gibbs sampling
    - Except for the lack of a negative phase, learning is similar to that in a Boltzmann machine
    - These metworks are naturally applicable to classification or unsupervised learning problems
    - They provide a link between connectionist learning and expert knowledge

## 1993

Hinton, G. E., & Zemel, R. (1993). Autoencoders, Minimum Description Length and Helmholtz Free Energy. Advances in Neural Information Processing Systems, 6. Retrieved from https://proceedings.neurips.cc/paper/1993/hash/9e3cfc48eccf81a0d57663e129aef3cb-Abstract.html

    - We derive an objective for training autoencoders based on the Minimum Description Length principle (MDL)
    - We aim to minimize the information required to describe both the code vector and the reconstruction error
    - This is minimized by choosing code vectors stochastically according to a Boltzmann distribution
    - The recognition weights approximate the Boltzmann distribution giving an upper bound on MDL
    - The generative weights define the energy of each possible code vector given the input vector

## 1994

Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Trans. Neural Networks, 5(2), 157–166. Retrieved from https://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf

    - It is experimentally known that RNNs poorly learn long-term dependencies (we also show this)
    - Theoretic result: either a dynamic system is resistant to noise, or efficiently trainable by GD
    - The gradient exponentially vanishes or explodes in RNNs
    - We propose alternatives to standard GD (time-weighted pseudo-Newton algorithm)

Zemel, R. (1994). A minimum description length framework for unsupervised learning. Retrieved from https://www.semanticscholar.org/paper/A-minimum-description-length-framework-for-learning-Zemel/b7b2bffdf5b62305bec4c0f1ea7e3c1ba66fccb5

    - A PhD thesis of R. Zemel
    - We describe unsupervised learning based on the Minimum Description Length (MDL) principle
    - It says to minimize the summed description length of the model and the data with respect to the model
    - It makes explicit a tradeoff between the accuracy of a representation and the succinctness
    - We derive objectives for self-supervised NNs according to MDL

## 1995

Dayan, P., Hinton, G. E., Neal, R. M., & Zemel, R. S. (1995). The Helmholtz Machine. Retrieved from https://www.cs.toronto.edu/~hinton/absps/helmholtz.pdf

    - A problem: For most of generative models, each pattern can be generated in exponentially many ways, so it is intractable to adjust the parameters to maximize the probability of the observed patterns
    - We overcome this by maximizing an easily computed lower bound on the probability of the observations
    - A Helmholtz Machine consists of multiple layers of binary stochastic units connected by two sets of weights
    - Top-down connections is a generative model, bottom-up connections is a recognition model

Hinton, G. E., Dayan, P., Frey, B., & Neal, R. (1995). The "wake-sleep" algorithm for unsupervised neural networks. Science. Retrieved from https://www.cs.toronto.edu/~hinton/absps/ws.pdf

    - We describe an unsupervised learning algorithm to fit a multilayer neural generative model
    - In the “wake” phase, neurons are driven by recognition connections and generative connections are adapted
    - In the “sleep” phase, neurons are driven by generative connections and recognition connections are adapted

Zemel, R. S., & Hinton, G. E. (1995). Learning population codes by minimizing description length. Neural Comput., 7(3), 549–564. Retrieved from https://dl.acm.org/doi/10.1162/neco.1995.7.3.549

    - We show how the minimum description length principle can be used to develop redundant population codes

## 1996

Saul, L. K., Jaakkola, T., & Jordan, M. I. (1996). Mean Field Theory for Sigmoid Belief Networks. arXiv, cs/9603102. Retrieved from https://arxiv.org/abs/cs/9603102v1

    - Problem: in large belief networks computing likelihoods is intractable
    - We provide a tractable approximation to the true probability distribution in sigmoid belief networks
    - We demonstrate the utility of this framework on handwritten digit recognition

## 1997

Bell, A. J., & Sejnowski, T. J. (1997). The “independent components” of natural scenes are edge filters. Vision Res., 37(23), 3327–3338. Retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2882863/pdf/nihms-197142.pdf

    - We propose the infomax network: a new unsupervised algorithm based on information maximization
    - Being trained on images, it produces sets of visual filters that are localized and oriented
    - The outputs of these filters are as independent as possible, since an algorithm performs ICA
    - Compared to PCA and ZCA, the ICA filters have more sparsely distributed outputs on natural scenes

Hinton, G. E., & Ghahramani, Z. (1997). Generative models for discovering sparse distributed representations. Philos. Trans. R. Soc. London, Ser. B. Retrieved from https://www.cs.toronto.edu/~hinton/absps/rgbn.pdf

    - We propose the Rectied Gaussian Belief Net: a hierarchical generative neural network
    - It uses bottom-up, top-down and lateral connections
    - It seems to work much better than the wake-sleep algorithm
    - It is very effective at discovering hierarchical sparse distributed representations
    - Its main disadvantage is that the recognition process involves Gibbs sampling

Hochreiter, S., & Schmidhuber, J. (1997). Flat Minima. Neural Comput., 9(1), 1–42. Retrieved from https://www.bioinf.jku.at/publications/older/3304.pdf

    - Bayesian argument suggests that flat minima correspond to "simple" networks and low expected overfitting
    - Our algorithm requires the computation of second order derivatives
    - But it has backprop's order of complexity
    - In stock market prediction, it outperforms backprop, weight decay and optimal brain surgeon

Rao, R. P. N., & Ballard, D. (1997). Dynamic Model of Visual Recognition Predicts Neural Response Properties in the Visual Cortex. Neural Comput. Retrieved from https://www.semanticscholar.org/paper/Dynamic-Model-of-Visual-Recognition-Predicts-Neural-Rao-Ballard/e3a83c2ed3af29a23ab342212d1ae9650a0c64a1

    -  We describe a hierarchical network model of visual recognition
    -  It dynamically combines input-driven bottom-up signals with expectation-driven top-down signals

## 1998

Prechelt, L. (1998). Early Stopping — But When? Neural Networks: Tricks of the Trade. Springer. Retrieved from https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf

    - We perform experiments with early stopping on 12 problems and 24 different network architectures 
    - We find that slower stopping criteria allow for small improvements in generalization (about 4% on average)
    - However, it costs much more training time (about factor 4 longer on average)

Lecun, Y., Bottou, L., Orr, G. B., & Muller, K.-r. (2000). Efficient BackProp. ResearchGate. Retrieved from https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf

    - Stochastic learning is faster than batch learning, often improves quality
    - Stochastic learning allow to track changes in online learning
    - However, many acceleration techniques (like conjugate gradient) only operate in batch learning
    - Shuffle the training set so that successive examples rarely belong to the same class
    - Focus hard examples (but not for data containing outliers)
    - Normalize the inputs and uncorrelate them if possible
    - Hyperbolic tangent often converge faster than sigmoid, also tanh(x) + ax may be good
    - Choose target values at the point of maximum second derivative of the sigmoid
    - Some recommendations for initialization are given (IMO see later works for this)
    - Some recommendations for separate learning rate for each weight
    - Radial basis functions (RBF) vs sigmoid units is discussed
    - SGD convergence and second-order methods are discussed in details
    - It is shown that classical second-order algoritms are impractical for NNs, modifications are proposed

Neal, R. M., & Hinton, G. E. (1998). A View of the EM Algorithm that Justifies Incremental, Sparse, and other Variants. Learning in Graphical Models. Springer. Retrieved from https://www.cs.toronto.edu/~hinton/absps/emk.pdf

    - We show that EM algorithm maximizes a function that resembles negative free energy
    - From this perspective, we can justify an incremental, sparse and other EM variants

Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1998). An Introduction to Variational Methods for Graphical Models. Learning in Graphical Models. Springer. Retrieved from https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf

## 1999

Rao, R. P. N., & Ballard, D. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nat. Neurosci. Retrieved from https://www.semanticscholar.org/paper/Predictive-coding-in-the-visual-cortex%3A-a-of-some-Rao-Ballard/a424ec3b8846f57b8ffdb566d272e28d5a525909

## 2001

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2001). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3(Jan), 601–608. Retrieved from https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

Ng, A., & Jordan, M. (2001). On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes. Advances in Neural Information Processing Systems, 14. Retrieved from https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html

## 2002

Hinton, H. (2002) Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation, 14, 1771-1800. - References - Scientific Research Publishing. (2024, March 10). Retrieved from https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf

## 2003

Bengio, Y., Paiement, J.-f., Vincent, P., Delalleau, O., Roux, N., & Ouimet, M. (2003). Out-of-Sample Extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering. Advances in Neural Information Processing Systems, 16. Retrieved from https://papers.nips.cc/paper_files/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html

    - We consider five types of unsupervised learning algorithms based on a spectral embedding
    - They are MDS, spectral clustering, Laplacian eigenmaps, Isomap and LLE
    - We present an extension for these methods
    - It allows one to apply a trained model to out-of-sample points without having to recompute eigenvectors
    - It introduces a notion of function induction and generalization error for these algorithms
    - We experiment on real high-dimensional data

Friston, K. J. (2003). Learning and inference in the brain. Neural Networks. Retrieved from https://www.semanticscholar.org/paper/Learning-and-inference-in-the-brain-Friston/0ea24ffe3bee9faeeead947c6c7ab00c99f5ccf2

Lee, T., & Mumford, D. (2003). Hierarchical Bayesian inference in the visual cortex. Journal of The Optical Society of America A-optics Image Science and Vision. Retrieved from https://www.semanticscholar.org/paper/Hierarchical-Bayesian-inference-in-the-visual-Lee-Mumford/5042cb5efad30f443adef472b8748e1b7bb0452f

Teh, Y. W., Welling, M., Osindero, S., & Hinton, G. E. (2004). Energy-based models for sparse overcomplete representations. Journal of Machine Learning Research, 4(7-8), 1235–1260. Retrieved from https://www.jmlr.org/papers/volume4/teh03a/teh03a.pdf

Welling, M., & Teh, Y. W. (2003). Approximate inference in Boltzmann machines. Artif. Intell., 143(1), 19–50. Retrieved from https://core.ac.uk/download/pdf/82455875.pdf

## 2004

Grandvalet, Y., & Bengio, Y. (2004). Semi-supervised Learning by Entropy Minimization. Advances in Neural Information Processing Systems, 17. Retrieved from https://papers.nips.cc/paper_files/paper/2004/hash/96f2b50b5d3613adf9c27049b2a888c7-Abstract.html

    - We present minimum entropy regularization for semi-supervised learning for any probabilistic classifier
    - Unlabeled examples are mostly beneficial when classes have small overlap
    - Experiments suggest that the minimum entropy regularization may be a serious contender to generative models

Knill, D., & Pouget, A. (2004). The Bayesian brain: the role of uncertainty in neural coding and computation. Trends Neurosci. Retrieved from https://www.semanticscholar.org/paper/The-Bayesian-brain%3A-the-role-of-uncertainty-in-and-Knill-Pouget/2a3146db2f3cb39ef37105d428eef964a253daf2

## 2005

Friston, K. J. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences. Retrieved from https://www.semanticscholar.org/paper/A-theory-of-cortical-responses-Friston/8fcbc38e7196b0dc8748f04cd6101e71f92c158e

## 2006

Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006). Online Passive-Aggressive Algorithms. Journal of Machine Learning Research, 7(19), 551–585. Retrieved from https://jmlr.csail.mit.edu/papers/v7/crammer06a.html

Friston, K. J., Kilner, J., & Harrison, L. (2006). A free energy principle for the brain. J. Physiol.-Paris. Retrieved from https://www.semanticscholar.org/paper/A-free-energy-principle-for-the-brain-Friston-Kilner/641a9d87b9636d4ef2e353a569ddede68bd29131

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

    - Kernel methods are fundamentally limited in their ability to learn complex high-dimensional functions
    - Kernel machines are shallow architectures, they can be very sample-inefficient
    - We analyze a limitation of kernel machines with a local kernel
    - We argue that deep architectures have the potential to generalize beyond immediate neighbors

Friston, K. J., & Stephan, K. (2007). Free-energy and the brain. Synthese. Retrieved from https://www.semanticscholar.org/paper/Free-energy-and-the-brain-Friston-Stephan/bf15128e48db7aaa9146328d965fc81dbdf4db3f

Ranzato, M., Poultney, C., Chopra, S., & Cun, Y. (2006). Efficient Learning of Sparse Representations with an Energy-Based Model. Advances in Neural Information Processing Systems, 19. Retrieved from https://proceedings.neurips.cc/paper/2006/hash/87f4d79e36d68c3031ccf6c55e9bbd39-Abstract.html

    - We train a linear encoder-decoder with sparsifying non-linearity in unsupervised way
    - The non-linearity turns a code vector into a quasi-binary sparse code vector
    - Learning proceeds in a two-phase EM-like fashion
    - We use the proposed method to initialize the first CNN layer to achieve SOTA on MNIST

## 2008

Friston, K. J. (2008). Hierarchical Models in the Brain. PLoS Comput. Biol. Retrieved from https://www.semanticscholar.org/paper/Hierarchical-Models-in-the-Brain-Friston/1a014a076cac3c7f5d81a084e296c095f9230437

Lee, H., Ekanadham, C., & Ng, A. (2007). Sparse deep belief net model for visual area V2. Advances in Neural Information Processing Systems, 20. Retrieved from https://papers.nips.cc/paper_files/paper/2007/hash/4daa3db355ef2b0e64b472968cb70f0d-Abstract.html

    - We develop a sparse variant of the deep belief networks
    - The first layer, similar to prior work on sparse coding and ICA, results in edge filters
    - The second layer learns "corner" features that mimic properties of visual cortical area V2

Le Roux, N., & Bengio, Y. (2008). Representational power of restricted boltzmann machines and deep belief networks. arXiv, 18254699. Retrieved from https://www.cl.uni-heidelberg.de/courses/ws14/deepl/LeRouxBengio07.pdf

    - We show that Restricted Boltzmann Machines are universal approximators of discrete distributions
    - Do we need a lot of layers in Deep Belief Networks?
    - Maybe the answer lies in the ability of a DBN to generalize better by having a more compact representation
    - The analysis suggests to investigate KL as an alternative to Contrastive Divergence for training each layer
    - Because it would take into account that more layers will be added

Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. Proceedings of the 25th International Conference on Machine Learning, 1096–1103. Retrieved from https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf

    - We propose a recipe for unsupervised learning based on the robustness to partial corruption of the input
    - Denoising autoencoders can be stacked to initialize deep architectures
    - We motivate this from a manifold learning, information theoretic and generative model perspectives
    - Experiments clearly show the surprising advantage of corrupting the input of autoencoders

Wainwright, M. J., & Jordan, M. I. (2008). Graphical Models, Exponential Families, and Variational Inference. MAL, 1(1-2), 1–305. Retrieved from https://www.nowpublishers.com/article/Details/MAL-001

## 2009

Bengio, Y. (2009). Learning Deep Architectures for AI. Foundations, 2(1), 1–55. Retrieved from https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf

    - We discuss the motivations and principles regarding learning algorithms for deep architectures
    - We discuss multiple levels of distributed representations of the data
    - We focus on the Deep Belief Networks, and their component elements, the Restricted Boltzmann Machine

Erhan, D., Manzagol, P.-A., Bengio, Y., Bengio, S., & Vincent, P. (2009). The Difficulty of Training Deep Architectures and the Effect of Unsupervised Pre-Training. Artificial Intelligence and Statistics. PMLR. Retrieved from https://proceedings.mlr.press/v5/erhan09a.html

    - Our experiments the positive effect of unsupervised pre-training and its role as a regularizer

Friston, K. J., & Kiebel, S. (2009). Predictive coding under the free-energy principle. Philosophical Transactions of the Royal Society B: Biological Sciences. Retrieved from https://www.semanticscholar.org/paper/Predictive-coding-under-the-free-energy-principle-Friston-Kiebel/6927ea92b0d759a75f0f696fa028155d6d9ee2ca

Friston, K. J., & Kiebel, S. (2009). Cortical circuits for perceptual inference. Neural Networks. Retrieved from https://www.semanticscholar.org/paper/Cortical-circuits-for-perceptual-inference-Friston-Kiebel/73881ecac61ef6a05847e0f110b4f3c6477a0d57

Friston, K. J. (2009). The free-energy principle: a rough guide to the brain? Trends in Cognitive Sciences. Retrieved from https://www.semanticscholar.org/paper/The-free-energy-principle%3A-a-rough-guide-to-the-Friston/a878886efacc6a5d742bf98bfc25c0734ce502b1

Friston, K. J., Daunizeau, J., & Kiebel, S. (2009). Reinforcement Learning or Active Inference? PLoS One. Retrieved from https://www.semanticscholar.org/paper/Reinforcement-Learning-or-Active-Inference-Friston-Daunizeau/5e07da2914783a21980ecc1cea688d1333e6b6e4

Salakhutdinov, R., & Hinton, G. (2009). Deep Boltzmann Machines. Artificial Intelligence and Statistics. PMLR. Retrieved from https://proceedings.mlr.press/v5/salakhutdinov09a.html

    - We propose a learning algorithm for Boltzmann machines with many layers of hidden variables
    - Data-dependent expectations are estimated using a variational approximation
    - Data-independent expectations are approximated using persistent Markov chains
    - The learning can be made more efficient by using a layer-by-layer "pre-training" phase
    - On MNIST and NORB deep Boltzmann machines learn good generative models

Larochelle, H., Bengio, Y., Louradour, J., & Lamblin, P. (2009). Exploring Strategies for Training Deep Neural Networks. Journal of Machine Learning Research, 1, 140. Retrieved from https://jmlr.org/papers/volume10/larochelle09a/larochelle09a.pdf

    - We confirmthat the greedy layer-wise unsupervised training strategy helps the optimization by
    - 1) Initializing weights in a region near a good local minimum
    - 2) Implicitly acts as a sort of regularization for high-level abstractions
    - We present a series of experiments demonstrating, for example, where the addition of more depth helps
    - We empirically explore simple variants of training algorithms
    - Highle resembles the paper "Greedy layer-wise training of deep networks"

Salakhutdinov, R. (2009). Learning Deep Generative Models (Dissertation). Retrieved from https://tspace.library.utoronto.ca/bitstream/1807/19226/3/Salakhutdinov_Ruslan_R_200910_PhD_thesis.pdf

    - A PhD thesis of Ruslan Salakhutdinov
    - In pt. 1 we describe Deep Belief Networks
    - In pt. 2 we describe Deep Boltzmann Machines

## 2010

Friston, K. J. (2010). The free-energy principle: a unified brain theory? Nat. Rev. Neurosci. Retrieved from https://www.semanticscholar.org/paper/The-free-energy-principle%3A-a-unified-brain-theory-Friston/1ed6a4a10589618d4f26350f1a296ee767ceff6b

Friston, K. J. (2010). Embodied Inference : or “ I think therefore I am , if I am what I think ”. Retrieved from https://www.semanticscholar.org/paper/Embodied-Inference-%3A-or-%E2%80%9C-I-think-therefore-I-am-%2C-Friston/c5b1e58ba8a5fde9607108e18c0b8ab2158cb1ba

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v9/glorot10a.html

    - Problem: standard GD from random initialization is doing poorly with deep neural networks
    - The logistic sigmoid activation is unsuited for DNNs, because it drives top hidden layer into saturation
    - Saturated units can slowly move out of saturation by themselves, explaining the plateaus seen when training NNs
    - We propose the hyperbolic tangent non-linearity that saturates less
    - For efficient training the singular values of the Jacobian associated with each layer should be close to 1
    - Based on this, we propose a new initialization scheme that brings substantially faster convergence

Kingma, D. P., & Cun, Y. (2010). Regularized estimation of image statistics by Score Matching. Advances in Neural Information Processing Systems, 23. Retrieved from https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html

    - We propose a version of the double-backpropagation algorithm for training high-dimensional density models
    - In addition, we introduce a regularization term for the Score Matching loss
    - Results are reported for image denoising and super-resolution

Martens, J. (2010). Deep learning via Hessian-free optimization. ICML'10: Proceedings of the 27th International Conference on International Conference on Machine Learning. Omnipress. Retrieved from https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf

Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. ICML'10: Proceedings of the 27th International Conference on International Conference on Machine Learning. Omnipress. Retrieved from https://www.cs.toronto.edu/%7Efritz/absps/reluICML.pdf

    - Noisy, rectified linear units approximate an infinite number of stochastic hidden units in RBM
    - These uints preserve information about relative intensities

Salakhutdinov, R., & Larochelle, H. (2010). Efficient Learning of Deep Boltzmann Machines. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v9/salakhutdinov10a/salakhutdinov10a.pdf

    - We present a new approximate inference algorithm for Deep Boltzmann Machines
    - It learns a separate "recognition" model to initialize latent variables in a single bottom-up pass
    - It performs well on visual recognition tasks: MNIST, OCR and NORB

Salakhutdinov, R. (2010). Learning Deep Boltzmann Machines using adaptive MCMC. ICML 2010 - Proceedings, 27th International Conference on Machine Learning, 943–950. Retrieved from https://www.cs.cmu.edu/~rsalakhu/papers/adapt.pdf

    - Deep Boltzmann Machine has an energy landscape with many local minima separated by high energy barriers
    - Gibbs sampler tends to get trapped in one local mode, which often results in unstable learning dynamics
    - We show a close connection between Fast PCD and adaptive MCMC
    - We propose a Coupled Adaptive Simulated Tempering algorithm to better explore a multimodal energy landscape
    - This improves learning of large-scale DBM’s

Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2010). Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. J. Mach. Learn. Res., 11, 3371–3408. Retrieved from https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

    - We propose to stack layers of denoising autoencoders which are trained locally
    - It bridges the performance gap between autoencoders and deep belief networks
    - Denoising autoencoders are able to learn Gabor-like edge detectors from natural image patches
    - Thus we establish the value of using a denoising criterion as a tractable unsupervised objective

## 2011

Carandini, M., & Heeger, D. J. (2011). Normalization as a canonical neural computation. Nat. Rev. Neurosci., 22108672. Retrieved from https://pubmed.ncbi.nlm.nih.gov/22108672

Friston, K. J. (2011). What Is Optimal about Motor Control? Neuron. Retrieved from https://www.semanticscholar.org/paper/What-Is-Optimal-about-Motor-Control-Friston/f3ba5b8e81eb4fe8fe5b381a65fd1003cd25d451

Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v15/glorot11a.html

    - This work extends the results of Nair and Hinton (2010) for the case of denoising autoencoders
    - Rectifying neurons yield equal or better performance than hyperbolic tangent networks
    - They create sparse representations with true zeros, which seem suitable for naturally sparse data
    - They can reach their best performance without requiring any unsupervised pre-training
    - They are also a better model of biological neurons

Huang, Y., & Rao, R. P. N. (2011). Predictive coding. Wiley Interdiscip. Rev. Cognit. Sci. Retrieved from https://www.semanticscholar.org/paper/Predictive-coding.-Huang-Rao/f1eb4088e2a1d433abb5627e18cd4d40886dc2fa

Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. Retrieved from https://proceedings.mlr.press/v15/larochelle11a.html

Ngiam, J., Chen, Z., Bhaskar, S., Koh, P., & Ng, A. (2011). Sparse Filtering. Advances in Neural Information Processing Systems, 24. Retrieved from https://papers.nips.cc/paper_files/paper/2011/hash/192fc044e74dffea144f9ac5dc9f3395-Abstract.html

    - We present Sparse filtering: an unsupervised feature learning method
    - It has only one hyperparemeter - the number of features to learn
    - Given a feature matrix, it involves first L2-normalizing it by rows, then by columns and finally summing up the absolute value of all entries (so, we optimize them for sparseness using the L1-penalty)
    - This method is faster than sparse coding, ICA, and sparse autoencoders
    - The sparse filtering objective can be viewed as a normalized version of the ICA objective

Oseledets, I. V. (2011). Tensor-Train Decomposition. SIAM J. Sci. Comput. Retrieved from https://epubs.siam.org/doi/abs/10.1137/090752286?journalCode=sjoce3

    - A new way to approximate tensor with a product of simpler tensors (eq. 1.2)

Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive Auto-Encoders: Explicit Invariance During Feature Extraction. International Conference on Machine Learning. Retrieved from https://icml.cc/2011/papers/455_icmlpaper.pdf

    - We add a well chosen penalty term to the classical reconstruction cost function in deterministic auto-encoders
    - This penalty is a Frobenius norm of the Jacobian matrix of the encoder activations
    - It achieves equal results or outperforms denoising auto-encoders
    - It can be seen as a link between deterministic and non-deterministic auto-encoders
    - It forms better representation that corresponds to lower-dimensional non-linear manifold

Saxe, A. M., Koh, P. W., Chen, Z., Bhand, M., & Ng, A. Y. (2011). On Random Weights and Unsupervised Feature Learning. Proceedings of the 28th International Conference on Machine Learning, 1089–1096. Retrieved from https://icml.cc/2011/papers/551_icmlpaper.pdf

    - Question: while it has been known that certain feature learning architectures can yield useful features for object recognition tasks even with untrained, random weights, why do random weights sometimes do so well?
    - We find that some CNNs can be frequency selective and translation invariant with random weights
    - We demonstrate the viability of using random weights to quickly evaluate candidate architectures
    - A large fraction of the SOTA methods performance can be attributed to the architecture alone

Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient langevin dynamics. ICML'11: Proceedings of the 28th International Conference on International Conference on Machine Learning. Omnipress. Retrieved from https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf

    - We propose to inject Gaussian noise with a specific variance into SGD optimization
    - This prevents collapse of SGD to MAP solution
    - Instead, it converges to samples from the true posterior as we polynomially anneal the stepsize
    - The resulting algorithm starts off being similar to stochastic optimization
    - Then it automatically transits to Langevin dynamics (which simulates samples from the posterior)
    - We propose a practical method to estimate when this transition happens
    - This can be seen as automatic "early stopping"
    - We test on a mixture of Gaussians, logistic regression and ICA with natural gradients

## 2012

Adams, R. A., Shipp, S., & Friston, K. J. (2012). Predictions not commands: active inference in the motor system. Brain Struct. Funct. Retrieved from https://www.semanticscholar.org/paper/Predictions-not-commands%3A-active-inference-in-the-Adams-Shipp/17b7e0b33847ca1c0546f714b75bdd768263b267

Alain, G., & Bengio, Y. (2012). What Regularized Auto-Encoders Learn from the Data Generating Distribution. arXiv, 1211.4246. Retrieved from https://arxiv.org/abs/1211.4246v5

    - The question: what does an auto-encoder learn about the data generating density?
    - Our main answer is that it estimates the score (first derivative of the log-density), and it also estimates the Hessian (second derivative of the log-density)
    - It locally characterizes the shape of the data generating density (local manifold structure of data)
    - It contradicts previous interpretations of reconstruction error as an energy function
    - Approximate Metropolis-Hastings MCMC can be setup to recover samples from the estimated distribution

Bastos, A., Usrey, W., Adams, R. A., Mangun, G., Fries, P., & Friston, K. J. (2012). Canonical Microcircuits for Predictive Coding. Neuron. Retrieved from https://www.semanticscholar.org/paper/Canonical-Microcircuits-for-Predictive-Coding-Bastos-Usrey/755bfd4f8060d5fcda64eaedb81a520ab7c8bdba

Bengio, Y., Courville, A., & Vincent, P. (2012). Representation Learning: A Review and New Perspectives. arXiv, 1206.5538. Retrieved from https://arxiv.org/abs/1206.5538v3

    - This paper is about representation learning
    - We discuss advances in probabilistic models, auto-encoders, manifold learning, and deep networks
    - We discuss the problem of the multimodality of the posterior P(h|x)
    - We cover many high-level generic priors that we believe could improve representation learning
    - The long-term objective is to discover learning algorithms that can disentangle underlying factors

Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. arXiv, 1206.5533. Retrieved from https://arxiv.org/abs/1206.5533v2

    - A practical guide for a common hyper-parameters for GD optimization
    - We describe methods to debug and visualize neural networks
    - We cover parallelism, sparse high-dimensional inputs, symbolic inputs and embeddings, multi-relational learning
    - We pose open questions on the difficulty of training deep architectures

Bengio, Y., Mesnil, G., Dauphin, Y., & Rifai, S. (2012). Better Mixing via Deep Representations. arXiv, 1207.4404. Retrieved from https://arxiv.org/abs/1207.4404v1

    - We test several hypotheses:
    - H1: a successfully trained DNN yields representation spaces in which Markov chains mix faster
    - H2: H1 is true because deeper representations can better disentangle the underlying factors of variation
    - H3: H2 is true because in disentangled representations samples fill the space uniformly
    - The experimental results were in agreement with these hypotheses
    - Empirically, at higher levels, good samples are obtained when interpolating between examples
    - Empirically, at higher levels, good samples are obtained when adding isotropic noise

Friston, K. J., Adams, R. A., Perrinet, L. U., & Breakspear, M. (2012). Perceptions as Hypotheses: Saccades as Experiments. Front. Psychol. Retrieved from https://www.semanticscholar.org/paper/Perceptions-as-Hypotheses%3A-Saccades-as-Experiments-Friston-Adams/04bc5da3f1e62a68f03692182ac4fededf9d1518

Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv, 1207.0580. Retrieved from https://arxiv.org/abs/1207.0580v1

    - Dropout method: randomly omitting half of the feature detectors (FD) on each training case
    - This prevents co-adaptations, when a FD is only helpful in the context of several other specific FDs
    - Instead, each neuron learns to detect a feature that is generally helpful
    - Gives big improvements on many benchmark tasks

Pascanu, R., Mikolov, T., & Bengio, Y. (2012). On the difficulty of training Recurrent Neural Networks. arXiv, 1211.5063. Retrieved from https://arxiv.org/abs/1211.5063v2

    - We attempt to improve the understanding of gradient vanishing and exploding in RNNs
    - We propose a gradient norm clipping strategy to deal with exploding gradients
    - We propose a soft constraint for the vanishing gradients problem

Rifai, S., Bengio, Y., Dauphin, Y., & Vincent, P. (2012). A Generative Process for Sampling Contractive Auto-Encoders. arXiv, 1206.6434. Retrieved from https://arxiv.org/abs/1206.6434v1

    - We propose a procedure for generating samples from contractive auto-encoders
    - It experimentally converges quickly and mix well between modes, compared to RBM and DBN
    - We propose to to train the second layer of contraction that pools lower-level features and learns to be invariant to the local directions of variation discovered in the first layer

Schoelkopf, B., Janzing, D., Peters, J., Sgouritsa, E., Zhang, K., & Mooij, J. (2012). On Causal and Anticausal Learning. arXiv, 1206.6471. Retrieved from https://arxiv.org/abs/1206.6471v1

    - To predict one variable from another, it helps to know the causal structure underlying the variables
    - We discuss causal (predict Effect from Cause) and anticausal (Cause from Effect) predictions
    - Hypothesis: under an independence assumption for causal mechanism and input, semi-supervised learning works better in anticausal or confounded problems
    - This can be useful for covariate shift, concept drift, transfer learning, semi-supervised learning

Swersky, K., Ranzato, M., Buchman, D., Marlin, B. M., & de Freitas, N. (2011). On Autoencoders and Score Matching for Energy Based Models. ResearchGate, 1201–1208. Retrieved from https://www.cs.toronto.edu/~ranzato/publications/Swersky_icml2011.pdf

    - We discuss continuous-data energy based models (EBMs)
    - Let the conditional distribution over the visible units is Gaussian
    - In this case, provide a link between EBM with score matching and a form of regularized autoencoder

## 2013

Ba, J., & Frey, B. (2013). Adaptive dropout for training deep neural networks. Advances in Neural Information Processing Systems, 26. Retrieved from https://papers.nips.cc/paper_files/paper/2013/hash/7b5b23f4aadf9513306bcd59afb6e4c9-Abstract.html

    - We propose standout method, when dropout is performed with a specific binary belief network
    - Both networks can be trained jointly

Baldi, P., & Sadowski, P. J. (2013). Understanding Dropout. Advances in Neural Information Processing Systems, 26. Retrieved from https://papers.nips.cc/paper_files/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html

    - We propose a general formalism for studying dropout
    - The averaging properties of dropout are characterized by three recursive equations
    - We also show how dropout performs SGD on a regularized error function

Bengio, Y. (2013). Deep Learning of Representations: Looking Forward. arXiv, 1305.0445. Retrieved from https://arxiv.org/abs/1305.0445v2

    - We present a few appealing directions of research towards deep learning challenges, including:
    - 1) Scaling computations
    - 2) Reducing the difficulties in optimizing parameters
    - 3) designing (or avoiding) expensive inference and sampling
    - 4) Helping to learn representations that better disentangle the unknown underlying factors of variation

Bengio, Y., Yao, L., Alain, G., & Vincent, P. (2013). Generalized Denoising Auto-Encoders as Generative Models. arXiv, 1305.6663. Retrieved from https://arxiv.org/abs/1305.6663v4

    - We propose a different probabilistic interpretation of Denoising Auto-Encoders
    - It is valid for any data type, any corruption process and any reconstruction loss
    - We can improve the sampling behavior by using the model itself to define the corruption process

Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv, 1308.3432. Retrieved from https://arxiv.org/abs/1308.3432v1

    - A question: can we "back-propagate" through stochastic neurons?
    - An existing approach is a special case of the REINFORCE algorithm
    - We propose an appoach to decompose neuron into a stochastic binary part and a smooth differentiable part
    - We also propose to inject additive or multiplicative noise in a differentiable computational graph
    - A fourth approach is a "straight-through estimator"
    - Experiments show that all the tested methods actually allow training to proceed
    - We describe several conclusions from the experiments

Bengio, Y., Thibodeau-Laufer, É., Alain, G., & Yosinski, J. (2013). Deep Generative Stochastic Networks Trainable by Backprop. arXiv, 1306.1091. Retrieved from https://arxiv.org/abs/1306.1091v5

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. Behav. Brain Sci. Retrieved from https://www.semanticscholar.org/paper/Whatever-next-Predictive-brains%2C-situated-agents%2C-Clark/69304ba2c5d2bff09f7059916ab0aa117bdbea41

Friston, K. J., & Friston, D. (2013). A Free Energy Formulation of Music Generation and Perception: Helmholtz Revisited. Retrieved from https://www.semanticscholar.org/paper/A-Free-Energy-Formulation-of-Music-Generation-and-Friston-Friston/b79995c7b34e639b273af4468702c53c9c3c9fc8

Friston, K. J., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. (2013). The anatomy of choice: active inference and agency. Front. Hum. Neurosci. Retrieved from https://www.semanticscholar.org/paper/The-anatomy-of-choice%3A-active-inference-and-agency-Friston-Schwartenbeck/3144baaeb4cab2bf48a8140de03080c03c4488b9

Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks. arXiv, 1312.6211. Retrieved from https://arxiv.org/abs/1312.6211v3

    - We found that using dropout is good at adapting to the new task, remembering the old task
    - As for comparing activation functions, different tasks and relationships between them result in very different rankings, this suggests that the choice of activation function should always be cross-validated

Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). Maxout Networks. arXiv, 1302.4389. Retrieved from https://arxiv.org/abs/1302.4389v4

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ...Bengio, Y. (2013). Challenges in Representation Learning: A report on three machine learning contests. arXiv, 1307.0414. Retrieved from https://arxiv.org/abs/1307.0414v1

    - We describe the ICML 2013 Workshop on Challenges in Representation Learning (datasets and results)
    - It was focused on the black box learning, the facial expression recognition and the multimodal learning
    - In this contest, unsupervised learning is important because of small amount of labeled data
    - The winner used sparse filtering for feature learning, RF for feature selection, SVM for classification
    - The second place rediscovered entropy regularization
    - Other competitors also obtained very competitive results with sparse filtering; this is interesting because sparse filtering has been perceived as an inexpensive and simple method that gives good but not optimal results
    - Sparse filtering features on the combination of the labeled and unlabeled data worked worse than learning the features on just the labeled data; this may be because the labeled data was drawn from the more difficult portion of the SVHN dataset

Gregor, K., Danihelka, I., Mnih, A., Blundell, C., & Wierstra, D. (2013). Deep AutoRegressive Networks. arXiv, 1310.8499. Retrieved from https://arxiv.org/abs/1310.8499v2

    - We propose Deep AutoRegressive Network (DARN)
    - This is a deep, generative autoencoder with stochastic layers equipped with autoregressive connections
    - This enable to sample from model quickly and exactly via ancestral sampling
    - We derive an efficient approximate parameter estimation method based on MDL and maximizing ELBO
    - We achieve SOTA generative performance on a number of datasets

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv, 1312.6114. Retrieved from https://arxiv.org/abs/1312.6114v11

    - We aim to learn a directed probabilistic model with continuous latent variables
    - A reparameterization of the ELBO yields an estimator that can be optimized using SGD
    - We fit an approximate inference model to the intractable posterior using this estimator

Friston, K. J. (2013). Life as we know it. J. R. Soc. Interface. Retrieved from https://www.semanticscholar.org/paper/Life-as-we-know-it-Friston/9d91dead9612e1c6a8dd9ae3793fe697565c744e

Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv, 1312.6120. Retrieved from https://arxiv.org/abs/1312.6120v3

    - Consider a DNN without activations
    - Despite the linearity, they have nonlinear GD dynamics, long plateaus followed by rapid transitions
    - Discussing some initial conditions on the weights that emerge during unsupervised pretraining
    - Propose a dynamical isometry (all singular values of the Jacobian concentrate near 1)
    - Propose orthonormal initialization
    - Faithful gradient propagation occurs in a special regime known as the edge of chaos

Sutskever, I., Martens, J., Dahl, G. E., & Hinton, G. E. (2013). On the importance of initialization and momentum in deep learning. International Conference on Machine Learning. Retrieved from https://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf

Uria, B., Murray, I., & Larochelle, H. (2013). A Deep and Tractable Density Estimator. arXiv, 1310.1757. Retrieved from https://arxiv.org/abs/1310.1757v2

Wager, S., Wang, S., & Liang, P. (2013). Dropout Training as Adaptive Regularization. arXiv, 1307.1493. Retrieved from https://arxiv.org/abs/1307.1493v2

Wang, S., & Manning, C. (2013). Fast dropout training. International Conference on Machine Learning. PMLR. Retrieved from https://proceedings.mlr.press/v28/wang13a.html

    - Problem: dropout makes training much slower
    - We consider an implied objective function of dropout and propose a method to speed up its optimization
    - We show how to do fast dropout training for classification, regression, and DNNs
    - Fast dropout often reaches the same validation set performance in a shorter time and in less iterations
    - Beyond dropout, our technique is extended for types of noise and small image transformations

## 2014

Bach, F. (2014). Breaking the Curse of Dimensionality with Convex Neural Networks. arXiv, 1412.8690. Retrieved from https://arxiv.org/abs/1412.8690v2

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv, 1409.0473. Retrieved from https://arxiv.org/abs/1409.0473v7

    - We conjecture that the use of a fixed-length vector between encoder and decoder is a bottleneck for NMT
    - We propose and attention mechanism: model automatically (soft-)searches for parts of a source sentence that are relevant to predicting a target word
    - Analysis reveals that the (soft-)alignments found by the model agree well with our intuition

Bengio, Y. (2014). How Auto-Encoders Could Provide Credit Assignment in Deep Networks via Target Propagation. arXiv, 1407.7906. Retrieved from https://arxiv.org/abs/1407.7906v3

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

Friston, K. J., Sengupta, B., & Auletta, G. (2014). Cognitive Dynamics: From Attractors to Active Inference. Proc. IEEE. Retrieved from https://www.semanticscholar.org/paper/Cognitive-Dynamics%3A-From-Attractors-to-Active-Friston-Sengupta/e3069e95026378d344e22766adac10490b053078

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ...Bengio, Y. (2014). Generative Adversarial Networks. arXiv, 1406.2661. Retrieved from https://arxiv.org/abs/1406.2661v1

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv, 1410.5401. Retrieved from https://arxiv.org/abs/1410.5401v2

Kingma, D. P., & Welling, M. (2014). Efficient Gradient-Based Inference through Transformations between Bayes Nets and Neural Nets. arXiv, 1402.0480. Retrieved from https://arxiv.org/abs/1402.0480v5

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv, 1412.6980. Retrieved from https://arxiv.org/abs/1412.6980v9

Koutník, J., Greff, K., Gomez, F., & Schmidhuber, J. (2014). A Clockwork RNN. arXiv, 1402.3511. Retrieved from https://arxiv.org/abs/1402.3511v1

Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2014). Random feedback weights support learning in deep neural networks. arXiv, 1411.0247. Retrieved from https://arxiv.org/abs/1411.0247v1

Maeda, S.-i. (2014). A Bayesian encourages dropout. arXiv, 1412.7003. Retrieved from https://arxiv.org/abs/1412.7003v3

    - We provide a Bayesian interpretation to dropout
    - The inference after dropout training can be considered as an approximate inference by Bayesian model averaging
    - This view enables us to optimize the dropout rate (Bayesian dropout)

Montúfar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the Number of Linear Regions of Deep Neural Networks. arXiv, 1402.1869. Retrieved from https://arxiv.org/abs/1402.1869v2

Neyshabur, B., Tomioka, R., & Srebro, N. (2014). In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning. arXiv, 1412.6614. Retrieved from https://arxiv.org/abs/1412.6614v4

Ozair, S., & Bengio, Y. (2014). Deep Directed Generative Autoencoders. arXiv, 1410.0630. Retrieved from https://arxiv.org/abs/1410.0630v1

    - We consider discrete data
    - The objective is to learn an encoder f that maps X to f(X) that has a much simpler distribution
    - Generating samples from the model is straightforward using ancestral sampling
    - We can pre-train and stack these encoders, gradually transforming the data distribution

Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. arXiv, 1401.4082. Retrieved from https://arxiv.org/abs/1401.4082v3

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). FitNets: Hints for Thin Deep Nets. arXiv, 1412.6550. Retrieved from https://arxiv.org/abs/1412.6550v4

Salimans, T., Kingma, D. P., & Welling, M. (2014). Markov Chain Monte Carlo and Variational Inference: Bridging the Gap. arXiv, 1410.6460. Retrieved from https://arxiv.org/abs/1410.6460v4

Schmidhuber, J. (2014). Deep Learning in Neural Networks: An Overview. arXiv, 1404.7828. Retrieved from https://arxiv.org/abs/1404.7828v4

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(56), 1929–1958. Retrieved from https://jmlr.org/papers/v15/srivastava14a.html

    - We show that recently proposed dropout improves the performance of NNss on supervised learning, tasks in vision, speech recognition, document classification and computational biology

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? arXiv, 1411.1792. Retrieved from https://arxiv.org/abs/1411.1792v1

    - An well-known fact is that the first-layer features of CNN appear to learn universally-applicable features
    - We define a way to quantify the degree to which a particular layer is general or specific
    - We then train pairs of CNNs on ImageNet and characterize the layer-by-layer transition from general to specific
    - We show two issues that cause performance degradation when using transferred features without fine-tuning:
    - 1) the specificity of the features themselves
    - 2) optimization difficulties due to splitting the base network between co-adapted layers
    - Initializing a network with transferred features from almost any number of layers boosts fine-tuning perfomance
    - This effect persists even after extensive fine-tuning
    - Transferring features even from distant tasks can be better than using random features

Zhang, S., Choromanska, A., & LeCun, Y. (2014). Deep learning with Elastic Averaging SGD. arXiv, 1412.6651. Retrieved from https://arxiv.org/abs/1412.6651v8

Zhao, P., & Zhang, T. (2014). Accelerating Minibatch Stochastic Gradient Descent using Stratified Sampling. arXiv, 1405.3080. Retrieved from https://arxiv.org/abs/1405.3080v1

## 2015

Baldassi, C., Ingrosso, A., Lucibello, C., Saglietti, L., & Zecchina, R. (2015). Subdominant Dense Clusters Allow for Simple Learning and High Computational Performance in Neural Networks with Discrete Synapses. arXiv, 1509.05753. Retrieved from https://arxiv.org/abs/1509.05753v1

Barrett, L. F., & Simmons, K. (2015). Interoceptive predictions in the brain. Nat. Rev. Neurosci. Retrieved from https://www.semanticscholar.org/paper/Interoceptive-predictions-in-the-brain-Barrett-Simmons/7a0b470ccc96aa1c8a95f0723c49e70cb18507b2

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A., & Bengio, Y. (2015). A Recurrent Latent Variable Model for Sequential Data. arXiv, 1506.02216. Retrieved from https://arxiv.org/abs/1506.02216v6

Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2015). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). arXiv, 1511.07289. Retrieved from https://arxiv.org/abs/1511.07289v5

Courbariaux, M., Bengio, Y., & David, J.-P. (2015). BinaryConnect: Training Deep Neural Networks with binary weights during propagations. arXiv, 1511.00363. Retrieved from https://arxiv.org/abs/1511.00363v3

Friston, K. J., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T. H. B., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive neuroscience. Retrieved from https://www.semanticscholar.org/paper/Active-inference-and-epistemic-value-Friston-Rigoli/57620e357ee348cd5ffa8eafa480e002a2aba06a

Friston, K. J., Levin, M., Sengupta, B., & Pezzulo, G. (2015). Knowing one's place: a free-energy approach to pattern regulation. J. R. Soc. Interface. Retrieved from https://www.semanticscholar.org/paper/Knowing-one's-place%3A-a-free-energy-approach-to-Friston-Levin/2c13294ccc0045d24fe0ae01a5ff6dd21d0566d1

Ge, R., Huang, F., Jin, C., & Yuan, Y. (2015). Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition. arXiv, 1503.02101. Retrieved from https://arxiv.org/abs/1503.02101v1

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv, 1512.03385. Retrieved from https://arxiv.org/abs/1512.03385v1

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv, 1502.01852. Retrieved from https://arxiv.org/abs/1502.01852v1

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv, 1502.03167. Retrieved from https://arxiv.org/abs/1502.03167v3

Joulin, A., & Mikolov, T. (2015). Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets. arXiv, 1503.01007. Retrieved from https://arxiv.org/abs/1503.01007v4

Kaiser, Ł., & Sutskever, I. (2015). Neural GPUs Learn Algorithms. arXiv, 1511.08228. Retrieved from https://arxiv.org/abs/1511.08228v3

Mishkin, D., & Matas, J. (2015). All you need is a good init. arXiv, 1511.06422. Retrieved from https://arxiv.org/abs/1511.06422v7

    - Layer-sequential unit-variance (LSUV) initialization:
    - 1) use orthonormal matrices
    - 2) normalize the variance of the output of each layer to be equal to one

Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. (2015). Tensorizing Neural Networks. arXiv, 1509.06569. Retrieved from https://arxiv.org/abs/1509.06569v2

    - Authors convert FCN weight matrices to Tensor Train format (TT-layer, TensorNet)
    - Number of parameters is reduced by a huge factor (up to 200000 times for VGG dense layers)
    - The expressive power is preserved

Peters, J., Bühlmann, P., & Meinshausen, N. (2015). Causal inference using invariant prediction: identification and confidence intervals. arXiv, 1501.01332. Retrieved from https://arxiv.org/abs/1501.01332v3

Pezzulo, G., Rigoli, F., & Friston, K. J. (2015). Active Inference, homeostatic regulation and adaptive behavioural control. Prog. Neurobiol. Retrieved from https://www.semanticscholar.org/paper/Active-Inference%2C-homeostatic-regulation-and-Pezzulo-Rigoli/c469f8a7f02015bc49e93df26c396228267a7e7b

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. arXiv, 1507.06228. Retrieved from https://arxiv.org/abs/1507.06228v2

    - We propose Highway networks inspired by LSTM
    - They use adaptive gating units to regulate the information flow, we call such paths "information highways"
    - Even with hundreds of layers, highway networks can be trained directly through SGD

Wiatowski, T., & Bölcskei, H. (2015). A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction. arXiv, 1512.06293. Retrieved from https://arxiv.org/abs/1512.06293v3

Wu, H., & Gu, X. (2015). Towards Dropout Training for Convolutional Neural Networks. arXiv, 1512.00242. Retrieved from https://arxiv.org/abs/1512.00242v1

    - For CNN, dropout is known to work well in fully-connected layers
    - However, its effect in convolutional and pooling layers is still not clear
    - We draw a connection between dropout before max-pooling and sampling from multinomial distribution
    - We propose probabilistic weighted pooling and achieve competitive results
    - We found that the effect of convolutional dropout is not trivial

## 2016

Abdi, M., & Nahavandi, S. (2016). Multi-Residual Networks: Improving the Speed and Accuracy of Residual Networks. arXiv, 1609.05672. Retrieved from https://arxiv.org/abs/1609.05672v4

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv, 1607.06450. Retrieved from https://arxiv.org/abs/1607.06450v1

    - Problem: batch normalization is dependent on batch size and is not trivial to apply to RNNs
    - We propose a Layer Normalization that works independently for every training case
    - It performs the same computation at training and test times, and is also straightforward to apply to RNNs
    - Like in BN, we also use scale and shift, after the normalization but before the non-linearity
    - Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks

Baldassi, C., Gerace, F., Lucibello, C., Saglietti, L., & Zecchina, R. (2016). Learning may need only a few bits of synaptic precision. arXiv, 1602.04129. Retrieved from https://arxiv.org/abs/1602.04129v2

Bottou, L., Curtis, F. E., & Nocedal, J. (2016). Optimization Methods for Large-Scale Machine Learning. arXiv, 1606.04838. Retrieved from https://arxiv.org/abs/1606.04838v3

Bruineberg, J., Kiverstein, J., & Rietveld, E. (2016). The anticipating brain is not a scientist: the free-energy principle from an ecological-enactive perspective. Synthese. Retrieved from https://www.semanticscholar.org/paper/The-anticipating-brain-is-not-a-scientist%3A-the-from-Bruineberg-Kiverstein/d923112dbf3c46d792b9d1172dd8fa69a68e3386

Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., ...Zecchina, R. (2016). Entropy-SGD: Biasing Gradient Descent Into Wide Valleys. arXiv, 1611.01838. Retrieved from https://arxiv.org/abs/1611.01838v5

    - Well-generalizable solutions lie in large flat regions with almost-zero eigenvalues in the Hessian
    - We propose Entropy-SGD optimizer that favors such solutions
    - Our algorithm resembles two nested loops of SGD
    - We use Langevin dynamics in the inner loop to compute the gradient of the local entropy
    - The new objective has a smoother energy landscape
    - Entropy-SGD obtains is comparable to competitive baselines and  gets a 2x speed-up over SGD

Cichocki, A., Lee, N., Oseledets, I. V., Phan, A.-H., Zhao, Q., & Mandic, D. (2016). Low-Rank Tensor Networks for Dimensionality Reduction and Large-Scale Optimization Problems: Perspectives and Challenges PART 1. arXiv, 1609.00893. Retrieved from https://arxiv.org/abs/1609.00893v3

    - A book (pt. 1) about Tucker and Tensor Train (TT) decompositions and their extensions or generalizations
    - This can be used to convert intractable huge-scale optimization problems into a set of smaller problems
    - Chapter 1: Introduction and Motivation
    - Chapter 2: Tensor Operations and Tensor Network Diagrams
    - Chapter 3: Constrained Tensor Decompositions: From Two-way to Multiway Component Analysis
    - Chapter 4: Tensor Train Decompositions: Graphical Interpretations and Algorithms
    - Chapter 5: Discussion and Conclusions

Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv, 1602.02830. Retrieved from https://arxiv.org/abs/1602.02830v3

Duvenaud, D., Maclaurin, D., & Adams, R. (2016). Early Stopping as Nonparametric Variational Inference. Artificial Intelligence and Statistics. PMLR. Retrieved from https://proceedings.mlr.press/v51/duvenaud16.html

    - We propose a Bayesian interpretation of SGD
    - Unconverged SGD yields a sequence of distributions which are variational approximations to the true posterior
    - This allows us to estimate a lower bound on the marginal likelihood of any model, even very large
    - This can be used for hyperparameter selection and early stopping without a validation set
    - The results are promising, but further refinements are likely to be necessary

Freeman, C. D., & Bruna, J. (2016). Topology and Geometry of Half-Rectified Network Optimization. arXiv, 1611.01540. Retrieved from https://arxiv.org/abs/1611.01540v4

    - We prove that single layer ReLU networks are asymptotically connected
    - We show that level sets remain connected throughout all the learning phase, suggesting a near convex behavior, but they become exponentially more curvy as the energy level decays

Friston, K. J., FitzGerald, T., Rigoli, F., Schwartenbeck, P., O'Doherty, J., & Pezzulo, G. (2016). Active inference and learning. Neurosci. Biobehav. Rev. Retrieved from https://www.semanticscholar.org/paper/Active-inference-and-learning-Friston-FitzGerald/3b3903f7914483e21576f9d098e611deef95ec45

Friston, K. J. (2016). I am therefore I think. Retrieved from https://www.semanticscholar.org/paper/I-am-therefore-I-think-Friston/2d450521f168fccbc3cd13112ca07159c7f1bd50

Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. (2016). Deep Networks with Stochastic Depth. arXiv, 1603.09382. Retrieved from https://arxiv.org/abs/1603.09382v3

Hohwy, J. (2016). The self-evidencing brain. Noûs. Retrieved from https://www.semanticscholar.org/paper/The-self-evidencing-brain-Hohwy/01aa6ef498431fb8c6d45e9375bb39a7a923b9bb

Im, D. J., Tao, M., & Branson, K. (2016). An empirical analysis of the optimization of deep network loss surfaces. arXiv, 1612.04010. Retrieved from https://arxiv.org/abs/1612.04010v4

    - We visualize the loss function by projecting them down to low-dimensional spaces
    - We show that optimization algorithms encounter and choose different descent directions at many saddle points
    - We hypothesize that each optimization algorithm makes characteristic choices at these saddle points

Jang, E., Gu, S., & Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax. arXiv, 1611.01144. Retrieved from https://arxiv.org/abs/1611.01144v5

Kaiser, Ł., & Bengio, S. (2016). Can Active Memory Replace Attention? arXiv, 1610.08613. Retrieved from https://arxiv.org/abs/1610.08613v2

    - Problem: by using softmax, attention focuses only on a single element of the memory

Kawaguchi, K. (2016). Deep Learning without Poor Local Minima. arXiv, 1605.07110. Retrieved from https://arxiv.org/abs/1605.07110v3

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. arXiv, 1609.04836. Retrieved from https://arxiv.org/abs/1609.04836v2

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ...Hadsell, R. (2016). Overcoming catastrophic forgetting in neural networks. arXiv, 1612.00796. Retrieved from https://arxiv.org/abs/1612.00796v2

Liao, Q., & Poggio, T. (2016). Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex. arXiv, 1604.03640. Retrieved from https://arxiv.org/abs/1604.03640v2

Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. Nat. Commun., 7(13276), 1–10. Retrieved from https://www.nature.com/articles/ncomms13276

Novikov, A., Trofimov, M., & Oseledets, I. (2016). Exponential Machines. arXiv, 1605.03795. Retrieved from https://arxiv.org/abs/1605.03795v3

    - Exponential Machines (ExM), a predictor that models all interactions of every order
    - The Tensor Train format regularizes an exponentially large tensor of parameters
    - SOTA performance on synthetic data with high-order interactions

Parikh, A. P., Täckström, O., Das, D., & Uszkoreit, J. (2016). A Decomposable Attention Model for Natural Language Inference. arXiv, 1606.01933. Retrieved from https://arxiv.org/abs/1606.01933v2

Rolfe, J. T. (2016). Discrete Variational Autoencoders. arXiv, 1609.02200. Retrieved from https://arxiv.org/abs/1609.02200v2

Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. arXiv, 1602.07868. Retrieved from https://arxiv.org/abs/1602.07868v3

    - a weight that decouples the length of vectors from their direction
    - we improve the conditioning of the optimization problem and we speed up convergence of SGD
    - is inspired by batch normalization but is more widely applicable
    - useful in supervised image recognition, generative modelling, and deep RL

Scellier, B., & Bengio, Y. (2016). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. arXiv, 1602.05179. Retrieved from https://arxiv.org/abs/1602.05179v5

Stoudenmire, E. M., & Schwab, D. J. (2016). Supervised Learning with Quantum-Inspired Tensor Networks. arXiv, 1605.05775. Retrieved from https://arxiv.org/abs/1605.05775v2

    - Algorithms for optimizin Tensor networks can be adapted to supervised learning tasks
    - One proposed solution is active memory: it allows the model to access and change all its memory
    - We present an extension of the Neural GPU model that yields good results for NMT
    - We clarify why the previous active memory model did not succeed on machine translation

Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). Instance Normalization: The Missing Ingredient for Fast Stylization. arXiv, 1607.08022. Retrieved from https://arxiv.org/abs/1607.08022v3

Xie, B., Liang, Y., & Song, L. (2016). Diverse Neural Network Learns True Target Functions. arXiv, 1611.03131. Retrieved from https://arxiv.org/abs/1611.03131v3

    - Do stationary points of the loss function learn the true target function (that is, generalize)?
    - We analyze one-hidden-layer ReLU neural networks
    - We show that neural networks with diverse units have no spurious local minima: the more diverse the unit weights, the more likely stationary points will result in small training loss and generalization error
    - We suggest a novel regularization function to promote unit diversity for potentially better generalization

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. arXiv, 1611.03530. Retrieved from https://arxiv.org/abs/1611.03530v2

    - We show that SOTA CNNs trained with SGD easily fit a random labeling and/or images of random noise
    - For random labeling, training time increases only by a small constant factor
    - This is unaffected by explicit regularization
    - Regularization such as weight decay, dropout, and data augmentation, do not adequately explain the generalization error of neural networks, it plays a rather different role in deep learning
    - We prove that simple two-layer ReLU of linear size can already represent any labeling of the training data
    - Appealing to linear models, we analyze how SGD acts as an implicit regularizer: for linear models, SGD always converges to a solution with small norm, hence, the algorithm itself is implicitly regularizing the solution
    - We suggest that more investigation is needed

## 2017

Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., ...Lacoste-Julien, S. (2017). A Closer Look at Memorization in Deep Networks. arXiv, 1706.05394. Retrieved from https://arxiv.org/abs/1706.05394v2

Balan, R., Singh, M., & Zou, D. (2017). Lipschitz Properties for Deep Convolutional Networks. arXiv, 1701.05217. Retrieved from https://arxiv.org/abs/1701.05217v1

Bartlett, P., Foster, D. J., & Telgarsky, M. (2017). Spectrally-normalized margin bounds for neural networks. arXiv, 1706.08498. Retrieved from https://arxiv.org/abs/1706.08498v2

Bengio, Y. (2017). The Consciousness Prior. arXiv, 1709.08568. Retrieved from https://arxiv.org/abs/1709.08568v2

Bojarski, M., Yeres, P., Choromanska, A., Choromanski, K., Firner, B., Jackel, L., & Muller, U. (2017). Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car. arXiv, 1704.07911. Retrieved from https://arxiv.org/abs/1704.07911v1

Brutzkus, A., Globerson, A., Malach, E., & Shalev-Shwartz, S. (2017). SGD Learns Over-parameterized Networks that Provably Generalize on Linearly Separable Data. arXiv, 1710.10174. Retrieved from https://arxiv.org/abs/1710.10174v1

Chang, B., Meng, L., Haber, E., Ruthotto, L., Begert, D., & Holtham, E. (2017). Reversible Architectures for Arbitrarily Deep Residual Neural Networks. arXiv, 1709.03698. Retrieved from https://arxiv.org/abs/1709.03698v2

Cichocki, A., Phan, A.-H., Zhao, Q., Lee, N., Oseledets, I. V., Sugiyama, M., & Mandic, D. (2017). Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations. Part 2 Applications and Future Perspectives. arXiv, 1708.09165. Retrieved from https://arxiv.org/abs/1708.09165v1

    -  A book (pt. 2) about tensor network models for super-compressed representation of data/parameters
    -  Emphasis is on the tensor train (TT) and Hierarchical Tucker (HT) decompositions
    -  Applied areas: regression and classification, eigenvalue decomposition, Riemannian optimization, DNNs
    -  Part 1 and Part 2 of this work can be used either as stand-alone separate texts

Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. arXiv, 1703.04933. Retrieved from https://arxiv.org/abs/1703.04933v2

Du, S. S., Lee, J. D., Tian, Y., Poczos, B., & Singh, A. (2017). Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima. arXiv, 1712.00779. Retrieved from https://arxiv.org/abs/1712.00779v2

Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. arXiv, 1707.04585. Retrieved from https://arxiv.org/abs/1707.04585v1

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., ...He, K. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv, 1706.02677. Retrieved from https://arxiv.org/abs/1706.02677v2

Guerguiev, J., Lillicrap, T. P., & Richards, B. A. (2017). Towards deep learning with segregated dendrites. eLife. Retrieved from https://elifesciences.org/articles/22901

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. arXiv, 1706.04599. Retrieved from https://arxiv.org/abs/1706.04599v2

Hoffer, E., Hubara, I., & Soudry, D. (2017). Train longer, generalize better: closing the generalization gap in large batch training of neural networks. arXiv, 1705.08741. Retrieved from https://arxiv.org/abs/1705.08741v2

Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). Snapshot Ensembles: Train 1, get M for free. arXiv, 1704.00109. Retrieved from https://arxiv.org/abs/1704.00109v1

Huang, F., Ash, J., Langford, J., & Schapire, R. (2017). Learning Deep ResNet Blocks Sequentially using Boosting Theory. arXiv, 1706.04964. Retrieved from https://arxiv.org/abs/1706.04964v4

Ioffe, S. (2017). Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. arXiv, 1702.03275. Retrieved from https://arxiv.org/abs/1702.03275v2

Jastrzębski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2017). Three Factors Influencing Minima in SGD. arXiv, 1711.04623. Retrieved from https://arxiv.org/abs/1711.04623v3

Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., & Uszkoreit, J. (2017). One Model To Learn Them All. arXiv, 1706.05137. Retrieved from https://arxiv.org/abs/1706.05137v1

Kaiser, Ł., Nachum, O., Roy, A., & Bengio, S. (2017). Learning to Remember Rare Events. arXiv, 1703.03129. Retrieved from https://arxiv.org/abs/1703.03129v1

Khrulkov, V., & Oseledets, I. (2017). Art of singular vectors and universal adversarial perturbations. arXiv, 1709.03582. Retrieved from https://arxiv.org/abs/1709.03582v2

    -  A new algorithm for constructing adversarial perturbations
    -  Computing (p, q)-singular vectors of the Jacobian matrices of hidden layers of a network

Khrulkov, V., Novikov, A., & Oseledets, I. (2017). Expressive power of recurrent neural networks. arXiv, 1711.00811. Retrieved from https://arxiv.org/abs/1711.00811v2

    - As known, deep Hierarchical Tucker CNNs have exponentially higher expressive power than shallow networks
    - We prove the same for RNNs with Tensor Train (TT) decomposition
    - We compare expressive powers of the HT- and TT-Networks
    - We implement the recurrent TT-Networks and provide numerical evidence of their expressivity

Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. arXiv, 1706.02515. Retrieved from https://arxiv.org/abs/1706.02515v5

Laurent, T., & von Brecht, J. (2017). Deep linear neural networks with arbitrary loss: All local minima are global. arXiv, 1712.01473. Retrieved from https://arxiv.org/abs/1712.01473v2

Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., & Sohl-Dickstein, J. (2017). Deep Neural Networks as Gaussian Processes. arXiv, 1711.00165. Retrieved from https://arxiv.org/abs/1711.00165v3

Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2017). Visualizing the Loss Landscape of Neural Nets. arXiv, 1712.09913. Retrieved from https://arxiv.org/abs/1712.09913v3

    - Simple visualization strategies fail to accurately capture the local geometry
    - We present a visualization method based on "filter normalization"
    - When networks become deep, loss surface turns from convex to chaotic, but skip connections prevent this
    - We measure non-convexity by calculating eigenvalues of the Hessian around local minima
    - We show that SGD optimization trajectories lie in an extremely low dimensional space
    - This can be explained by the presence of large, nearly convex regions in the loss landscape

Liu, T., Lugosi, G., Neu, G., & Tao, D. (2017). Algorithmic stability and hypothesis complexity. arXiv, 1702.08712. Retrieved from https://arxiv.org/abs/1702.08712v2

Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv, 1711.05101. Retrieved from https://arxiv.org/abs/1711.05101v3

Lu, Y., Zhong, A., Li, Q., & Dong, B. (2017). Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations. arXiv, 1710.10121. Retrieved from https://arxiv.org/abs/1710.10121v3

Ma, S., Bassily, R., & Belkin, M. (2017). The Power of Interpolation: Understanding the Effectiveness of SGD in Modern Over-parametrized Learning. arXiv, 1712.06559. Retrieved from https://arxiv.org/abs/1712.06559v3

Mahsereci, M., Balles, L., Lassner, C., & Hennig, P. (2017). Early Stopping without a Validation Set. arXiv, 1703.09580. Retrieved from https://arxiv.org/abs/1703.09580v3

    - We propose a cheap and scalable early stopping criterion based on local statistics of the gradients
    - It does not require a validation set, thus enabling the optimizer to use all available training data
    - We test on linear and MLP models

Molchanov, D., Ashukha, A., & Vetrov, D. (2017). Variational Dropout Sparsifies Deep Neural Networks. arXiv, 1701.05369. Retrieved from https://arxiv.org/abs/1701.05369v3

Nguyen, Q., & Hein, M. (2017). The loss surface of deep and wide neural networks. arXiv, 1704.08045. Retrieved from https://arxiv.org/abs/1704.08045v2

Oord, A. v. d., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. arXiv, 1711.00937. Retrieved from https://arxiv.org/abs/1711.00937v2

Peng, K.-C., Wu, Z., & Ernst, J. (2017). Zero-Shot Deep Domain Adaptation. arXiv, 1707.01922. Retrieved from https://arxiv.org/abs/1707.01922v5

Pennington, J., Schoenholz, S. S., & Ganguli, S. (2017). Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. arXiv, 1711.04735. Retrieved from https://arxiv.org/abs/1711.04735v1

    - Authors compute analytically the entire singular value distribution of a DNN’s input-output Jacobian
    - ReLU networks are incapable of dynamical isometry (see https://arxiv.org/abs/1312.6120)
    - Sigmoidal networks with orthogonal weight initialization can achieve isometry and outperform ReLU nets
    - DNNs achieving dynamical isometry learn orders of magnitude faster than networks that do not

Ramstead, M., Badcock, P. B., & Friston, K. J. (2017). Answering Schrödinger's question: A free-energy formulation. Phys. Life Rev. Retrieved from https://www.semanticscholar.org/paper/Answering-Schr%C3%B6dinger's-question%3A-A-free-energy-Ramstead-Badcock/cbf4040cb14a019ff3556fad5c455e99737f169f

Reddi, S. J., Zaheer, M., Sra, S., Poczos, B., Bach, F., Salakhutdinov, R., & Smola, A. J. (2017). A Generic Approach for Escaping Saddle points. arXiv, 1709.01434. Retrieved from https://arxiv.org/abs/1709.01434v1

Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., & Bottou, L. (2017). Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. arXiv, 1706.04454. Retrieved from https://arxiv.org/abs/1706.04454v3

Safran, I., & Shamir, O. (2017). Spurious Local Minima are Common in Two-Layer ReLU Neural Networks. arXiv, 1712.08968. Retrieved from https://arxiv.org/abs/1712.08968v3

Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual Learning with Deep Generative Replay. arXiv, 1705.08690. Retrieved from https://arxiv.org/abs/1705.08690v3

Smith, L. N., & Topin, N. (2017). Exploring loss function topology with cyclical learning rates. arXiv, 1702.04283. Retrieved from https://arxiv.org/abs/1702.04283v1

Smith, L. N., & Topin, N. (2017). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv, 1708.07120. Retrieved from https://arxiv.org/abs/1708.07120v3

Smith, S. L., & Le, Q. V. (2017). A Bayesian Perspective on Generalization and Stochastic Gradient Descent. arXiv, 1710.06451. Retrieved from https://arxiv.org/abs/1710.06451v3

Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2017). Don't Decay the Learning Rate, Increase the Batch Size. arXiv, 1711.00489. Retrieved from https://arxiv.org/abs/1711.00489v2

Soudry, D., & Hoffer, E. (2017). Exponentially vanishing sub-optimal local minima in multilayer neural networks. arXiv, 1702.05777. Retrieved from https://arxiv.org/abs/1702.05777v5

Taki, M. (2017). Deep Residual Networks and Weight Initialization. arXiv, 1709.02956. Retrieved from https://arxiv.org/abs/1709.02956v1

    - ResNets are relatively insensitive to choice of initial weights
    - how batch normalization improves backpropagation in ResNet
    - we propose new weight initialization distribution to prevent exploding gradients

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ...Polosukhin, I. (2017). Attention Is All You Need. arXiv, 1706.03762. Retrieved from https://arxiv.org/abs/1706.03762v7

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. arXiv, 1710.10903. Retrieved from https://arxiv.org/abs/1710.10903v3

Wiatowski, T., Grohs, P., & Bölcskei, H. (2017). Energy Propagation in Deep Convolutional Neural Networks. arXiv, 1704.03636. Retrieved from https://arxiv.org/abs/1704.03636v3

Wu, L., Zhu, Z., & E, W. (2017). Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes. arXiv, 1706.10239. Retrieved from https://arxiv.org/abs/1706.10239v2

Xie, D., Xiong, J., & Pu, S. (2017). All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation. arXiv, 1703.01827. Retrieved from https://arxiv.org/abs/1703.01827v3

    - Problem: how to train deep nets without any shortcuts/identity mappings?
    - Solution: regularizer which utilizes orthonormality and a backward error modulation mechanism.

You, Y., Zhang, Z., Hsieh, C.-J., Demmel, J., & Keutzer, K. (2017). ImageNet Training in Minutes. arXiv, 1709.05011. Retrieved from https://arxiv.org/abs/1709.05011v10

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. arXiv, 1703.06114. Retrieved from https://arxiv.org/abs/1703.06114v3

## 2018

Agarap, A. F. (2018). Deep Learning using Rectified Linear Units (ReLU). arXiv, 1803.08375. Retrieved from https://arxiv.org/abs/1803.08375v2

Allen-Zhu, Z., Li, Y., & Song, Z. (2018). A Convergence Theory for Deep Learning via Over-Parameterization. arXiv, 1811.03962. Retrieved from https://arxiv.org/abs/1811.03962v5

    - Study the theory of multi-layer networks
    - Proof that SGD can find global minima on the training objective of over-parameterized DNNs
    - Key insight is that in a neighborhood of the random initialization, the opt. landscape is almost convex
    - This implies an equivalence between over-parameterized finite width NNs and neural tangent kernel
    - Our theory at least applies to FCN, CNN and ResNet

Athiwaratkun, B., Finzi, M., Izmailov, P., & Wilson, A. G. (2018). There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average. arXiv, 1806.05594. Retrieved from https://arxiv.org/abs/1806.05594v3

Balestriero, R., & Baraniuk, R. (2018). Mad Max: Affine Spline Insights into Deep Learning. arXiv, 1805.06576. Retrieved from https://arxiv.org/abs/1805.06576v5

    - A large class of DNs can be written as a composition of maxaffine spline operators (MASOs)
    - This links DNs to the theory of vector quantization (VQ) and K-means clustering
    - Propose a simple penalty term to loss function to significantly improve performance

Bartlett, P. L., Helmbold, D. P., & Long, P. M. (2018). Gradient descent with identity initialization efficiently learns positive definite linear transformations by deep residual networks. arXiv, 1802.06093. Retrieved from https://arxiv.org/abs/1802.06093v4

Belkin, M., Ma, S., & Mandal, S. (2018). To understand deep learning we need to understand kernel learning. arXiv, 1802.01396. Retrieved from https://arxiv.org/abs/1802.01396v3

Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2018). Invertible Residual Networks. arXiv, 1811.00995. Retrieved from https://arxiv.org/abs/1811.00995v3

Bengio, Y., Lodi, A., & Prouvost, A. (2018). Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. arXiv, 1811.06128. Retrieved from https://arxiv.org/abs/1811.06128v2

Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence. arXiv, 1801.10112. Retrieved from https://arxiv.org/abs/1801.10112v3

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. arXiv, 1806.07366. Retrieved from https://arxiv.org/abs/1806.07366v5

    - Continuous-depth residual networks and continuous-time latent variable models, continuous normalizing flows
    - The derivative of the hidden state is parameterized
    - The output of the network is computed using a blackbox differential equation solver
    - Have constant memory cost
    - Adapt evaluation strategy to each input
    - Can explicitly trade numerical precision for speed

Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., Willems, L., Saharia, C., Nguyen, T. H., & Bengio, Y. (2018). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. arXiv, 1810.08272. Retrieved from https://arxiv.org/abs/1810.08272v4

Cooper, Y. (2018). The loss landscape of overparameterized neural networks. arXiv, 1804.10200. Retrieved from https://arxiv.org/abs/1804.10200v1

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2018). Universal Transformers. arXiv, 1807.03819. Retrieved from https://arxiv.org/abs/1807.03819v3

Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially No Barriers in Neural Network Energy Landscape. arXiv, 1803.00885. Retrieved from https://arxiv.org/abs/1803.00885v5

Du, S. S., Hu, W., & Lee, J. D. (2018). Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced. arXiv, 1806.00900. Retrieved from https://arxiv.org/abs/1806.00900v2

Fort, S., & Scherlis, A. (2018). The Goldilocks zone: Towards better understanding of neural network loss landscapes. arXiv, 1807.02581. Retrieved from https://arxiv.org/abs/1807.02581v2

Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. arXiv, 1803.03635. Retrieved from https://arxiv.org/abs/1803.03635v5

Friston, K. (2018). Am I Self-Conscious? (Or Does Self-Organization Entail Self-Consciousness?). Front. Psychol., 9, 348034. Retrieved from https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00579/full

Galloway, A., Tanay, T., & Taylor, G. W. (2018). Adversarial Training Versus Weight Decay. arXiv, 1804.03308. Retrieved from https://arxiv.org/abs/1804.03308v3

Golmant, N., Vemuri, N., Yao, Z., Feinberg, V., Gholami, A., Rothauge, K., ...Gonzalez, J. (2018). On the Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent. arXiv, 1811.12941. Retrieved from https://arxiv.org/abs/1811.12941v1

Gotmare, A., Keskar, N. S., Xiong, C., & Socher, R. (2018). A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation. arXiv, 1810.13243. Retrieved from https://arxiv.org/abs/1810.13243v1

Gu, J., Hassan, H., Devlin, J., & Li, V. O. K. (2018). Universal Neural Machine Translation for Extremely Low Resource Languages. arXiv, 1802.05368. Retrieved from https://arxiv.org/abs/1802.05368v2

Hahn, S., & Choi, H. (2018). Understanding Dropout as an Optimization Trick. arXiv, 1806.09783. Retrieved from https://arxiv.org/abs/1806.09783v3

Hanin, B., & Rolnick, D. (2018). How to Start Training: The Effect of Initialization and Architecture. arXiv, 1803.01719. Retrieved from https://arxiv.org/abs/1803.01719v3

    - Study failure modes for early training in deep ReLU nets:
    - 1) exploding or vanishing mean activation length
    - 2) exponentially large variance of activation length
    - For FCN, the cure of 1) require a specific init, and the cure of 2) require a specific constraint
    - For ResNets, the cure of 1) require a specific scaling, then 2) also gets cured

Hernández-García, A., & König, P. (2018). Do deep nets really need weight decay and dropout? arXiv, 1802.07042. Retrieved from https://arxiv.org/abs/1802.07042v3

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., & Bengio, Y. (2018). Learning deep representations by mutual information estimation and maximization. arXiv, 1808.06670. Retrieved from https://arxiv.org/abs/1808.06670v5

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv, 1803.05407. Retrieved from https://arxiv.org/abs/1803.05407v3

Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. arXiv, 1806.07572. Retrieved from https://arxiv.org/abs/1806.07572v4

Kaiser, Ł., & Bengio, S. (2018). Discrete Autoencoders for Sequence Models. arXiv, 1801.09797. Retrieved from https://arxiv.org/abs/1801.09797v1

Karakida, R., Akaho, S., & Amari, S.-i. (2018). Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. arXiv, 1806.01316. Retrieved from https://arxiv.org/abs/1806.01316v3

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2018). Effect of Depth and Width on Local Minima in Deep Learning. arXiv, 1811.08150. Retrieved from https://arxiv.org/abs/1811.08150v4

Kawaguchi, K., & Bengio, Y. (2018). Depth with Nonlinearity Creates No Bad Local Minima in ResNets. arXiv, 1810.09038. Retrieved from https://arxiv.org/abs/1810.09038v3

Kirchhoff, M. D., Parr, T., Palacios, E., Friston, K. J., & Kiverstein, J. (2018). The Markov blankets of life: autonomy, active inference and the free energy principle. J. R. Soc. Interface. Retrieved from https://www.semanticscholar.org/paper/The-Markov-blankets-of-life%3A-autonomy%2C-active-and-Kirchhoff-Parr/e1d3f7eb3bf11a0881d9417df8071ca663eefbaf

Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2018). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. arXiv, 1810.00825. Retrieved from https://arxiv.org/abs/1810.00825v3

Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. arXiv, 1804.08838. Retrieved from https://arxiv.org/abs/1804.08838v1

Liang, S., Sun, R., Lee, J. D., & Srikant, R. (2018). Adding One Neuron Can Eliminate All Bad Local Minima. arXiv, 1805.08671. Retrieved from https://arxiv.org/abs/1805.08671v1

Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018). Don't Use Large Mini-Batches, Use Local SGD. arXiv, 1808.07217. Retrieved from https://arxiv.org/abs/1808.07217v6

Liu, J., & Xu, L. (2018). Accelerating Stochastic Gradient Descent Using Antithetic Sampling. arXiv, 1810.03124. Retrieved from https://arxiv.org/abs/1810.03124v1

Martin, C. H., & Mahoney, M. W. (2018). Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning. arXiv, 1810.01075. Retrieved from https://arxiv.org/abs/1810.01075v1

Matthews, A. G. d. G., Rowland, M., Hron, J., Turner, R. E., & Ghahramani, Z. (2018). Gaussian Process Behaviour in Wide Deep Neural Networks. arXiv, 1804.11271. Retrieved from https://arxiv.org/abs/1804.11271v2

Mei, S., Montanari, A., & Nguyen, P.-M. (2018). A Mean Field View of the Landscape of Two-Layers Neural Networks. arXiv, 1804.06561. Retrieved from https://arxiv.org/abs/1804.06561v2

Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2018). Do Deep Generative Models Know What They Don't Know? arXiv, 1810.09136. Retrieved from https://arxiv.org/abs/1810.09136v3

Neal, B., Mittal, S., Baratin, A., Tantia, V., Scicluna, M., Lacoste-Julien, S., & Mitliagkas, I. (2018). A Modern Take on the Bias-Variance Tradeoff in Neural Networks. arXiv, 1810.08591. Retrieved from https://arxiv.org/abs/1810.08591v4

Nouiehed, M., & Razaviyayn, M. (2018). Learning Deep Models: Critical Points and Local Openness. arXiv, 1803.02968. Retrieved from https://arxiv.org/abs/1803.02968v2

Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, Ł., Shazeer, N., Ku, A., & Tran, D. (2018). Image Transformer. arXiv, 1802.05751. Retrieved from https://arxiv.org/abs/1802.05751v3

Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. arXiv, 1804.08771. Retrieved from https://arxiv.org/abs/1804.08771v2

    - BLEU is a parameterized metric
    - These parameters are often not reported
    - The main culprit is different tokenization and normalization schemes applied to the reference
    - The author provide a new tool, SacreBLEU, to use a common BLEU scheme

Ruthotto, L., & Haber, E. (2018). Deep Neural Networks Motivated by Partial Differential Equations. arXiv, 1804.04272. Retrieved from https://arxiv.org/abs/1804.04272v2

Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? arXiv, 1805.11604. Retrieved from https://arxiv.org/abs/1805.11604v5

Scaman, K., & Virmaux, A. (2018). Lipschitz regularity of deep neural networks: analysis and efficient estimation. arXiv, 1805.10965. Retrieved from https://arxiv.org/abs/1805.10965v2

Shamir, O. (2018). Are ResNets Provably Better than Linear Predictors? arXiv, 1804.06739. Retrieved from https://arxiv.org/abs/1804.06739v4

Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. arXiv, 1803.02155. Retrieved from https://arxiv.org/abs/1803.02155v2

Tarnowski, W., Warchoł, P., Jastrzębski, S., Tabor, J., & Nowak, M. A. (2018). Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function. arXiv, 1809.08848. Retrieved from https://arxiv.org/abs/1809.08848v3

    - In ResNets dynamical isometry (https://arxiv.org/abs/1312.6120) is achievable for any activation function
    - Use Free Probability and Random Matrix Theories (FPT & RMT)
    - Study initial and late phases of the learning processes

Thorpe, M., & van Gennip, Y. (2018). Deep Limits of Residual Neural Networks. arXiv, 1810.11741. Retrieved from https://arxiv.org/abs/1810.11741v4

    - Study ResNet as a discretisation of an ODE
    - Some convergence studies that connect the discrete setting to a continuum problem

Wang, W., Sun, Y., Eriksson, B., Wang, W., & Aggarwal, V. (2018). Wide Compression: Tensor Ring Nets. arXiv, 1802.09052. Retrieved from https://arxiv.org/abs/1802.09052v1

    - Tensor Ring (TR) factorizations to compress existing MLPs and CNNs
    - with little or no quality degredation on image classification

Wu, Y., & He, K. (2018). Group Normalization. arXiv, 1803.08494. Retrieved from https://arxiv.org/abs/1803.08494v3

Xiao, L., Bahri, Y., Sohl-Dickstein, J., Schoenholz, S. S., & Pennington, J. (2018). Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks. arXiv, 1806.05393. Retrieved from https://arxiv.org/abs/1806.05393v2

    - Are residual connections and batch normalization necessary for very deep nets?
    - No, just use a Delta-Orthogonal initialization and appropriate (in this case, tanh) nonlinearity.
    - This research is based on a mean field theory and dynamical isometry (https://arxiv.org/abs/1312.6120)

Xing, C., Arpit, D., Tsirigotis, C., & Bengio, Y. (2018). A Walk with SGD. arXiv, 1802.08770. Retrieved from https://arxiv.org/abs/1802.08770v4

Yuille, A. L., & Liu, C. (2018). Deep Nets: What have they ever done for Vision? arXiv, 1805.04025. Retrieved from https://arxiv.org/abs/1805.04025v4

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

    - a novel initialization strategy for weight normalized networks with and without residual connections
    - is based on mean field approximation
    - outperforms existing methods in generalization, robustness to hyper-parameters and variance between seeds

Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2019). Benign Overfitting in Linear Regression. arXiv, 1906.11300. Retrieved from https://arxiv.org/abs/1906.11300v3

Belkin, M., Hsu, D., & Xu, J. (2019). Two models of double descent for weak features. arXiv, 1903.07571. Retrieved from https://arxiv.org/abs/1903.07571v2

Bengio, Y., Deleu, T., Rahaman, N., Ke, R., Lachapelle, S., Bilaniuk, O., ...Pal, C. (2019). A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms. arXiv, 1901.10912. Retrieved from https://arxiv.org/abs/1901.10912v2

Dean, T., Fan, C., Lewis, F. E., & Sano, M. (2019). Biological Blueprints for Next Generation AI Systems. arXiv, 1912.00421. Retrieved from https://arxiv.org/abs/1912.00421v1

Ding, T., Li, D., & Sun, R. (2019). Sub-Optimal Local Minima Exist for Neural Networks with Almost All Non-Linear Activations. arXiv, 1911.01413. Retrieved from https://arxiv.org/abs/1911.01413v3

Farquhar, S., & Gal, Y. (2019). A Unifying Bayesian View of Continual Learning. arXiv, 1902.06494. Retrieved from https://arxiv.org/abs/1902.06494v1

Fort, S., & Jastrzebski, S. (2019). Large Scale Structure of Neural Network Loss Landscapes. arXiv, 1906.04724. Retrieved from https://arxiv.org/abs/1906.04724v1

Fort, S., & Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes. arXiv, 1910.05929. Retrieved from https://arxiv.org/abs/1910.05929v1

Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep Ensembles: A Loss Landscape Perspective. arXiv, 1912.02757. Retrieved from https://arxiv.org/abs/1912.02757v2

Fort, S., Nowak, P. K., Jastrzebski, S., & Narayanan, S. (2019). Stiffness: A New Perspective on Generalization in Neural Networks. arXiv, 1901.09491. Retrieved from https://arxiv.org/abs/1901.09491v3

Ghorbani, B., Krishnan, S., & Xiao, Y. (2019). An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. arXiv, 1901.10159. Retrieved from https://arxiv.org/abs/1901.10159v1

Gu, J., Wang, Y., Cho, K., & Li, V. O. K. (2019). Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations. arXiv, 1906.01181. Retrieved from https://arxiv.org/abs/1906.01181v1

Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2019). Surprises in High-Dimensional Ridgeless Least Squares Interpolation. arXiv, 1903.08560. Retrieved from https://arxiv.org/abs/1903.08560v5

He, F., Liu, T., & Tao, D. (2019). Why ResNet Works? Residuals Generalize. arXiv, 1904.01367. Retrieved from https://arxiv.org/abs/1904.01367v1

Jiang, A. H., Wong, D. L.-K., Zhou, G., Andersen, D. G., Dean, J., Ganger, G. R., ...Pillai, P. (2019). Accelerating Deep Learning by Focusing on the Biggest Losers. arXiv, 1910.00762. Retrieved from https://arxiv.org/abs/1910.00762v1

Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2019). Subspace Inference for Bayesian Deep Learning. arXiv, 1907.07504. Retrieved from https://arxiv.org/abs/1907.07504v1

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2019). Every Local Minimum Value is the Global Minimum Value of Induced Model in Non-convex Machine Learning. arXiv, 1904.03673. Retrieved from https://arxiv.org/abs/1904.03673v3

Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2019). Hyperbolic Image Embeddings. arXiv, 1904.02239. Retrieved from https://arxiv.org/abs/1904.02239v2

    - Hyperbolic embeddings as an alternative to Euclidean and spherical embeddings
    - Hyperbolic spaces are more suitable for embedding data with such hierarchical structure
    - Experiments with few-shot learning and person re-identification demonstrate these embeddings are beneficial
    - Propose an approach to evaluate the hyperbolicity of a dataset using Gromov δ-hyperbolicity

Kosiorek, A. R., Sabour, S., Teh, Y. W., & Hinton, G. E. (2019). Stacked Capsule Autoencoders. arXiv, 1906.06818. Retrieved from https://arxiv.org/abs/1906.06818v2

Labach, A., Salehinejad, H., & Valaee, S. (2019). Survey of Dropout Methods for Deep Neural Networks. arXiv, 1904.13310. Retrieved from https://arxiv.org/abs/1904.13310v2

Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. arXiv, 1902.06720. Retrieved from https://arxiv.org/abs/1902.06720v4

Lezcano-Casado, M., & Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group. arXiv, 1901.08428. Retrieved from https://arxiv.org/abs/1901.08428v3

    - A reparametrization to perform unconstrained optimizaion with orthogonal and unitary constraints
    - We apply our results to RNNs with orthogonal recurrent weights, yielding a new architecture called EXPRNN
    - Faster, accurate, and more stable convergence
    - https://github.com/pytorch/pytorch/issues/48144

Liang, S., Sun, R., & Srikant, R. (2019). Revisiting Landscape Analysis in Deep Neural Networks: Eliminating Decreasing Paths to Infinity. arXiv, 1912.13472. Retrieved from https://arxiv.org/abs/1912.13472v1

Liu, T., Chen, M., Zhou, M., Du, S. S., Zhou, E., & Zhao, T. (2019). Towards Understanding the Importance of Shortcut Connections in Residual Networks. arXiv, 1909.04653. Retrieved from https://arxiv.org/abs/1909.04653v3

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2019). On the Variance of the Adaptive Learning Rate and Beyond. arXiv, 1908.03265. Retrieved from https://arxiv.org/abs/1908.03265v4

Ma, X., Zhang, P., Zhang, S., Duan, N., Hou, Y., Song, D., & Zhou, M. (2019). A Tensorized Transformer for Language Modeling. arXiv, 1906.09777. Retrieved from https://arxiv.org/abs/1906.09777v3

    - We propose Multi-linear attention with Block-Term Tensor Decomposition (BTD)
    - This not only largely compress the model parameters but also obtain performance improvements

Millidge, B. (2019). Deep Active Inference as Variational Policy Gradients. arXiv, 1907.03876. Retrieved from https://arxiv.org/abs/1907.03876v1

Nakamura, K., & Hong, B.-W. (2019). Adaptive Weight Decay for Deep Neural Networks. arXiv, 1907.08931. Retrieved from https://arxiv.org/abs/1907.08931v2

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep Double Descent: Where Bigger Models and More Data Hurt. arXiv, 1912.02292. Retrieved from https://arxiv.org/abs/1912.02292v1

Neal, B. (2019). On the Bias-Variance Tradeoff: Textbooks Need an Update. arXiv, 1912.08286. Retrieved from https://arxiv.org/abs/1912.08286v1

Peluchetti, S., & Favaro, S. (2019). Infinitely deep neural networks as diffusion processes. arXiv, 1905.11065. Retrieved from https://arxiv.org/abs/1905.11065v3

    - For deep nets with iid weight init, the dependency on the input vanishes as depth increases to infinity.
    - Under some assumptions, infinitely deep ResNets converge to SDEs (diffusion processes)
    - They do not suffer from the above property

Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019). Micro-Batch Training with Batch-Channel Normalization and Weight Standardization. arXiv, 1903.10520. Retrieved from https://arxiv.org/abs/1903.10520v2

Rangamani, A., Nguyen, N. H., Kumar, A., Phan, D., Chin, S. H., & Tran, T. D. (2019). A Scale Invariant Flatness Measure for Deep Network Minima. arXiv, 1902.02434. Retrieved from https://arxiv.org/abs/1902.02434v1

Shen, X., Tian, X., Liu, T., Xu, F., & Tao, D. (2019). Continuous Dropout. arXiv, 1911.12675. Retrieved from https://arxiv.org/abs/1911.12675v1

Simsekli, U., Sagun, L., & Gurbuzbalaban, M. (2019). A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks. arXiv, 1901.06053. Retrieved from https://arxiv.org/abs/1901.06053v1

Siu, C. (2019). Residual Networks Behave Like Boosting Algorithms. arXiv, 1909.11790. Retrieved from https://arxiv.org/abs/1909.11790v1

Sohl-Dickstein, J., & Kawaguchi, K. (2019). Eliminating all bad Local Minima from Loss Landscapes without even adding an Extra Unit. arXiv, 1901.03909. Retrieved from https://arxiv.org/abs/1901.03909v1

Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., & Chao, L. S. (2019). Learning Deep Transformer Models for Machine Translation. arXiv, 1906.01787. Retrieved from https://arxiv.org/abs/1906.01787v1

Wang, J., Chen, Y., Chakraborty, R., & Yu, S. X. (2019). Orthogonal Convolutional Neural Networks. arXiv, 1911.12207. Retrieved from https://arxiv.org/abs/1911.12207v3

    - Orthogonal convolution: filter orthogonality with doubly block-Toeplitz matrix representation
    - Outperforms the kernel orthogonality, learns more diverse and expressive features

Wen, Y., Luk, K., Gazeau, M., Zhang, G., Chan, H., & Ba, J. (2019). An Empirical Study of Large-Batch Stochastic Gradient Descent with Structured Covariance Noise. arXiv, 1902.08234. Retrieved from https://arxiv.org/abs/1902.08234v4

Yang, G., Pennington, J., Rao, V., Sohl-Dickstein, J., & Schoenholz, S. S. (2019). A Mean Field Theory of Batch Normalization. arXiv, 1902.08129. Retrieved from https://arxiv.org/abs/1902.08129v2

Yang, G. (2019). Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes. arXiv, 1910.12478. Retrieved from https://arxiv.org/abs/1910.12478v3

Yun, C., Sra, S., & Jadbabaie, A. (2019). Are deep ResNets provably better than linear predictors? arXiv, 1907.03922. Retrieved from https://arxiv.org/abs/1907.03922v2

Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. arXiv, 1901.09321. Retrieved from https://arxiv.org/abs/1901.09321v2

Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S. J., Kumar, S., & Sra, S. (2019). Why are Adaptive Methods Good for Attention Models? arXiv, 1912.03194. Retrieved from https://arxiv.org/abs/1912.03194v2

## 2020

Agarwal, C., D'souza, D., & Hooker, S. (2020). Estimating Example Difficulty Using Variance of Gradients. arXiv, 2008.11600. Retrieved from https://arxiv.org/abs/2008.11600v4

Bachlechner, T., Majumder, B. P., Mao, H. H., Cottrell, G. W., & McAuley, J. (2020). ReZero is All You Need: Fast Convergence at Large Depth. arXiv, 2003.04887. Retrieved from https://arxiv.org/abs/2003.04887v2

Chan, K. H. R., Yu, Y., You, C., Qi, H., Wright, J., & Ma, Y. (2020). Deep Networks from the Principle of Rate Reduction. arXiv, 2010.14765. Retrieved from https://arxiv.org/abs/2010.14765v1

Chen, L., Min, Y., Belkin, M., & Karbasi, A. (2020). Multiple Descent: Design Your Own Generalization Curve. arXiv, 2008.01036. Retrieved from https://arxiv.org/abs/2008.01036v7

Chen, Z., Deng, L., Wang, B., Li, G., & Xie, Y. (2020). A Comprehensive and Modularized Statistical Framework for Gradient Norm Equality in Deep Neural Networks. arXiv, 2001.00254. Retrieved from https://arxiv.org/abs/2001.00254v1

Choe, Y. J., Ham, J., & Park, K. (2020). An Empirical Study of Invariant Risk Minimization. arXiv, 2004.05007. Retrieved from https://arxiv.org/abs/2004.05007v2

D'Ascoli, S., Refinetti, M., Biroli, G., & Krzakala, F. (2020). Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime. arXiv, 2003.01054. Retrieved from https://arxiv.org/abs/2003.01054v2

Domingos, P. (2020). Every Model Learned by Gradient Descent Is Approximately a Kernel Machine. arXiv, 2012.00152. Retrieved from https://arxiv.org/abs/2012.00152v1

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ...Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv, 2010.11929. Retrieved from https://arxiv.org/abs/2010.11929v2

Faghri, F., Duvenaud, D., Fleet, D. J., & Ba, J. (2020). A Study of Gradient Variance in Deep Learning. arXiv, 2007.04532. Retrieved from https://arxiv.org/abs/2007.04532v1

Fang, C., Lee, J. D., Yang, P., & Zhang, T. (2020). Modeling from Features: a Mean-field Framework for Over-parameterized Deep Neural Networks. arXiv, 2007.01452. Retrieved from https://arxiv.org/abs/2007.01452v1

    - A new framework to analyze neural network training
    - We capture the evolution of an over-parameterized DNN trained by Gradient Descent
    - Global convergence proof for over-parameterized DNN in the mean-field regime

Fort, S., Dziugaite, G. K., Paul, M., Kharaghani, S., Roy, D. M., & Ganguli, S. (2020). Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel. arXiv, 2010.15110. Retrieved from https://arxiv.org/abs/2010.15110v1

Furrer, D., van Zee, M., Scales, N., & Schärli, N. (2020). Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures. arXiv, 2007.08970. Retrieved from https://arxiv.org/abs/2007.08970v3

Gorbunov, E., Danilova, M., & Gasnikov, A. (2020). Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping. arXiv, 2005.10785. Retrieved from https://arxiv.org/abs/2005.10785v2

Hu, W., Xiao, L., & Pennington, J. (2020). Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks. arXiv, 2001.05992. Retrieved from https://arxiv.org/abs/2001.05992v1

    - Proof that orthogonal initialization speeds up convergence
    - With it, the width for efficient convergence is independent of the depth (without it does not)
    - Is related to the principle of dynamical isometry (https://arxiv.org/abs/1312.6120)

Huang, K., Wang, Y., Tao, M., & Zhao, T. (2020). Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? -- A Neural Tangent Kernel Perspective. arXiv, 2002.06262. Retrieved from https://arxiv.org/abs/2002.06262v2

Jastrzebski, S., Szymczak, M., Fort, S., Arpit, D., Tabor, J., Cho, K., & Geras, K. (2020). The Break-Even Point on Optimization Trajectories of Deep Neural Networks. arXiv, 2002.09572. Retrieved from https://arxiv.org/abs/2002.09572v1

Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. arXiv, 2005.08926. Retrieved from https://arxiv.org/abs/2005.08926v2

    - Problem in neural ODEs: no mechanism for adjusting the trajectory based on subsequent observations
    - We demonstrate how this may be resolved through the mathematics of controlled differential equations
    - This is applicable to the partially observed irregularly-sampled multivariate time series
    - SOTA performance against similar (ODE or RNN based) models in empirical studies on a range of datasets
    - Theoretical results demonstrating universal approximation

Lengerich, B., Xing, E. P., & Caruana, R. (2020). Dropout as a Regularizer of Interaction Effects. arXiv, 2007.00823. Retrieved from https://arxiv.org/abs/2007.00823v2

Liu, L., Liu, X., Gao, J., Chen, W., & Han, J. (2020). Understanding the Difficulty of Training Transformers. arXiv, 2004.08249. Retrieved from https://arxiv.org/abs/2004.08249v3

Liu, C., Zhu, L., & Belkin, M. (2020). Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. arXiv, 2003.00307. Retrieved from https://arxiv.org/abs/2003.00307v2

Millidge, B., Tschantz, A., Seth, A. K., & Buckley, C. L. (2020). On the Relationship Between Active Inference and Control as Inference. arXiv, 2006.12964. Retrieved from https://arxiv.org/abs/2006.12964v3

Millidge, B., Tschantz, A., & Buckley, C. L. (2020). Whence the Expected Free Energy? arXiv, 2004.08128. Retrieved from https://arxiv.org/abs/2004.08128v5

Millidge, B., Tschantz, A., & Buckley, C. L. (2020). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. arXiv, 2006.04182. Retrieved from https://arxiv.org/abs/2006.04182v5

Mundt, M., Hong, Y., Pliushch, I., & Ramesh, V. (2020). A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning. arXiv, 2009.01797. Retrieved from https://arxiv.org/abs/2009.01797v3

Muthukumar, V., Narang, A., Subramanian, V., Belkin, M., Hsu, D., & Sahai, A. (2020). Classification vs regression in overparameterized regimes: Does the loss function matter? arXiv, 2005.08054. Retrieved from https://arxiv.org/abs/2005.08054v2

Timothy P. Lillicrap, #., Adam Santoro, #., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. Nat. Rev. Neurosci., 32303713. Retrieved from https://pubmed.ncbi.nlm.nih.gov/32303713

Lu, Y., Ma, C., Lu, Y., Lu, J., & Ying, L. (2020). A Mean-field Analysis of Deep ResNet and Beyond: Towards Provable Optimization Via Overparameterization From Depth. arXiv, 2003.05508. Retrieved from https://arxiv.org/abs/2003.05508v2

    - Question: why do ResNets achieve zero training loss, while optimization landscape is highly non-convex?
    - We propose a new continuum limit of deep ResNets with a good landscape where every local minimizer is global
    - We apply existing mean-field analyses of two-layer networks to deep networks
    - We propose several novel training schemes which result in strong empirical performance

Melkman, A. A., Guo, S., Ching, W.-K., Liu, P., & Akutsu, T. (2020). On the Compressive Power of Boolean Threshold Autoencoders. arXiv, 2004.09735. Retrieved from https://arxiv.org/abs/2004.09735v1

Mixon, D. G., Parshall, H., & Pi, J. (2020). Neural collapse with unconstrained features. arXiv, 2011.11619. Retrieved from https://arxiv.org/abs/2011.11619v1

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training. arXiv, 2008.08186. Retrieved from https://arxiv.org/abs/2008.08186v2

Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., & Lajoie, G. (2020). Gradient Starvation: A Learning Proclivity in Neural Networks. arXiv, 2011.09468. Retrieved from https://arxiv.org/abs/2011.09468v4

Raunak, V., Dalmia, S., Gupta, V., & Metze, F. (2020). On Long-Tailed Phenomena in Neural Machine Translation. arXiv, 2010.04924. Retrieved from https://arxiv.org/abs/2010.04924v1

    - Problem: NMT models struggle with generating low-frequency tokens
    - Penalizing  low-confidence predictions hurts beam search performance
    - We propose Anti-Focal loss, a generalization of Focal loss and cross-entropy
    - Anti-Focal loss allocates less relative loss to low-confidence predictions
    - It leads to significant gains over cross-entropy, especially on the generation of low-frequency words

Queiruga, A. F., Erichson, N. B., Taylor, D., & Mahoney, M. W. (2020). Continuous-in-Depth Neural Networks. arXiv, 2008.02389. Retrieved from https://arxiv.org/abs/2008.02389v1

Sankar, A. R., Khasbage, Y., Vigneswaran, R., & Balasubramanian, V. N. (2020). A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization. arXiv, 2012.03801. Retrieved from https://arxiv.org/abs/2012.03801v2

Shen, D., Zheng, M., Shen, Y., Qu, Y., & Chen, W. (2020). A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation. arXiv, 2009.13818. Retrieved from https://arxiv.org/abs/2009.13818v2

Sun, R., Li, D., Liang, S., Ding, T., & Srikant, R. (2020). The Global Landscape of Neural Networks: An Overview. arXiv, 2007.01429. Retrieved from https://arxiv.org/abs/2007.01429v1

Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Reinforcement Learning through Active Inference. arXiv, 2002.12636. Retrieved from https://arxiv.org/abs/2002.12636v1

Wang, L., Shen, B., Zhao, N., & Zhang, Z. (2020). Is the Skip Connection Provable to Reform the Neural Network Loss Landscape? arXiv, 2006.05939. Retrieved from https://arxiv.org/abs/2006.05939v1

Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. arXiv, 2002.08791. Retrieved from https://arxiv.org/abs/2002.08791v4

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ...Liu, T.-Y. (2020). On Layer Normalization in the Transformer Architecture. arXiv, 2002.04745. Retrieved from https://arxiv.org/abs/2002.04745v2

Yang, G. (2020). Tensor Programs II: Neural Tangent Kernel for Any Architecture. arXiv, 2006.14548. Retrieved from https://arxiv.org/abs/2006.14548v4

Yang, G. (2020). Tensor Programs III: Neural Matrix Laws. arXiv, 2009.10685. Retrieved from https://arxiv.org/abs/2009.10685v3

Yang, Z., Yu, Y., You, C., Steinhardt, J., & Ma, Y. (2020). Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. arXiv, 2002.11328. Retrieved from https://arxiv.org/abs/2002.11328v3

Zhao, P., Chen, P.-Y., Das, P., Ramamurthy, K. N., & Lin, X. (2020). Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. arXiv, 2005.00060. Retrieved from https://arxiv.org/abs/2005.00060v2

## 2021

Aguilera, M., Millidge, B., Tschantz, A., & Buckley, C. L. (2021). How particular is the physics of the free energy principle? arXiv, 2105.11203. Retrieved from https://arxiv.org/abs/2105.11203v3

Belkin, M. (2021). Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. arXiv, 2105.14368. Retrieved from https://arxiv.org/abs/2105.14368v1

Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T.-Y., ...Zoph, B. (2021). Revisiting ResNets: Improved Training and Scaling Strategies. arXiv, 2103.07579. Retrieved from https://arxiv.org/abs/2103.07579v1

Benton, G. W., Maddox, W. J., Lotfi, S., & Wilson, A. G. (2021). Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling. arXiv, 2102.13042. Retrieved from https://arxiv.org/abs/2102.13042v2

Berariu, T., Czarnecki, W., De, S., Bornschein, J., Smith, S., Pascanu, R., & Clopath, C. (2021). A study on the plasticity of neural networks. arXiv, 2106.00042. Retrieved from https://arxiv.org/abs/2106.00042v2

Bingham, G., & Miikkulainen, R. (2021). AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks. arXiv, 2109.08958. Retrieved from https://arxiv.org/abs/2109.08958v2

    - A weight initialization algorithm that automatically adapts to different architectures
    - Scales the weights by tracking the mean and variance of signals as they propagate through the network
    - Improves performance of convolutional, residual, and transformer networks

Bond-Taylor, S., Leach, A., Long, Y., & Willcocks, C. G. (2021). Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. arXiv, 2103.04922. Retrieved from https://arxiv.org/abs/2103.04922v4

Cao, S. (2021). Choose a Transformer: Fourier or Galerkin. arXiv, 2105.14995. Retrieved from https://arxiv.org/abs/2105.14995v4

Cohen, A.-S., Cont, R., Rossier, A., & Xu, R. (2021). Scaling Properties of Deep Residual Networks. arXiv, 2105.12245. Retrieved from https://arxiv.org/abs/2105.12245v2

    - We investigate the scaling behavior of trained ResNet weights as the number of layers increases
    - Found at least three different scaling regimes
    - In two of these regimes, the properties may be described in terms of a class of ODEs or SDEs

Dar, Y., Muthukumar, V., & Baraniuk, R. G. (2021). A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning. arXiv, 2109.02355. Retrieved from https://arxiv.org/abs/2109.02355v1

Ding, Z., Chen, S., Li, Q., & Wright, S. (2021). Overparameterization of deep ResNet: zero loss and mean-field analysis. arXiv, 2105.14417. Retrieved from https://arxiv.org/abs/2105.14417v3

    - Study ResNet convergence in the infinite-depth and infinite-width regime
    - GD becomes a gradient flow for a probability distribution that is characterized by a PDE
    - Results suggest that the training of the large enough ResNet gives a near-zero loss
    - Estimates of the depth and width needed to reduce the loss below a given threshold

Federici, M., Tomioka, R., & Forré, P. (2021). An Information-theoretic Approach to Distribution Shifts. arXiv, 2106.03783. Retrieved from https://arxiv.org/abs/2106.03783v2

Fortuin, V. (2021). Priors in Bayesian Deep Learning: A Review. arXiv, 2105.06868. Retrieved from https://arxiv.org/abs/2105.06868v3

Gilmer, J., Ghorbani, B., Garg, A., Kudugunta, S., Neyshabur, B., Cardoze, D., ...Firat, O. (2021). A Loss Curvature Perspective on Training Instability in Deep Learning. arXiv, 2110.04369. Retrieved from https://arxiv.org/abs/2110.04369v1

Hinton, G. (2021). How to represent part-whole hierarchies in a neural network. arXiv, 2102.12627. Retrieved from https://arxiv.org/abs/2102.12627v1

Han, X. Y., Papyan, V., & Donoho, D. L. (2021). Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path. arXiv, 2106.02073. Retrieved from https://arxiv.org/abs/2106.02073v4

Hua, T., Wang, W., Xue, Z., Ren, S., Wang, Y., & Zhao, H. (2021). On Feature Decorrelation in Self-Supervised Learning. arXiv, 2105.00470. Retrieved from https://arxiv.org/abs/2105.00470v2

Izmailov, P., Nicholson, P., Lotfi, S., & Wilson, A. G. (2021). Dangers of Bayesian Model Averaging under Covariate Shift. arXiv, 2106.11905. Retrieved from https://arxiv.org/abs/2106.11905v2

Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). What Are Bayesian Neural Network Posteriors Really Like? arXiv, 2104.14421. Retrieved from https://arxiv.org/abs/2104.14421v1

Lanillos, P., Meo, C., Pezzato, C., Meera, A. A., Baioumy, M., Ohata, W., ...Tani, J. (2021). Active Inference in Robotics and Artificial Agents: Survey and Challenges. arXiv, 2112.01871. Retrieved from https://arxiv.org/abs/2112.01871v1

Larsen, B. W., Fort, S., Becker, N., & Ganguli, S. (2021). How many degrees of freedom do we need to train deep networks: a loss landscape perspective. arXiv, 2107.05802. Retrieved from https://arxiv.org/abs/2107.05802v2

Liu, F., Suykens, J. A. K., & Cevher, V. (2021). On the Double Descent of Random Features Models Trained with SGD. arXiv, 2110.06910. Retrieved from https://arxiv.org/abs/2110.06910v6

Liu, M., Chen, L., Du, X., Jin, L., & Shang, M. (2021). Activated Gradients for Deep Neural Networks. arXiv, 2107.04228. Retrieved from https://arxiv.org/abs/2107.04228v1

Meunier, L., Delattre, B., Araujo, A., & Allauzen, A. (2021). A Dynamical System Perspective for Lipschitz Neural Networks. arXiv, 2110.12690. Retrieved from https://arxiv.org/abs/2110.12690v2

Millidge, B., Seth, A., & Buckley, C. L. (2021). Predictive Coding: a Theoretical and Experimental Review. arXiv, 2107.12979. Retrieved from https://arxiv.org/abs/2107.12979v4

Nado, Z., Gilmer, J. M., Shallue, C. J., Anil, R., & Dahl, G. E. (2021). A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes. arXiv, 2102.06356. Retrieved from https://arxiv.org/abs/2102.06356v3

Ørebæk, O.-E., & Geitle, M. (2021). Exploring the Hyperparameters of XGBoost Through 3D Visualizations. AAAI Spring Symposium Combining Machine Learning with Knowledge Engineering. Retrieved from https://www.semanticscholar.org/paper/Exploring-the-Hyperparameters-of-XGBoost-Through-3D-%C3%98reb%C3%A6k-Geitle/4b5895a52efa17c60b6bbd693f80594c0e10440c

Ortiz, J., Evans, T., & Davison, A. J. (2021). A visual introduction to Gaussian Belief Propagation. arXiv, 2107.02308. Retrieved from https://arxiv.org/abs/2107.02308v1

Rame, A., Dancette, C., & Cord, M. (2021). Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization. arXiv, 2109.02934. Retrieved from https://arxiv.org/abs/2109.02934v3

Roberts, D. A., Yaida, S., & Hanin, B. (2021). The Principles of Deep Learning Theory. arXiv, 2106.10165. Retrieved from https://arxiv.org/abs/2106.10165v2

Sander, M. E., Ablin, P., Blondel, M., & Peyré, G. (2021). Momentum Residual Neural Networks. arXiv, 2102.07870. Retrieved from https://arxiv.org/abs/2102.07870v3

Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. arXiv, 2102.11174. Retrieved from https://arxiv.org/abs/2102.11174v3

Schneider, F., Dangel, F., & Hennig, P. (2021). Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks. arXiv, 2102.06604. Retrieved from https://arxiv.org/abs/2102.06604v2

Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Towards Causal Representation Learning. arXiv, 2102.11107. Retrieved from https://arxiv.org/abs/2102.11107v1

Shang, Y., Duan, B., Zong, Z., Nie, L., & Yan, Y. (2021). Lipschitz Continuity Guided Knowledge Distillation. arXiv, 2108.12905. Retrieved from https://arxiv.org/abs/2108.12905v1

Shleifer, S., Weston, J., & Ott, M. (2021). NormFormer: Improved Transformer Pretraining with Extra Normalization. arXiv, 2110.09456. Retrieved from https://arxiv.org/abs/2110.09456v2

Silver, D., Singh, S., Precup, D., & Sutton, R. (2021). Reward is enough. Artif. Intell. Retrieved from https://web.eecs.umich.edu/~baveja/Papers/RewardIsEnough.pdf

Smith, S. L., Dherin, B., Barrett, D. G. T., & De, S. (2021). On the Origin of Implicit Regularization in Stochastic Gradient Descent. arXiv, 2101.12176. Retrieved from https://arxiv.org/abs/2101.12176v1

Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., ...Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture for Vision. arXiv, 2105.01601. Retrieved from https://arxiv.org/abs/2105.01601v4

Yang, G., & Hu, E. J. (2021). Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks. International Conference on Machine Learning. PMLR. Retrieved from https://proceedings.mlr.press/v139/yang21c.html

Yin, M., Sui, Y., Liao, S., & Yuan, B. (2021). Towards Efficient Tensor Decomposition-Based DNN Model Compression with Optimization Framework. arXiv, 2107.12422. Retrieved from https://arxiv.org/abs/2107.12422v1

    - Problem: compressing CNNs with Tensor train (TT) and Tensor ring (TR) suffers significant accuracy loss
    - A new approach requires a specific training procedure
    - very high compression performance with high accuracy
    - Also works for RNNs

Zhao, M., Liu, Z., Luan, S., Zhang, S., Precup, D., & Bengio, Y. (2021). A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning. arXiv, 2106.02097. Retrieved from https://arxiv.org/abs/2106.02097v3

Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., & Qu, Q. (2021). A Geometric Analysis of Neural Collapse with Unconstrained Features. arXiv, 2105.02375. Retrieved from https://arxiv.org/abs/2105.02375v1

Ziyin, L., Li, B., Simon, J. B., & Ueda, M. (2021). SGD with a Constant Large Learning Rate Can Converge to Local Maxima. arXiv, 2107.11774. Retrieved from https://arxiv.org/abs/2107.11774v4

## 2022

Ahn, K., Zhang, J., & Sra, S. (2022). Understanding the unstable convergence of gradient descent. arXiv, 2204.01050. Retrieved from https://arxiv.org/abs/2204.01050v2

Amid, E., Anil, R., Kotłowski, W., & Warmuth, M. K. (2022). Learning from Randomly Initialized Neural Network Features. arXiv, 2202.06438. Retrieved from https://arxiv.org/abs/2202.06438v1

Arjevani, Y., & Field, M. (2022). Annihilation of Spurious Minima in Two-Layer ReLU Networks. arXiv, 2210.06088. Retrieved from https://arxiv.org/abs/2210.06088v1

Bai, Q., Rosenberg, S., & Xu, W. (2022). A Geometric Understanding of Natural Gradient. arXiv, 2202.06232. Retrieved from https://arxiv.org/abs/2202.06232v3

Christof, C., & Kowalczyk, J. (2022). On the Omnipresence of Spurious Local Minima in Certain Neural Network Training Problems. arXiv, 2202.12262. Retrieved from https://arxiv.org/abs/2202.12262v2

Cohen, J. M., Ghorbani, B., Krishnan, S., Agarwal, N., Medapati, S., Badura, M., ...Gilmer, J. (2022). Adaptive Gradient Methods at the Edge of Stability. arXiv, 2207.14484. Retrieved from https://arxiv.org/abs/2207.14484v1

Ergen, T., Neyshabur, B., & Mehta, H. (2022). Convexifying Transformers: Improving optimization and understanding of transformer networks. arXiv, 2211.11052. Retrieved from https://arxiv.org/abs/2211.11052v1

Fedus, W., Dean, J., & Zoph, B. (2022). A Review of Sparse Expert Models in Deep Learning. arXiv, 2209.01667. Retrieved from https://arxiv.org/abs/2209.01667v1

Fort, S., Cubuk, E. D., Ganguli, S., & Schoenholz, S. S. (2022). What does a deep neural network confidently perceive? The effective dimension of high certainty class manifolds and their low confidence boundaries. arXiv, 2210.05546. Retrieved from https://arxiv.org/abs/2210.05546v1

Friston, K. J., Ramstead, M. J. D., Kiefer, A. B., Tschantz, A., Buckley, C. L., Albarracin, M., ...René, G. (2022). Designing Ecosystems of Intelligence from First Principles. arXiv, 2212.01354. Retrieved from https://arxiv.org/abs/2212.01354v2

Friston, K., Da Costa, L., Sajid, N., Heins, C., Ueltzhoffer, K., Pavliotis, G., & Parr, T. (2022). The free energy principle made simpler but not too simple. Phys. Rep. Retrieved from https://www.semanticscholar.org/paper/The-free-energy-principle-made-simpler-but-not-too-Friston-Costa/e54427100b2de8187fe3b96303653b6220aaad44

Isomura, T., Shimazaki, H., & Friston, K. J. (2022). Canonical neural networks perform active inference. Commun. Biol., 5(55), 1–15. Retrieved from https://www.nature.com/articles/s42003-021-02994-2

Hafner, D., Lee, K.-H., Fischer, I., & Abbeel, P. (2022). Deep Hierarchical Planning from Pixels. arXiv, 2206.04114. Retrieved from https://arxiv.org/abs/2206.04114v1

Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv, 2212.13345. Retrieved from https://arxiv.org/abs/2212.13345v1

Juneja, J., Bansal, R., Cho, K., Sedoc, J., & Saphra, N. (2022). Linear Connectivity Reveals Generalization Strategies. arXiv, 2205.12411. Retrieved from https://arxiv.org/abs/2205.12411v5

Li, Y. (2022). A Short Survey of Systematic Generalization. arXiv, 2211.11956. Retrieved from https://arxiv.org/abs/2211.11956v1

Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi, S. J., ...Kumar, S. (2022). The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers. arXiv, 2210.06313. Retrieved from https://arxiv.org/abs/2210.06313v2

Liu, Z., Michaud, E. J., & Tegmark, M. (2022). Omnigrok: Grokking Beyond Algorithmic Data. arXiv, 2210.01117. Retrieved from https://arxiv.org/abs/2210.01117v2

Malladi, S., Wettig, A., Yu, D., Chen, D., & Arora, S. (2022). A Kernel-Based View of Language Model Fine-Tuning. arXiv, 2210.05643. Retrieved from https://arxiv.org/abs/2210.05643v4

Millidge, B., Salvatori, T., Song, Y., Bogacz, R., & Lukasiewicz, T. (2022). Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation? arXiv, 2202.09467. Retrieved from https://arxiv.org/abs/2202.09467v1

Mohamadi, M. A., Bae, W., & Sutherland, D. J. (2022). A Fast, Well-Founded Approximation to the Empirical Neural Tangent Kernel. arXiv, 2206.12543. Retrieved from https://arxiv.org/abs/2206.12543v3

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv, 2201.02177. Retrieved from https://arxiv.org/abs/2201.02177v1

Rangwani, H., Aithal, S. K., Mishra, M., & Babu, R. V. (2022). Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data. arXiv, 2212.13827. Retrieved from https://arxiv.org/abs/2212.13827v1

Ramesh, A., Kirsch, L., van Steenkiste, S., & Schmidhuber, J. (2022). Exploring through Random Curiosity with General Value Functions. arXiv, 2211.10282. Retrieved from https://arxiv.org/abs/2211.10282v1

Ramstead, M. J. D., Sakthivadivel, D. A. R., Heins, C., Koudahl, M., Millidge, B., Da Costa, L., ...Friston, K. J. (2022). On Bayesian Mechanics: A Physics of and by Beliefs. arXiv, 2205.11543. Retrieved from https://arxiv.org/abs/2205.11543v4

Sander, M. E., Ablin, P., & Peyré, G. (2022). Do Residual Neural Networks discretize Neural Ordinary Differential Equations? arXiv, 2205.14612. Retrieved from https://arxiv.org/abs/2205.14612v2

    - Are discrete dynamics defined by a ResNet close to the continuous one of a Neural ODE?
    - Several theoretical results
    - A simple technique to train ResNets without storing activations
    - Recover the approximated activations during the backward pass by using a reverse-time Euler scheme
    - Fine-tuning very deep ResNets without memory consumption in the residual layers

Sutton, R. S., Bowling, M., & Pilarski, P. M. (2022). The Alberta Plan for AI Research. arXiv, 2208.11173. Retrieved from https://arxiv.org/abs/2208.11173v3

Sutton, R. S., Machado, M. C., Holland, G. Z., Szepesvari, D., Timbers, F., Tanner, B., & White, A. (2022). Reward-Respecting Subtasks for Model-Based Reinforcement Learning. arXiv, 2202.03466. Retrieved from https://arxiv.org/abs/2202.03466v4

Vanchurin, V., Wolf, Y. I., Katsnelson, M. I., & Koonin, E. V. (2022). Toward a theory of evolution as multilevel learning. Proc. Natl. Acad. Sci. U.S.A., 119(6), e2120037119. Retrieved from https://doi.org/10.1073/pnas.2120037119

Wang, H., Ma, S., Huang, S., Dong, L., Wang, W., Peng, Z., ...Wei, F. (2022). Foundation Transformers. arXiv, 2210.06423. Retrieved from https://arxiv.org/abs/2210.06423v2

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., & Wei, F. (2022). DeepNet: Scaling Transformers to 1,000 Layers. arXiv, 2203.00555. Retrieved from https://arxiv.org/abs/2203.00555v1

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., ...Gao, J. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv, 2203.03466. Retrieved from https://arxiv.org/abs/2203.03466v2

Zhou, J., You, C., Li, X., Liu, K., Liu, S., Qu, Q., & Zhu, Z. (2022). Are All Losses Created Equal: A Neural Collapse Perspective. arXiv, 2210.02192. Retrieved from https://arxiv.org/abs/2210.02192v2

Zhou, J., Li, X., Ding, T., You, C., Qu, Q., & Zhu, Z. (2022). On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features. arXiv, 2203.01238. Retrieved from https://arxiv.org/abs/2203.01238v2

Zhu, Z., Liu, F., Chrysos, G. G., & Cevher, V. (2022). Robustness in deep learning: The good (width), the bad (depth), and the ugly (initialization). arXiv, 2209.07263. Retrieved from https://arxiv.org/abs/2209.07263v4

## 2023

Altintas, G. S., Bachmann, G., Noci, L., & Hofmann, T. (2023). Disentangling Linear Mode-Connectivity. arXiv, 2312.09832. Retrieved from https://arxiv.org/abs/2312.09832v1

Andriushchenko, M., Croce, F., Müller, M., Hein, M., & Flammarion, N. (2023). A Modern Look at the Relationship between Sharpness and Generalization. arXiv, 2302.07011. Retrieved from https://arxiv.org/abs/2302.07011v2

Araujo, A., Havens, A., Delattre, B., Allauzen, A., & Hu, B. (2023). A Unified Algebraic Perspective on Lipschitz Neural Networks. arXiv, 2303.03169. Retrieved from https://arxiv.org/abs/2303.03169v2

Arbel, J., Pitas, K., Vladimirova, M., & Fortuin, V. (2023). A Primer on Bayesian Neural Networks: Review and Debates. arXiv, 2309.16314. Retrieved from https://arxiv.org/abs/2309.16314v1

Bombari, S., Kiyani, S., & Mondelli, M. (2023). Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels. arXiv, 2302.01629. Retrieved from https://arxiv.org/abs/2302.01629v2

Isomura, T. (2023). Bayesian mechanics of self-organising systems. arXiv, 2311.10216. Retrieved from https://arxiv.org/abs/2311.10216v1

Cai, C., Hy, T. S., Yu, R., & Wang, Y. (2023). On the Connection Between MPNN and Graph Transformer. arXiv, 2301.11956. Retrieved from https://arxiv.org/abs/2301.11956v4

Chen, L., Lukasik, M., Jitkrittum, W., You, C., & Kumar, S. (2023). It's an Alignment, Not a Trade-off: Revisiting Bias and Variance in Deep Models. arXiv, 2310.09250. Retrieved from https://arxiv.org/abs/2310.09250v1

Cirone, N. M., Lemercier, M., & Salvi, C. (2023). Neural signature kernels as infinite-width-depth-limits of controlled ResNets. arXiv, 2303.17671. Retrieved from https://arxiv.org/abs/2303.17671v2

    - We consider randomly initialized controlled ResNets defined as Euler-discretizations of neural controlled differential equations, a unified architecture which enconpasses both RNNs and ResNets
    - Study convergence in the infinite-depth regime

Dang, H., Tran, T., Osher, S., Tran-The, H., Ho, N., & Nguyen, T. (2023). Neural Collapse in Deep Linear Networks: From Balanced to Imbalanced Data. arXiv, 2301.00437. Retrieved from https://arxiv.org/abs/2301.00437v5

Dubois, Y., Hashimoto, T., & Liang, P. (2023). Evaluating Self-Supervised Learning via Risk Decomposition. arXiv, 2302.03068. Retrieved from https://arxiv.org/abs/2302.03068v3

Fu, S., & Wang, D. (2023). Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach. arXiv, 2310.06112. Retrieved from https://arxiv.org/abs/2310.06112v2

Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking Distribution Shift in Tabular Data with TableShift. arXiv, 2312.07577. Retrieved from https://arxiv.org/abs/2312.07577v3

Gao, P., Xu, Q., Wen, P., Shao, H., Yang, Z., & Huang, Q. (2023). A Study of Neural Collapse Phenomenon: Grassmannian Frame, Symmetry and Generalization. arXiv, 2304.08914. Retrieved from https://arxiv.org/abs/2304.08914v2

Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2023). A mathematical perspective on Transformers. arXiv, 2312.10794. Retrieved from https://arxiv.org/abs/2312.10794v3

Giannou, A., Rajput, S., & Papailiopoulos, D. (2023). The Expressive Power of Tuning Only the Normalization Layers. arXiv, 2302.07937. Retrieved from https://arxiv.org/abs/2302.07937v2

Goyle, V., Krishnaswamy, P., Ravikumar, K. G., Chattopadhyay, U., & Goyle, K. (2023). Neural Machine Translation For Low Resource Languages. arXiv, 2304.07869. Retrieved from https://arxiv.org/abs/2304.07869v2

    - In low-resouce NMT, there is no comprehensive survey done to identify what approaches work well
    - We take mBART as a baseline
    - We applt techniques like transfer learning, back translation and focal loss, and verify their effect

Isomura, T., Kotani, K., Jimbo, Y., & Friston, K. J. (2023). Experimental validation of the free-energy principle with in vitro neural networks. Nat. Commun., 14(4547), 1–15. Retrieved from https://www.nature.com/articles/s41467-023-40141-z

Jordan, K. (2023). Calibrated Chaos: Variance Between Runs of Neural Network Training is Harmless and Inevitable. arXiv, 2304.01910. Retrieved from https://arxiv.org/abs/2304.01910v1

Kreisler, I., Nacson, M. S., Soudry, D., & Carmon, Y. (2023). Gradient Descent Monotonically Decreases the Sharpness of Gradient Flow Solutions in Scalar Networks and Beyond. arXiv, 2305.13064. Retrieved from https://arxiv.org/abs/2305.13064v1

Laurent, O., Aldea, E., & Franchi, G. (2023). A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors. arXiv, 2310.08287. Retrieved from https://arxiv.org/abs/2310.08287v1

Lee, J. H., & Vijayan, S. (2023). Having Second Thoughts? Let's hear it. arXiv, 2311.15356. Retrieved from https://arxiv.org/abs/2311.15356v1

Li, W., Peng, Y., Zhang, M., Ding, L., Hu, H., & Shen, L. (2023). Deep Model Fusion: A Survey. arXiv, 2309.15698. Retrieved from https://arxiv.org/abs/2309.15698v1

Liu, Z., Xu, Z., Jin, J., Shen, Z., & Darrell, T. (2023). Dropout Reduces Underfitting. arXiv, 2303.01500. Retrieved from https://arxiv.org/abs/2303.01500v2

Lyu, K., Jin, J., Li, Z., Du, S. S., Lee, J. D., & Hu, W. (2023). Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking. arXiv, 2311.18817. Retrieved from https://arxiv.org/abs/2311.18817v1

Marion, P., Wu, Y.-H., Sander, M. E., & Biau, G. (2023). Implicit regularization of deep residual networks towards neural ODEs. arXiv, 2309.01213. Retrieved from https://arxiv.org/abs/2309.01213v2

    - Proof that if ResNet is initialized as a discretization of a neural ODE, then it holds throughout training

Merrill, W., Tsilivis, N., & Shukla, A. (2023). A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks. arXiv, 2303.11873. Retrieved from https://arxiv.org/abs/2303.11873v1

Pan, L., & Cao, X. (2023). Towards Understanding Neural Collapse: The Effects of Batch Normalization and Weight Decay. arXiv, 2309.04644. Retrieved from https://arxiv.org/abs/2309.04644v2

Peng, Z., Qi, L., Shi, Y., & Gao, Y. (2023). A Theoretical Explanation of Activation Sparsity through Flat Minima and Adversarial Robustness. arXiv, 2309.03004. Retrieved from https://arxiv.org/abs/2309.03004v4

Simon, J. B., Karkada, D., Ghosh, N., & Belkin, M. (2023). More is Better in Modern Machine Learning: when Infinite Overparameterization is Optimal and Overfitting is Obligatory. arXiv, 2311.14646. Retrieved from https://arxiv.org/abs/2311.14646v2

Súkeník, P., Mondelli, M., & Lampert, C. (2023). Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model. arXiv, 2305.13165. Retrieved from https://arxiv.org/abs/2305.13165v1

Vasilescu, M. A. O. (2023). Causal Deep Learning: Causal Capsules and Tensor Transformers. arXiv, 2301.00314. Retrieved from https://arxiv.org/abs/2301.00314v1

    - NNs and tensor factorization methods may perform causal inference, or simply perform regression
    - A new deep neural network composed of a hierarchy of autoencoders
    - This results in a hierarchy of kernel tensor factor models
    - Forward causal questions (what if?) and inverse causal questions (why?) are addressed

Verwimp, E., Aljundi, R., Ben-David, S., Bethge, M., Cossu, A., Gepperth, A., ...van de Ven, G. M. (2023). Continual Learning: Applications and the Road Forward. arXiv, 2311.11908. Retrieved from https://arxiv.org/abs/2311.11908v2

Wang, M., Pan, Y., Xu, Z., Yang, X., Li, G., & Cichocki, A. (2023). Tensor Networks Meet Neural Networks: A Survey and Future Perspectives. arXiv, 2302.09019. Retrieved from https://arxiv.org/abs/2302.09019v2

    - Tensor networks (TNs) were introduced to solve the curse of dimensionality in large-scale tensors
    - We refer to the combinations of NNs and TNs as tensorial neural networks (TNNs)
    - Three primary aspects: network compression, information fusion, and quantum circuit simulation
    - Methods for improving TNNs, implementing TNNs, future directions

Xie, L., Wei, L., Zhang, X., Bi, K., Gu, X., Chang, J., & Tian, Q. (2023). Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models. arXiv, 2306.08641. Retrieved from https://arxiv.org/abs/2306.08641v1

Xu, Z., Wang, Y., Frei, S., Vardi, G., & Hu, W. (2023). Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data. arXiv, 2310.02541. Retrieved from https://arxiv.org/abs/2310.02541v1

Yang, G., & Littwin, E. (2023). Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit. arXiv, 2308.01814. Retrieved from https://arxiv.org/abs/2308.01814v2

Yang, G., Yu, D., Zhu, C., & Hayou, S. (2023). Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. arXiv, 2310.02244. Retrieved from https://arxiv.org/abs/2310.02244v5

Ye, J., Zhu, Z., Liu, F., Shokri, R., & Cevher, V. (2023). Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks. arXiv, 2310.20579. Retrieved from https://arxiv.org/abs/2310.20579v1

Zhao, M., Alver, S., van Seijen, H., Laroche, R., Precup, D., & Bengio, Y. (2023). Consciousness-Inspired Spatio-Temporal Abstractions for Better Generalization in Reinforcement Learning. arXiv, 2310.00229. Retrieved from https://arxiv.org/abs/2310.00229v3

Zheng, C., Wu, G., Bao, F., Cao, Y., Li, C., & Zhu, J. (2023). Revisiting Discriminative vs. Generative Classifiers: Theory and Implications. arXiv, 2302.02334. Retrieved from https://arxiv.org/abs/2302.02334v2

Zhu, L., Liu, C., Radhakrishnan, A., & Belkin, M. (2023). Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning. arXiv, 2306.04815. Retrieved from https://arxiv.org/abs/2306.04815v2

## 2024

Anthony, Q., Tokpanov, Y., Glorioso, P., & Millidge, B. (2024). BlackMamba: Mixture of Experts for State-Space Models. arXiv, 2402.01771. Retrieved from https://arxiv.org/abs/2402.01771v1

Guo, L., Ross, K., Zhao, Z., Andriopoulos, G., Ling, S., Xu, Y., & Dong, Z. (2024). Cross Entropy versus Label Smoothing: A Neural Collapse Perspective. arXiv, 2402.03979. Retrieved from https://arxiv.org/abs/2402.03979v2

Huang, Q., Wake, N., Sarkar, B., Durante, Z., Gong, R., Taori, R., ...Gao, J. (2024). Position Paper: Agent AI Towards a Holistic Intelligence. arXiv, 2403.00833. Retrieved from https://arxiv.org/abs/2403.00833v1

Humayun, A. I., Balestriero, R., & Baraniuk, R. (2024). Deep Networks Always Grok and Here is Why. arXiv, 2402.15555. Retrieved from https://arxiv.org/abs/2402.15555v1

Paolo, G., Gonzalez-Billandon, J., & Kégl, B. (2024). A call for embodied AI. arXiv, 2402.03824. Retrieved from https://arxiv.org/abs/2402.03824v1

Song, Y., Millidge, B., Salvatori, T., Lukasiewicz, T., Xu, Z., & Bogacz, R. (2024). Inferring neural activity before plasticity as a foundation for learning beyond backpropagation. Nat. Neurosci., 27, 348–358. Retrieved from https://www.nature.com/articles/s41593-023-01514-1

## OOD

Li, D., Yang, Y., Song, Y.-Z., & Hospedales, T. M. (2017). Deeper, Broader and Artier Domain Generalization. arXiv, 1710.03077. Retrieved from https://arxiv.org/abs/1710.03077v1

    - A problem is to learn domain-agnostic model from multiple training domains, and apply to unseen domain
    - Motivation: target domains may have sparse data for training
    - We develop a low-rank parameterized CNN model for end-to-end Domain Generalization
    - This model is based on Tucker decomposition to reduce number of parameters
    - Every weight tensor for a given domain is the sum of a domain specific tensor and a domain agnostic tensor
    - We develop a Domain Generalization benchmark covering photo, sketch, cartoon and painting domains
    - Our method outperforms existing DG alternatives

Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2018). Moment Matching for Multi-Source Domain Adaptation. arXiv, 1812.01754. Retrieved from https://arxiv.org/abs/1812.01754v4

    - We aim to transfer knowledge learned from multiple labeled source domains to an unlabeled target domain
    - We collect a DomainNet dataset with 6 domains, ~0.6 million images, 345 categories
    - We propose Moment Matching for Multi-Source Domain Adaptation (M3SDA)
    - It consists of three components: feature extractor, moment matching component, and classifiers
    - Moment matching component dynamically aligns moments of domains feature distributions (fig. 3)

Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv, 1907.02893. Retrieved from https://arxiv.org/abs/1907.02893v3

    - Problem: training data may contain spurious correlations (we do not expect them to hold in the future)
    - We assume that the training data is collected into distinct, separate environments
    - We want to learn correlations that are stable across training environments
    - We propose Invariant Risk Minimization (IRM) with the idea: to learn invariances across environments, find a data representation such that the optimal classifier on top of that representation matches for all environments
    - We pose the constrained optimization problem and simplify it into the practical version

Krueger, D., Caballero, E., Jacobsen, J.-H., Zhang, A., Binas, J., Zhang, D., ...Courville, A. (2020). Out-of-Distribution Generalization via Risk Extrapolation (REx). arXiv, 2003.00688. Retrieved from https://arxiv.org/abs/2003.00688v5

    - We point out that shifts at test time may be more extreme in magnitude than shifts between training domains
    - We formulate domain generalization (or OOD generalization) as optimizing the worst-case performance over a perturbation set of possible test domains
    - Our method minimax Risk Extrapolation is an extension of distributionally robust optimization (fig. 1)
    - It can uncover invariant relationships between X and Y (maintained across all domains)
    - We demonstrate that REx solves invariant prediction tasks where IRM fails due to covariate shift

Gulrajani, I., & Lopez-Paz, D. (2020). In Search of Lost Domain Generalization. arXiv, 2007.01434. Retrieved from https://arxiv.org/abs/2007.01434v1

    - We point at inconsistencies in experimental conditions for testing various Domain Generalization methods
    - We realize that model selection is non-trivial for domain generalization tasks
    - A domain generalization algorithm should be responsible for specifying a model selection method
    - A model selection policy should have no access to the test domain
    - We implement DOMAINBED, a testbed for domain generalization
    - It includes 7 multi-domain datasets, 9 baseline algorithms, and 3 model selection criteria
    - Carefully designed ERM shows SOTA performance across all datasets

Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., ...Schmidt, L. (2021). Robust fine-tuning of zero-shot models. arXiv, 2109.01903. Retrieved from https://arxiv.org/abs/2109.01903v3

    - Problem: fine-tuning CLIP or ALIGN reduce robustness to distribution shifts
    - We propose to ensemble the weights of the zero-shot and fine-tuned models (WiSE-FT)
    - We test it on a set of image distribution shifts
    - This comes at no additional computational cost during fine-tuning or inference

Arpit, D., Wang, H., Zhou, Y., & Xiong, C. (2021). Ensemble of Averages: Improving Model Selection and Boosting Performance in Domain Generalization. arXiv, 2110.10832. Retrieved from https://arxiv.org/abs/2110.10832v4

Cha, J., Chun, S., Lee, K., Cho, H.-C., Park, S., Lee, Y., & Park, S. (2021). SWAD: Domain Generalization by Seeking Flat Minima. arXiv, 2102.08604. Retrieved from https://arxiv.org/abs/2102.08604v4

Zhang, J., & Bottou, L. (2022). Learning useful representations for shifting tasks and distributions. arXiv, 2212.07346. Retrieved from https://arxiv.org/abs/2212.07346v3

Ramé, A., Kirchmeyer, M., Rahier, T., Rakotomamonjy, A., Gallinari, P., & Cord, M. (2022). Diverse Weight Averaging for Out-of-Distribution Generalization. arXiv, 2205.09739. Retrieved from https://arxiv.org/abs/2205.09739v2

Zhang, J., & Bottou, L. (2024). Fine-tuning with Very Large Dropout. arXiv, 2403.00946. Retrieved from https://arxiv.org/abs/2403.00946v1

    - It is known that nsemble techniques involving multiple data distributions gives richer representations 
    - We investigate the use of very high dropout rates instead of ensembles to obtain such rich representations
    - Training a DNN from scratch using such dropout rates is virtually impossible
    - However, fine-tuning under such conditions is possible
    - It achieves out-of-distribution performances that exceed those of both ensembles and weight averaging
    - We also provide interesting insights on representations and intrinsically linear nature of fine-tuning

## Continual LLM

Zhang, H., Gui, L., Zhai, Y., Wang, H., Lei, Y., & Xu, R. (2023). COPF: Continual Learning Human Preference through Optimal Policy Fitting. arXiv, 2310.15694. Retrieved from https://arxiv.org/abs/2310.15694v4

Sun, Y., Wang, S., Li, Y., Feng, S., Tian, H., Wu, H., & Wang, H. (2019). ERNIE 2.0: A Continual Pre-training Framework for Language Understanding. arXiv, 1907.12412. Retrieved from https://arxiv.org/abs/1907.12412v2

Jang, J., Ye, S., Yang, S., Shin, J., Han, J., Kim, G., ...Seo, M. (2021). Towards Continual Knowledge Learning of Language Models. arXiv, 2110.03215. Retrieved from https://arxiv.org/abs/2110.03215v4

Zhang, Z., Fang, M., Chen, L., Namazi-Rad, M.-R., & Wang, J. (2023). How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances. arXiv, 2310.07343. Retrieved from https://arxiv.org/abs/2310.07343v1

## Other TODO

Jelodar, H., Wang, Y., Yuan, C., Feng, X., Jiang, X., Li, Y., & Zhao, L. (2017). Latent Dirichlet Allocation (LDA) and Topic modeling: models, applications, a survey. arXiv, 1711.04305. Retrieved from https://arxiv.org/abs/1711.04305v2

Babu, G. J., Banks, D., Cho, H., Han, D., Sang, H., & Wang, S. (2021). A Statistician Teaches Deep Learning. arXiv, 2102.01194. Retrieved from https://arxiv.org/abs/2102.01194v2

Musgrave, K., Belongie, S., & Lim, S.-N. (2020). A Metric Learning Reality Check. arXiv, 2003.08505. Retrieved from https://arxiv.org/abs/2003.08505v3

Dudzik, A., & Veličković, P. (2022). Graph Neural Networks are Dynamic Programmers. arXiv, 2203.15544. Retrieved from https://arxiv.org/abs/2203.15544v3

Ibarz, B., Kurin, V., Papamakarios, G., Nikiforou, K., Bennani, M., Csordás, R., ...Veličković, P. (2022). A Generalist Neural Algorithmic Learner. arXiv, 2209.11142. Retrieved from https://arxiv.org/abs/2209.11142v2

Rodionov, G., & Prokhorenkova, L. (2023). Neural Algorithmic Reasoning Without Intermediate Supervision. arXiv, 2306.13411. Retrieved from https://arxiv.org/abs/2306.13411v2

Rodionov, G., & Prokhorenkova, L. (2024). Discrete Neural Algorithmic Reasoning. arXiv, 2402.11628. Retrieved from https://arxiv.org/abs/2402.11628v1

Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., ...Chen, W. (2024). ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. arXiv, 2403.03853. Retrieved from https://arxiv.org/abs/2403.03853v2
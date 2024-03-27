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

Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. Neural Comput., 7(1), 108–116. Retrieved from https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf

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

Williams, C. (1996). Computing with Infinite Networks. Advances in Neural Information Processing Systems, 9. Retrieved from https://papers.nips.cc/paper_files/paper/1996/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html

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

## 2000

Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. arXiv, physics/0004057. Retrieved from https://arxiv.org/abs/physics/0004057v1

    - To find which features of X play a role in the prediction, we may want to find a short code for X that preserves the maximum information about Y
    - We squeeze the information that X provides about Y through a "bottleneck" formed by a limited set of codewords
    - We derive equations and an iterative algorithm for finding representations of the signal that capture its relevant structure

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

Bottou, L. (2010). Large-Scale Machine Learning with Stochastic Gradient Descent. Proceedings of COMPSTAT'2010. Physica-Verlag HD. Retrieved from https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16

Dekel, O., Gilad-Bachrach, R., Shamir, O., & Xiao, L. (2010). Optimal Distributed Online Prediction using Mini-Batches. arXiv, 1012.1367. Retrieved from https://arxiv.org/abs/1012.1367v2

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

Zinkevich, M., Weimer, M., Li, L., & Smola, A. (2010). Parallelized Stochastic Gradient Descent. Advances in Neural Information Processing Systems, 23. Retrieved from https://papers.nips.cc/paper_files/paper/2010/hash/abea47ba24142ed16b7d8fbf2c740e0d-Abstract.html

    - We present the first parallel SGD algorithm that is perfectly suited to MapReduce settings

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

Daniely, A., Linial, N., & Shalev-Shwartz, S. (2013). From average case complexity to improper learning complexity. arXiv, 1311.2272. Retrieved from https://arxiv.org/abs/1311.2272v2

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

Pascanu, R., Montufar, G., & Bengio, Y. (2013). On the number of response regions of deep feed forward networks with piece-wise linear activations. arXiv, 1312.6098. Retrieved from https://arxiv.org/abs/1312.6098v5

Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv, 1312.6120. Retrieved from https://arxiv.org/abs/1312.6120v3

    - Consider a DNN without activations
    - Despite the linearity, they have nonlinear GD dynamics, long plateaus followed by rapid transitions
    - Discussing some initial conditions on the weights that emerge during unsupervised pretraining
    - Propose a dynamical isometry (all singular values of the Jacobian concentrate near 1)
    - Propose orthonormal initialization
    - Faithful gradient propagation occurs in a special regime known as the edge of chaos

Sutskever, I., Martens, J., Dahl, G. E., & Hinton, G. E. (2013). On the importance of initialization and momentum in deep learning. International Conference on Machine Learning. Retrieved from https://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf

    - We show that SGD with momentum can be successfully applied to train DNNs
    - We use a well-designed random initialization and a particular type of slowly increasing schedule
    - Poorly initialized networks cannot be trained with momentum
    - Well-initialized networks perform worse when the momentum is absent or poorly tuned

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
    - We propose the saddle-free Newton method, that can rapidly escape high dimensional saddle points
    - It has superior optimization performance for RNNs

Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation. arXiv, 1410.8516. Retrieved from https://arxiv.org/abs/1410.8516v6

    - We think that a good representation is one in which the data has a distribution that is easy to model
    - We propose Non-linear Independent Component Estimation (NICE) for modeling complex high-dimensional densities
    - It learns factorized latent distribution, i.e. independent latent variables
    - This approach yields good generative models on four image datasets and can be used for inpainting

Friston, K. J., Sengupta, B., & Auletta, G. (2014). Cognitive Dynamics: From Attractors to Active Inference. Proc. IEEE. Retrieved from https://www.semanticscholar.org/paper/Cognitive-Dynamics%3A-From-Attractors-to-Active-Friston-Sengupta/e3069e95026378d344e22766adac10490b053078

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ...Bengio, Y. (2014). Generative Adversarial Networks. arXiv, 1406.2661. Retrieved from https://arxiv.org/abs/1406.2661v1

Goodfellow, I. J., Vinyals, O., & Saxe, A. M. (2014). Qualitatively characterizing neural network optimization problems. arXiv, 1412.6544. Retrieved from https://arxiv.org/abs/1412.6544v6

    - We analyze why model NNs are overcoming local optima
    - We find that on a straight path from initialization to solution, a variety of SOTA NNs never encounter any significant obstacles

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv, 1410.5401. Retrieved from https://arxiv.org/abs/1410.5401v2

Kingma, D. P., & Welling, M. (2014). Efficient Gradient-Based Inference through Transformations between Bayes Nets and Neural Nets. arXiv, 1402.0480. Retrieved from https://arxiv.org/abs/1402.0480v5

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv, 1412.6980. Retrieved from https://arxiv.org/abs/1412.6980v9

    - We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objectives
    - It is scalable and invariant to diagonal rescaling of the gradients
    - It is appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients
    - We provide a theoretical analysis of Adam’s convergence in online convex programming
    - We discuss AdaMax, a variant of Adam based on the infinity norm

Kingma, D. P., Rezende, D. J., Mohamed, S., & Welling, M. (2014). Semi-Supervised Learning with Deep Generative Models. arXiv, 1406.5298. Retrieved from https://arxiv.org/abs/1406.5298v2

Kistler, N. (2014). Derrida's random energy models. From spin glasses to the extremes of correlated random fields. arXiv, 1412.0958. Retrieved from https://arxiv.org/abs/1412.0958v1

    - This is notes for a mini-course on the extremes of correlated random fields
    - Derrida's random energy models are described
    - Gaussian hierarchical field are described

Koutník, J., Greff, K., Gomez, F., & Schmidhuber, J. (2014). A Clockwork RNN. arXiv, 1402.3511. Retrieved from https://arxiv.org/abs/1402.3511v1

Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2014). Random feedback weights support learning in deep neural networks. arXiv, 1411.0247. Retrieved from https://arxiv.org/abs/1411.0247v1

    - Backpropagation requires biologically implausible transport of individual synaptic weight information
    - We find that a network can learn with random feedback connections instead of backpropagation
    - This means replacing W^T by a matrix of fixed random weights
    - We call this feedback alignment
    - This new mechanism performs as quickly and accurately as backpropagation on a variety of problems

Maddison, C. J., Tarlow, D., & Minka, T. (2014). A* Sampling. arXiv, 1411.0030. Retrieved from https://arxiv.org/abs/1411.0030v2

Maeda, S.-i. (2014). A Bayesian encourages dropout. arXiv, 1412.7003. Retrieved from https://arxiv.org/abs/1412.7003v3

    - We provide a Bayesian interpretation to dropout
    - The inference after dropout training can be considered as an approximate inference by Bayesian model averaging
    - This view enables us to optimize the dropout rate (Bayesian dropout)

Montúfar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the Number of Linear Regions of Deep Neural Networks. arXiv, 1402.1869. Retrieved from https://arxiv.org/abs/1402.1869v2

    - We study the advantage of depth for neural networks with piecewise linear activation functions
    - We calculate the maximal number of linear regions of the functions computed by a neural network
    - The depth yields an exponential number of input regions mapped to the same output

Neyshabur, B., Tomioka, R., & Srebro, N. (2014). In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning. arXiv, 1412.6614. Retrieved from https://arxiv.org/abs/1412.6614v4

    - We tried to force neural net overfitting by adding random label noise to the data
    - We show on single-hidden-layer networks that size does not behave as a capacity control parameter
    - We suggest that optimization is introducing some implicit regularization by trying to find a solution with small "complexity", for some notion of complexity, perhaps norm
    - We demonstrate how implicit L2 weight decay in an infinite two-layer network gives rise to a "convex neural net" with L1 (not L2) regularization in the top layer

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

    - We propose a Gaussian Dropout that multiplies the outputs of the neurons by Gaussian random noise
    - We show that dropout improves the performance of NNss on supervised learning, tasks in vision, speech recognition, document classification and computational biology

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

Choromanska, A., LeCun, Y., & Arous, G. B. (2015). Open Problem: The landscape of the loss surfaces of multilayer networks. Conference on Learning Theory. PMLR. Retrieved from https://proceedings.mlr.press/v40/Choromanska15.html

    - We pose an open problem: is it possible to establish a connection between the loss function of the neural networks and the Hamiltonian of the spherical spin-glass models under realistic assumptions?
    - The central problem is to eliminate unrealistic assumptions of variable independence

Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2015). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). arXiv, 1511.07289. Retrieved from https://arxiv.org/abs/1511.07289v5

Courbariaux, M., Bengio, Y., & David, J.-P. (2015). BinaryConnect: Training Deep Neural Networks with binary weights during propagations. arXiv, 1511.00363. Retrieved from https://arxiv.org/abs/1511.00363v3

Figurnov, M., Ibraimova, A., Vetrov, D., & Kohli, P. (2015). PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions. arXiv, 1504.08362. Retrieved from https://arxiv.org/abs/1504.08362v4

Friston, K. J., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T. H. B., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive neuroscience. Retrieved from https://www.semanticscholar.org/paper/Active-inference-and-epistemic-value-Friston-Rigoli/57620e357ee348cd5ffa8eafa480e002a2aba06a

Friston, K. J., Levin, M., Sengupta, B., & Pezzulo, G. (2015). Knowing one's place: a free-energy approach to pattern regulation. J. R. Soc. Interface. Retrieved from https://www.semanticscholar.org/paper/Knowing-one's-place%3A-a-free-energy-approach-to-Friston-Levin/2c13294ccc0045d24fe0ae01a5ff6dd21d0566d1

Gal, Y., & Ghahramani, Z. (2015). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. arXiv, 1506.02142. Retrieved from https://arxiv.org/abs/1506.02142v6

    - We show that dropout training in DNNs is an approximate Bayesian inference in deep Gaussian processes
    - We propose Monte Carlo dropout: N stochastic forward passes with averaging
    - This allows to estimate the model uncertainty
    - We show that model uncertainty is indispensable for classification tasks
    - We also discuss model uncertainty in RL

Ge, R., Huang, F., Jin, C., & Yuan, Y. (2015). Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition. arXiv, 1503.02101. Retrieved from https://arxiv.org/abs/1503.02101v1

    - We consider non-convex functions with exponentially many local minima and saddle points, for example, orthogonal tensor decomposition problem, that is the key step in spectral learning for many latent variable models
    - We identify a property of non-convex functions which we call "strict saddle"
    - Intuitively, it guarantees local progress if we have access to the Hessian information
    - With strict saddle property SGD converges to a local minimum in a polynomial number of iterations

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv, 1512.03385. Retrieved from https://arxiv.org/abs/1512.03385v1

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv, 1502.01852. Retrieved from https://arxiv.org/abs/1502.01852v1

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv, 1503.02531. Retrieved from https://arxiv.org/abs/1503.02531v1

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv, 1502.03167. Retrieved from https://arxiv.org/abs/1502.03167v3

Joulin, A., & Mikolov, T. (2015). Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets. arXiv, 1503.01007. Retrieved from https://arxiv.org/abs/1503.01007v4

Kaiser, Ł., & Sutskever, I. (2015). Neural GPUs Learn Algorithms. arXiv, 1511.08228. Retrieved from https://arxiv.org/abs/1511.08228v3

Lee, T. S. (2015). The visual system's internal model of the world. Proc. IEEE Inst. Electr. Electron. Eng., 26566294. Retrieved from https://pubmed.ncbi.nlm.nih.gov/26566294

Li, Y., Yosinski, J., Clune, J., Lipson, H., & Hopcroft, J. (2015). Convergent Learning: Do different neural networks learn the same representations? arXiv, 1511.07543. Retrieved from https://arxiv.org/abs/1511.07543v3

Mishkin, D., & Matas, J. (2015). All you need is a good init. arXiv, 1511.06422. Retrieved from https://arxiv.org/abs/1511.06422v7

    - We propose a layer-sequential unit-variance (LSUV) initialization:
    - 1) use orthonormal matrices
    - 2) sequentially normalize the variance of the output of each layer to be equal to one

Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. (2015). Tensorizing Neural Networks. arXiv, 1509.06569. Retrieved from https://arxiv.org/abs/1509.06569v2

    - We convert FCN weight matrices to Tensor Train format (we call it TT-layer, TensorNet)
    - Number of parameters is reduced by a huge factor (up to 200000 times for VGG dense layers)
    - The expressive power is preserved

Peters, J., Bühlmann, P., & Meinshausen, N. (2015). Causal inference using invariant prediction: identification and confidence intervals. arXiv, 1501.01332. Retrieved from https://arxiv.org/abs/1501.01332v3

Pezzulo, G., Rigoli, F., & Friston, K. J. (2015). Active Inference, homeostatic regulation and adaptive behavioural control. Prog. Neurobiol. Retrieved from https://www.semanticscholar.org/paper/Active-Inference%2C-homeostatic-regulation-and-Pezzulo-Rigoli/c469f8a7f02015bc49e93df26c396228267a7e7b

Smith, L. N. (2015). Cyclical Learning Rates for Training Neural Networks. arXiv, 1506.01186. Retrieved from https://arxiv.org/abs/1506.01186v6

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. arXiv, 1507.06228. Retrieved from https://arxiv.org/abs/1507.06228v2

    - We propose Highway networks inspired by LSTM
    - They use adaptive gating units to regulate the information flow, we call such paths "information highways"
    - Even with hundreds of layers, highway networks can be trained directly through SGD

Tishby, N., & Zaslavsky, N. (2015). Deep Learning and the Information Bottleneck Principle. arXiv, 1503.02406. Retrieved from https://arxiv.org/abs/1503.02406v1

    - We suggest a novel information theoretic analysis of DNNs based on the information bottleneck (IB) principle
    - DNN can be quantified by the mutual information between the layers and the input and output variables
    - The goal of the network is to optimize the IB tradeoff between compression and prediction
    - We view DNNs as an successive (Markovian) relevant compression of the input
    - We discuss advantages of this new view

Wiatowski, T., & Bölcskei, H. (2015). A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction. arXiv, 1512.06293. Retrieved from https://arxiv.org/abs/1512.06293v3

Wu, H., & Gu, X. (2015). Towards Dropout Training for Convolutional Neural Networks. arXiv, 1512.00242. Retrieved from https://arxiv.org/abs/1512.00242v1

    - For CNN, dropout is known to work well in fully-connected layers
    - However, its effect in convolutional and pooling layers is still not clear
    - We draw a connection between dropout before max-pooling and sampling from multinomial distribution
    - We propose probabilistic weighted pooling and achieve competitive results
    - We found that the effect of convolutional dropout is not trivial

Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., ...Bengio, Y. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. arXiv, 1502.03044. Retrieved from https://arxiv.org/abs/1502.03044v3

    - We introduce an attention based model for image captioning with either soft or hard attention
    - Soft deterministic attention is trainable by SGD
    - Hard stochastic attention is trainable by maximizing an approximate ELBO or equivalently by REINFORCE
    - We achieve SOTA on Flickr8k, Flickr30k and MS COCO

## 2016

Abdi, M., & Nahavandi, S. (2016). Multi-Residual Networks: Improving the Speed and Accuracy of Residual Networks. arXiv, 1609.05672. Retrieved from https://arxiv.org/abs/1609.05672v4

Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2016). Deep Variational Information Bottleneck. arXiv, 1612.00410. Retrieved from https://arxiv.org/abs/1612.00410v7

    - We propose Deep Variational Information Bottleneck (Deep VIB): a variational approximation to the information bottleneck that allows us to parameterize the information bottleneck model with a NN and efficiently train it
    - VIB objective outperform other forms of regularization in terms of generalization performance and robustness to adversarial attack
    - We describe a connection to VAE

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv, 1607.06450. Retrieved from https://arxiv.org/abs/1607.06450v1

    - Problem: batch normalization is dependent on batch size and is not trivial to apply to RNNs
    - We propose a Layer Normalization that works independently for every training case
    - It performs the same computation at training and test times, and is also straightforward to apply to RNNs
    - Like in BN, we also use scale and shift, after the normalization but before the non-linearity
    - Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks

Baldassi, C., Gerace, F., Lucibello, C., Saglietti, L., & Zecchina, R. (2016). Learning may need only a few bits of synaptic precision. arXiv, 1602.04129. Retrieved from https://arxiv.org/abs/1602.04129v2

Bottou, L., Curtis, F. E., & Nocedal, J. (2016). Optimization Methods for Large-Scale Machine Learning. arXiv, 1606.04838. Retrieved from https://arxiv.org/abs/1606.04838v3

    - In this review we discuss the past, present, and future of numerical optimization in ML applications
    - We discuss two main streams of research:
    - 1) techniques that diminish noise in the stochastic directions
    - 2) methods that make use of second-order derivative approximations

Bruineberg, J., Kiverstein, J., & Rietveld, E. (2016). The anticipating brain is not a scientist: the free-energy principle from an ecological-enactive perspective. Synthese. Retrieved from https://www.semanticscholar.org/paper/The-anticipating-brain-is-not-a-scientist%3A-the-from-Bruineberg-Kiverstein/d923112dbf3c46d792b9d1172dd8fa69a68e3386

Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., ...Zecchina, R. (2016). Entropy-SGD: Biasing Gradient Descent Into Wide Valleys. arXiv, 1611.01838. Retrieved from https://arxiv.org/abs/1611.01838v5

    - Well-generalizable solutions lie in large flat regions with almost-zero eigenvalues in the Hessian
    - We propose Entropy-SGD optimizer that favors such solutions
    - Our algorithm resembles two nested loops of SGD
    - We use Langevin dynamics in the inner loop to compute the gradient of the local entropy
    - The new objective has a smoother energy landscape
    - Entropy-SGD obtains is comparable to competitive baselines and  gets a 2x speed-up over SGD

Chen, X., Kingma, D. P., Salimans, T., Duan, Y., Dhariwal, P., Schulman, J., ...Abbeel, P. (2016). Variational Lossy Autoencoder. arXiv, 1611.02731. Retrieved from https://arxiv.org/abs/1611.02731v2

Cichocki, A., Lee, N., Oseledets, I. V., Phan, A.-H., Zhao, Q., & Mandic, D. (2016). Low-Rank Tensor Networks for Dimensionality Reduction and Large-Scale Optimization Problems: Perspectives and Challenges PART 1. arXiv, 1609.00893. Retrieved from https://arxiv.org/abs/1609.00893v3

    - A book (pt. 1) about Tucker and Tensor Train (TT) decompositions and their extensions or generalizations
    - This can be used to convert intractable huge-scale optimization problems into a set of smaller problems
    - Chapter 1: Introduction and Motivation
    - Chapter 2: Tensor Operations and Tensor Network Diagrams
    - Chapter 3: Constrained Tensor Decompositions: From Two-way to Multiway Component Analysis
    - Chapter 4: Tensor Train Decompositions: Graphical Interpretations and Algorithms
    - Chapter 5: Discussion and Conclusions

Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv, 1602.02830. Retrieved from https://arxiv.org/abs/1602.02830v3

Daniely, A., Frostig, R., & Singer, Y. (2016). Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity. Advances in Neural Information Processing Systems, 29. Retrieved from https://proceedings.neurips.cc/paper_files/paper/2016/hash/abea47ba24142ed16b7d8fbf2c740e0d-Abstract.html

    - We introduce the notion of a computation skeleton, an acyclic graph that can describe NN computations
    - A skeleton defines a class H of functions obtained from the skeleton’s structure
    - We analyze the set of functions that can be expressed by varying the weights of the last layer
    - (in this case the objective is convex)
    - We show that with high probability over the choice of initial network weights, any function in H can be approximated by selecting the final layer’s weights
    - It follows that random weight initialization often yields a favorable starting point for optimization

Duvenaud, D., Maclaurin, D., & Adams, R. (2016). Early Stopping as Nonparametric Variational Inference. Artificial Intelligence and Statistics. PMLR. Retrieved from https://proceedings.mlr.press/v51/duvenaud16.html

    - We propose a Bayesian interpretation of SGD
    - Unconverged SGD yields a sequence of distributions which are variational approximations to the true posterior
    - This allows us to estimate a lower bound on the marginal likelihood of any model, even very large
    - This can be used for hyperparameter selection and early stopping without a validation set
    - The results are promising, but further refinements are likely to be necessary

Edwards, H., & Storkey, A. (2016). Towards a Neural Statistician. arXiv, 1606.02185. Retrieved from https://arxiv.org/abs/1606.02185v2

    - We take seriously the idea of working with datasets, rather than datapoints, as the key objects to model
    - We demonstrate an extension to VAE called a neural statistician
    - It is trained to produce statistics that encapsulate a generative model for each dataset
    - It unsupervisedly learns representations, or statistics, of datasets
    - This enables efficient few-shot learning from new datasets

Freeman, C. D., & Bruna, J. (2016). Topology and Geometry of Half-Rectified Network Optimization. arXiv, 1611.01540. Retrieved from https://arxiv.org/abs/1611.01540v4

    - We prove that single layer ReLU networks are asymptotically connected
    - We show that level sets remain connected throughout all the learning phase, suggesting a near convex behavior, but they become exponentially more curvy as the energy level decays

Friston, K. J., FitzGerald, T., Rigoli, F., Schwartenbeck, P., O'Doherty, J., & Pezzulo, G. (2016). Active inference and learning. Neurosci. Biobehav. Rev. Retrieved from https://www.semanticscholar.org/paper/Active-inference-and-learning-Friston-FitzGerald/3b3903f7914483e21576f9d098e611deef95ec45

Friston, K. J. (2016). I am therefore I think. Retrieved from https://www.semanticscholar.org/paper/I-am-therefore-I-think-Friston/2d450521f168fccbc3cd13112ca07159c7f1bd50

Hardt, M., & Ma, T. (2016). Identity Matters in Deep Learning. arXiv, 1611.04231. Retrieved from https://arxiv.org/abs/1611.04231v3

    - We give a strikingly simple proof that arbitrarily deep LINEAR ResNets have no spurious local optima
    - (the same result for standard linear DNNs is substantially more delicate)
    - We show that ResNets with ReLU can express a dataset of size N with R classes given O(n log n + r^2) parameters
    - We show SOTA on all-convolutional networks (no batch norm, dropout, or max pool) on CIFAR10, CIFAR100, ImageNet

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ...Lerchner, A. (2016). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. OpenReview. Retrieved from https://openreview.net/pdf?id=Sy2fzU9gl

    - We propose beta-VAE, a modification of VAE that relies on tuning a single hyperparameter beta
    - Beta can be tuned using weakly labelled data or through heuristic visual inspection
    - Beta-VAE reduces to VAE if beta = 1
    - beta-VAE allows to discover interpretable factorised latent representations from images without supervision
    - It achieves SOTA on disentangled factor learning on a variety of datasets (celebA, faces and chairs)
    - We devise a protocol to quantitatively compare the degree of disentanglement learnt by different models

Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. (2016). Deep Networks with Stochastic Depth. arXiv, 1603.09382. Retrieved from https://arxiv.org/abs/1603.09382v3

Hohwy, J. (2016). The self-evidencing brain. Noûs. Retrieved from https://www.semanticscholar.org/paper/The-self-evidencing-brain-Hohwy/01aa6ef498431fb8c6d45e9375bb39a7a923b9bb

Im, D. J., Tao, M., & Branson, K. (2016). An empirical analysis of the optimization of deep network loss surfaces. arXiv, 1612.04010. Retrieved from https://arxiv.org/abs/1612.04010v4

    - We visualize the loss function by projecting them down to low-dimensional spaces
    - We show that optimization algorithms encounter and choose different descent directions at many saddle points
    - We hypothesize that each optimization algorithm makes characteristic choices at these saddle points

Jang, E., Gu, S., & Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax. arXiv, 1611.01144. Retrieved from https://arxiv.org/abs/1611.01144v5

    - Stochastic NNs rarely use categorical latent variables due to the inability to backpropagate through samples
    - We introduce Gumbel-Softmax, a continuous distribution on the simplex that can approximate categorical samples
    - Its gradients can be easily computed via the reparameterization trick
    - It can be used to efficiently train semi-supervised models

Kaiser, Ł., & Bengio, S. (2016). Can Active Memory Replace Attention? arXiv, 1610.08613. Retrieved from https://arxiv.org/abs/1610.08613v2

Kawaguchi, K. (2016). Deep Learning without Poor Local Minima. arXiv, 1605.07110. Retrieved from https://arxiv.org/abs/1605.07110v3

    - We prove the following statements for linear DNNs  with squared loss function:
    - 1) the function is non-convex and non-concave
    - 2) there are only global minima and saddle points, no local minima
    - 3) there exist "bad" saddle points (where the Hessian has no negative eigenvalue) for deep networks
    - The bad local minima would arise in a deep nonlinear model but only as an effect of adding nonlinearities

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. arXiv, 1609.04836. Retrieved from https://arxiv.org/abs/1609.04836v2

    -  It has been observed that large batch size leads to degradation in the quality of the model
    -  We show that large-batch methods tend to converge to sharp minima with poor generalization
    -  Small-batch methods consistently converge to flat minima due to the inherent noise in the gradient estimation
    -  We discuss strategies to improve large-batch training

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ...Hadsell, R. (2016). Overcoming catastrophic forgetting in neural networks. arXiv, 1612.00796. Retrieved from https://arxiv.org/abs/1612.00796v2

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2016). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. arXiv, 1612.01474. Retrieved from https://arxiv.org/abs/1612.01474v3   
  
    - We propose non-Bayesian deep ensembles  
    - They are an alternative to Bayesian NNs (e.g. variational inference or MCMC methods)  
    - They are simple, parallelizable and yields high quality  
    - They produce well-calibrated uncertainty estimates  
    - They express higher uncertainty on OOD examples

Li, S., Jiao, J., Han, Y., & Weissman, T. (2016). Demystifying ResNet. arXiv, 1611.01186. Retrieved from https://arxiv.org/abs/1611.01186v2

    - It was empirically observed that shortcuts that have depth 2 results in smaller training error, while it is not true for shortcut of depth 1 or 3
    - We prove that shortcuts that have depth 2 yields depth-invariant condition number of the Hessian
    - Shortcuts of higher depth result in an extremely flat (high-order) stationary point initially
    - The shortcut 1 has a condition number exploding to infinity as the number of layers grows
    - We experimentally show that initializing the network to small weights with shortcut 2 achieves significantly better results than random Gaussian Xavier initialization, orthogonal initialization, and shortcuts of deeper depth

Liao, Q., & Poggio, T. (2016). Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex. arXiv, 1604.03640. Retrieved from https://arxiv.org/abs/1604.03640v2

Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv, 1608.03983. Retrieved from https://arxiv.org/abs/1608.03983v5

    - We propose a simple warm restart technique for SGD
    - With this technique we demonstate SOTA on CIFAR-10 and CIFAR-100 datasets
    - Future work should consider warm restarts for other algorithms such as Adam

Novikov, A., Trofimov, M., & Oseledets, I. (2016). Exponential Machines. arXiv, 1605.03795. Retrieved from https://arxiv.org/abs/1605.03795v3

    - Exponential Machines (ExM), a predictor that models all interactions of every order
    - The Tensor Train format regularizes an exponentially large tensor of parameters
    - SOTA performance on synthetic data with high-order interactions

Parikh, A. P., Täckström, O., Das, D., & Uszkoreit, J. (2016). A Decomposable Attention Model for Natural Language Inference. arXiv, 1606.01933. Retrieved from https://arxiv.org/abs/1606.01933v2

Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., & Ganguli, S. (2016). Exponential expressivity in deep neural networks through transient chaos. arXiv, 1606.05340. Retrieved from https://arxiv.org/abs/1606.05340v2

Rolfe, J. T. (2016). Discrete Variational Autoencoders. arXiv, 1609.02200. Retrieved from https://arxiv.org/abs/1609.02200v2

Sagun, L., Bottou, L., & LeCun, Y. (2016). Eigenvalues of the Hessian in Deep Learning: Singularity and Beyond. arXiv, 1611.07476. Retrieved from https://arxiv.org/abs/1611.07476v2

    - We show that the eigenvalue distribution in NNs is seen to be composed of two parts:
    - 1) the bulk which is concentrated around zero and indicating how overparametrized the system is
    - 2) the edges which are scattered away from zero and depend on the input data
    - (this paper seems to be an early version of https://arxiv.org/abs/1706.04454v3)

Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. arXiv, 1602.07868. Retrieved from https://arxiv.org/abs/1602.07868v3

    - a weight that decouples the length of vectors from their direction
    - we improve the conditioning of the optimization problem and we speed up convergence of SGD
    - is inspired by batch normalization but is more widely applicable
    - useful in supervised image recognition, generative modelling, and deep RL

Scellier, B., & Bengio, Y. (2016). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. arXiv, 1602.05179. Retrieved from https://arxiv.org/abs/1602.05179v5

    - We introduce Equilibrium Propagation, a learning framework for energy-based models
    - It involves computing the gradient of an objective, but is different from backpropagation
    - The first phase is when the prediction is made, and the second phase is when the target is revealed
    - The only local difference between the two phases is whether synaptic changes are allowed or not

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

    - We conduct a series of experiments that contrast the learning dynamics of DNNs on real vs. noise data
    - DNNs learn simple patterns first, before memorizing, taking advantage of patterns shared by multiple examples
    - For appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data

Balan, R., Singh, M., & Zou, D. (2017). Lipschitz Properties for Deep Convolutional Networks. arXiv, 1701.05217. Retrieved from https://arxiv.org/abs/1701.05217v1

    - We hope to see a small change in the feature vector with respect to a deformation on the input signal
    - The key step is to derive the Lipschitz properties, tt is desired to have a formula for the Lipschitz bound
    - We compare different methods for computing the Lipschitz constants

Bartlett, P., Foster, D. J., & Telgarsky, M. (2017). Spectrally-normalized margin bounds for neural networks. arXiv, 1706.08498. Retrieved from https://arxiv.org/abs/1706.08498v2

    - We calculate DNN's Lipschitz constant: the product of the spectral norms of the weight matrices
    - It highly correlates with a generalization bound (difference between train and test scores?)
    - We experiment on data with original or random labels
    - SGD selects predictors whose complexity scales with the difficulty of the learning task

Bengio, Y. (2017). The Consciousness Prior. arXiv, 1709.08568. Retrieved from https://arxiv.org/abs/1709.08568v2

Bojarski, M., Yeres, P., Choromanska, A., Choromanski, K., Firner, B., Jackel, L., & Muller, U. (2017). Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car. arXiv, 1704.07911. Retrieved from https://arxiv.org/abs/1704.07911v1

Brutzkus, A., Globerson, A., Malach, E., & Shalev-Shwartz, S. (2017). SGD Learns Over-parameterized Networks that Provably Generalize on Linearly Separable Data. arXiv, 1710.10174. Retrieved from https://arxiv.org/abs/1710.10174v1

    - Current generalization bounds for NNs fail to explain good generalization in the over-parameterized regime
    - We study two-layer NN with Leaky ReLU trained with SGD on linearly separable data
    - Will a large network will overfit in such a case or not?
    - SGD both finds a global minimum, and avoids overfitting despite the high capacity
    - For this case, we prove convergence rates of SGD to a global minimum independent of the network size
    - This is the first theoretical demonstration that SGD can avoid overfitting

Chang, B., Meng, L., Haber, E., Ruthotto, L., Begert, D., & Holtham, E. (2017). Reversible Architectures for Arbitrarily Deep Residual Neural Networks. arXiv, 1709.03698. Retrieved from https://arxiv.org/abs/1709.03698v2

Cichocki, A., Phan, A.-H., Zhao, Q., Lee, N., Oseledets, I. V., Sugiyama, M., & Mandic, D. (2017). Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations. Part 2 Applications and Future Perspectives. arXiv, 1708.09165. Retrieved from https://arxiv.org/abs/1708.09165v1

    -  A book (pt. 2) about tensor network models for super-compressed representation of data/parameters
    -  Emphasis is on the tensor train (TT) and Hierarchical Tucker (HT) decompositions
    -  Applied areas: regression and classification, eigenvalue decomposition, Riemannian optimization, DNNs
    -  Part 1 and Part 2 of this work can be used either as stand-alone separate texts

Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. arXiv, 1703.04933. Retrieved from https://arxiv.org/abs/1703.04933v2

    - We argue that most notions of flatness are problematic for deep models to explain generalization
    - Given ReLU DNN with flat minima, we can build equivalent model corresponding to arbitrarily sharper minima using the inherent symmetries of the architecture
    - If we allow to reparametrize a function, the geometry of its parameters can change drastically without affecting its generalization properties

Du, S. S., Lee, J. D., Tian, Y., Poczos, B., & Singh, A. (2017). Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima. arXiv, 1712.00779. Retrieved from https://arxiv.org/abs/1712.00779v2

    - We consider 1-layer CNN with ReLU and take the same network with fixed weights as target function
    - We prove that with Gaussian input, there is a spurious local minimizer
    - GD with weight normalization converges to global minimum or to bad local minimum
    - We see that GD dynamics has two phases: it starts off slow, but converges much faster after several iterations

George, D., Lehrach, W., Kansky, K., Lázaro-Gredilla, M., Laan, C., Marthi, B., ...Phoenix, D. (2017). A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs. Science. Retrieved from https://www.cs.jhu.edu/~ayuille/JHUcourses/ProbabilisticModelsOfVisualCognition2019/Lec23/nov19lecture/GeorgeCAPCHAS.pdf

    - We introduce a hierarchical probabilistic generative model called the Recursive Cortical Network (RCN)
    - RCN incorporates message-passing based inference based on neuroscience insights
    - RCN handles recognition, segmentation and reasoning
    - To parse a scene, RCN maintains hierarchical graphs for object instances at multiple locations
    - A backward pass can reject many object hypotheses that were falsely identified in the forward pass
    - It outperforms DNNs on a scene text recognition benchmark while being 300-fold more data efficient
    - It breaks the defense of modern text-based CAPTCHAs

Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. arXiv, 1707.04585. Retrieved from https://arxiv.org/abs/1707.04585v1

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., ...He, K. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv, 1706.02677. Retrieved from https://arxiv.org/abs/1706.02677v2

    - We adopt a linear scaling rule for adjusting learning rates as a function of minibatch size
    - We develop a new warmup scheme that overcomes optimization challenges early in training
    - Our implementation achieves ∼90% scaling efficiency when moving from 8 to 256 GPUs

Guerguiev, J., Lillicrap, T. P., & Richards, B. A. (2017). Towards deep learning with segregated dendrites. eLife. Retrieved from https://elifesciences.org/articles/22901

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. arXiv, 1706.04599. Retrieved from https://arxiv.org/abs/1706.04599v2

Hoffer, E., Hubara, I., & Soudry, D. (2017). Train longer, generalize better: closing the generalization gap in large batch training of neural networks. arXiv, 1705.08741. Retrieved from https://arxiv.org/abs/1705.08741v2

Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). Snapshot Ensembles: Train 1, get M for free. arXiv, 1704.00109. Retrieved from https://arxiv.org/abs/1704.00109v1

    - We train a single neural network, converging to several local minima along its optimization path
    - We leverage recent work on cyclic learning rate schedules to obtain repeated rapid convergence
    - The resulting technique, Snapshot Ensembling, compares favorably with traditional network ensembles

Huang, F., Ash, J., Langford, J., & Schapire, R. (2017). Learning Deep ResNet Blocks Sequentially using Boosting Theory. arXiv, 1706.04964. Retrieved from https://arxiv.org/abs/1706.04964v4

Ioffe, S. (2017). Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. arXiv, 1702.03275. Retrieved from https://arxiv.org/abs/1702.03275v2

Jain, P., Kakade, S. M., Kidambi, R., Netrapalli, P., & Sidford, A. (2017). Accelerating Stochastic Gradient Descent For Least Squares Regression. arXiv, 1704.08227. Retrieved from https://arxiv.org/abs/1704.08227v2

    - We discuss least squares regression as a special case of stochastic approximation where we have access to a stochastic first order oracle (stochastic gradient)
    - We rethink what acceleration (momentum?) has to offer when working with a stochastic gradient
    - We propose an accelerated stochastic gradient method (ASGD)
    - This paper presents the first known provable analysis of the claim that fast gradient methods are stable when dealing with statistical errors, in contrast to previous negative results

Jastrzębski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2017). Three Factors Influencing Minima in SGD. arXiv, 1711.04623. Retrieved from https://arxiv.org/abs/1711.04623v3

    - We investigate the previously proposed approximation of SGD by a stochastic differential equation (SDE)
    - Higher values of the ratio of learning rate to batch size lead to wider minima and often better generalization
    - We study this theoretically and confirm experimentally
    - Learning rate schedules can be replaced with batch size schedules

Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., & Uszkoreit, J. (2017). One Model To Learn Them All. arXiv, 1706.05137. Retrieved from https://arxiv.org/abs/1706.05137v1

Kaiser, Ł., Nachum, O., Roy, A., & Bengio, S. (2017). Learning to Remember Rare Events. arXiv, 1703.03129. Retrieved from https://arxiv.org/abs/1703.03129v1

    - How to make memory-augmented DNNs better at life-long and one-shot learning of rare events?
    - We present a large-scale life-long memory module that exploits fast nearest-neighbor algorithms
    - It provide the ability to remember and do life-long one-shot learning: it can remember training examples shown many thousands of steps in the past and it can successfully generalize from them
    - We try it with RNNs and image classification CNNs
    - We set a new SOTA for one-shot learning on the Omniglot dataset

Khrulkov, V., & Oseledets, I. (2017). Art of singular vectors and universal adversarial perturbations. arXiv, 1709.03582. Retrieved from https://arxiv.org/abs/1709.03582v2

    -  We propose a new algorithm for constructing adversarial perturbations
    -  To do this we compute (p, q)-singular vectors of the Jacobian matrices of hidden layers of a network

Khrulkov, V., Novikov, A., & Oseledets, I. (2017). Expressive power of recurrent neural networks. arXiv, 1711.00811. Retrieved from https://arxiv.org/abs/1711.00811v2

    - As known, deep Hierarchical Tucker CNNs have exponentially higher expressive power than shallow networks
    - We prove the same for RNNs with Tensor Train (TT) decomposition
    - We compare expressive powers of the HT- and TT-Networks
    - We implement the recurrent TT-Networks and provide numerical evidence of their expressivity

Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. arXiv, 1706.02515. Retrieved from https://arxiv.org/abs/1706.02515v5

    - We propose self-normalizing neural networks (SNNs) with SELU activation function
    - Given some (not very realistic) assumptions about input independence we prove that in SNNs activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance, and vanishing and exploding gradients are impossible
    - This enables to train FCNs with many layers
    - SNNs significantly outperformed all competing FNN methods at 121 UCI tasks

Laurent, T., & Brecht, J. (2018). Deep Linear Networks with Arbitrary Loss: All Local Minima Are Global. International Conference on Machine Learning. PMLR. Retrieved from https://proceedings.mlr.press/v80/laurent18a.html

    - Consider a DNN with the following properties:
    - 1) Arbitrary convex differentiable loss
    - 2) Hidden layers are either at least as wide as the input layer, or at least as wide as the output layer
    - We proof that in this case all local minima are global minima
    - If the loss is convex and Lipschitz but not differentiable then DNN can have sub-optimal local minima

Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., & Sohl-Dickstein, J. (2017). Deep Neural Networks as Gaussian Processes. arXiv, 1711.00165. Retrieved from https://arxiv.org/abs/1711.00165v3

    - It is known that shallow infinite-width NNs are equivalent to gaussian processes (GP)
    - We derive the exact equivalence between infinitely wide deep networks and GPs
    - We develop a computationally efficient pipeline to compute the covariance function for these GPs
    - We use the resulting NNGPs to perform Bayesian inference for wide DNNs on MNIST and CIFAR10
    - GP predictions typically outperform those of finite-width networks
    - One benefit in using a GP is that, due to its Bayesian nature, all predictions have uncertainty estimates
    - We observe that the NNGP uncertainty estimate is highly correlated with prediction error
    - We intend to look into scalability for larger learning tasks

Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2017). Visualizing the Loss Landscape of Neural Nets. arXiv, 1712.09913. Retrieved from https://arxiv.org/abs/1712.09913v3

    - Simple visualization strategies fail to accurately capture the local geometry
    - We present a visualization method based on "filter normalization"
    - When networks become deep, loss surface turns from convex to chaotic, but skip connections prevent this
    - We measure non-convexity by calculating eigenvalues of the Hessian around local minima
    - We show that SGD optimization trajectories lie in an extremely low dimensional space
    - This can be explained by the presence of large, nearly convex regions in the loss landscape

Liu, T., Lugosi, G., Neu, G., & Tao, D. (2017). Algorithmic stability and hypothesis complexity. arXiv, 1702.08712. Retrieved from https://arxiv.org/abs/1702.08712v2

Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv, 1711.05101. Retrieved from https://arxiv.org/abs/1711.05101v3

    - Common implementations of adaptive GD algorithms employ L2 regularization, often calling it "weight decay"
    - However, weight decay is equivalent to L2 regularization only for standard SGD
    - We propose to decouple the weight decay from the optimization steps taken w.r.t. the loss function
    - It decouples the optimal choice of weight decay factor from the setting of the LR for SGD and Adam
    - It substantially improves Adam’s generalization performance (AdamW), allowing it to compete with SGD with momentum on image classification datasets (on which it was previously typically outperformed by the latter)

Lu, Y., Zhong, A., Li, Q., & Dong, B. (2017). Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations. arXiv, 1710.10121. Retrieved from https://arxiv.org/abs/1710.10121v3

Ma, S., Bassily, R., & Belkin, M. (2017). The Power of Interpolation: Understanding the Effectiveness of SGD in Modern Over-parametrized Learning. arXiv, 1712.06559. Retrieved from https://arxiv.org/abs/1712.06559v3

    - We study why SGD converges so fast
    - We show that there is a critical batch size m∗ such that:
    - 1) SGD iteration with mini-batch size m ≤ m∗ is nearly equivalent to m iterations of mini-batch size 1 (linear scaling regime). Doubling the batch size in this regime will roughly halve the number of iterations needed
    - 2) SGD iteration with mini-batch m > m∗ is nearly equivalent to a full gradient descent iteration (saturation regime). Increasing batch size in this regime becomes much less beneficial
    - A critical batch size is nearly independent of the data size

Mahsereci, M., Balles, L., Lassner, C., & Hennig, P. (2017). Early Stopping without a Validation Set. arXiv, 1703.09580. Retrieved from https://arxiv.org/abs/1703.09580v3

    - We propose a cheap and scalable early stopping criterion based on local statistics of the gradients
    - It does not require a validation set, thus enabling the optimizer to use all available training data
    - We test on linear and MLP models

Mallya, A., & Lazebnik, S. (2017). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. arXiv, 1711.05769. Retrieved from https://arxiv.org/abs/1711.05769v2

Molchanov, D., Ashukha, A., & Vetrov, D. (2017). Variational Dropout Sparsifies Deep Neural Networks. arXiv, 1701.05369. Retrieved from https://arxiv.org/abs/1701.05369v3

    - We extend a recently proposed Variational Dropout to the case when dropout rates are unbounded
    - We propose a way to reduce the variance of the gradient estimator
    - Individual dropout rates per weight leads to extremely sparse solutions both in FCNs and CNNs
    - This effect is similar to automatic relevance determination
    - We reduce the number of parameters up to 68 times on VGG with a negligible decrease of accuracy

Nguyen, Q., & Hein, M. (2017). The loss surface of deep and wide neural networks. arXiv, 1704.08045. Retrieved from https://arxiv.org/abs/1704.08045v2

    - Consider a FCN with the following properties:
    - 1) squared loss
    - 2) analytic activation function
    - 3) the number of hidden units of one layer is larger than the number of training points
    - 4) the network structure from this layer on is pyramidal
    - We prove that in this case all local minima are global

Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. arXiv, 1705.08039. Retrieved from https://arxiv.org/abs/1705.08039v2

    - Problem: no method yet exists that is able to compute embeddings of large graph-structured data – such as social networks, knowledge graphs or taxonomies – without loss of information
    - We propose to compute embeddings in hyperbolic space (the Poincaré ball)
    - Informally, hyperbolic space can be thought of as a continuous version of trees
    - It is naturally equipped to model hierarchical structures
    - Poincaré embeddings are successful in lexical entailment on WordNet and in predicting links in graphs where they outperform Euclidean embeddings, especially in low dimensions

Oord, A. v. d., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. arXiv, 1711.00937. Retrieved from https://arxiv.org/abs/1711.00937v2

Peng, K.-C., Wu, Z., & Ernst, J. (2017). Zero-Shot Deep Domain Adaptation. arXiv, 1707.01922. Retrieved from https://arxiv.org/abs/1707.01922v5

Pennington, J., Schoenholz, S. S., & Ganguli, S. (2017). Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. arXiv, 1711.04735. Retrieved from https://arxiv.org/abs/1711.04735v1

    - We compute analytically the entire singular value distribution of a DNN’s input-output Jacobian
    - ReLU networks are incapable of dynamical isometry (see https://arxiv.org/abs/1312.6120)
    - Sigmoidal networks with orthogonal weight initialization can achieve isometry and outperform ReLU nets
    - DNNs achieving dynamical isometry learn orders of magnitude faster than networks that do not

Ramstead, M., Badcock, P. B., & Friston, K. J. (2017). Answering Schrödinger's question: A free-energy formulation. Phys. Life Rev. Retrieved from https://www.semanticscholar.org/paper/Answering-Schr%C3%B6dinger's-question%3A-A-free-energy-Ramstead-Badcock/cbf4040cb14a019ff3556fad5c455e99737f169f

Reddi, S. J., Zaheer, M., Sra, S., Poczos, B., Bach, F., Salakhutdinov, R., & Smola, A. J. (2017). A Generic Approach for Escaping Saddle points. arXiv, 1709.01434. Retrieved from https://arxiv.org/abs/1709.01434v1

    - Problem: second-order methods effectively escape saddle points, but require expensive Hessian-based computations
    - We alternate between a first-order and a second-order optimizers, using the latter only close to saddle points
    - This minimizes Hessian-based computations and yields convergence results competitive to SOTA

Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., & Bottou, L. (2017). Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. arXiv, 1706.04454. Retrieved from https://arxiv.org/abs/1706.04454v3

    - In DL, we empirically show that the spectrum of the Hessian is composed of two parts:
    - 1) The bulk centered near zero, and 2) outliers away from the bulk
    - We analyze sevaral findings, hoping it would shed light on the geometry of high-dimensional non-convex spaces
    - Small and large batch gradient descent appear to converge to different basins of attraction but we show that they are in fact connected through their flat region and so belong to the same basin

Safran, I., & Shamir, O. (2017). Spurious Local Minima are Common in Two-Layer ReLU Neural Networks. arXiv, 1712.08968. Retrieved from https://arxiv.org/abs/1712.08968v3

    - We consider ReLU networks with one hidden layer and squared loss
    - We study spurious local minima in this case (not clearly described). The results imply that in high input dimensions, nearly all target networks (what is this?) of the relevant sizes lead to spurious local minima

Shalev-Shwartz, S., Shamir, O., & Shammah, S. (2017). Failures of Gradient-Based Deep Learning. arXiv, 1703.07950. Retrieved from https://arxiv.org/abs/1703.07950v2

    - We describe 4 types of simple problems, for which gradient descent suffers from significant difficulties
    - These difficulties are not necessarily related to local minima or saddle points, but are related to:
    - 1) Insufficient information in the gradients about the underlying target function
    - 2) Low signal-to-noise ratio
    - 3) Bad conditioning
    - 4) Flatness in the activations

Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual Learning with Deep Generative Replay. arXiv, 1705.08690. Retrieved from https://arxiv.org/abs/1705.08690v3

Shwartz-Ziv, R., & Tishby, N. (2017). Opening the Black Box of Deep Neural Networks via Information. arXiv, 1703.00810. Retrieved from https://arxiv.org/abs/1703.00810v3

    - We visualize DNNs based on Information Bottleneck (IB) theory and stucy the SGD dynamics:
    - 1) Most of the training epochs in standard DL are spent on compression of the input to efficient representation and not on fitting the training labels
    - 2) Then SGD switches from a fast drift to smaller training error into a stochastic relaxation, or random diffusion, and the representation compression phase begins
    - 3) Finally, the converged layers lie on or very close to the Information Bottleneck (IB) theoretical bound
    - This provides a new explanation for the computational benefit of the hidden layers

Smith, L. N., & Topin, N. (2017). Exploring loss function topology with cyclical learning rates. arXiv, 1702.04283. Retrieved from https://arxiv.org/abs/1702.04283v1

    - We apply Cyclical Learning Rates (CLR) and linear network interpolation between checkpoints, and observe counterintuitive increases and decreases in training loss and instances of rapid training
    - Cyclical Learning Rates can produce greater testing accuracy than traditional training 

Smith, L. N., & Topin, N. (2017). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv, 1708.07120. Retrieved from https://arxiv.org/abs/1708.07120v3

    - We describe "super-convergence", where CNNs can be trained an order of magnitude faster than with standard methods
    - We train NNs with one LR cycle and a large maximum LR
    - Large LRs regularize the training, hence requiring a reduction of all other forms of regularization, because the amount of regularization must be balanced for each dataset and architecture
    - We derive a method to compute an estimate of the optimal learning rate
    - Super-convergence provides a greater boost in performance when the amount of labeled training data is limited
    - The philosophy behind CLR is a combination of curriculum learning and simulated annealing

Smith, S. L., & Le, Q. V. (2017). A Bayesian Perspective on Generalization and Stochastic Gradient Descent. arXiv, 1710.06451. Retrieved from https://arxiv.org/abs/1710.06451v3

    - Why NNs can easily memorize randomly labeled training data, but generalizing well on real labels?
    - We show that the same phenomenon occurs in small linear models
    - This is because Bayesian evidence penalizes sharp minima but is invariant to model parameterization
    - The noise introduced by small mini-batches drives the parameters towards minima whose evidence is large
    - We show that the optimum batch size is proportional to both the learning rate and the size of the training set

Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2017). Don't Decay the Learning Rate, Increase the Batch Size. arXiv, 1711.00489. Retrieved from https://arxiv.org/abs/1711.00489v2

    - Decaying the learning rate is simulated annealing
    - Instead, we propose to increase the batch size during training
    - This has the potential to dramatically reduce model training times

Soudry, D., & Hoffer, E. (2017). Exponentially vanishing sub-optimal local minima in multilayer neural networks. arXiv, 1702.05777. Retrieved from https://arxiv.org/abs/1702.05777v5

Taki, M. (2017). Deep Residual Networks and Weight Initialization. arXiv, 1709.02956. Retrieved from https://arxiv.org/abs/1709.02956v1

    - ResNets are relatively insensitive to choice of initial weights
    - How does batch normalization improve backpropagation in ResNet?
    - We propose a new weight initialization distribution to prevent exploding gradients

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ...Polosukhin, I. (2017). Attention Is All You Need. arXiv, 1706.03762. Retrieved from https://arxiv.org/abs/1706.03762v7

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. arXiv, 1710.10903. Retrieved from https://arxiv.org/abs/1710.10903v3

Whittington, J. C. R., & Bogacz, R. (2017). An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. Neural Comput., 28333583. Retrieved from https://pubmed.ncbi.nlm.nih.gov/28333583

Wiatowski, T., Grohs, P., & Bölcskei, H. (2017). Energy Propagation in Deep Convolutional Neural Networks. arXiv, 1704.03636. Retrieved from https://arxiv.org/abs/1704.03636v3

Wu, L., Zhu, Z., & E, W. (2017). Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes. arXiv, 1706.10239. Retrieved from https://arxiv.org/abs/1706.10239v2

    - We find that local minima with large volume of attractors often lead good generalization performance
    - We show that the volume of basin of attraction of good minima dominates over that of poor minima, which guarantees optimization methods with random initialization to converge to good minima
    - This is irrelevant with the types of optimization methods, in contrast with previous understanding which attribute a good generalization to some particular optimizers (SGD) or regularizations (dropout)
    - Low-complexity solutions have a small norm of Hessian matrix with respect to model parameters

Xie, D., Xiong, J., & Pu, S. (2017). All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation. arXiv, 1703.01827. Retrieved from https://arxiv.org/abs/1703.01827v3

    - Problem: how to train deep nets without any shortcuts/identity mappings?
    - Solution: regularizer which utilizes orthonormality and a backward error modulation mechanism

You, Y., Zhang, Z., Hsieh, C.-J., Demmel, J., & Keutzer, K. (2017). ImageNet Training in Minutes. arXiv, 1709.05011. Retrieved from https://arxiv.org/abs/1709.05011v10

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. arXiv, 1703.06114. Retrieved from https://arxiv.org/abs/1703.06114v3

## 2018

Agarap, A. F. (2018). Deep Learning using Rectified Linear Units (ReLU). arXiv, 1803.08375. Retrieved from https://arxiv.org/abs/1803.08375v2

    - A joke paper with a lot of citations due to misleading title
    - We just replace softmax with ReLU for classification neural networks
    - We apply cross-entropy loss (IMO this is strage since predicted probabilities are now arbitrary numbers)
    - We test on MNIST, Fashion-MNIST, WDBC
    - The quality of our method is WORSE than standard softmax

Allen-Zhu, Z., Li, Y., & Song, Z. (2018). A Convergence Theory for Deep Learning via Over-Parameterization. arXiv, 1811.03962. Retrieved from https://arxiv.org/abs/1811.03962v5

    - We study the theory of multi-layer networks
    - We proove that SGD can find global minima on the training objective of over-parameterized DNNs
    - Key insight is that in a neighborhood of the random initialization, the opt. landscape is almost convex
    - This implies an equivalence between over-parameterized finite width NNs and neural tangent kernel
    - Our theory at least applies to FCN, CNN and ResNet

Atanov, A., Ashukha, A., Struminsky, K., Vetrov, D., & Welling, M. (2018). The Deep Weight Prior. arXiv, 1810.06943. Retrieved from https://arxiv.org/abs/1810.06943v6

Atanov, A., Ashukha, A., Molchanov, D., Neklyudov, K., & Vetrov, D. (2018). Uncertainty Estimation via Stochastic Batch Normalization. arXiv, 1802.04893. Retrieved from https://arxiv.org/abs/1802.04893v2

    - We show that Batch Normalization maximizes the ELBO for a certain probabilistic model
    - We design an algorithm which acts consistently during train and test, however, inference becomes inefficient
    - We propose Stochastic Batch Normalization – an efficient approximation of proper inference procedure
    - It successfully extends Dropout and Deep Ensembles methods
    - It allows a scalable uncertainty estimation
    - We show its performance on OOD uncertainty estimation on MNIST and CIFAR10

Athiwaratkun, B., Finzi, M., Izmailov, P., & Wilson, A. G. (2018). There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average. arXiv, 1806.05594. Retrieved from https://arxiv.org/abs/1806.05594v3

    - We discuss consistency regularization that is known to be successful in semi-supervised learning
    - We show that SGD struggles to converge to a single point on the consistency loss
    - Instead, it continues to explore many solutions with high distances apart
    - This leads to changes in predictions on the test data
    - We propose to train consistency-based methods with Stochastic Weight Averaging
    - We propose fast-SWA which averages multiple points within each cycle of a cyclical LR
    - We achieve the best known semi-supervised results on CIFAR-10 and CIFAR-100

Balestriero, R., & Baraniuk, R. (2018). Mad Max: Affine Spline Insights into Deep Learning. arXiv, 1805.06576. Retrieved from https://arxiv.org/abs/1805.06576v5

    - A large class of DNs can be written as a composition of maxaffine spline operators (MASOs)
    - This links DNs to the theory of vector quantization (VQ) and K-means clustering
    - We propose a simple penalty term to loss function to significantly improve performance

Bartlett, P. L., Helmbold, D. P., & Long, P. M. (2018). Gradient descent with identity initialization efficiently learns positive definite linear transformations by deep residual networks. arXiv, 1802.06093. Retrieved from https://arxiv.org/abs/1802.06093v4

    - We analyze deep linear neural networks (without non-linearities) when target function Φ is also linear
    - While, in practice, DNNs are non-linear, analysis of the linear case can provide a tractable way to gain insights
    - We study GD convergence for Φ with different properfies and regularization of DNN towards identity
    - We show that GD fails to converge for Φ whose distance from the identity is a larger constant

Belkin, M., Ma, S., & Mandal, S. (2018). To understand deep learning we need to understand kernel learning. arXiv, 1802.01396. Retrieved from https://arxiv.org/abs/1802.01396v3

    - We show that strong performance of overfitted classifiers is not a unique feature of deep learning
    - Kernel machines trained to have nearly-zero training error perform very well on test data
    - Non-smooth Laplacian kernels easily fit random labels, but smooth Gaussian kernels do not
    - However, overfitted Laplacian and Gaussian classifiers have similar performance on test
    - This suggests that generalization is tied to the properties of the kernel function, not of the optimization
    - This indicates a need for new theoretical ideas for understanding properties of classical kernel methods
    - Note that kernel methods can be viewed as a special type of two-layer neural networks with a fixed first layer

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2018). Reconciling modern machine learning practice and the bias-variance trade-off. arXiv, 1812.11118. Retrieved from https://arxiv.org/abs/1812.11118v2

    - We introduce the "double descent" curve instead of U-shaped bias-variance curve for DNNs
    - We show that double descent exists for a wide spectrum of models and datasets

Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2018). Invertible Residual Networks. arXiv, 1811.00995. Retrieved from https://arxiv.org/abs/1811.00995v3

Bengio, Y., Lodi, A., & Prouvost, A. (2018). Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. arXiv, 1811.06128. Retrieved from https://arxiv.org/abs/1811.06128v2

Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence. arXiv, 1801.10112. Retrieved from https://arxiv.org/abs/1801.10112v3

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. arXiv, 1806.07366. Retrieved from https://arxiv.org/abs/1806.07366v5

    - We propose continuous-depth ResNets, continuous-time latent variable models, continuous normalizing flows
    - The derivative of the hidden state is parameterized
    - The output of the network is computed using a blackbox differential equation solver
    - They have constant memory cost
    - We can adapt evaluation strategy to each input
    - We can explicitly trade numerical precision for speed

Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., Willems, L., Saharia, C., Nguyen, T. H., & Bengio, Y. (2018). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. arXiv, 1810.08272. Retrieved from https://arxiv.org/abs/1810.08272v4

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2018). Universal Transformers. arXiv, 1807.03819. Retrieved from https://arxiv.org/abs/1807.03819v3

Du, S. S., Lee, J. D., Li, H., Wang, L., & Zhai, X. (2018). Gradient Descent Finds Global Minima of Deep Neural Networks. arXiv, 1811.03804. Retrieved from https://arxiv.org/abs/1811.03804v4

    - We prove that GD achieves zero training loss in polynomial time for a deep over-parameterized ResNet
    - The Gram matrix is stable throughout the training process
    - We obtain a similar convergence result for convolutional ResNet

George, D., Lavin, A., Guntupalli, J. S., Mely, D., Hay, N., & Lazaro-Gredilla, M. (2018). Cortical Microcircuits from a Generative Vision Model. arXiv, 1808.01058. Retrieved from https://arxiv.org/abs/1808.01058v1

    - Based on RCN, we derive a family of anatomically instantiated and functional cortical circuit models
    - It is derived by comparing the computational requirements with known anatomical constraints
    - It suggests precise functional roles for the feedforward, feedback and lateral connections

Draxler, F., Veschgini, K., Salmhofer, M., & Hamprecht, F. A. (2018). Essentially No Barriers in Neural Network Energy Landscape. arXiv, 1803.00885. Retrieved from https://arxiv.org/abs/1803.00885v5

    - We construct continuous paths between minima of DNNs on CIFAR10 and CIFAR100
    - These paths are flat in both the training and test landscapes
    - This implies that minima are points on a single connected manifold of low loss

Du, S. S., Hu, W., & Lee, J. D. (2018). Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced. arXiv, 1806.00900. Retrieved from https://arxiv.org/abs/1806.00900v2

Fort, S., & Scherlis, A. (2018). The Goldilocks zone: Towards better understanding of neural network loss landscapes. arXiv, 1807.02581. Retrieved from https://arxiv.org/abs/1807.02581v2

    - We explore the loss landscape of FCNs and CNNs with ReLU and tanh non-linearities on MNIST and CIFAR-10
    - We see the excess of local convexity in a range of configuration space radii we call the Goldilocks zone
    - Local convexity of an initialization is predictive of training speed
    - We hypothesize that the Goldilocks zone contains high density of suitable initialization configurations

Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. arXiv, 1803.03635. Retrieved from https://arxiv.org/abs/1803.03635v5

Friston, K. (2018). Am I Self-Conscious? (Or Does Self-Organization Entail Self-Consciousness?). Front. Psychol., 9, 348034. Retrieved from https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00579/full

Galloway, A., Tanay, T., & Taylor, G. W. (2018). Adversarial Training Versus Weight Decay. arXiv, 1804.03308. Retrieved from https://arxiv.org/abs/1804.03308v3

Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. arXiv, 1802.10026. Retrieved from https://arxiv.org/abs/1802.10026v4

    - We show that optima (posterior modes) of DNN loss surfaces are connected by simple curves over which training and test accuracy are nearly constant (mode connectivity)
    - We introduce a training procedure to discover these high-accuracy pathways between modes
    - We propose Fast Geometric Ensembling to train ensembles in the time required to train a single model
    - We surpass other Snapshot Ensembles techniques on CIFAR-10, CIFAR-100, and ImageNet

Garriga-Alonso, A., Rasmussen, C. E., & Aitchison, L. (2018). Deep Convolutional Networks as shallow Gaussian Processes. arXiv, 1808.05587. Retrieved from https://arxiv.org/abs/1808.05587v2

    - We draw a connection between infinitely-wide CNNs and a Gaussian process (GP)
    - The equivalent kernel can be computed efficiently
    - We demonstrate the performance increase coming from adding translation-invariant structure to the GP prior
    - Without computing any gradients, and without augmenting the training set, we obtain 0.84% error rate on the MNIST, setting a new record for nonparametric GP-based methods

Golmant, N., Vemuri, N., Yao, Z., Feinberg, V., Gholami, A., Rothauge, K., ...Gonzalez, J. (2018). On the Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent. arXiv, 1811.12941. Retrieved from https://arxiv.org/abs/1811.12941v1

    - We empirically investigate large-batch training across wide range of network architectures and problem domains
    - As the batch size becomes larger, there are three main phases of scaling behavior for convergence speed:
    - 1) Linear: increasing the batch size results in linear gains in convergence speed
    - 2) Diminishing returns: improving wall-clock training time at the expense of greater total computational cost
    - 3) Stagnation: increasing batch size results in marginal or non-existent reductions in convergence speed
    - So, increasing the batch size beyond a certain point (that is usually substantially below the capacity of current systems) yields no improvement in wall-clock time to convergence, even for a system with perfect parallelism
    - Increasing the batch size leads to a significant increase in generalization error
    - Model architecture and data complexity is important about this, not a dataset size

Gotmare, A., Keskar, N. S., Xiong, C., & Socher, R. (2018). A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation. arXiv, 1810.13243. Retrieved from https://arxiv.org/abs/1810.13243v1

    - We use mode connectivity and CCA to improve understanding of cosine annealing, learning rate warmup and knowledge distillation, and hypothesize reasons for the success of the heuristics
    - The reasons often quoted for the success of cosine annealing are not substantiated by our experiments (escaping local minima after restarts might be an oversimplification)
    - The effect of LR warmup is to prevent the deeper layers from creating training instability, since it limits weight changes in the deeper layers; freezing them achieves similar outcomes as warmup
    - The latent knowledge shared by the teacher in distillation is primarily disbursed in the deeper layers

Gu, J., Hassan, H., Devlin, J., & Li, V. O. K. (2018). Universal Neural Machine Translation for Extremely Low Resource Languages. arXiv, 1802.05368. Retrieved from https://arxiv.org/abs/1802.05368v2

Gur-Ari, G., Roberts, D. A., & Dyer, E. (2018). Gradient Descent Happens in a Tiny Subspace. arXiv, 1812.04754. Retrieved from https://arxiv.org/abs/1812.04754v1

    - This paper shows that the gradient over training time lies primarily in the subspace spanned by the top few largest eigenvalues of the Hessian H (a sentence from https://arxiv.org/pdf/1910.05929v1)
    - This implies that most of the descent directions lie along extremely low dimensional subspaces of high local positive curvature
    - Exploration in the vastly larger number of other directions utilizes a small portion of the gradient

Kidambi, R., Netrapalli, P., Jain, P., & Kakade, S. M. (2018). On the insufficiency of existing momentum schemes for Stochastic Optimization. arXiv, 1803.05591. Retrieved from https://arxiv.org/abs/1803.05591v2

    - We discuss momentum methods such as heavy ball (HB) and Nesterov’s accelerated gradient descent (NAG)
    - Question: are they optimal even with batchsize of 1? (this case is called stochastic first order oracle, SFO)
    - We describe a linear regression problem instance where it is indeed possible to improve upon SGD, and ASGD (https://arxiv.org/abs/1704.08227) achieves this improvement, but HB (with any step size and momentum) cannot achieve any improvement over SGD; the same holds true for NAG as well
    - We conclude that HB and NAG’s improved performance is attributed to mini-batching and these methods will often struggle to improve over SGD with small constant batch sizes
    - ASGD provides a distinct advantage in training deep networks over SGD, HB and NAG

Hahn, S., & Choi, H. (2018). Understanding Dropout as an Optimization Trick. arXiv, 1806.09783. Retrieved from https://arxiv.org/abs/1806.09783v3

Hanin, B., & Rolnick, D. (2018). How to Start Training: The Effect of Initialization and Architecture. arXiv, 1803.01719. Retrieved from https://arxiv.org/abs/1803.01719v3

    - Study failure modes for early training in deep ReLU nets:
    - 1) exploding or vanishing mean activation length
    - 2) exponentially large variance of activation length
    - For FCN, the cure of 1) require a specific init, and the cure of 2) require a specific constraint
    - For ResNets, the cure of 1) require a specific scaling, then 2) also gets cured

Hernández-García, A., & König, P. (2018). Do deep nets really need weight decay and dropout? arXiv, 1802.07042. Retrieved from https://arxiv.org/abs/1802.07042v3

    - Recently is was suggested that explicit regularization may not be as important as widely believed
    - We perform ablations on weight decay and dropout
    - We find that they may not be necessary, their generalization gain can be achieved by data augmentation alone

Hewitt, L. B., Nye, M. I., Gane, A., Jaakkola, T., & Tenenbaum, J. B. (2018). The Variational Homoencoder: Learning to learn high capacity generative models from few examples. arXiv, 1807.08919. Retrieved from https://arxiv.org/abs/1807.08919v1

    - Hierarchical Bayesian methods can unify many related tasks as inference within a single generative model
    - When this generative model is a powerful DNN, we show that existing learning techniques typically fail to effectively use latent variables
    - We develop a a Variational Homoencoder (VHE): a modification of the VAE in which encoded observations are decoded to new elements from the same class
    - It produces a hierarchical latent variable model which better utilises latent variables
    - Using VHE, we learn a hierarchical PixelCNN on the Omniglot dataset and achieve strong one-shot performance

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., & Bengio, Y. (2018). Learning deep representations by mutual information estimation and maximization. arXiv, 1808.06670. Retrieved from https://arxiv.org/abs/1808.06670v5

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv, 1803.05407. Retrieved from https://arxiv.org/abs/1803.05407v3

    - We propose Stochastic Weight Averaging (SWA): simple averaging of multiple checkpoints that leads to better generalization than conventional training
    - It finds much flatter solutions than SGD and approximates Fast Geometric Ensembling (FGE)
    - SWA is extremely easy to implement, improves generalization, and has almost no computational overhead

Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. arXiv, 1806.07572. Retrieved from https://arxiv.org/abs/1806.07572v4

    - We find that the evolution of an NN during training can be described by the Neural Tangent Kernel (NTK)
    - This makes it possible to study the training of NNs in function space instead of parameter space
    - We observe NTK behavior for wide networks, and compare it to the infinite-width limit
    - We suggest a theoretical motivation for early stopping

Jastrzębski, S., Kenton, Z., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2018). On the Relation Between the Sharpest Directions of DNN Loss and the SGD Step Length. arXiv, 1807.05031. Retrieved from https://arxiv.org/abs/1807.05031v6

    - This paper shows that the maximal Hessian eigenvalue grows, peaks and then declines during training
    - This implies that gradient descent trajectories tend to enter higher positive curvature regions of the loss landscape before eventually finding the desired flatter regions

Kaiser, Ł., & Bengio, S. (2018). Discrete Autoencoders for Sequence Models. arXiv, 1801.09797. Retrieved from https://arxiv.org/abs/1801.09797v1

Karakida, R., Akaho, S., & Amari, S.-i. (2018). Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. arXiv, 1806.01316. Retrieved from https://arxiv.org/abs/1806.01316v3

    - The landscape of the parameter space of DNN is defined by the Fisher information matrix (FIM)
    - We investigate the asymptotic statistics of FIM eigenvalues and reveal that most of them are close to zero while the maximum eigenvalue takes a huge value; so, the landscape of the parameter space of DNN is locally flat in most dimensions, but strongly distorted in others
    - Small eigenvalues that induce flatness are connected to a measure of generalization ability
    - The maximum eigenvalue that induces the distortion enables us to estimate a LR

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2018). Effect of Depth and Width on Local Minima in Deep Learning. arXiv, 1811.08150. Retrieved from https://arxiv.org/abs/1811.08150v4

    - We study DNNs with squared loss without the strong overparameterization and simplification assumptions
    - We show that the quality of local minima improves toward the global minimum as depth and width increase

Kawaguchi, K., & Bengio, Y. (2018). Depth with Nonlinearity Creates No Bad Local Minima in ResNets. arXiv, 1810.09038. Retrieved from https://arxiv.org/abs/1810.09038v3

    - One can consider a map that takes a classical machine-learning model (a basis-function model with an arbitrary fixed basis or set of features) as input, and outputs a deep version of the classical model. One can then ask what structure this "deepening" map preserves
    - We prove that in a type of deep ResNets, depth with nonlinearity (i.e., the "deepening" map from the set of basis-function models to the set of deep ResNets) does not create "bad" local minima

Kirchhoff, M. D., Parr, T., Palacios, E., Friston, K. J., & Kiverstein, J. (2018). The Markov blankets of life: autonomy, active inference and the free energy principle. J. R. Soc. Interface. Retrieved from https://www.semanticscholar.org/paper/The-Markov-blankets-of-life%3A-autonomy%2C-active-and-Kirchhoff-Parr/e1d3f7eb3bf11a0881d9417df8071ca663eefbaf

Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2018). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. arXiv, 1810.00825. Retrieved from https://arxiv.org/abs/1810.00825v3

Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. arXiv, 1804.08838. Retrieved from https://arxiv.org/abs/1804.08838v1

    - We train networks not in their native parameter space, but instead in a smaller, randomly oriented subspace
    - We provide new cartography of the objective landscapes wandered by parameterized models
    - As a result, we see that many problems (datasets) have small intrinsic dimensions
    - It allows quantitative comparison of problem difficulty
    - For example, solving the inverted pendulum problem is 100 times easier than classifying digits from MNIST
    - This allows to obtain an upper bound on the minimum description length of a solution
    - The result is a simple approach for compressing networks, in some cases by more than 100 times

Liang, S., Sun, R., Lee, J. D., & Srikant, R. (2018). Adding One Neuron Can Eliminate All Bad Local Minima. arXiv, 1805.08671. Retrieved from https://arxiv.org/abs/1805.08671v1

    - We study the landscape of NNs for binary classification
    - Under mild assumptions, we prove that after adding one special neuron with a skip connection to the output, or one special neuron per layer, every local minimum is a global minimum
    - So, the distance between any NN and a good NN (with no spurious local minima) is just a neuron away: a class of good NNs is rather “dense” in the class of all NNs
    - This is the first result that no spurious local minimum exists for a wide class of DNNs

Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. arXiv, 1804.08838. Retrieved from https://arxiv.org/abs/1804.08838v1

Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018). Don't Use Large Mini-Batches, Use Local SGD. arXiv, 1808.07217. Retrieved from https://arxiv.org/abs/1808.07217v6

    - Problem: models trained with large batches often do not generalize well
    - We provide the first comprehensive empirically study of the trade-offs in local SGD for DL - when varying the number of workers, number of local steps and mini-batch sizes
    - We propose post-local SGD to address the generalization issue of large-batch training

Liu, J., & Xu, L. (2018). Accelerating Stochastic Gradient Descent Using Antithetic Sampling. arXiv, 1810.03124. Retrieved from https://arxiv.org/abs/1810.03124v1

    - We propose the Antithetic Sampling to reduce the variance of stochastic gradient
    - We make stochastic gradients in a mini-batch negatively correlated as much as possible
    - For this, we just need to calculate the antithetic samples in advance
    - Our stochastic gradient is still an unbiased estimator of full gradient

Martin, C. H., & Mahoney, M. W. (2018). Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning. arXiv, 1810.01075. Retrieved from https://arxiv.org/abs/1810.01075v1

    - We analyze the empirical spectral density of DNN layer matrices
    - They show signatures of traditionally-regularized statistical models, even without explicit regularization
    - Based on Random Matrix Theory, we develop a theory to identify 5+1 Phases of Training, corresponding to increasing amounts of Implicit Self-Regularization
    - For small/old DNNs the Implicit Self-Regularization is like traditional Tikhonov regularization
    - For SOTA DNNs we identify a novel form of Heavy-Tailed Self-Regularization, similar to the self-organization seen in the statistical physics  of disordered systems
    - We can cause a small model to exhibit all 5+1 phases of training simply by changing the batch size
    - Large-batch SGD leads to less-well implicitly-regularized models, explaining the the generalization gap
    - In light of our results, we have a much better understanding of why VC theory does not apply to NNs
    - This also suggests why transfer learning is so effective
    - Our practical theory opens the door to address very practical questions

Matthews, A. G. d. G., Rowland, M., Hron, J., Turner, R. E., & Ghahramani, Z. (2018). Gaussian Process Behaviour in Wide Deep Neural Networks. arXiv, 1804.11271. Retrieved from https://arxiv.org/abs/1804.11271v2

    - We study the relationship between deep FCNs and Gaussian processes
    - We empirically study the distance between finite networks and their Gaussian process analogues
    - We compare exact Gaussian process inference and MCMC inference for finite Bayesian NNs
    - A practical recommendation following from our study is that the Bayesian DL community should routinely compare their results to Gaussian processes with the kernels studied in this paper
    - In some scenarios, the Gaussian process behaviour may not be desired because it implies a lack of a hierarchical representation and a Gaussian statistical assumption; we highlight promising ideas to prevent such behaviour

Mei, S., Montanari, A., & Nguyen, P.-M. (2018). A Mean Field View of the Landscape of Two-Layers Neural Networks. arXiv, 1804.06561. Retrieved from https://arxiv.org/abs/1804.06561v2

Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2018). Do Deep Generative Models Know What They Don't Know? arXiv, 1810.09136. Retrieved from https://arxiv.org/abs/1810.09136v3

Neal, B., Mittal, S., Baratin, A., Tantia, V., Scicluna, M., Lacoste-Julien, S., & Mitliagkas, I. (2018). A Modern Take on the Bias-Variance Tradeoff in Neural Networks. arXiv, 1810.08591. Retrieved from https://arxiv.org/abs/1810.08591v4

    - We measure prediction bias and variance of FCNs
    - We find that both bias and variance can decrease as the number of parameters grows
    - We decompose variance into variance due to optimization and variance due to training set sampling
    - Variance due to optimization monotonically decreases with width in the over-parameterized regime

Nouiehed, M., & Razaviyayn, M. (2018). Learning Deep Models: Critical Points and Local Openness. arXiv, 1803.02968. Retrieved from https://arxiv.org/abs/1803.02968v2

Novak, R., Xiao, L., Lee, J., Bahri, Y., Yang, G., Hron, J., ...Sohl-Dickstein, J. (2018). Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes. arXiv, 1810.05148. Retrieved from https://arxiv.org/abs/1810.05148v4

    - There is a previously identified equivalence between wide FCNs and Gaussian processes
    - We derive an analogous equivalence for multi-layer CNNs both with and without pooling layers
    - We also introduce a Monte Carlo method to estimate the GP corresponding to a given NN architecture
    - Translation equivariance, beneficial in finite channel CNNs, is guaranteed to play no role in the Bayesian treatment of the infinite channel limit
    - We confirm that regular SGD-trained CNNs can significantly outperform their corresponding GPs, suggesting advantages from SGD training compared to fully Bayesian parameter estimation

Papyan, V. (2018). The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD Training and Sample Size. arXiv, 1811.07062. Retrieved from https://arxiv.org/abs/1811.07062v2

Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, Ł., Shazeer, N., Ku, A., & Tran, D. (2018). Image Transformer. arXiv, 1802.05751. Retrieved from https://arxiv.org/abs/1802.05751v3

Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. arXiv, 1804.08771. Retrieved from https://arxiv.org/abs/1804.08771v2

    - BLEU is a parameterized metric
    - These parameters are often not reported
    - The main culprit is different tokenization and normalization schemes applied to the reference
    - The author provide a new tool, SacreBLEU, to use a common BLEU scheme

Ruthotto, L., & Haber, E. (2018). Deep Neural Networks Motivated by Partial Differential Equations. arXiv, 1804.04272. Retrieved from https://arxiv.org/abs/1804.04272v2

Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? arXiv, 1805.11604. Retrieved from https://arxiv.org/abs/1805.11604v5

    - We find that in a certain sense BatchNorm does not reduce internal covariate shift
    - We find that BatchNorm makes the optimization landscape significantly more smooth, so we can use larger LR
    - The Lipschitzness of both the loss and the gradients are improved with BatchNorm
    - A number of other natural normalization techniques have a similar (and, sometime, even stronger) effect
    - It could be the case that the smoothening effect of BatchNorm’s encourages converging to more flat minima

Scaman, K., & Virmaux, A. (2018). Lipschitz regularity of deep neural networks: analysis and efficient estimation. arXiv, 1805.10965. Retrieved from https://arxiv.org/abs/1805.10965v2

Shamir, O. (2018). Are ResNets Provably Better than Linear Predictors? arXiv, 1804.06739. Retrieved from https://arxiv.org/abs/1804.06739v4

    - We prove that that ResNet optimization landscape contains NO local minima with value above what can be obtained with a linear predictor (namely a 1-layer network)
    - We use minimal or no assumptions on the network architecture, data distribution, or loss function
    - We show a certain architectural tweak that allows SGD to achieve loss close or better than any linear predictor

Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. arXiv, 1803.02155. Retrieved from https://arxiv.org/abs/1803.02155v2

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The Implicit Bias of Gradient Descent on Separable Data. Journal of Machine Learning Research, 19(70), 1–57. Retrieved from https://www.jmlr.org/papers/v19/18-188.html

Tarnowski, W., Warchoł, P., Jastrzębski, S., Tabor, J., & Nowak, M. A. (2018). Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function. arXiv, 1809.08848. Retrieved from https://arxiv.org/abs/1809.08848v3

    - In ResNets dynamical isometry (https://arxiv.org/abs/1312.6120) is achievable for any activation function
    - We use Free Probability and Random Matrix Theories (FPT & RMT)
    - We study initial and late phases of the learning processes

Thorpe, M., & van Gennip, Y. (2018). Deep Limits of Residual Neural Networks. arXiv, 1810.11741. Retrieved from https://arxiv.org/abs/1810.11741v4

    - We study ResNet as a discretisation of an ODE
    - We study convergence to connect the discrete setting to a continuum problem

Wang, W., Sun, Y., Eriksson, B., Wang, W., & Aggarwal, V. (2018). Wide Compression: Tensor Ring Nets. arXiv, 1802.09052. Retrieved from https://arxiv.org/abs/1802.09052v1

    - We propose Tensor Ring (TR) factorizations to compress existing MLPs and CNNs
    - It compresses with little or no quality degredation on image classification

Wu, Y., & He, K. (2018). Group Normalization. arXiv, 1803.08494. Retrieved from https://arxiv.org/abs/1803.08494v3

    - Problem: Batch Normalization (BN) works poorly with small batches
    - We present Group Normalization (GN) as a simple alternative to BN
    - GN divides the channels into groups and computes within each group the mean and variance for normalization
    - This computation is independent of batch size
    - On ResNet-50 trained in ImageNet with batch size of 2, GN has 10.6% lower error than its BN counterpart
    - When using typical batch sizes, GN is comparably good with BN
    - GN can be naturally transferred from pre-training to fine-tuning (while fine-tuning BN works poorly)

Xiao, L., Bahri, Y., Sohl-Dickstein, J., Schoenholz, S. S., & Pennington, J. (2018). Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks. arXiv, 1806.05393. Retrieved from https://arxiv.org/abs/1806.05393v2

    - Are residual connections and batch normalization necessary for very deep nets?
    - No, just use a Delta-Orthogonal initialization and appropriate (in this case, tanh) nonlinearity.
    - This research is based on a mean field theory and dynamical isometry (https://arxiv.org/abs/1312.6120)

Xing, C., Arpit, D., Tsirigotis, C., & Bengio, Y. (2018). A Walk with SGD. arXiv, 1802.08770. Retrieved from https://arxiv.org/abs/1802.08770v4

    - We show the qualitatively different roles of LR and batch-size in DNN optimization and generalization
    - We find that the loss interpolation between parameters before and after each training iteration’s update is roughly convex with a minimum (valley floor) in between for most of the training
    - This means that SGD moves in valley like regions "bouncing between walls at a height"
    - While a large LR maintains a large height from the valley floor, a small batch size injects noise facilitating exploration; this mechanism is crucial for generalization

Yao, Z., Gholami, A., Lei, Q., Keutzer, K., & Mahoney, M. W. (2018). Hessian-based Analysis of Large Batch Training and Robustness to Adversaries. arXiv, 1802.08241. Retrieved from https://arxiv.org/abs/1802.08241v4

    - We study large batch size training through the lens of the Hessian
    - Large batch size training converges to points with noticeably higher Hessian spectrum, and such points have poor robustness to adversarial perturbation
    - We study the connection between robust optimization (https://arxiv.org/abs/1706.06083, that is a min-max optimization problem), and large batch size training
    - Robust optimization is antithetical to large batch training, in the sense that it favors areas with small spectrum (aka flat minimas), and robust training is a saddle-free optimization problem almost everywhere

Yuille, A. L., & Liu, C. (2018). Deep Nets: What have they ever done for Vision? arXiv, 1805.04025. Retrieved from https://arxiv.org/abs/1805.04025v4

Yun, C., Sra, S., & Jadbabaie, A. (2018). Small nonlinearities in activation functions create bad local minima in neural networks. arXiv, 1802.03487. Retrieved from https://arxiv.org/abs/1802.03487v4

    - We prove that for ReLU DNNs and almost all practical datasets (specifically, if linear models cannot perfectly fit the data) there exist infinitely many local minima that are not global
    - We summarize what is known so far, and in our paper we make the least restrictive assumptions
    - We also tackle more general nonlinear activation functions
    - We present some other theoretical results

Yun, C., Sra, S., & Jadbabaie, A. (2018). Efficiently testing local optimality and escaping saddles for ReLU networks. arXiv, 1809.10858. Retrieved from https://arxiv.org/abs/1809.10858v2

    - We provide a theoretical algorithm for checking local optimality and escaping saddles at nondifferentiable points of empirical risks of two-layer ReLU networks
    - Such nondifferentiable points lie in a set of measure zero, so one may be tempted to overlook them as "non-generic", however, when studying critical points we cannot do so, as they are precisely such "non-generic" points

Zaeemzadeh, A., Rahnavard, N., & Shah, M. (2018). Norm-Preservation: Why Residual Networks Can Become Extremely Deep? arXiv, 1805.07477. Retrieved from https://arxiv.org/abs/1805.07477v5

    - We prove that the skip connections facilitate preserving the norm of the gradient, and lead to stable backprop
    - As more residual blocks are stacked, the norm-preservation of the network is enhanced
    - We validate this experimentally
    - We propose Procrustes ResNets: an method to regularize the singular values of the convolution operator and making the ResNet’s transition layers extra norm-preserving
    - This can be used as a guide for training deeper networks and can also inspire new deeper architectures

Zhang, L., & Schaeffer, H. (2018). Forward Stability of ResNet and Its Variants. arXiv, 1811.09885. Retrieved from https://arxiv.org/abs/1811.09885v1

Zhang, J., Liu, T., & Tao, D. (2018). An Information-Theoretic View for Deep Learning. arXiv, 1804.09060. Retrieved from https://arxiv.org/abs/1804.09060v8

    - Question: does it always hold that a deeper network leads to better performance?
    - We derive an upper bound on the expected generalization error depending on depth
    - This shows that as depth increases, the expected generalization error will decrease exponentially
    - Some more results

Zhang, C., Öztireli, C., Mandt, S., & Salvi, G. (2018). Active Mini-Batch Sampling using Repulsive Point Processes. arXiv, 1804.02772. Retrieved from https://arxiv.org/abs/1804.02772v2

## 2019

Allen-Zhu, Z., & Li, Y. (2019). What Can ResNet Learn Efficiently, Going Beyond Kernels? arXiv, 1905.10337. Retrieved from https://arxiv.org/abs/1905.10337v3

    - We prove there are functions that with the same number of training examples, the test error obtained by NNs can be much smaller than any kernel method, including neural tangent kernels (NTK)
    - The main intuition is that DNNs can implicitly perform hierarchical learning reducing the sample complexity

Allen-Zhu, Z., & Li, Y. (2019). Can SGD Learn Recurrent Neural Networks with Provable Generalization? arXiv, 1902.01028. Retrieved from https://arxiv.org/abs/1902.01028v2

    - We discuss optimization and generalization capabilities of RNN
    - We show that RNN with SGD can learn some notable concept class efficiently, meaning that both time and sample complexity scale polynomially in the input length

Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. arXiv, 1907.02893. Retrieved from https://arxiv.org/abs/1907.02893v3

    - Problem: ML models learn spurious correlations stemming from data biases
    - We propose Invariant Risk Minimization (IRM) method
    - It estimates invariant predictors from multiple training environments to enable OOD generalization
    - It tries to find a representation such that the optimal classifier on top of it matches for all environments

Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., & Wang, R. (2019). On Exact Computation with an Infinitely Wide Neural Net. arXiv, 1904.11955. Retrieved from https://arxiv.org/abs/1904.11955v2

Arpit, D., Campos, V., & Bengio, Y. (2019). How to Initialize your Network? Robust Initialization for WeightNorm & ResNets. arXiv, 1906.02341. Retrieved from https://arxiv.org/abs/1906.02341v2

    - We propose a novel initialization strategy for weight normalized networks with and without residual connections
    - It is based on mean field approximation
    - It outperforms existing methods in generalization, robustness to hyper-parameters and variance between seeds

Ash, J. T., & Adams, R. P. (2019). On Warm-Starting Neural Network Training. arXiv, 1910.08475. Retrieved from https://arxiv.org/abs/1910.08475v3

    - We discuss warm-starting for contunual learning
    - In practice it seems to yield poorer quality than fresh random initializations as new data arrive, even though the final training losses are similar
    - We provide the "shrink and perturb" trick that overcomes this pathology in several important situations

Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2019). Benign Overfitting in Linear Regression. arXiv, 1906.11300. Retrieved from https://arxiv.org/abs/1906.11300v3

    - Question: why DNNs seem to predict well, even with a perfect fit to noisy training data
    - We study when the same happens in linear regression
    - We show that overparameterization is essential for benign overfitting in linear regression
    - We show that data that lies in a large but finite dimensional space exhibits the benign overfitting phenomenon with a much wider range of covariance properties than data that lies in an infinite dimensional space

Belkin, M., Hsu, D., & Xu, J. (2019). Two models of double descent for weak features. arXiv, 1903.07571. Retrieved from https://arxiv.org/abs/1903.07571v2

    - We show that the double descent curve can be observed in two simple random features models with the least squares/least norm predictor
    - We show that double descent occurs when features are plentiful but individually too "weak"

Bengio, Y., Deleu, T., Rahaman, N., Ke, R., Lachapelle, S., Bilaniuk, O., ...Pal, C. (2019). A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms. arXiv, 1901.10912. Retrieved from https://arxiv.org/abs/1901.10912v2

Dean, T., Fan, C., Lewis, F. E., & Sano, M. (2019). Biological Blueprints for Next Generation AI Systems. arXiv, 1912.00421. Retrieved from https://arxiv.org/abs/1912.00421v1

Farquhar, S., & Gal, Y. (2019). A Unifying Bayesian View of Continual Learning. arXiv, 1902.06494. Retrieved from https://arxiv.org/abs/1902.06494v1

Fort, S., & Jastrzebski, S. (2019). Large Scale Structure of Neural Network Loss Landscapes. arXiv, 1906.04724. Retrieved from https://arxiv.org/abs/1906.04724v1

    - We model the NN`s loss surface as a union of n-dimensional manifolds that we call n-wedges
    - We show that common regularizers (learning rate, batch size, L2 regularization, dropout, network width) all influence the optimization trajectory in a similar way
    - We see surprising effects in high dimensions, when our intuition about hills and valleys in 2D often fails us
    - We also critically examine the recently popular Stochastic Weight Averaging (SWA) technique

Fort, S., & Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes. arXiv, 1910.05929. Retrieved from https://arxiv.org/abs/1910.05929v1

    - We discuss 4 previously found local properties of loss landscape in DNNs
    - We develop a simple theoretical model of gradients and Hessians, justified by numerical experiments
    - This model simultaneously accounts for all 4 of these surprising and seemingly unrelated properties
    - This model makes connections with diverse topics in neural networks, random matrix theory, and spin glasses, including the neural tangent kernel, BBP phase transitions, and Derrida’s random energy model

Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep Ensembles: A Loss Landscape Perspective. arXiv, 1912.02757. Retrieved from https://arxiv.org/abs/1912.02757v2

    - Why Bayesian neural networks do not perform as well as deep ensembles in practice?
    - We show that the ability of random initializations to explore entirely different modes is unmatched by popular subspace sampling (Bayesian) methods
    - In other words, the functions sampled along a single training trajectory or subspace thereof tend to be very similar in predictions (while potential far away in the weight space), whereas functions sampled from different randomly initialized trajectories tend to be very diverse

Fort, S., Nowak, P. K., Jastrzebski, S., & Narayanan, S. (2019). Stiffness: A New Perspective on Generalization in Neural Networks. arXiv, 1901.09491. Retrieved from https://arxiv.org/abs/1901.09491v3

    - We study generalization through the lens of stiffness: we measure how stiff a neural network is by analyzing how a small gradient step based on one input example affects the loss on another input example
    - We demonstrate the connection between stiffness and generalization, and observe its dependence on LR
    - Some more results

Ghorbani, B., Krishnan, S., & Xiao, Y. (2019). An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. arXiv, 1901.10159. Retrieved from https://arxiv.org/abs/1901.10159v1

    - We develop a tool to study the evolution of the entire Hessian spectrum throughout the optimization process
    - We find interesting details such that rapid appearance of large isolated eigenvalues in the spectrum
    - We analyze how the outlier eigenvalues affect the speed of optimization
    - We believe our tool and style of analysis will open up new avenues of research in optimization

Greff, K., Kaufman, R. L., Kabra, R., Watters, N., Burgess, C., Zoran, D., ...Lerchner, A. (2019). Multi-Object Representation Learning with Iterative Variational Inference. arXiv, 1903.00450. Retrieved from https://arxiv.org/abs/1903.00450v3

Gu, J., Wang, Y., Cho, K., & Li, V. O. K. (2019). Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations. arXiv, 1906.01181. Retrieved from https://arxiv.org/abs/1906.01181v1

    - Problem: zero-shot translation (translating between language pairs on which a system has never been trained), is an emergent property; however, naive training for zero-shot translation easily fails and is sensitive to hyperparemeters; the performance typically lags far behind the more conventional pivot-based approach which translates twice using a third language as a pivot
    - We show that this issue is a consequence of capturing spurious correlation
    - We propose decoder pre-training and back-translation approaches to improve zero-shot translation

Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2019). Surprises in High-Dimensional Ridgeless Least Squares Interpolation. arXiv, 1903.08560. Retrieved from https://arxiv.org/abs/1903.08560v5

    - SOTA NNs appear to be interpolators: estimators that achieve zero training error with low testing error
    - We study minimum L2 norm interpolation in high-dimensional least squares regression
    - We study linear model and one-layer NN
    - We recover several phenomena including the "double descent"

He, F., Liu, T., & Tao, D. (2019). Why ResNet Works? Residuals Generalize. arXiv, 1904.01367. Retrieved from https://arxiv.org/abs/1904.01367v1

    - We prove that skip connections does not increase the hypothesis complexity (expressive power?) of the NNs
    - We study some generalization bounds
    - We conclude that we need to use regularization terms to control the magnitude of the norms of weight matrices not to increase too much, which justifes the standard technique of weight decay

Jiang, A. H., Wong, D. L.-K., Zhou, G., Andersen, D. G., Dean, J., Ganger, G. R., ...Pillai, P. (2019). Accelerating Deep Learning by Focusing on the Biggest Losers. arXiv, 1910.00762. Retrieved from https://arxiv.org/abs/1910.00762v1

    - We propose Selective-Backprop: prioritizing examples with high loss at each iteration, and skipping others
    - This accelerates training by reducing the number of backprops
    - Selective-Backprop converges to target error rates up to 3.5x faster than with standard SGD, and 1.02-1.8x faster than a SOTA importance sampling approach
    - Further acceleration of 26% can be achieved by also skipping forward passes of low priority examples

Jiao, L., & Zhao, J. (2019). A Survey on the New Generation of Deep Learning in Image Processing. IEEE Access, 7, 172231–172263. Retrieved from https://ieeexplore.ieee.org/document/8917633

Izmailov, P., Maddox, W. J., Kirichenko, P., Garipov, T., Vetrov, D., & Wilson, A. G. (2019). Subspace Inference for Bayesian Deep Learning. arXiv, 1907.07504. Retrieved from https://arxiv.org/abs/1907.07504v1

    - We construct low-dimensional subspaces of parameter space which contain diverse sets of high performing models
    - We perform Bayesian model averaging over the induced posterior in these subspaces
    - This produces accurate predictions and well calibrated predictive uncertainty for regression and classification

Kawaguchi, K., & Huang, J. (2019). Gradient Descent Finds Global Minima for Generalizable Deep Neural Networks of Practical Sizes. arXiv, 1908.02419. Retrieved from https://arxiv.org/abs/1908.02419v3

    - We prove that GD can find a global minimum for DNNs of sizes commonly encountered in practice
    - This only requires the practical degrees of over-parameterization (several orders of magnitude smaller than that required by the previous theories)
    - Our theory only requires the number of trainable parameters to increase linearly as the training set grows
    - Such DNNs are shown to generalize well to unseen test samples with a natural dataset

Kawaguchi, K., Huang, J., & Kaelbling, L. P. (2019). Every Local Minimum Value is the Global Minimum Value of Induced Model in Non-convex Machine Learning. arXiv, 1904.03673. Retrieved from https://arxiv.org/abs/1904.03673v3

    - We prove, under mild assumptions, that every local minimum achieves the globally optimal value of the perturbable gradient basis model at any differentiable point (what?)

Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2019). Hyperbolic Image Embeddings. arXiv, 1904.02239. Retrieved from https://arxiv.org/abs/1904.02239v2

    - Hyperbolic embeddings as an alternative to Euclidean and spherical embeddings
    - Hyperbolic spaces are more suitable for embedding data with such hierarchical structure
    - Experiments with few-shot learning and person re-identification demonstrate these embeddings are beneficial
    - Propose an approach to evaluate the hyperbolicity of a dataset using Gromov δ-hyperbolicity

Kosiorek, A. R., Sabour, S., Teh, Y. W., & Hinton, G. E. (2019). Stacked Capsule Autoencoders. arXiv, 1906.06818. Retrieved from https://arxiv.org/abs/1906.06818v2

Labach, A., Salehinejad, H., & Valaee, S. (2019). Survey of Dropout Methods for Deep Neural Networks. arXiv, 1904.13310. Retrieved from https://arxiv.org/abs/1904.13310v2

    - We provide an overview of dropout methods, including:
    - We summarize approaches for theoretically explaining the function of dropout methods
    - We describe dropout methods for CNNs
    - We describe dropout for compressing NNs
    - We describe Monte Carlo dropout and related work

Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2019). The Omniglot challenge: a 3-year progress report. arXiv, 1902.03477. Retrieved from https://arxiv.org/abs/1902.03477v2

Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. arXiv, 1902.06720. Retrieved from https://arxiv.org/abs/1902.06720v4

    - We build on Neural tangent kernel work and study infinitely-wide NNs
    - Infinitely-wide NNs are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters
    - Gradient-based training of infinitely-wide NNs with a squared loss produces test set predictions drawn from a Gaussian process with a particular compositional kernel

Lezcano-Casado, M., & Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group. arXiv, 1901.08428. Retrieved from https://arxiv.org/abs/1901.08428v3

    - A reparametrization to perform unconstrained optimizaion with orthogonal and unitary constraints
    - We apply our results to RNNs with orthogonal recurrent weights, yielding a new architecture called EXPRNN
    - Faster, accurate, and more stable convergence
    - https://github.com/pytorch/pytorch/issues/48144

Liang, S., Sun, R., & Srikant, R. (2019). Revisiting Landscape Analysis in Deep Neural Networks: Eliminating Decreasing Paths to Infinity. arXiv, 1912.13472. Retrieved from https://arxiv.org/abs/1912.13472v1

    - We highlight that even without bad local minima, an optimization may diverge to infinity (extremely big parameter values) if there are paths leading to infinity, along which the loss function decreases
    - If we restrict parameters, this may introduce additional bad local minima on the boundaries
    - We consider a large class of over-parameterized deep neural networks with appropriate regularizers
    - For them, we prove that the loss function has no bad local minima and no decreasing paths to infinity

Liu, T., Chen, M., Zhou, M., Du, S. S., Zhou, E., & Zhao, T. (2019). Towards Understanding the Importance of Shortcut Connections in Residual Networks. arXiv, 1909.04653. Retrieved from https://arxiv.org/abs/1909.04653v3

    - We study a two-layer non-overlapping convolutional ResNet
    - The corresponding optimization problem has a spurious local optimum
    - However, GD with proper normalization avoids it and converges to a global optimum in polynomial time

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2019). On the Variance of the Adaptive Learning Rate and Beyond. arXiv, 1908.03265. Retrieved from https://arxiv.org/abs/1908.03265v4

    - Why warmup is essential for adaptive stochastic optimization algorithms like RMSprop and Adam?
    - We show that their variance is problematically large in the early stage
    - We propose Rectified Adam (RAdam) by introducing a term to rectify the variance of the adaptive LR

Ma, X., Zhang, P., Zhang, S., Duan, N., Hou, Y., Song, D., & Zhou, M. (2019). A Tensorized Transformer for Language Modeling. arXiv, 1906.09777. Retrieved from https://arxiv.org/abs/1906.09777v3

    - We propose Multi-linear attention with Block-Term Tensor Decomposition (BTD)
    - This not only largely compress the model parameters but also obtain performance improvements

Millidge, B. (2019). Deep Active Inference as Variational Policy Gradients. arXiv, 1907.03876. Retrieved from https://arxiv.org/abs/1907.03876v1

Nakamura, K., & Hong, B.-W. (2019). Adaptive Weight Decay for Deep Neural Networks. arXiv, 1907.08931. Retrieved from https://arxiv.org/abs/1907.08931v2

    - We propose adaptive weight-decay (AdaDecay) where the gradient norms are normalized within each layer and the degree of regularization for each parameter is proportional to the magnitude of its gradient using the sigmoid
    - We show the effectiveness on MNIST, Fashion-MNIST, and CIFAR-10

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep Double Descent: Where Bigger Models and More Data Hurt. arXiv, 1912.02292. Retrieved from https://arxiv.org/abs/1912.02292v1

    - We discuss previously found "double-descent" phenomenon
    - We find that a variety of modern DL tasks exhibit this phenomenon
    - We define the effective model complexity (EMC) of a training procedure as the maximum number of samples on which it can achieve close to zero training error and hypothesize that double descent occurs as a function of the EMC
    - Indeed we observe that double descent occurs also as a function of the number of training epochs
    - We identify certain regimes where increasing the number of train samples actually hurts test performance

Neal, B. (2019). On the Bias-Variance Tradeoff: Textbooks Need an Update. arXiv, 1912.08286. Retrieved from https://arxiv.org/abs/1912.08286v1

    - A PhD thesis of Brady Neal
    - We review the history of the bias-variance tradeoff
    - We show a lack of a bias-variance tradeoff in NNs
    - We observe a similar phenomenon in deep RL

Papyan, V. (2019). Measurements of Three-Level Hierarchical Structure in the Outliers in the Spectrum of Deepnet Hessians. arXiv, 1901.08244. Retrieved from https://arxiv.org/abs/1901.08244v1

    - We study a known fact that spectrum of the Hessian of DNNs contains outliers
    - We clarify the source of these outliers, comparing to previous works
    - We find a way to approximate the principal subspace of the Hessian using certain "averaging" operations, avoiding the need for high-dimensional eigenanalysis

Peluchetti, S., & Favaro, S. (2019). Infinitely deep neural networks as diffusion processes. arXiv, 1905.11065. Retrieved from https://arxiv.org/abs/1905.11065v3

    - For deep nets with iid weight init, the dependency on the input vanishes as depth increases to infinity
    - Under some assumptions, infinitely deep ResNets converge to SDEs (diffusion processes)
    - They do not suffer from the above property

Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019). Micro-Batch Training with Batch-Channel Normalization and Weight Standardization. arXiv, 1903.10520. Retrieved from https://arxiv.org/abs/1903.10520v2

    - We propose Weight Standardization (WS) and Batch-Channel Normalization (BCN)
    - This bring two success factors of BatchNorm into micro-batch training:
    - 1) the smoothing effects on the loss landscape
    - 2) the ability to avoid harmful elimination singularities (neuron saturation?)
    - The latter problem is not solved by Layer Normalization or Group Normalization in micro-batch training
    - WS and BCN with micro-batch training is even able to match or outperform BN with large-batch training

Rangamani, A., Nguyen, N. H., Kumar, A., Phan, D., Chin, S. H., & Tran, T. D. (2019). A Scale Invariant Flatness Measure for Deep Network Minima. arXiv, 1902.02434. Retrieved from https://arxiv.org/abs/1902.02434v1

    - Question: flatness of minima empirically provides better generalization, but most measures of sharpness/flatness are not invariant to rescaling of the network parameters
    - We propose a Hessian-based measure for flatness that is invariant to rescaling
    - We confirm that Large-Batch SGD minima are indeed sharper than Small-Batch SGD minima (there was also another work that shows that Large-Batch SGD converges to solution with another Hessian properties)

Shen, X., Tian, X., Liu, T., Xu, F., & Tao, D. (2019). Continuous Dropout. arXiv, 1911.12675. Retrieved from https://arxiv.org/abs/1911.12675v1

    - We extend the traditional binary dropout to continuous dropout inspired by neuroscience
    - We compare it with binary dropout, adaptive dropout, and DropConnect, and show that it performs better

Simsekli, U., Sagun, L., & Gurbuzbalaban, M. (2019). A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks. arXiv, 1901.06053. Retrieved from https://arxiv.org/abs/1901.06053v1

    - The gradient noise in SGD is often considered to be Gaussian
    - This enables SGD to be analyzed as a stochastic differential equation (SDE) driven by a Brownian motion
    - We show that in deep learning the gradient noise is highly non-Gaussian and admits heavy-tails
    - We investigate this in varying network architectures and sizes, loss functions, and datasets
    - Instead of Brownian motion, we e propose to analyze SGD as an SDE driven by a Lévy motion
    - Such SDEs can incur "jumps", which force the SDE transition from narrow minima to wider minima
    - This sheds more light on the belief that SGD prefers wide minima

Siu, C. (2019). Residual Networks Behave Like Boosting Algorithms. arXiv, 1909.11790. Retrieved from https://arxiv.org/abs/1909.11790v1

    - We show that ResNets with standard training are equivalent to boosting feature representation
    - Inspired by Online Boosting, we modify the ResNet with an additional learnable shrinkage parameter
    - We propose a ResNet-DT (neural decision tree residual network) and test it on the datasets from UCI repository

Sohl-Dickstein, J., & Kawaguchi, K. (2019). Eliminating all bad Local Minima from Loss Landscapes without even adding an Extra Unit. arXiv, 1901.03909. Retrieved from https://arxiv.org/abs/1901.03909v1

    - A one-list paper!
    - We find a way to remove all bad local minima from any loss landscape, so long as the global minimum has a loss of zero (seems like they add two more learnable parameters)
    - Pathologies (diverging go infinity) can continue to exist in losses modified in a such fashion
    - We leave it to the reader to judge whether removing local minima in this fashion is trivial, deep, or both

Sukhbaatar, S., Grave, E., Lample, G., Jegou, H., & Joulin, A. (2019). Augmenting Self-attention with Persistent Memory. arXiv, 1907.01470. Retrieved from https://arxiv.org/abs/1907.01470v1

    - We propose a new model that solely consists of attention layers (no FF layers)
    - We augment the self-attention with persistent memory vectors that play a similar role as the FF layer

Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., & Chao, L. S. (2019). Learning Deep Transformer Models for Machine Translation. arXiv, 1906.01787. Retrieved from https://arxiv.org/abs/1906.01787v1

    - We show that deep transformer encoders may outperform their wide counterparts
    - The proper use of layer normalization is the key to learning deep encoders
    - We propose an approach based on dynamic linear combination of layers (DLCL) to memorizing the features extracted from all preceding layers
    - We successfully train a 30-layer encoder that is currently the deepest encoder in NMT

Wang, J., Chen, Y., Chakraborty, R., & Yu, S. X. (2019). Orthogonal Convolutional Neural Networks. arXiv, 1911.12207. Retrieved from https://arxiv.org/abs/1911.12207v3

    - We propose orthogonal convolution: filter orthogonality with doubly block-Toeplitz matrix representation
    - It outperforms the kernel orthogonality, learns more diverse and expressive features

Wen, Y., Luk, K., Gazeau, M., Zhang, G., Chan, H., & Ba, J. (2019). An Empirical Study of Large-Batch Stochastic Gradient Descent with Structured Covariance Noise. arXiv, 1902.08234. Retrieved from https://arxiv.org/abs/1902.08234v4

    - We address the problem of improving generalization in large-batch training without elongating training duration
    - We propose to add covariance noise to the gradients
    - We do some theoretical studies

Yang, G., Pennington, J., Rao, V., Sohl-Dickstein, J., & Schoenholz, S. S. (2019). A Mean Field Theory of Batch Normalization. arXiv, 1902.08129. Retrieved from https://arxiv.org/abs/1902.08129v2

    - We show that the batch normalization (BN) itself is the cause of gradient explosion
    - Vanilla BN networks without skip connections are not trainable at large depths for common initialization schemes
    - Gradient explosion can be reduced by tuning the network close to the linear regime
    - It is possible to perform exact Bayesian inference in the case of wide neural networks with BN

Yang, G. (2019). Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation. arXiv, 1902.04760. Retrieved from https://arxiv.org/abs/1902.04760v3

    - We introduce a notion of a Tensor Program that can express most neural network computations where all dimensions are large compared to input and output dimensions
    - A Tensor Program is indeed a program over a specific set of functions: matrix transpose, matrix-vector multiplication, linear combination of vectors, and coordinatewise application of any nonlinearity
    - Such tensor programs can express the computation in most NN scenarios, but not all (one example is layer normalization, however we can still deal with it, and the paper shows how)
    - This framework describes the convergence of random NNs (CNN, RNN, ResNet, attention, BN) to Gaussian processes
    - We discuss the applicability of the gradient independence assumption
    - The convergence of the Neural Tangent Kernel is also a part of our framework
    - Our framework is general enough to rederive classical random matrix results as well as recent results in neural network Jacobian singular values
    - We hope our work opens a way toward design of even stronger Gaussian Processess, initialization schemes and deeper understanding of SGD dynamics

Yang, G. (2019). Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes. arXiv, 1910.12478. Retrieved from https://arxiv.org/abs/1910.12478v3

    - It has been shown that wide NNs with random weights are Gaussian processes
    - We show that this extends to most of modern feedforward or recurrent neural networks
    - This work serves as a tutorial on the tensor programs technique (https://arxiv.org/abs/1902.04760)
    - We provide open-source implementations of the Gaussian Process kernels of simple RNN, GRU, transformer, and batchnorm+ReLU network

Yun, C., Sra, S., & Jadbabaie, A. (2019). Are deep ResNets provably better than linear predictors? arXiv, 1907.03922. Retrieved from https://arxiv.org/abs/1907.03922v2

Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. arXiv, 1901.09321. Retrieved from https://arxiv.org/abs/1901.09321v2

    - Normalization layers are believed to stabilize training, enable higher LR, improve generalization
    - We show that none of the perceived benefits is unique to normalization
    - We propose Fixup initialization via properly rescaling a standard initialization
    - Training ResNets with Fixup is as stable as training with normalization
    - Fixup allows to achieve SOTA on image classification and NMT

Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S. J., Kumar, S., & Sra, S. (2019). Why are Adaptive Methods Good for Attention Models? arXiv, 1912.03194. Retrieved from https://arxiv.org/abs/1912.03194v2

    - Why are adaptive methods like Clipped SGD/Adam good for attention models?
    - We show that heavy-tails of the noise in stochastic gradients is one cause of SGD’s poor performance
    - We provide tight upper and lower convergence bounds for adaptive gradient methods under heavy-tailed noise
    - Though clipping speeds up SGD, it does not close the gap between SGD and Adam
    - We develop an adaptive coordinate-wise clipping algorithm
    - We experimentally show that it outperforms Adam on BERT training tasks

## 2020

Allen-Zhu, Z., & Li, Y. (2020). Backward Feature Correction: How Deep Learning Performs Deep (Hierarchical) Learning. arXiv, 2001.04413. Retrieved from https://arxiv.org/abs/2001.04413v6

Agarwal, C., D'souza, D., & Hooker, S. (2020). Estimating Example Difficulty Using Variance of Gradients. arXiv, 2008.11600. Retrieved from https://arxiv.org/abs/2008.11600v4

    - We propose Variance of Gradients (VoG) as a metric to rank data by difficulty
    - Data points with high VoG scores are far more difficult for the model to learn and may require memorization
    - VoG provides insight into the learning cycle of the model
    - VoG is a valuable and efficient ranking for out-of-distribution detection

Alemi, A. A., Morningstar, W. R., Poole, B., Fischer, I., & Dillon, J. V. (2020). VIB is Half Bayes. arXiv, 2011.08711. Retrieved from https://arxiv.org/abs/2011.08711v1

Ashukha, A., Lyzhov, A., Molchanov, D., & Vetrov, D. (2020). Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning. arXiv, 2002.06470. Retrieved from https://arxiv.org/abs/2002.06470v4

    - We point out pitfalls of existing metrics for in-domain uncertainty estimation
    - We introduce the deep ensemble equivalent score (DEE)
    - We compare different ensembling techniques and show that many of them are equivalent to an ensemble of only few independently trained networks in terms of test performance

Bachlechner, T., Majumder, B. P., Mao, H. H., Cottrell, G. W., & McAuley, J. (2020). ReZero is All You Need: Fast Convergence at Large Depth. arXiv, 2003.04887. Retrieved from https://arxiv.org/abs/2003.04887v2

    - We propose ReZero: a modification of ResNet by adding a learnable (initially zero) multiplier to residual blocks
    - ReZero initializes each layer to perform the identity operation and satisfies initial dynamical isometry
    - ReZero effectively propagates signals through deep network
    - We are the first to train Transformers over 100 layers without LR warm-up, LayerNorm or auxiliary losses

Chan, K. H. R., Yu, Y., You, C., Qi, H., Wright, J., & Ma, Y. (2020). Deep Networks from the Principle of Rate Reduction. arXiv, 2010.14765. Retrieved from https://arxiv.org/abs/2010.14765v1

    - We construct a CNN layer-by-layer by optimizing the rate reduction of learned features
    - We do this one gradient ascent iteration per layer
    - This "white box" network has precise optimization, statistical, and geometric interpretation
    - This framework justifies the role of multi-channel lifting and sparse coding in early stage of CNN
    - So constructed CNN can learn a good discriminative representation even without any backprop training

Chen, M., Bai, Y., Lee, J. D., Zhao, T., Wang, H., Xiong, C., & Socher, R. (2020). Towards Understanding Hierarchical Learning: Benefits of Neural Representations. arXiv, 2006.13436. Retrieved from https://arxiv.org/abs/2006.13436v2

    - Not easy to understand what happens in this paper!
    - Seems like they use randomly initialized neural network as a source of deep features to train a shallow model
    - They demonstrate that intermediate neural representations can be advantageous over raw inputs

Chen, L., Min, Y., Belkin, M., & Karbasi, A. (2020). Multiple Descent: Design Your Own Generalization Curve. arXiv, 2008.01036. Retrieved from https://arxiv.org/abs/2008.01036v7

    - We show a multiple descent: a generalization curve with many peaks can exist for linear regression
    - Locations of those peaks can be explicitly controlled
    - On the other hand, we rarely observe complex generalization curves in practice
    - We conclude that realistic generalization curves arise from specific interactions between data properties and the inductive biases of algorithms

Chen, Z., Deng, L., Wang, B., Li, G., & Xie, Y. (2020). A Comprehensive and Modularized Statistical Framework for Gradient Norm Equality in Deep Neural Networks. arXiv, 2001.00254. Retrieved from https://arxiv.org/abs/2001.00254v1

    - We propose a novel metric called Block Dynamical Isometry, which measures the change of gradient norm
    - We propose a highly modularized statistical framework based on free probability
    - With our metric and framework we analyze extensive initialization, normalization, and network structures
    - Based on our analysis, we:
    - 1) Improve an activation function selection strategy for initialization techniques
    - 2) Propose a new configuration for weight normalization
    - 3) Propose a depth-aware way to derive coefficients in SeLU
    - 4) Propose a second moment normalization, which is theoretically 30% faster than BatchNorm without accuracy loss

Choe, Y. J., Ham, J., & Park, K. (2020). An Empirical Study of Invariant Risk Minimization. arXiv, 2004.05007. Retrieved from https://arxiv.org/abs/2004.05007v2

    - We empirically investigate IRMv1, which is the first practical algorithm to approximately solve IRM
    - We use ColoredMNIST and Stanford Sentiment Treebank (SST-2) datasets
    - IRMv1 performs better as the gap between training environments grows larger
    - IRMv1 can perform good even when the association between shape and label is only approximately invariant
    - IRMv1 can perform good with spurious token-to-label correlations in text classification tasks

D'Ascoli, S., Refinetti, M., Biroli, G., & Krzakala, F. (2020). Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime. arXiv, 2003.01054. Retrieved from https://arxiv.org/abs/2003.01054v2

Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv, 2002.06305. Retrieved from https://arxiv.org/abs/2002.06305v1

    - Problem: fine-tuning BERT is seed-dependent
    - We examine the effects of random initialization and random training order on fine-tuning BERT
    - We find that both contribute comparably, some weight initializations perform well across all tasks explored
    - Many fine-tuning trials diverge part of the way through training
    - We offer best practices for practitioners to stop training less promising runs early

Domingos, P. (2020). Every Model Learned by Gradient Descent Is Approximately a Kernel Machine. arXiv, 2012.00152. Retrieved from https://arxiv.org/abs/2012.00152v1

    - NNs trained by SGD, regardless of architecture, are approximately equivalent to kernel machines
    - Kernel machine store a subset of the training data points and match them to the query using the kernel
    - So, NNs are effectively a superposition of the training examples
    - This contrasts with the standard view of DL as a method for discovering representations from data
    - Our result also has significant implications for boosting algorithms, probabilistic graphical models and convex optimization
    - Also some discussion here: https://www.reddit.com/r/MachineLearning/comments/k8h01q/r_wide_neural_networks_are_feature_learners_not/?rdt=59640

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ...Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv, 2010.11929. Retrieved from https://arxiv.org/abs/2010.11929v2

Ellis, K., Wong, C., Nye, M., Sable-Meyer, M., Cary, L., Morales, L., ...Tenenbaum, J. B. (2020). DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning. arXiv, 2006.08381. Retrieved from https://arxiv.org/abs/2006.08381v1

Faghri, F., Duvenaud, D., Fleet, D. J., & Ba, J. (2020). A Study of Gradient Variance in Deep Learning. arXiv, 2007.04532. Retrieved from https://arxiv.org/abs/2007.04532v1

Fang, C., Lee, J. D., Yang, P., & Zhang, T. (2020). Modeling from Features: a Mean-field Framework for Over-parameterized Deep Neural Networks. arXiv, 2007.01452. Retrieved from https://arxiv.org/abs/2007.01452v1

    - A new framework to analyze neural network training
    - We capture the evolution of an over-parameterized DNN trained by Gradient Descent
    - Global convergence proof for over-parameterized DNN in the mean-field regime

Fort, S., Dziugaite, G. K., Paul, M., Kharaghani, S., Roy, D. M., & Ganguli, S. (2020). Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel. arXiv, 2010.15110. Retrieved from https://arxiv.org/abs/2010.15110v1

Furrer, D., van Zee, M., Scales, N., & Schärli, N. (2020). Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures. arXiv, 2007.08970. Retrieved from https://arxiv.org/abs/2007.08970v3

George, D., Lázaro-Gredilla, M., & Guntupalli, J. S. (2020). From CAPTCHA to Commonsense: How Brain Can Teach Us About Artificial Intelligence. Front. Comput. Neurosci., 14, 554097. Retrieved from https://www.frontiersin.org/articles/10.3389/fncom.2020.554097/full

Geva, M., Schuster, R., Berant, J., & Levy, O. (2020). Transformer Feed-Forward Layers Are Key-Value Memories. arXiv, 2012.14913. Retrieved from https://arxiv.org/abs/2012.14913v2

    - Feedforward layers in transformer-based language models operate as key-value memories
    - Each key correlates with a set of human-interpretable input patterns, such as n-grams or semantic topic
    - Each value can induce a distribution over the output vocabulary
    - This distribution correlates with the next-token distribution in the upper layers of the model
    - The learned patterns are human-interpretable
    - Lower layers tend to capture shallow patterns, while upper layers learn more semantic ones
    - The output of a feed-forward layer is a composition of its memories

Gorbunov, E., Danilova, M., & Gasnikov, A. (2020). Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping. arXiv, 2005.10785. Retrieved from https://arxiv.org/abs/2005.10785v2

    - We propose clipped-SSTM: a first-order method for smooth convex stochastic optimization
    - Clipped-SSTM is for heavy-tailed distributed noise in stochastic gradients
    - Clipped-SSTM is based on SGD and and gradient clipping
    - We derive some theoretical bounds

Hu, W., Xiao, L., & Pennington, J. (2020). Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks. arXiv, 2001.05992. Retrieved from https://arxiv.org/abs/2001.05992v1

    - Proof that orthogonal initialization speeds up convergence
    - With it, the width for efficient convergence is independent of the depth (without it does not)
    - Is related to the principle of dynamical isometry (https://arxiv.org/abs/1312.6120)

Huang, K., Wang, Y., Tao, M., & Zhao, T. (2020). Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? -- A Neural Tangent Kernel Perspective. arXiv, 2002.06262. Retrieved from https://arxiv.org/abs/2002.06262v2

Jastrzebski, S., Szymczak, M., Fort, S., Arpit, D., Tabor, J., Cho, K., & Geras, K. (2020). The Break-Even Point on Optimization Trajectories of Deep Neural Networks. arXiv, 2002.09572. Retrieved from https://arxiv.org/abs/2002.09572v1

    - We show that the key properties of the loss surface are strongly influenced by SGD in the early training phase
    - Using a large LR in the early training phase is beneficial from the optimization perspective
    - It reduces the variance of the gradient, and improves the conditioning of the covariance of gradients
    - Using a low LR in the early training phase results in bad conditioning even with BatchNorm

Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. arXiv, 2005.08926. Retrieved from https://arxiv.org/abs/2005.08926v2

    - Problem in neural ODEs: no mechanism for adjusting the trajectory based on subsequent observations
    - We demonstrate how this may be resolved through the mathematics of controlled differential equations
    - This is applicable to the partially observed irregularly-sampled multivariate time series
    - SOTA performance against similar (ODE or RNN based) models in empirical studies on a range of datasets
    - Theoretical results demonstrating universal approximation

Lengerich, B., Xing, E. P., & Caruana, R. (2020). Dropout as a Regularizer of Interaction Effects. arXiv, 2007.00823. Retrieved from https://arxiv.org/abs/2007.00823v2

    - We show that Dropout regularizes against higher-order interactions
    - So, high Dropout is useful when we need stronger regularization against spurious high-order interactions
    - When NNs are trained on data with important interaction effects, the optimal Dropout rate is lower
    - Weight decay and early stopping do not achieve Dropout’s regularization against high-order interactions
    - Caution should be exercised when interpreting Dropout-based uncertainty measures

Liu, L., Liu, X., Gao, J., Chen, W., & Han, J. (2020). Understanding the Difficulty of Training Transformers. arXiv, 2004.08249. Retrieved from https://arxiv.org/abs/2004.08249v3

    - It is known that SGD fails to train transformers effectively
    - We show that unbalanced gradients are not the root cause of this problem
    - We discover an effect inside the transformer that amplifies small parameter perturbations
    - We propose Admin (Adaptive model initialization) to stabilize the early stage’s training

Liu, C., Zhu, L., & Belkin, M. (2020). Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. arXiv, 2003.00307. Retrieved from https://arxiv.org/abs/2003.00307v2

    - We ovserve that NN optimization is generally not convex, even locally
    - Convexity is not the right framework for analysis of over-parameterized systems
    - Instead, wide NNs satisfy a variant of the Polyak-Lojasiewicz condition on most of the parameter space
    - This guarantees an efficient convergence by SGD to a global minimum
    - This is closely related to the condition number of the NTK

Lobacheva, E., Chirkova, N., Kodryan, M., & Vetrov, D. (2020). On Power Laws in Deep Ensembles. arXiv, 2007.08483. Retrieved from https://arxiv.org/abs/2007.08483v2

    - It was shown (https://arxiv.org/abs/2002.06470) that calibrated negative log-likelihood (CNLL) of a deep ensemble measures its quality of uncertainty estimation
    - We examine CNLL of a deep ensemble as a function of the ensemble size and the member network size
    - Under several conditions it follows a power law w. r. t. ensemble size or member network size
    - We find that one large network may perform worse than an ensemble of several medium-size networks with the same total number of parameters (we call this ensemble a memory split)
    - With these results we can guess the possible gain from the ensembling and the optimal memory split

Millidge, B., Tschantz, A., Seth, A. K., & Buckley, C. L. (2020). On the Relationship Between Active Inference and Control as Inference. arXiv, 2006.12964. Retrieved from https://arxiv.org/abs/2006.12964v3

Millidge, B., Tschantz, A., & Buckley, C. L. (2020). Whence the Expected Free Energy? arXiv, 2004.08128. Retrieved from https://arxiv.org/abs/2004.08128v5

Millidge, B., Tschantz, A., & Buckley, C. L. (2020). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. arXiv, 2006.04182. Retrieved from https://arxiv.org/abs/2006.04182v5

Mundt, M., Hong, Y., Pliushch, I., & Ramesh, V. (2020). A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning. arXiv, 2009.01797. Retrieved from https://arxiv.org/abs/2009.01797v3

Muthukumar, V., Narang, A., Subramanian, V., Belkin, M., Hsu, D., & Sahai, A. (2020). Classification vs regression in overparameterized regimes: Does the loss function matter? arXiv, 2005.08054. Retrieved from https://arxiv.org/abs/2005.08054v2

    - We analyze the overparameterized regime under the linear model with Gaussian features
    - In this case every training sample is a support vector
    - Consequently, the outcome of GD optimization is the same whether we use the hinge, square or logistic loss
    - On the other hand, the choice of test loss function results in a significant asymptotic difference: some overparameterized predictors will generalize poorly for square loss but well for 0-1 loss

Timothy P. Lillicrap, #., Adam Santoro, #., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. Nat. Rev. Neurosci., 32303713. Retrieved from https://pubmed.ncbi.nlm.nih.gov/32303713

Lu, Y., Ma, C., Lu, Y., Lu, J., & Ying, L. (2020). A Mean-field Analysis of Deep ResNet and Beyond: Towards Provable Optimization Via Overparameterization From Depth. arXiv, 2003.05508. Retrieved from https://arxiv.org/abs/2003.05508v2

    - Question: why do ResNets achieve zero training loss, while optimization landscape is highly non-convex?
    - We propose a new continuum limit of deep ResNets with a good landscape where every local minimizer is global
    - We apply existing mean-field analyses of two-layer networks to deep networks
    - We propose several novel training schemes which result in strong empirical performance

Melkman, A. A., Guo, S., Ching, W.-K., Liu, P., & Akutsu, T. (2020). On the Compressive Power of Boolean Threshold Autoencoders. arXiv, 2004.09735. Retrieved from https://arxiv.org/abs/2004.09735v1

Mixon, D. G., Parshall, H., & Pi, J. (2020). Neural collapse with unconstrained features. arXiv, 2011.11619. Retrieved from https://arxiv.org/abs/2011.11619v1

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training. arXiv, 2008.08186. Retrieved from https://arxiv.org/abs/2008.08186v2

    - One of the standard workflow practices is training beyond zero-error to zero-loss
    - We discuss a Terminal Phase of Training (TPT), when training accuracy is 1, but training loss is still lowering
    - We measure TPT for 3 deep architectures and 7 classification datasets
    - We expose a pervasive inductive bias we call Neural Collapse
    - 1) Last layer class representations collapse to points
    - 2) These points collapse to the vertices of a Simplex Equiangular Tight Frame (fig. 1)
    - Convergence to this simple structure is beneficial: it improves test performance and adversarial robustness

Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., & Lajoie, G. (2020). Gradient Starvation: A Learning Proclivity in Neural Networks. arXiv, 2011.09468. Retrieved from https://arxiv.org/abs/2011.09468v4

    - We provide a theoretical explanation for Gradient Starvation
    - It arises when loss is minimized by capturing only a subset of relevant features
    - Other predictive features fail to be discovered
    - Such a situation can be expected given certain statistical structure in training data
    - We propose Spectral Decoupling (SD): a regularization method aimed at decoupling feature learning dynamics
    - We experiment on classification and adversarial attack tasks

Raunak, V., Dalmia, S., Gupta, V., & Metze, F. (2020). On Long-Tailed Phenomena in Neural Machine Translation. arXiv, 2010.04924. Retrieved from https://arxiv.org/abs/2010.04924v1

    - Problem: NMT models struggle with generating low-frequency tokens
    - Penalizing low-confidence predictions hurts beam search performance
    - We propose Anti-Focal loss, a generalization of Focal loss and cross-entropy
    - Anti-Focal loss allocates less relative loss to low-confidence predictions
    - It leads to significant gains over cross-entropy, especially on the generation of low-frequency words

Queiruga, A. F., Erichson, N. B., Taylor, D., & Mahoney, M. W. (2020). Continuous-in-Depth Neural Networks. arXiv, 2008.02389. Retrieved from https://arxiv.org/abs/2008.02389v1

Sankar, A. R., Khasbage, Y., Vigneswaran, R., & Balasubramanian, V. N. (2020). A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization. arXiv, 2012.03801. Retrieved from https://arxiv.org/abs/2012.03801v2

    - We study the eigenspectra of the Hessian at each DNN layer
    - We propose a new regularizer: Layerwise Hessian Trace Regularization (HTR)
    - It forces Stochastic Gradient Descent to converge to flatter minima

Shen, D., Zheng, M., Shen, Y., Qu, Y., & Chen, W. (2020). A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation. arXiv, 2009.13818. Retrieved from https://arxiv.org/abs/2009.13818v2

    - Problem: adversarial training has been shown effective but requires expensive computation
    - We propose cutoff: a data augmentation strategy, where part of the input sentence is erased
    - For these samples we use a Jensen-Shannon Divergence consistency loss
    - We apply cutoff to both understanding and generation problems, including GLUE and NMT
    - Cutoff consistently outperforms adversarial training

Sun, R. (2020). Optimization for Deep Learning: An Overview. J. Oper. Res. Soc. China, 8(2), 249–294. doi: 10.1007/s40305-020-00309-6 https://www.ise.ncsu.edu/fuzzy-neural/wp-content/uploads/sites/9/2020/08/Optimization-for-deep-learning.pdf   
  
    - A survey paper. We discuss:  
    - The issue of undesirable spectrum, including gradient explosion/vanishing  
    - Solutions such as careful initialization, normalization, and skip connections  
    - SGD, adaptive gradient methods, and existing theoretical results  
    - Results on global landscape, mode connectivity, lottery ticket hypothesis and NTK

Sun, R., Li, D., Liang, S., Ding, T., & Srikant, R. (2020). The Global Landscape of Neural Networks: An Overview. arXiv, 2007.01429. Retrieved from https://arxiv.org/abs/2007.01429v1   
  
    - A survey paper. We discuss:  
    - That wide NNs may have sub-optimal local minima under certain assumptions  
    - Geometric properties of wide NNs  
    - Some modifications that eliminate sub-optimal local minima and/or decreasing paths to infinity  
    - Visualization and empirical explorations of the loss landscape  
    - Some convergence results  
    - Compared to another survey ("Optimization for Deep Learning: An Overview" from the same author), this article focuses on global landscape and contains formal theorem statements, while they covered many aspects of neural net optimization and did not present formal theorems.

Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Reinforcement Learning through Active Inference. arXiv, 2002.12636. Retrieved from https://arxiv.org/abs/2002.12636v1

Tschantz, A., Millidge, B., Seth, A. K., & Buckley, C. L. (2020). Control as Hybrid Inference. arXiv, 2007.05838. Retrieved from https://arxiv.org/abs/2007.05838v1

Wang, L., Shen, B., Zhao, N., & Zhang, Z. (2020). Is the Skip Connection Provable to Reform the Neural Network Loss Landscape? arXiv, 2006.05939. Retrieved from https://arxiv.org/abs/2006.05939v1

Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. arXiv, 2002.08791. Retrieved from https://arxiv.org/abs/2002.08791v4

    - We propose MultiSWAG to significantly improve Deep Ensembles  
    - MultiSWAG alleviates double descent  
    - Deep Ensembles are not a competing approach to Bayesian inference, but are a mechanism for Bayesian marginalization that provides a better approximation to the Bayesian predictive distribution than standard Bayesian approaches  
    - Fitting random labels (https://arxiv.org/abs/1611.03530) can be understood by reasoning about prior distributions over functions, and are not specific to NNs  
    - Gaussian processes can also perfectly fit images with random labels, yet generalize on the noise-free problem

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ...Liu, T.-Y. (2020). On Layer Normalization in the Transformer Architecture. arXiv, 2002.04745. Retrieved from https://arxiv.org/abs/2002.04745v2

    - We theoretically study why the LR warm-up stage is essential for transformers  
    - We show that in a recently proposed Pre-LN Transformer the gradients are well-behaved at initialization  
    - Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time

Yang, G. (2020). Tensor Programs II: Neural Tangent Kernel for Any Architecture. arXiv, 2006.14548. Retrieved from https://arxiv.org/abs/2006.14548v4

    - We review the tensor programs technique
    - We prove that infinitely-wide NN of any architecture has its NTK converge to a deterministic limit
    - We demonstrate how to calculate this limit
    - We decribe a Simple GIA Check to check gradient independence assumption (GIA) used in NTK
    - When Simple GIA Check fails, we show GIA can result in wrong answers
    - We implement infinite-width NTKs of RNN, transformer, and batch normalization in a repo

Yang, G. (2020). Tensor Programs III: Neural Matrix Laws. arXiv, 2009.10685. Retrieved from https://arxiv.org/abs/2009.10685v3

    - We study intinitely-wide NNs with a random matrix theory and derive the Free Independence Principle (FIP)
    - It justifies the calculation of Jacobian singular value distribution of intinitely-wide NN
    - It gives a new justification of gradient independence assumption used for calculating NTK
    - We generalize the Master Theorems from previous works

Yang, G., & Hu, E. J. (2020). Feature Learning in Infinite-Width Neural Networks. arXiv, 2011.14522. Retrieved from https://arxiv.org/abs/2011.14522v3

    - This is "Tensor Programs IV" (another 11-list version: https://proceedings.mlr.press/v139/yang21c/yang21c.pdf)
    - Using the Tensor Programs we adapt NTK to the case of pre-training and transfer learning such as with BERT
    - We compute an infinite-width limits on Word2Vec and few-shot learning on Omniglot via MAML
    - Such feature-learning limit outperforms both the NTK and the finite-width neural networks
    - We classify a space of NN parametrizations that generalizes standard, NTK, and Mean Field parametrizations
    - https://www.reddit.com/r/MachineLearning/comments/k8h01q/r_wide_neural_networks_are_feature_learners_not/
    - The title really should be something like “To Explain Pretraining and Transfer Learning, Wide Neural Networks Should Be Thought of as Feature Learners, Not Kernel Machines” but that’s really long
    - So, NNs can be kernel machines, but we can understand them better as feature learners
    - More precisely, the same NN can have different infinite-width limits, depending on the parametrization
    - A big contribution of this paper is classifying what kind of limits are possible

Yang, Z., Yu, Y., You, C., Steinhardt, J., & Ma, Y. (2020). Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. arXiv, 2002.11328. Retrieved from https://arxiv.org/abs/2002.11328v3

    - We study double descent and confirm that variance is unimodal or bell-shaped  
    - This occurs robustly for all models we considered  
    - Accuracy drops on OOD data comes from increased bias  
    - Deeper models decrease bias and increase variance (for both IID and OOD data)  
    - Increasing model depth may help combat the drop in OOD accuracy

Zhao, M., Zhu, Y., Shareghi, E., Vulić, I., Reichart, R., Korhonen, A., & Schütze, H. (2020). A Closer Look at Few-Shot Crosslingual Transfer: The Choice of Shots Matters. arXiv, 2012.15682. Retrieved from https://arxiv.org/abs/2012.15682v2

    - We conduct a largescale study of few-shot crosslingual transfer on diverse NLP tasks and languages
    - We find that the model exhibits a high degree of sensitivity to the selection of few shots
    - We provide an analysis of success and failure cases
    - We find that a straightforward fine-tuning outperforms several SOTA few-shot approaches

Zhao, P., Chen, P.-Y., Das, P., Ramamurthy, K. N., & Lin, X. (2020). Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. arXiv, 2005.00060. Retrieved from https://arxiv.org/abs/2005.00060v2

Zhou, W., Lin, B. Y., & Ren, X. (2020). IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization. arXiv, 2005.02178. Retrieved from https://arxiv.org/abs/2005.02178v2

    - It was shown that isotropic embeddings can significantly improve performance
    - We study how isotropic (unit-variance and uncorrelated) are output [CLS] embeddings of pre-trained language models
    - We find high variance in their standard deviation, and high correlation between different dimensions
    - We propose isotropic batch normalization (IsoBN) regularization that penalizes dominating principal components
    - IsoBN allows to learn more isotropic representations in fine-tuning
    - We achieve improvement on the average of seven NLU tasks

## 2021

Aguilera, M., Millidge, B., Tschantz, A., & Buckley, C. L. (2021). How particular is the physics of the free energy principle? arXiv, 2105.11203. Retrieved from https://arxiv.org/abs/2105.11203v3

Belkin, M. (2021). Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. arXiv, 2105.14368. Retrieved from https://arxiv.org/abs/2105.14368v1

    - A review paper on the foundations of DL through the prism of interpolation and over-parameterization

Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T.-Y., ...Zoph, B. (2021). Revisiting ResNets: Improved Training and Scaling Strategies. arXiv, 2103.07579. Retrieved from https://arxiv.org/abs/2103.07579v1

Benton, G. W., Maddox, W. J., Lotfi, S., & Wilson, A. G. (2021). Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling. arXiv, 2102.13042. Retrieved from https://arxiv.org/abs/2102.13042v2

Berariu, T., Czarnecki, W., De, S., Bornschein, J., Smith, S., Pascanu, R., & Clopath, C. (2021). A study on the plasticity of neural networks. arXiv, 2106.00042. Retrieved from https://arxiv.org/abs/2106.00042v2

    - We focus on plasticity, namely the ability of the model to keep learning; when NNs lose this ability?
    - For example, PackNet (Mallya & Lazebnik, 2017) eventually gets to a point where all neurons are frozen and learning is not possible anymore; alternatively, learning might become less data efficient (negative forward transfer)
    - We build on https://arxiv.org/abs/1910.08475, provide a hypothesis about it and study the implications

Bingham, G., & Miikkulainen, R. (2021). AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks. arXiv, 2109.08958. Retrieved from https://arxiv.org/abs/2109.08958v2

    - A weight initialization algorithm that automatically adapts to different architectures
    - Scales the weights by tracking the mean and variance of signals as they propagate through the network
    - Improves performance of convolutional, residual, and transformer networks

Bond-Taylor, S., Leach, A., Long, Y., & Willcocks, C. G. (2021). Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. arXiv, 2103.04922. Retrieved from https://arxiv.org/abs/2103.04922v4

    - A survey paper
    - Deep generative models reserch has fragmented into various approaches we review
    - They make trade-offs including run-time, diversity, and architectural restrictions

Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. arXiv, 2102.06171. Retrieved from https://arxiv.org/abs/2102.06171v1

    - Problem: BatchNorm has many undesirable properties (is computationally expensive, perform poorly when the batch size is too small, introduces a train-test discrepancy, is often the cause of subtle implementation errors, cannot be used for some tasks due to interaction between training examples - we discuss it in Appendix B)
    - Problem: Normalizer-Free networks are often unstable for large learning rates or strong data augmentations
    - We propose NFNets: Normalizer-Free networks with an adaptive gradient clipping technique to overcome instabilities
    - With our NFNets, we achieve SOTA on ImageNet, fast convergence, better fine-tuning performance

Cao, S. (2021). Choose a Transformer: Fourier or Galerkin. arXiv, 2105.14995. Retrieved from https://arxiv.org/abs/2105.14995v4

    - We apply self-attention to a data-driven operator learning problem related to PDE
    - We present three operator learning experiments
    - We demonstrate that the softmax normalization is sufficient but not necessary
    - We propose Fourier Transformer (FT) with the Fourier-type encoder, and the Galerkin Transformer (GT) with the Galerkin-type encoder (?) to improve quality in PDE-related operator learning tasks
    - https://scaomath.github.io/blog/galerkin-transformer/

Chen, Y., Huang, W., Nguyen, L. M., & Weng, T.-W. (2021). On the Equivalence between Neural Network and Support Vector Machine. arXiv, 2111.06063. Retrieved from https://arxiv.org/abs/2111.06063v2

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

Kanavati, F., & Tsuneki, M. (2021). Partial transfusion: on the expressive influence of trainable batch norm parameters for transfer learning. arXiv, 2102.05543. Retrieved from https://arxiv.org/abs/2102.05543v1

    - It is typically recommended to fine-tune the model with the BatchNorm layers kept in inference mode
    - We find that fine-tuning only the scale and shift weights of the BatchNorm leads to similar performance
as to fine-tuning all of the weights, with the added benefit of faster convergence

Lafon, M., & Thomas, A. (2021). Understanding the Double Descent Phenomenon in Deep Learning. arXiv, 2403.10459. Retrieved from https://arxiv.org/abs/2403.10459v1

Lanillos, P., Meo, C., Pezzato, C., Meera, A. A., Baioumy, M., Ohata, W., ...Tani, J. (2021). Active Inference in Robotics and Artificial Agents: Survey and Challenges. arXiv, 2112.01871. Retrieved from https://arxiv.org/abs/2112.01871v1

Larsen, B. W., Fort, S., Becker, N., & Ganguli, S. (2021). How many degrees of freedom do we need to train deep networks: a loss landscape perspective. arXiv, 2107.05802. Retrieved from https://arxiv.org/abs/2107.05802v2

Liu, F., Suykens, J. A. K., & Cevher, V. (2021). On the Double Descent of Random Features Models Trained with SGD. arXiv, 2110.06910. Retrieved from https://arxiv.org/abs/2110.06910v6

Liu, M., Chen, L., Du, X., Jin, L., & Shang, M. (2021). Activated Gradients for Deep Neural Networks. arXiv, 2107.04228. Retrieved from https://arxiv.org/abs/2107.04228v1

Lobacheva, E., Kodryan, M., Chirkova, N., Malinin, A., & Vetrov, D. (2021). On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay. arXiv, 2106.15739. Retrieved from https://arxiv.org/abs/2106.15739v3

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

Courtois, A., Morel, J.-M., & Arias, P. (2022). Can neural networks extrapolate? Discussion of a theorem by Pedro Domingos. arXiv, 2211.03566. Retrieved from https://arxiv.org/abs/2211.03566v1

    - We discuss a theorem proved in https://arxiv.org/abs/2012.00152
    - We extend the proof to the discrete setting and the multi-dimensional case
    - It is unclear how the interpretability of the NTK would apply to high dimensional settings such as images
    - Our experiments seem to confirm Domingos’ interpretation of his theorem

Engel, A., Wang, Z., Sarwate, A. D., Choudhury, S., & Chiang, T. (2022). TorchNTK: A Library for Calculation of Neural Tangent Kernels of PyTorch Models. arXiv, 2205.12372. Retrieved from https://arxiv.org/abs/2205.12372v1

Ergen, T., Neyshabur, B., & Mehta, H. (2022). Convexifying Transformers: Improving optimization and understanding of transformer networks. arXiv, 2211.11052. Retrieved from https://arxiv.org/abs/2211.11052v1

Fedus, W., Dean, J., & Zoph, B. (2022). A Review of Sparse Expert Models in Deep Learning. arXiv, 2209.01667. Retrieved from https://arxiv.org/abs/2209.01667v1

Fort, S., Cubuk, E. D., Ganguli, S., & Schoenholz, S. S. (2022). What does a deep neural network confidently perceive? The effective dimension of high certainty class manifolds and their low confidence boundaries. arXiv, 2210.05546. Retrieved from https://arxiv.org/abs/2210.05546v1

Friston, K. J., Ramstead, M. J. D., Kiefer, A. B., Tschantz, A., Buckley, C. L., Albarracin, M., ...René, G. (2022). Designing Ecosystems of Intelligence from First Principles. arXiv, 2212.01354. Retrieved from https://arxiv.org/abs/2212.01354v2

Friston, K., Da Costa, L., Sajid, N., Heins, C., Ueltzhoffer, K., Pavliotis, G., & Parr, T. (2022). The free energy principle made simpler but not too simple. Phys. Rep. Retrieved from https://www.semanticscholar.org/paper/The-free-energy-principle-made-simpler-but-not-too-Friston-Costa/e54427100b2de8187fe3b96303653b6220aaad44

Isomura, T., Shimazaki, H., & Friston, K. J. (2022). Canonical neural networks perform active inference. Commun. Biol., 5(55), 1–15. Retrieved from https://www.nature.com/articles/s42003-021-02994-2

Ivanova, A. A., Schrimpf, M., Anzellotti, S., Zaslavsky, N., Fedorenko, E., & Isik, L. (2022). Beyond linear regression: mapping models in cognitive neuroscience should align with research goals. arXiv, 2208.10668. Retrieved from https://arxiv.org/abs/2208.10668v1

Hafner, D., Lee, K.-H., Fischer, I., & Abbeel, P. (2022). Deep Hierarchical Planning from Pixels. arXiv, 2206.04114. Retrieved from https://arxiv.org/abs/2206.04114v1

Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv, 2212.13345. Retrieved from https://arxiv.org/abs/2212.13345v1

Irie, K., Csordás, R., & Schmidhuber, J. (2022). The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention. arXiv, 2202.05798. Retrieved from https://arxiv.org/abs/2202.05798v2

    - It is known that linear layers in NNs trained by GD can be expressed as a key/value/query-attention operation
    - It stores training datapoints and outputs unnormalised dot attention over the entire training experience
    - (none of the mathematical results we’ll discuss is novel)
    - This dual formulation allows for visualising attention weights over all training patterns, given a test input
    - This is not easy: the memory storage requirement forces us to conduct experiments with small datasets
    - Our analysis is not applicable to models which are already trained
    - We experiment on image classification (single-task, multi-task, continual learning) and language modeling
    - We observe many interesting patterns in various scenarios

Juneja, J., Bansal, R., Cho, K., Sedoc, J., & Saphra, N. (2022). Linear Connectivity Reveals Generalization Strategies. arXiv, 2205.12411. Retrieved from https://arxiv.org/abs/2205.12411v5

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., ...Farhadi, A. (2022). Matryoshka Representation Learning. arXiv, 2205.13147. Retrieved from https://arxiv.org/abs/2205.13147v4

    - We propose Matryoshka Representation Learning (MRL) to learn coarse-to-fine representations
    - This allows shorter embeddings and improvements for long-tail few-shot classification
    - This is a flexible representation that can adapt to multiple downstream tasks with varying computational resources

Li, Y. (2022). A Short Survey of Systematic Generalization. arXiv, 2211.11956. Retrieved from https://arxiv.org/abs/2211.11956v1

Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi, S. J., ...Kumar, S. (2022). The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers. arXiv, 2210.06313. Retrieved from https://arxiv.org/abs/2210.06313v2

Liu, Z., Michaud, E. J., & Tegmark, M. (2022). Omnigrok: Grokking Beyond Algorithmic Data. arXiv, 2210.01117. Retrieved from https://arxiv.org/abs/2210.01117v2

Malladi, S., Wettig, A., Yu, D., Chen, D., & Arora, S. (2022). A Kernel-Based View of Language Model Fine-Tuning. arXiv, 2210.05643. Retrieved from https://arxiv.org/abs/2210.05643v4

Millidge, B., Salvatori, T., Song, Y., Bogacz, R., & Lukasiewicz, T. (2022). Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation? arXiv, 2202.09467. Retrieved from https://arxiv.org/abs/2202.09467v1

Mohamadi, M. A., Bae, W., & Sutherland, D. J. (2022). A Fast, Well-Founded Approximation to the Empirical Neural Tangent Kernel. arXiv, 2206.12543. Retrieved from https://arxiv.org/abs/2206.12543v3

Novak, R., Sohl-Dickstein, J., & Schoenholz, S. S. (2022). Fast Finite Width Neural Tangent Kernel. arXiv, 2206.08720. Retrieved from https://arxiv.org/abs/2206.08720v1

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv, 2201.02177. Retrieved from https://arxiv.org/abs/2201.02177v1

Rangwani, H., Aithal, S. K., Mishra, M., & Babu, R. V. (2022). Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data. arXiv, 2212.13827. Retrieved from https://arxiv.org/abs/2212.13827v1

Ramesh, A., Kirsch, L., van Steenkiste, S., & Schmidhuber, J. (2022). Exploring through Random Curiosity with General Value Functions. arXiv, 2211.10282. Retrieved from https://arxiv.org/abs/2211.10282v1

Rissanen, S., Heinonen, M., & Solin, A. (2022). Generative Modelling With Inverse Heat Dissipation. arXiv, 2206.13397. Retrieved from https://arxiv.org/abs/2206.13397v7

Ramstead, M. J. D., Sakthivadivel, D. A. R., Heins, C., Koudahl, M., Millidge, B., Da Costa, L., ...Friston, K. J. (2022). On Bayesian Mechanics: A Physics of and by Beliefs. arXiv, 2205.11543. Retrieved from https://arxiv.org/abs/2205.11543v4

Sander, M. E., Ablin, P., & Peyré, G. (2022). Do Residual Neural Networks discretize Neural Ordinary Differential Equations? arXiv, 2205.14612. Retrieved from https://arxiv.org/abs/2205.14612v2

    - Are discrete dynamics defined by a ResNet close to the continuous one of a Neural ODE?
    - Several theoretical results
    - A simple technique to train ResNets without storing activations
    - Recover the approximated activations during the backward pass by using a reverse-time Euler scheme
    - Fine-tuning very deep ResNets without memory consumption in the residual layers

Sutton, R. S., Bowling, M., & Pilarski, P. M. (2022). The Alberta Plan for AI Research. arXiv, 2208.11173. Retrieved from https://arxiv.org/abs/2208.11173v3

Sutton, R. S., Machado, M. C., Holland, G. Z., Szepesvari, D., Timbers, F., Tanner, B., & White, A. (2022). Reward-Respecting Subtasks for Model-Based Reinforcement Learning. arXiv, 2202.03466. Retrieved from https://arxiv.org/abs/2202.03466v4

Träuble, F., Goyal, A., Rahaman, N., Mozer, M., Kawaguchi, K., Bengio, Y., & Schölkopf, B. (2022). Discrete Key-Value Bottleneck. arXiv, 2207.11240. Retrieved from https://arxiv.org/abs/2207.11240v3

    - We tackle the problem of catastrophic forgetting during fine-tuning
    - We propose an architecture built upon a discrete bottleneck of separate and learnable key-value codes
    - The encoded input is used to select the nearest keys, and the corresponding values are fed to the decoder
    - This enables localized and context-dependent model updates
    - It allows learning under distribution shifts and reduces the complexity of the hypothesis class
    - We evaluate on class-incremental learning scenarios wide variety of pre-trained models, outperforming baselines

Vanchurin, V., Wolf, Y. I., Katsnelson, M. I., & Koonin, E. V. (2022). Toward a theory of evolution as multilevel learning. Proc. Natl. Acad. Sci. U.S.A., 119(6), e2120037119. Retrieved from https://doi.org/10.1073/pnas.2120037119

Wang, H., Ma, S., Huang, S., Dong, L., Wang, W., Peng, Z., ...Wei, F. (2022). Foundation Transformers. arXiv, 2210.06423. Retrieved from https://arxiv.org/abs/2210.06423v2

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., & Wei, F. (2022). DeepNet: Scaling Transformers to 1,000 Layers. arXiv, 2203.00555. Retrieved from https://arxiv.org/abs/2203.00555v1

Weng, Lilian. (Sep 2022). Some math behind neural tangent kernel. Lil’Log. https://lilianweng.github.io/posts/2022-09-08-ntk/.

    - A blogpost review on a small number of core papers in NTK
    - The goal is to show all the math behind NTK in a clear and easy-to-follow format

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., ...Schmidt, L. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. arXiv, 2203.05482. Retrieved from https://arxiv.org/abs/2203.05482v3

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., ...Gao, J. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv, 2203.03466. Retrieved from https://arxiv.org/abs/2203.03466v2

Yuan, H., Yuan, Z., Tan, C., Huang, F., & Huang, S. (2022). HyPe: Better Pre-trained Language Model Fine-tuning with Hidden Representation Perturbation. arXiv, 2212.08853. Retrieved from https://arxiv.org/abs/2212.08853v2

    - Problem: fine-tuning LMs poses problems such as over-fitting or representation collapse
    - We propose HyPe fine-tuning technique: we inject random noise between transformer layers
    - This outperforms vanilla fine-tuning with negligible computational overheads

Zhou, J., You, C., Li, X., Liu, K., Liu, S., Qu, Q., & Zhu, Z. (2022). Are All Losses Created Equal: A Neural Collapse Perspective. arXiv, 2210.02192. Retrieved from https://arxiv.org/abs/2210.02192v2

Zhou, J., Li, X., Ding, T., You, C., Qu, Q., & Zhu, Z. (2022). On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features. arXiv, 2203.01238. Retrieved from https://arxiv.org/abs/2203.01238v2

Zhu, Z., Liu, F., Chrysos, G. G., & Cevher, V. (2022). Robustness in deep learning: The good (width), the bad (depth), and the ugly (initialization). arXiv, 2209.07263. Retrieved from https://arxiv.org/abs/2209.07263v4

## 2023

Abbas, Z., Zhao, R., Modayil, J., White, A., & Machado, M. C. (2023). Loss of Plasticity in Continual Deep Reinforcement Learning. arXiv, 2303.07507. Retrieved from https://arxiv.org/abs/2303.07507v1

Altintas, G. S., Bachmann, G., Noci, L., & Hofmann, T. (2023). Disentangling Linear Mode-Connectivity. arXiv, 2312.09832. Retrieved from https://arxiv.org/abs/2312.09832v1

Andriushchenko, M., Croce, F., Müller, M., Hein, M., & Flammarion, N. (2023). A Modern Look at the Relationship between Sharpness and Generalization. arXiv, 2302.07011. Retrieved from https://arxiv.org/abs/2302.07011v2

Araujo, A., Havens, A., Delattre, B., Allauzen, A., & Hu, B. (2023). A Unified Algebraic Perspective on Lipschitz Neural Networks. arXiv, 2303.03169. Retrieved from https://arxiv.org/abs/2303.03169v2

Arbel, J., Pitas, K., Vladimirova, M., & Fortuin, V. (2023). A Primer on Bayesian Neural Networks: Review and Debates. arXiv, 2309.16314. Retrieved from https://arxiv.org/abs/2309.16314v1

Barabási, D. L., Bianconi, G., Bullmore, E., Burgess, M., Chung, S., Eliassi-Rad, T., ...Buzsáki, G. (2023). Neuroscience needs Network Science. arXiv, 2305.06160. Retrieved from https://arxiv.org/abs/2305.06160v2

Bombari, S., Kiyani, S., & Mondelli, M. (2023). Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels. arXiv, 2302.01629. Retrieved from https://arxiv.org/abs/2302.01629v2

Irie, K., Csordás, R., & Schmidhuber, J. (2023). Automating Continual Learning. arXiv, 2312.00276. Retrieved from https://arxiv.org/abs/2312.00276v1

    - We propose Automated Continual Learning (ACL) to tackle the problem of catastrophic forgetting
    - ACL encodes good performance on both old and new tasks into its meta-learning objectives
    - ACL requires training settings similar to those of few-shot/meta learning problems
    - We demonstrate that ACL effectively solves "in-context catastrophic forgetting"

Isomura, T. (2023). Bayesian mechanics of self-organising systems. arXiv, 2311.10216. Retrieved from https://arxiv.org/abs/2311.10216v1

Cai, C., Hy, T. S., Yu, R., & Wang, Y. (2023). On the Connection Between MPNN and Graph Transformer. arXiv, 2301.11956. Retrieved from https://arxiv.org/abs/2301.11956v4

Chen, L., Lukasik, M., Jitkrittum, W., You, C., & Kumar, S. (2023). It's an Alignment, Not a Trade-off: Revisiting Bias and Variance in Deep Models. arXiv, 2310.09250. Retrieved from https://arxiv.org/abs/2310.09250v1

Cirone, N. M., Lemercier, M., & Salvi, C. (2023). Neural signature kernels as infinite-width-depth-limits of controlled ResNets. arXiv, 2303.17671. Retrieved from https://arxiv.org/abs/2303.17671v2

    - We consider randomly initialized controlled ResNets defined as Euler-discretizations of neural controlled differential equations, a unified architecture which enconpasses both RNNs and ResNets
    - Study convergence in the infinite-depth regime

Dang, H., Tran, T., Osher, S., Tran-The, H., Ho, N., & Nguyen, T. (2023). Neural Collapse in Deep Linear Networks: From Balanced to Imbalanced Data. arXiv, 2301.00437. Retrieved from https://arxiv.org/abs/2301.00437v5

Du, Y., & Nguyen, D. (2023). Measuring the Instability of Fine-Tuning. arXiv, 2302.07778. Retrieved from https://arxiv.org/abs/2302.07778v2

    - Problem: fine-tuning pre-trained LMs is sensitive to random seed, especially on small datasets
    - Most previous studies measure only the standard deviation of performance scores between runs
    - We analyze six other measures quantifying instability at different levels of granularity
    - We reassess existing instability mitigation methods

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

Jacobs, B., & Stein, D. (2023). Pearl's and Jeffrey's Update as Modes of Learning in Probabilistic Programming. arXiv, 2309.07053. Retrieved from https://arxiv.org/abs/2309.07053v2

Jordan, K. (2023). Calibrated Chaos: Variance Between Runs of Neural Network Training is Harmless and Inevitable. arXiv, 2304.01910. Retrieved from https://arxiv.org/abs/2304.01910v1

    - Problem: typical NN trainings have substantial variance in test-set performance between repeated runs
    - This impedes hyperparameter comparison and training reproducibility
    - We show that while variance on test-sets is high, variance on test-distributions is low
    - This variance is harmless and unavoidable
    - We conduct preliminary studies of distribution-shift, fine-tuning, data augmentation and learning rate through the lens of variance between runs

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

Schaeffer, R., Khona, M., Robertson, Z., Boopathy, A., Pistunova, K., Rocks, J. W., ...Koyejo, O. (2023). Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle. arXiv, 2303.14151. Retrieved from https://arxiv.org/abs/2303.14151v1

Shen, E., Farhadi, A., & Kusupati, A. (2023). Are "Hierarchical" Visual Representations Hierarchical? arXiv, 2311.05784. Retrieved from https://arxiv.org/abs/2311.05784v2

    - We create HierNet, 12 datasets spanning 3 kinds of hierarchy from the BREEDs subset of ImageNet
    - We evaluate Hyperbolic and Matryoshka Representations
    - We conclude that they do not capture hierarchy better than the standard representations
    - But they can assist in other aspects like search efficiency and interpretability

Shen, K., Guo, J., Tan, X., Tang, S., Wang, R., & Bian, J. (2023). A Study on ReLU and Softmax in Transformer. arXiv, 2302.06461. Retrieved from https://arxiv.org/abs/2302.06461v1

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

Wortsman, M., Lee, J., Gilmer, J., & Kornblith, S. (2023). Replacing softmax with ReLU in Vision Transformers. arXiv, 2309.08586. Retrieved from https://arxiv.org/abs/2309.08586v2

Wu, J., Yang, T., Hao, X., Hao, J., Zheng, Y., Wang, W., & Taylor, M. E. (2023). PORTAL: Automatic Curricula Generation for Multiagent Reinforcement Learning. AAMAS '23: Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems. International Foundation for Autonomous Agents and Multiagent Systems. Retrieved from https://dl.acm.org/doi/10.5555/3545946.3598967

Xie, L., Wei, L., Zhang, X., Bi, K., Gu, X., Chang, J., & Tian, Q. (2023). Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models. arXiv, 2306.08641. Retrieved from https://arxiv.org/abs/2306.08641v1

Xu, Z., Wang, Y., Frei, S., Vardi, G., & Hu, W. (2023). Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data. arXiv, 2310.02541. Retrieved from https://arxiv.org/abs/2310.02541v1

Yang, G., & Littwin, E. (2023). Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit. arXiv, 2308.01814. Retrieved from https://arxiv.org/abs/2308.01814v2

Yang, G., Yu, D., Zhu, C., & Hayou, S. (2023). Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. arXiv, 2310.02244. Retrieved from https://arxiv.org/abs/2310.02244v5

Ye, J., Zhu, Z., Liu, F., Shokri, R., & Cevher, V. (2023). Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks. arXiv, 2310.20579. Retrieved from https://arxiv.org/abs/2310.20579v1

Zhao, M., Alver, S., van Seijen, H., Laroche, R., Precup, D., & Bengio, Y. (2023). Consciousness-Inspired Spatio-Temporal Abstractions for Better Generalization in Reinforcement Learning. arXiv, 2310.00229. Retrieved from https://arxiv.org/abs/2310.00229v3

Zheng, C., Wu, G., Bao, F., Cao, Y., Li, C., & Zhu, J. (2023). Revisiting Discriminative vs. Generative Classifiers: Theory and Implications. arXiv, 2302.02334. Retrieved from https://arxiv.org/abs/2302.02334v2

Zhu, L., Liu, C., Radhakrishnan, A., & Belkin, M. (2023). Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning. arXiv, 2306.04815. Retrieved from https://arxiv.org/abs/2306.04815v2

## 2024

Afkanpour, A., Khazaie, V. R., Ayromlou, S., & Forghani, F. (2024). Can Generative Models Improve Self-Supervised Representation Learning? arXiv, 2403.05966. Retrieved from https://arxiv.org/abs/2403.05966v1

Anthony, Q., Tokpanov, Y., Glorioso, P., & Millidge, B. (2024). BlackMamba: Mixture of Experts for State-Space Models. arXiv, 2402.01771. Retrieved from https://arxiv.org/abs/2402.01771v1

Guo, L., Ross, K., Zhao, Z., Andriopoulos, G., Ling, S., Xu, Y., & Dong, Z. (2024). Cross Entropy versus Label Smoothing: A Neural Collapse Perspective. arXiv, 2402.03979. Retrieved from https://arxiv.org/abs/2402.03979v2

Hirono, Y., Tanaka, A., & Fukushima, K. (2024). Understanding Diffusion Models by Feynman's Path Integral. arXiv, 2403.11262. Retrieved from https://arxiv.org/abs/2403.11262v1

Huang, Q., Wake, N., Sarkar, B., Durante, Z., Gong, R., Taori, R., ...Gao, J. (2024). Position Paper: Agent AI Towards a Holistic Intelligence. arXiv, 2403.00833. Retrieved from https://arxiv.org/abs/2403.00833v1

Humayun, A. I., Balestriero, R., & Baraniuk, R. (2024). Deep Networks Always Grok and Here is Why. arXiv, 2402.15555. Retrieved from https://arxiv.org/abs/2402.15555v1

Pagliardini, M., Mohtashami, A., Fleuret, F., & Jaggi, M. (2024). DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging. arXiv, 2402.02622. Retrieved from https://arxiv.org/abs/2402.02622v2

Paolo, G., Gonzalez-Billandon, J., & Kégl, B. (2024). A call for embodied AI. arXiv, 2402.03824. Retrieved from https://arxiv.org/abs/2402.03824v1

Peters, B., DiCarlo, J. J., Gureckis, T., Haefner, R., Isik, L., Tenenbaum, J., ...Kriegeskorte, N. (2024). How does the primate brain combine generative and discriminative computations in vision? arXiv, 2401.06005. Retrieved from https://arxiv.org/abs/2401.06005v1

Sohl-Dickstein, J. (2024). The boundary of neural network trainability is fractal. arXiv, 2402.06184. Retrieved from https://arxiv.org/abs/2402.06184v1

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
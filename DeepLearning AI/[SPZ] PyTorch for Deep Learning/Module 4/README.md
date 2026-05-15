**Module 4 Resources**

**Activation functions (theory & diversity)**

- Ramachandran et al., _Searching for Activation Functions_ (Swish). Introduces an automated search that discovers Swish, explaining why smooth, non-monotonic activations can outperform ReLU on modern nets. [arXiv](https://arxiv.org/abs/1710.05941)
- Roboflow - _What is an Activation Function?_ A practical roundup of common activations (ReLU, GELU, Swish, Mish) with when/why to use each. [Roboflow Blog](https://blog.roboflow.com/activation-function-computer-vision)
- Activation Functions - EXPLAINED! A fast, visual tour of activation shapes, intuition, and trade-offs in training. [YouTube](https://www.youtube.com/watch?v=s-V7gKrsels)

**Convolution layers, kernels, padding/stride, pooling**

- CS231n notes: Intuitive walkthrough of convolution, stride, padding, pooling, and shape tracking with worked examples. [cs231n.github.io](https://cs231n.github.io/convolutional-networks)
- Dumoulin & Visin: A guide to convolution arithmetic. Exact formulas for output sizes (incl. transposed convs) to compute dimensions layer by layer. [arXiv](https://arxiv.org/abs/1603.07285)
- Distill: Computing Receptive Fields. Clear visuals showing how receptive fields grow across CNN stacks and why it matters. [Distill](https://distill.pub/2019/computing-receptive-fields)

**CIFAR-10 & the CIFAR "family"**

- CIFAR-10/100 (official page). Dataset specs, class lists, and downloads for the canonical small-image benchmarks. [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html)
- SVHN (Stanford). A real-world street-view digits dataset that's tougher than MNIST and great for testing robustness to natural noise.[huggingface](https://huggingface.co/datasets/Genius-Society/svhn)
- STL-10 (Stanford). CIFAR-like classes at higher resolution with an unlabeled split for unsupervised/representation learning. [Computer Science](https://cs.stanford.edu/~acoates/stl10)

**Dropout**

- Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR). The classic paper showing how randomly dropping units reduces co-adaptation and improves generalization. [Journal of Machine Learning Research](https://jmlr.org/papers/v15/srivastava14a.html)
- Dropout layer in Neural Network - Quick Explained. Short demo of how dropout behaves at train vs. eval and why it helps. [YouTube](https://www.youtube.com/watch?v=Fv8O4MvanJY)
- DEV Community - Dropout in Neural Networks. Beginner-friendly explanation with code snippets and practical tips. [DEV Community](https://dev.to/fotiecodes/dropout-in-neural-networks-simplified-explanation-for-beginners-2oj6)

**Architectures: theory, practice & design (how to choose, design, optimize)**

- Designing Network Design Spaces (RegNet). Systematically explores architecture families to derive simple depth/width rules under FLOPs/latency constraints. [openaccess.thecvf.com](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf)
- Google AI Blog - EfficientNet. Shows how compound scaling (depth/width/resolution) guides model sizing for a given compute budget. [research.google](https://research.google/blog/efficientnet-improving-accuracy-and-efficiency-through-automl-and-model-scaling)
- CS231n: Training Neural Networks I. Concrete heuristics for picking architectures and tuning regularization/optimizers to make CNNs train reliably. [youtube.com](https://www.youtube.com/watch?v=wEoyxE0GP2M)

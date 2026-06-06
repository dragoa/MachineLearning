**Module 5 Resources**

**Metrics & losses (macro vs micro, when to use what)**
- scikit-learn: model evaluation guide (averaging & metric choice). Defines precision/recall/F1 with macro/micro/weighted averaging and explains when each averaging scheme is appropriate. [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Google ML Crash Course (precision/recall/F1). A quick visual refresher on confusion matrices and how precision/recall/F1 trade off in practice. [Google for Developers](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

**Learning rate & LR schedulers**
- Jeremy Jordan — “Setting the learning rate”. Intuition and recipes for LR range tests, step/cosine schedules, and picking stable starting values. [Jeremy Jordan](https://www.jeremyjordan.me/nn-learning-rate)

**Deep-learning hypers (regularization, batch size, optimizers)**
- Karpathy — “A Recipe for Training Neural Networks”. A pragmatic checklist of what to tune first (data/baselines), how to debug, and when to add regularization. [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe)
- Ruder — “Overview of gradient descent optimization algorithms”. Clear comparisons of SGD + momentum vs Adagrad/RMSprop/Adam and guidance on when to prefer each. [arXiv](https://arxiv.org/abs/1609.04747) [ruder.io](https://www.ruder.io/optimizing-gradient-descent)
- Batch size: stability, generalization & speed. Practical heuristics on choosing batch size to balance noise, convergence speed, memory, and throughput. [MachineLearningMastery.com](https://www.machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size)

**Bayesian optimization (for HPO)**
- Will Koehrsen — conceptual explainer.Gentle, math-light introduction to Bayesian optimization for hyperparameter tuning with concrete examples. [willkoehrsen.github.io](https://willkoehrsen.github.io/bayesian/machine%20learning/explanation/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning)
- Distill — “Exploring Bayesian Optimization”. Interactive intuition for Gaussian processes, acquisition functions, and exploration–exploitation trade-offs. [Distill](https://distill.pub/2020/bayesian-optimization)

**Importance of architecture parameters (depth/width, kernels, scaling)**
- Paper (RegNet): Designing Network Design Spaces. Shows how simple depth/width rules define performant model families under FLOPs/latency constraints. [arXiv](https://arxiv.org/abs/2003.13678)
- Google Research Blog — EfficientNet (compound scaling). Explains scaling depth, width, and resolution together to match a compute budget. [research.google](https://research.google/blog/efficientnet-improving-accuracy-and-efficiency-through-automl-and-model-scaling)
- Paperspace blog — “CNN dimensions & performance”. Links kernel sizes, width/depth, and feature map shapes to accuracy, memory, and compute cost. [Paperspace by DigitalOcean Blog](https://blog.paperspace.com/convolutional-neural-network-dimensions-model-performance)

**Efficiency & multi-criteria model selection (memory, latency, throughput, power)**
- Horace He. Making Deep Learning go Brrrr. Principles for identifying whether you’re compute-, memory-bandwidth-, or overhead-bound and what to optimize first. [horace.io](https://horace.io/brrr_intro.html)
- Speechmatics — “How to Accurately Time CUDA Kernels in PyTorch”. How to measure latency correctly with warm-ups, CUDA events, and synchronizations. [blog.speechmatics.com](https://blog.speechmatics.com/cuda-timings)
- NVIDIA Dev Blog — “Understanding Overhead and Latency in Nsight Systems”. Reading profiler traces to separate copy, compute, and idle time. [NVIDIA Developer](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems)
- ml.energy — “Measuring GPU Energy: Best Practices”. Practical guidance on tracking energy/power (NVML), aligning measurements to workload segments, and reporting. [ml.energy](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices)

**Module 3 Resources**

**Learning from Data**
- MIT CSAIL · Data-Centric AI (lecture notes/overview). Frames model-centric vs data-centric workflows and lays out a practical recipe for improving data quality through iteration, labeling standards, and targeted error analysis. [https://datacentricai.org/](https://dcai.csail.mit.edu/2024/data-centric-model-centric/)

- CACM: The Principles of Data-Centric AI. A quotable set of principles for dataset design, versioning, labeling consistency, augmentation, and evaluation that shift focus from models to data. [https://cacm.acm.org/](https://cacm.acm.org/research/the-principles-of-data-centric-ai/)

- ArXiv survey: Data-Centric Artificial Intelligence. A broad taxonomy and literature map covering data assessment, curation, augmentation, weak supervision, and governance for ML. [https://arxiv.org/](https://arxiv.org/html/2212.11854v4)


**Data Organization & Leakage Prevention**
- Scikit-learn cross-validation guide. Explains when to use KFold, StratifiedKFold, GroupKFold, and time-aware splits, plus common leakage pitfalls to avoid. https://scikit-learn.org/stable/modules/cross_validation.html

- Data leakage in ML (MachineLearningMastery). Concrete tabular examples of target/leaky features and how to prevent leakage via proper pipelines and validation design. [https://machinelearningmastery.com/data-leakage-machine-learning/](https://machinelearningmastery.com/data-leakage-machine-learning/)


**Oxford Flowers 102 Dataset**

- Official VGG page (Oxford). Dataset description, class list, image counts, official splits, and download links for the 102-category flowers set. https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

- Dataset card (Huggingface). Quick specs, features, loading snippet, and links to benchmarks/SOTA for Flowers-102. [https://huggingface.co/datasets/oxford_flowers102](https://huggingface.co/datasets/Voxel51/OxfordFlowers102)

- Original paper reference (overview). Background on how the dataset was collected and the evaluation protocol used in the original work. [https://www.robots.ox.ac.uk/~vgg/](https://www.researchgate.net/publication/221551861_Automated_Flower_Classification_over_a_Large_Number_of_Classes)


**Preprocessing & Transform Pipelines**

- CS231n notes/slides (data preprocessing). Intuitive treatment of mean subtraction, standardization, resizing/cropping, and why to keep preprocessing consistent between train and test. https://cs231n.stanford.edu/ https://cs231n.stanford.edu/slides/2023/lecture_7.pdf

- Raschka: Feature scaling/normalization (blog primer). Clear, math-light comparison of standardization vs normalization and when each is appropriate. https://sebastianraschka.com/Articles/2014_about_feature_scaling.html

- Voxel51 blog: image preprocessing best practices. Practical guidance on resizing strategies, normalization choices, and handling domain shifts with visualization in mind. https://voxel51.com/blog/image-preprocessing-best-practices-to-optimize-your-ai-workflows


**Data Augmentation & Robustness**

- Roboflow — What is Data Augmentation? The Ultimate Guide. Hands-on catalogue of common CV augmentations with parameter tips, caveats, and example use cases. https://blog.roboflow.com/data-augmentation/

- Stanford AI Lab blog — Automating Data Augmentation. Readable overview of search-based policies (AutoAugment/RandAugment) and their trade-offs in practice. https://ai.stanford.edu/blog/data-augmentation/

- YouTube — Albumentations tutorial (PyTorch-focused). Step-by-step video showing fast image augmentations, composing pipelines, and visualizing results. https://www.youtube.com/watch?v=rAdLwKJBvPM

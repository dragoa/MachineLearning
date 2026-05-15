# PyTorch – Convolutional Neural Networks

> Notes from *PyTorch for Deep Learning Professional Certificate* · DeepLearning.AI  
> Module: Core Neural Network Components

---

## Table of Contents

1. [CNNs Part 1: Filters, Patterns & Feature Maps](#1-cnns-part-1-filters-patterns--feature-maps)
2. [CNNs Part 2: The Full Architecture](#2-cnns-part-2-the-full-architecture)
3. [Training a CNN for Image Classification](#3-training-a-cnn-for-image-classification)
4. [Dynamic Graphs](#4-dynamic-graphs)
5. [Modular Architectures](#5-modular-architectures)
6. [Model Inspection & Debugging](#6-model-inspection--debugging)

---

## 1. CNNs Part 1: Filters, Patterns & Feature Maps

### Why CNNs?

**The problem with linear layers:** They treat every pixel as independent. When looking at a butterfly or flower image, a linear layer sees thousands of separate numbers with no understanding that neighboring pixels form features like wings, antennae, or petals.

### Biological Inspiration

CNNs are inspired by Hubel & Wiesel's 1962 discovery that neurons in the visual cortex respond to specific patterns (like oriented lines). CNNs mimic this by using **filters** to detect features in images.

### What is a Convolution?

A **filter** (or kernel) is a small grid of numbers (e.g., 3×3) that slides across an image. At each position:

1. Multiply filter values with pixel values underneath
2. Sum all the results
3. Add a bias term
4. This becomes the new pixel value

**Visual example:**
```
Image pixels:           Filter:              
┌─────────────┐         ┌─────────┐         
│ 191 126  78 │         │ -2 -2 -2│         
│  98  61  76 │    ×    │  1  1  1│    
│  94  91  92 │         │ -1 -1 -1│         
└─────────────┘         └─────────┘         

Step 1: Multiply        Step 2: Sum all     Step 3: Add bias
┌─────────────┐         -382 - 252 - 156    
│-382-252-156 │         + 98 + 122 + 152    Sum + bias = 45
│  98 122 152 │         +188 +  91 + 184    
│ 188  91 184 │         ─────────────────    
└─────────────┘         = 45 (before bias)   

Result: New pixel value = 45
```

**Visual Example:**

```
Image pixels:              Filter:                Result:
┌─────────────────┐       ┌─────────────┐        
│ 191  126   78   │       │ -2  -2  -2  │        
│  98   61   76   │   ×   │  1   1   1  │   →   45
│  94   91   92   │       │ -1  -1  -1  │        
└─────────────────┘       └─────────────┘        

Calculation:
191×(-2) + 126×(-2) + 78×(-2) +
98×(1)   + 61×(1)   + 76×(1)  +
94×(-1)  + 91×(-1)  + 92×(-1) + bias = 45
```

**Example filter behavior:**
- **Vertical edge detector:** Highlights contrast between left/right neighbors
- **Horizontal edge detector:** Highlights contrast between top/bottom neighbors

```
Vertical Edge Filter:          Horizontal Edge Filter:
┌─────────────┐                ┌─────────────┐
│ -1   0   1  │                │ -1  -2  -1  │
│ -2   0   2  │                │  0   0   0  │
│ -1   0   1  │                │  1   2   1  │
└─────────────┘                └─────────────┘
     ↓                              ↓
Detects vertical edges        Detects horizontal edges
(left-right contrast)         (top-bottom contrast)
```

### Why This Matters for Classification

When identifying a butterfly, you don't analyze every pixel — you notice wing shapes, vein patterns, color sections. **Filters help highlight the patterns that distinguish a monarch from a swallowtail or a butterfly from a beetle.**

### The Key Power of CNNs

Instead of hand-designing filters, **the model learns which filters work best** during training. Each filter's weights are tuned to detect specific patterns that help classify your images.

---

## 2. CNNs Part 2: The Full Architecture

### Core Components

A complete CNN combines three types of layers:

#### 1. Convolutional Layers (`nn.Conv2d`)

**Key parameters:**
- `in_channels` – Input channels (3 for RGB, 1 for grayscale)
- `out_channels` – Number of filters to learn
- `kernel_size` – Filter size (typically 3×3)
- `stride` – How far to move the filter each step (1 = every pixel)
- `padding` – Add zeros around edges so corner pixels can be centered

**Padding visualization:**
```
Original image corner:        With padding=1:
┌──────┐                     ┌────────────┐
│ A  B │                     │ 0  0  0  0 │
│ C  D │                     │ 0  A  B  0 │
└──────┘                     │ 0  C  D  0 │
                             │ 0  0  0  0 │
Cannot center 3×3            └────────────┘
filter on pixel A            Now we can center
                             filter on A!
```

**Output:** Feature maps (activation maps) showing where each filter responds strongly

#### 2. Activation Function (`nn.ReLU`)

Sets negative values to zero. Helps the model learn complex, non-linear patterns.

#### 3. Pooling Layers (`nn.MaxPool2d`)

**Purpose:** Reduce spatial size while keeping important information

**How it works:**
- Divide feature map into 2×2 regions
- Keep only the maximum value from each region
- Output is 1/4 the original size

**Visual example:**
```
Original 4×4 Feature Map:        After MaxPool (2×2):
┌────────────────────┐           ┌──────────┐
│  0   64  128  128  │           │ 192  144 │
│ 48  192  144  144  │    →      │ 255  168 │
│142  226  168    0  │           └──────────┘
│255    0    0   64  │
└────────────────────┘

Each 2×2 region:                 Keeps maximum:
┌─────────┐                      ┌─────┐
│  0   64 │ → max = 192          │ 192 │
│ 48  192 │                      └─────┘
└─────────┘
┌─────────┐                      ┌─────┐
│128  128 │ → max = 144          │ 144 │
│144  144 │                      └─────┘
└─────────┘
```

**Why?**
- Makes the network more efficient (less data to process)
- Makes it more robust to small shifts/changes in the image
- Filters already extracted key features — pooling just compresses them

#### 4. Fully Connected Layers (`nn.Linear`)

After convolution and pooling extract features, the fully connected layer combines all features into a final classification.

### Architecture Flow

```
                    CNN PIPELINE
                    
Input Image                          Output
(28×28×1)                           (10 classes)
    │                                    ▲
    ├──► Conv2d (32 filters, 3×3)        │
    │         │                          │
    │         ▼                          │
    │    28×28×32 feature maps           │
    │         │                          │
    │         ▼                          │
    ├──► ReLU + MaxPool2d                │
    │         │                          │
    │         ▼                          │
    │    14×14×32 (halved)               │
    │         │                          │
    ├──► Conv2d (64 filters, 3×3)        │
    │         │                          │
    │         ▼                          │
    │    14×14×64 feature maps           │
    │         │                          │
    │         ▼                          │
    ├──► ReLU + MaxPool2d                │
    │         │                          │
    │         ▼                          │
    │     7×7×64 (halved)                │
    │         │                          │
    │         ▼                          │
    └──► Flatten ──► 3,136 values        │
              │                          │
              ▼                          │
         Fully Connected ────────────────┘
            (Linear)
```

### Why This Works

1. **Early layers** detect simple features (edges, corners)
2. **Middle layers** combine features (shapes, textures)
3. **Deep layers** recognize complex patterns (wings, petals)
4. **Final layer** makes the classification decision

---

## 3. Training a CNN for Image Classification

### Dataset: 32×32 Color Images

Training on diverse images: flowers, insects, small animals (15 classes total)

### Architecture Details

**Three convolutional blocks:**
- Block 1: 3 channels (RGB) → 32 filters → 16×16 after pooling
- Block 2: 32 → 64 filters → 8×8 after pooling
- Block 3: 64 → 128 filters → 4×4 after pooling

**Fully connected layers:**
- Flatten: 128 × 4 × 4 = 2,048 values
- FC1: 2,048 → 512 neurons
- Dropout (0.5)
- FC2: 512 → 15 classes

### Regularization Techniques

#### Dropout (`nn.Dropout`)

**The Problem: Co-adaptation**

Neurons can learn shortcuts instead of robust features. Classic example: a model classifying huskies vs wolves learned to look for *snow in the background* instead of actual animal features.

**How Dropout Helps:**
- Randomly deactivates ~50% of neurons during training
- Forces the network to learn multiple independent features
- Prevents over-reliance on any single pattern
- Typical rates: 0.2 to 0.5
- Applied after activation functions, not before final layer

**Dropout in action (0.5 = 50% drop rate):**
```
Without Dropout:              With Dropout (training):
┌───────────────────────┐          ┌───────────────────────┐
│ ⚫ ⚫ ⚫ ⚫ ⚫ ⚫  │          │ ⚫ ⭕ ⚫ ⭕ ⚫ ⭕  │
│ ⚫ ⚫ ⚫ ⚫ ⚫ ⚫  │          │ ⭕ ⚫ ⭕ ⚫ ⭕ ⚫  │
│ ⚫ ⚫ ⚫ ⚫ ⚫ ⚫  │   →      │ ⚫ ⭕ ⚫ ⭕ ⚫ ⭕  │
└───────────────────────┘          └───────────────────────┘
All neurons active           ⚫ = active
                             ⭕ = dropped (zeroed)

Network must learn robust features that work
even when some neurons are randomly disabled
```

**Important distinction:**
- **Dataset problem:** If all wolves have snow and dogs don't, that's a data issue
- **Co-adaptation:** When the model relies on spurious correlations instead of generalizing

#### Weight Decay

**The Problem: Overfitting through Large Weights**

Very large weights can indicate the model is memorizing training data rather than learning generalizable features.

**How Weight Decay Helps:**
- Adds a small penalty for large weights
- Encourages simpler, more robust solutions
- Works differently from dropout (penalizes weights, not neurons)

### Training Results

After just 10 epochs:
- Training loss steadily decreases
- Validation accuracy improves
- Model correctly classifies low-resolution 32×32 images with surprising accuracy

---

## 4. Dynamic Graphs

### What is a Computation Graph?

Every operation (multiply, add, ReLU, etc.) is recorded as a node in a **computation graph**. During training, PyTorch walks backwards through this graph (backpropagation) to compute gradients and update parameters.

### Static vs. Dynamic: The Core Difference

| Aspect | Static (older frameworks) | Dynamic (PyTorch) |
|--------|---------------------------|-------------------|
| **When graph is built** | Before any data runs | On the fly, during forward pass |
| **Structure** | Fixed, locked at definition | Rebuilt fresh every forward pass |
| **Branching/loops** | Must be pre-declared | Plain Python `if` / `for` |
| **Debugging** | Special debug mode required | Just use `print()` |
| **Performance** | Highly optimizable | Small overhead, huge flexibility gain |

### `nn.Sequential` – The Static-Like Approach

Creates a fixed pipeline where every operation happens in order.

**Constraints:**
- No branching with `if` statements
- No loops that adapt to data
- Cannot print intermediate values
- Every input follows the exact same path

**When to use it:**
- Simple, linear architectures
- When you want clean, readable layer stacking
- Building reusable blocks

### `nn.Module` – The Dynamic Approach

Splits responsibility cleanly:

| Method | Purpose |
|--------|---------|
| `__init__` | Define architecture (layers, blocks) |
| `forward` | Define the flow (the dynamic graph) |

### What Dynamic Graphs Enable

#### 1. Conditional Branching
```
if is_flower(x):
    return flower_layers(x)
else:
    return butterfly_layers(x)
```
The `if` statement doesn't just control logic — it **shapes the graph itself**.

#### 2. Variable-Length Inputs
Process sentences of 3 words or 50 words without pre-defining maximum length.

#### 3. In-Line Debugging
```
x = conv1(x)
if x.std() < 0.1:
    print(f"Low variance detected: {x.std()}")
x = conv2(x)
```

#### 4. Adaptive Processing
```
if is_easy_case(x):
    return small_network(x)   # 2 layers
else:
    return full_network(x)    # 50 layers
```

### Key Insight

> **Static frameworks make you think like a compiler.  
> PyTorch lets you think like a Python programmer.**

**Static Graph (older frameworks):**
```
Define entire graph upfront → Lock structure → Run data through
┌─────────────────────────────────────────────────┐
│  Conv → ReLU → Pool → Conv → ReLU → Pool → FC  │ ← Fixed path
└─────────────────────────────────────────────────┘
Every input must follow this exact path
```

**Dynamic Graph (PyTorch):**
```
Each forward pass builds fresh graph based on actual execution

Run 1:  Input → flower_check → [True]  → flower_layers → Output
                                 ↓
                          Builds graph A

Run 2:  Input → flower_check → [False] → butterfly_layers → Output
                                 ↓
                          Builds graph B (different!)
```

Each time `forward` runs:
1. PyTorch builds a fresh computation graph based on the actual execution path
2. Uses it for backpropagation
3. Discards it

The next run can follow a completely different path.

---

## 5. Modular Architectures

### The Problem with Flat Code

As models grow, explicitly defining every layer becomes:
- **Repetitive:** Copy-pasting conv-relu-pool blocks
- **Error-prone:** Easy to mismatch layer names (conv4/relu5?)
- **Hard to modify:** Adding a layer means updating both `__init__` and `forward`

### Solution 1: `nn.Sequential` for Blocks

Group related layers into logical blocks:

```
self.features = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

self.classifier = nn.Sequential(
    nn.Linear(128 * 4 * 4, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
```

**Benefits:**
- Clean separation of concerns
- Shorter `forward` method
- Easier to add/remove layers within a block

**Limitations:**
- Still no branching/loops inside a block
- Single input → single output per block

### Solution 2: Custom `nn.Module` Blocks

Create reusable building blocks:

```
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.block(x)
```

Then use it:
```
self.features = nn.Sequential(
    ConvBlock(3, 32),
    ConvBlock(32, 64),
    ConvBlock(64, 128),
    ConvBlock(128, 256)  # Easy to add
)
```

**Benefits:**
- Extremely reusable
- Easy to experiment (add/remove blocks)
- Changes propagate automatically
- Can add complexity (BatchNorm) in one place

### Workflow Tips

1. **Start explicit** – Write everything out to understand it
2. **Look for patterns** – Find repeated structures
3. **Refactor** – Create blocks and modules
4. **Keep flexibility** – Use custom modules when you need dynamic behavior

---

## 6. Model Inspection & Debugging

### Viewing Model Structure

#### Basic inspection
```
print(model)
```
Shows layer names and types, but not parameter counts or shapes.

#### Count total parameters
```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

#### See each layer's parameters
```
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

**Output example:**
```
conv1.weight: torch.Size([32, 3, 3, 3])
conv1.bias: torch.Size([32])
conv2.weight: torch.Size([64, 32, 3, 3])
conv2.bias: torch.Size([64])
fc1.weight: torch.Size([512, 2048])
fc1.bias: torch.Size([512])
```

### Exploring Module Hierarchy

PyTorch organizes models as a tree:
- **Root:** Your model (e.g., `SimpleCNN`)
- **Modules:** Top-level blocks (e.g., `features`, `classifier`)
- **SubModules:** Layers inside blocks (e.g., `conv1`, `relu1`)

**Module hierarchy visualization:**
```
                    SimpleCNN (root)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    features         classifier       dropout
        │                │
  ┌─────┼─────┐     ┌────┼────┐
  │     │     │     │    │    │
conv1  relu1 pool1  fc1  fc2  relu
  │
┌─┴─┐
│   │
W   b   (weights, biases)
```

#### `named_children()` – One level deep
```
SimpleCNN
├── features      (sees this)
├── classifier    (sees this)
└── dropout       (sees this)
    └── conv1     (does NOT see this)
```

#### `named_modules()` – Full tree
```
SimpleCNN
├── features      (sees this)
│   ├── conv1     (sees this)
│   ├── relu1     (sees this)
│   └── pool1     (sees this)
├── classifier    (sees this)
│   ├── fc1       (sees this)
│   └── fc2       (sees this)
└── dropout       (sees this)
```

### Debugging Shape Mismatches

**Common error:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied 
(32x2048 and 1024x512)
```

**Solution: Trace shapes through forward pass**
```
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.features(x)
    print(f"After features: {x.shape}")
    x = x.flatten(1)
    print(f"Flattened: {x.shape}")
    x = self.classifier(x)
    print(f"Output: {x.shape}")
    return x
```

**Typical output:**
```
Input:         [1, 3, 32, 32]    ← Batch=1, RGB=3, 32×32 image
                     │
After conv1:   [1, 32, 32, 32]   ← 32 feature maps, same size
                     │
After pool1:   [1, 32, 16, 16]   ← Halved by MaxPool
                     │
After conv2:   [1, 64, 16, 16]   ← 64 feature maps
                     │
After pool2:   [1, 64, 8, 8]     ← Halved again
                     │
After conv3:   [1, 128, 8, 8]    ← 128 feature maps
                     │
After pool3:   [1, 128, 4, 4]    ← Halved again
                     │
After flatten: [1, 2048]         ← 128×4×4 = 2,048 values
                     │
Output:        [1, 15]           ← 15 class predictions

Spatial dimensions: 32 → 16 → 8 → 4
Feature depth:      3 → 32 → 64 → 128
```

### Key Debugging Patterns

1. **Shape mismatches:** Add `print(x.shape)` after each layer
2. **Parameter counts:** Use `named_parameters()` to verify layer sizes
3. **Module structure:** Use `named_modules()` to understand nesting
4. **Vanishing gradients:** Check if activations are too small (`x.std() < threshold`)
5. **Exploding gradients:** Check if activations are too large

---

## Summary: What You've Mastered

✅ **CNNs:** How filters extract features, build feature maps, and enable hierarchical learning  
✅ **Architecture:** Convolution → ReLU → Pooling → Fully Connected pipeline  
✅ **Regularization:** Dropout prevents co-adaptation, weight decay prevents overfitting  
✅ **Dynamic Graphs:** PyTorch's flexibility enables branching, debugging, and adaptive models  
✅ **Modular Design:** `nn.Sequential` and custom blocks keep code clean and reusable  
✅ **Debugging:** Inspect parameters, trace shapes, understand your model's internals  

---

## What's Next?

- Advanced PyTorch tools
- Working with visual data using Torchvision
- Text models and NLP architectures
- Hyperparameter optimization
- Building real-world ML pipelines

---

*Course: PyTorch for Deep Learning Professional Certificate · DeepLearning.AI*

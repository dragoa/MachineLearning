##### Definition

==They define how the model is built.==

**Number of layers** The number of layers of a network defines its depth. ==Shallow networks are easier to interpret and computationally efficient==, but they might struggle with complex datasets and often require manual feature engineering. ==Deeper networks can extract more complex features== but involve more training parameters, longer training times, and a higher risk of overfitting, particularly with small datasets.

![[shallow_deep_nn.png]]

**Number of neurons per layer** More neurons can lead to better results with complex datasets, but also mean more weights to learn, higher memory usage and higher computational costs. For convolutional neural networks you can also specify the number of filters, the filter size, and the padding.

![[few_many_hidden_neurons.png]]

**Activation functions** [[Activation functions]] determine neuron output and introduce non-linearity, helping the network learn complex patterns.

![[activation_functions.png]]

In general, default to ==ReLU for hidden layers==, and choose between ==sigmoid or softmax in the final layer== depending on whether you have a binary or multiclass classifier, respectively.

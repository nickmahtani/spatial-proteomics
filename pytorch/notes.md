# PyTorch Notes

Notes from Sebastian Raschka's [PyTorch in 1 Hour](https://sebastianraschka.com/teaching/pytorch-1h/) tutorial.

## What is PyTorch

- Open-source deep learning library
- Automatic differentiation engine:
    - Allows us to use backpropagation
- Main components:
    - **Tensor library**: efficient computing
        - Extends the concept of array-oriented programming (NumPy) and adds accelerated computation on GPUs
        - Seamless switch between CPUs and GPUs
    - **Automatic differentiation engine**: differentiate computations automatically
        - aka autograd
        - Automatic computation of gradients for tensor operations
        - Simplifies backpropagation and model optimization
    - **Deep learning library**: utilities that use both the above and trains deep learning models
        - Building blocks
        - Loss functions
        - Optimizers
        - Pre-trained models
- API (Application Programming Interface):
    - Rules for how different pieces of software talk to each other
    - All the functions and classes
    - Communication without understanding of the internals (menu)

### Defining Deep Learning

- Tasks that AI wants to be good at:
    - Understanding natural language
    - Recognizing patterns
    - Making decisions
- ML:
    - Subfield of AI
    - Enable computers to learn from data
    - Make predictions/decisions
    - ML engineers develop algorithms that can:
        - Identify patterns
        - Learn from historical data
        - Improve their performance over time with more data and feedback
    - Things are no longer rule-based if/else systems
- DL is a subcategory of ML focused on implementation of deep neural networks:
    - Inspired by how the brain works
    - Multiple layers of artificial neurons/nodes
    - Can model complex, nonlinear relationships in data
    - ML was good for simple pattern recognition but DL is good for unstructured data like:
        - Images
        - Audio
        - Text

### MPS vs CUDA

- They are the bridges that let PyTorch talk to the GPU hardware
- CUDA is NVIDIA tech to allow computations on NVIDIA GPUs
- MPS is Metal Performance Shaders
    - It allows PyTorch to offload computations to the GPU cores in the Apple silicon chip

## Understanding Tensors

- Mathematical objects that generalize vectors and matrices to higher dimensions
- Can be characterized by their order/rank: number of dimensions
- Data containers for multidimensional data
- PyTorch can:
    - Create
    - Manipulate
    - Compute with these tensors efficiently
- If created from integers → the tensor objects have the default 64-bit integer data type
- If we create with float → 32-bit data type
- The 32-bit float has enough precision for deep learning tasks and it consumes less memory and computational resources
    - It speeds up model training and inference

### Automatic Differentiation Engine

- Provides functions to compute gradients in dynamic computational graphs automatically
    - **Computational graph**:
        - Directed graph to express and visualize mathematical expressions
        - Lays out the sequence of calculations needed to compute the output of a neural network
        - It computes the required gradients for backpropagation (the main training algorithm for neural networks)
        - It's a visual/structural way of representing a sequence of mathematical operations
        - Tracks what happens to data step by step so that PyTorch can figure out how to compute gradients for backpropagation
        - It does not need us to define the full graph — it records what we do and updates it dynamically. The graph is rebuilt fresh every forward pass.
        - It is a record of operations that makes automatic differentiation possible
- **Gradients**:
    - Training neural networks requires the adjustment of weights
    - The gradient tells you which direction to nudge each weight and by how much to reduce the error
    - Why it needs computational graphs:
        - The neural net may have millions of parameters and hundreds of operations chained together
        - The gradient of a final loss w.r.t. a single weight depends on every operation that comes after it
        - PyTorch needs to remember the full chain of operations so it can work backward through them and apply the chain rule
- The main training algorithm for neural networks is computing the gradients via backpropagation. Backpropagation is the process of computing gradients.
    - **Training loop**:
        - Forward pass: get the first prediction from the data
        - Compute loss: compare prediction to the true answer using a loss function
        - Backward pass/backpropagation: compute the gradient of the loss w.r.t. every weight
        - Update weights: the optimizer uses the gradients to nudge the weights in the right direction
- **Gradient**: a vector containing all of the partial derivatives of a multivariate function

## PyTorch as a Library for Implementing Deep Neural Nets

- **Layer**:
    - One step of computation; a linear transformation (weights + bias)
    - Followed by an activation function like ReLU (Rectified Linear Unit) or sigmoid
        - **ReLU**: only pass through positive inputs
            - It breaks linearity. A simple kink that lets the network learn curves and complex boundaries instead of just straight lines
            - Gradient is either 0 or 1
        - **Sigmoid**: used in older networks but for very large or small inputs, the gradient is nearly 0. Makes learning very slow (vanishing gradient problem)
        - Activation function name comes from neuroscience: the function that decides how much this neuron activates. The gatekeeper that determines what signal gets passed to the next layer
    - A single layer can only learn linear boundaries (straight lines). But if you stack layers with nonlinear activations in between, the model is able to learn complex decision boundaries

### A Bit About Classes

- If I want each instance of a class to have its own data, I will need a `__init__`
- For neural networks, we always need it because that's where we define our layers
- `self` is what makes variables global within the class
- `super().__init__()` calls the parent class's constructor
- The class is a blueprint; when we create an object we are instantiating (creating an object/instance)

### Logits

- Logits are raw output scores of a neural network before applying a sigmoid or a softmax
- The name comes from logistic function (another name for sigmoid)
- The input to the logistic function is the logit
- The logit function is the inverse of the sigmoid

### Weights

- When we create a layer with `nn.Linear`, PyTorch initializes weights with small random numbers using the Kaiming initialization (1/sqrt(2))
- If they weren't random and started at the same value, the network would never learn anything useful — this is called the symmetry problem
- Random initialization breaks the symmetry. Each unit in the layer will learn different features and specialize over time

### Useful Coding Tips

- `torch.rand(1, 50)`: creates 1 random input sample with 50 features
    - The values are random numbers between 0 and 1
    - `(1, 50)` is the shape: 1 row, 50 columns
    - The first dimension is batch size and the second needs to match with the network's first layer
- NumPy arrays are different from tensors in that they cannot:
    - Track gradients with autograd
    - Be run on GPUs
- Seeding random numbers:
    - `manual_seed` needs to be called multiple times if you want it to start at the same seed for a particular cell
- Capital letters represent matrices (2D data with multiple samples). Lowercase represent vectors or scalars

### Forward Pass

- Forward pass:
    - `model(X)`: it is feeding the input through all the layers
    - It is calling the forward method: `model.forward(X)`
- `grad_fn`: the function that will be used to compute the gradient
    - Every time an operation is done on a tensor with `requires_grad=True`, a `grad_fn` gets attached
    - Each `grad_fn` is a node in the graph
    - If `grad_fn = None`, it means the tensor was a leaf node (raw weights or input data). Nothing was computed to produce it
- If it is there, it shows that PyTorch is tracking for backpropagation
- `grad_fn=<AddmmBackward0>`
    - Addmm: Add matrix multiply

### Using the Model for Inference/Prediction Only

- We do not have to construct a computational graph for backpropagation
    - Performs unnecessary computations and consumes additional memory
    - We can use the `torch.no_grad()` context manager to avoid keeping track of the gradients
        - A context manager is a Python feature that automatically handles setup and cleanup using the `with` keyword
        - If you use `with`, you do not have to close files when you open them
        - When the block ends, gradient tracking turns on automatically if you use `with`
        - `dim=1` means it will be summed across the classes (the columns will disappear). Matters more when we have a bunch of samples

### Softmax

- It is like sigmoid but when you have more than two classes
- Softmax takes a vector of logits and converts them all into probabilities that sum to 1
- Sigmoid was good for binary classification

### Loss Functions

- The loss functions in PyTorch usually combine the softmax operation with the negative log-likelihood loss in a single class
- This is for numerical efficiency and stability
- e.g. `cross_entropy`: it does the softmax and then it calculates the -log
    - -log is chosen because it gives a small loss when predicted probability for the correct class is high and big loss when it's low
    - When prediction is perfect, it gives 0
    - It penalizes bad predictions more harshly than it rewards good ones, so it makes the model learn faster
    - It has clean gradients that work well for optimization
    - The name comes from information theory. Entropy measures the uncertainty in a distribution. Cross means between two distributions (truth vs prediction)
- The last layer of the neural network has one output neuron per class. The output dimension matches the number of classes

### Data Loaders

- A DataLoader takes the dataset and feeds it to the model in small batches during training instead of all at once
- It handles 3 things:
    - **Batching**: uses `__getitem__` to group samples into chunks of batch size
    - **Shuffling**: randomizes the order each epoch so the model doesn't memorize the sequence of the data
    - **Iterating**: loop through the batches in the training loop
- The dunder methods (`__`):
    - Special Python methods that let the class work with built-in Python features
    - `__len__` allows us to use `len()` on the class object
    - `__getitem__` allows us to use square bracket indexing
- **Epochs**:
    - At each epoch, all the cards are dealt out in the batch number
    - Shuffle happens once per epoch
    - Each epoch sees the 1000 samples just in different order. Each time it sees the same samples, the weights have been updated a bit
    - So it learns something new from them
- `num_workers` controls how many separate processes load the data in parallel
    - So perhaps while the GPU is busy training the current batch, the CPU can be loading and preparing the next batch in the background
    - In practice, if you have a substantially smaller batch as the last batch in a training epoch it can disturb the convergence during training, so we will drop the last batch with `drop_last=True`
    - Issues with using `num_workers > 0`:
        - Overhead problem:
            - Starting extra processes takes time
            - Creation, giving a copy of the dataset, coordination

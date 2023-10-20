import torch

from torch import nn
from torch import optim

from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit

from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.classifiers.loss import CLossCrossEntropy
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.features import CNormalizerMinMax

from secml.array import CArray


class Net(nn.Module):
    """
    Creation of the multiclass classifier
    """
    def __init__(self, n_features, n_hidden, n_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)  # fully connected layer with n_hidden neurons
        self.fc2 = nn.Linear(n_hidden, n_classes)  # fully connected layer with n_classes neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        x = self.fc2(x)  # Pass the output through the second layer
        return x


def pgd_l2_untargeted(x, y, clf, eps, alpha, steps):
    """
    Now we implement PGD for an UNTARGETED attack step by step
    untarget attack -> I want to maximize the error
    :param x: double: starting x position
    :param y: double: starting y position
    :param clf: CClassifierPyTorch wrapper
    :param eps: double: radius of max perturbation
    :param alpha: double: step size
    :param steps: int: number of iterations
    :return:
    """
    # defining the loss function
    loss_func = CLossCrossEntropy()
    # adversarial point at first a copy of the original sample
    x_adv = x.deepcopy()

    # we use a CArray to store intermediate results
    path = CArray.zeros((steps + 1, x.shape[1]))
    path[0, :] = x_adv  # store initial point

    # we iterate multiple times to repeat the gradient descent step
    for i in range(steps):
        # we calculate the output of the model (not the loss)
        scores = clf.decision_function(x_adv)

        # we compute the gradient of the loss w.r.t. the clf logits
        loss_gradient = loss_func.dloss(y_true=y, score=scores)

        # we compute gradient of the clf logits w.r.t. the input
        clf_gradient = clf.grad_f_x(x_adv, y)

        # With the chain rule, we compute the gradient of the CE loss w.r.t. the input of the network
        gradient = clf_gradient * loss_gradient

        # normalize the gradient (takes only the direction and discards the magnitude) (remeber to avoid division by 0)
        if gradient.norm() != 0:
            gradient /= gradient.norm()

        # apply the gradient descent step, by summing the normalized gradient (multiplied by the stepsize) to the sample
        x_adv = x_adv + alpha * gradient

        # re-project inside epsilon-ball
        delta = x_adv - x
        if delta.norm() > eps:
            delta = delta / delta.norm() * eps
            x_adv = x + delta

        # force input bounds
        x_adv = x_adv.clip(0, 1)

        # store point in the path
        path[i + 1, :] = x_adv

    return x_adv, clf.predict(x_adv), path


random_state = 999

# number of features
n_features = 2
# number of samples
n_samples = 1250
# centers of the clusters
centers = [[-2, 0], [2, -2], [2, 2]]
# standard deviation of the clusters
cluster_std = 0.8
# number of classes
n_classes = len(centers)
print(f"Number of classes: {n_classes}")

# generate the dataset using SecML library
dataset = CDLRandomBlobs(n_features=n_features,
                         centers=centers,
                         cluster_std=cluster_std,
                         n_samples=n_samples,
                         random_state=random_state).load()

# number of training set samples
n_tr = 1000
# number of test set samples
n_ts = 250

# split in training and test
splitter = CTrainTestSplit(
    train_size=n_tr, test_size=n_ts, random_state=random_state)
tr, ts = splitter.split(dataset)

# normalize the data between 0 and 1
nmz = CNormalizerMinMax()
tr.X = nmz.fit_transform(tr.X)
ts.X = nmz.transform(ts.X)

# creation of the torch model
net = Net(n_features=n_features, n_classes=n_classes, n_hidden=100)

# Define a loss function (Cross-Entropy) and an optimizer (Stochastic Gradient Descent - SGD)
criterion = nn.CrossEntropyLoss()  # Cross-Entropy loss is commonly used for multiclass classification
optimizer = optim.SGD(net.parameters(),
                      lr=0.001, momentum=0.9)

# wrap torch model in CClassifierPyTorch class
clf = CClassifierPyTorch(model=net,
                         loss=criterion,
                         optimizer=optimizer,
                         input_shape=(n_features,),
                         random_state=random_state)

# fit the classifier with training data
clf.fit(tr.X, tr.Y)

# compute predictions on a test set
y_pred = clf.predict(ts.X)

# Evaluate the accuracy of the classifier
metric = CMetricAccuracy()
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)

print("Accuracy on test set: {:.2%}".format(acc))

index = 0
point = ts[index, :]
x0, y0 = point.X, point.Y
steps = 200
eps = 0.3
alpha = 0.1

print(f"Starting point has label: {y0.item()}")
x_adv, y_adv, attack_path = pgd_l2_untargeted(x0, y0, clf, eps, alpha, steps)
print(f"Adversarial point has label: {y_adv.item()}")

import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import math
import auto_diff as ad
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

max_len = 28

LOG_LOSS_INTERVAL = 100

def linear_layer(X: ad.Node, w: ad.Node) -> ad.Node:
    # mainly used for classifier head
    output_node = ad.matmul(node_A=X, node_B=w)
    return output_node

def mlp(X: ad.Node, w1: ad.Node, w2: ad.Node) -> ad.Node:
    X1 = ad.matmul(X, w1)
    X_activated = ad.relu(X1)
    return ad.matmul(X_activated, w2)

def self_attention(X: ad.Node, wq: ad.Node, wk: ad.Node, wv: ad.Node, wo: ad.Node, model_dim: int) -> ad.Node:
    Q = ad.matmul(node_A=X, node_B=wq)
    K = ad.matmul(node_A=X, node_B=wk)
    V = ad.matmul(node_A=X, node_B=wv)
    S = ad.matmul(Q, ad.transpose(K, dim0=-1, dim1=-2)) /  math.sqrt(model_dim) # Q @ K^T / sqrt(dk)
    S_prob = ad.softmax(node_A=S, dim=-1)
    O = ad.matmul(node_A=ad.matmul(node_A=S_prob, node_B=V), node_B=wo)
    return O



def transformer(X: ad.Node, nodes: List[ad.Node], 
                      D: int, L: int, eps, B, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    D: int
        Dimension of the model (hidden size).
    L: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    """TODO: Your code here"""
    wp, wq, wk, wv, wo, w1, w2, wc = nodes
    projected_X = ad.matmul(node_A=X, node_B=wp)
    residual = projected_X
    ln1_normed_X = ad.layernorm(node_A=projected_X, normalized_shape=[D], eps=eps)
    self_attn_X = self_attention(X=ln1_normed_X, wq=wq, wk=wk, wv=wv, wo=wo, model_dim=D) + residual # residual
    residual = self_attn_X # update residual
    ln2_normed_X = ad.layernorm(node_A=self_attn_X, normalized_shape=[D], eps=eps)
    mlp_X = mlp(X=ln2_normed_X, w1=w1, w2=w2) + residual # 2nd residual
    classified_X = linear_layer(X=mlp_X, w=wc)
    return ad.mean(node_A=classified_X, dim=(1), keepdim=False)
    



def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""
    softmax_Z = ad.softmax(Z, dim=-1)
    log_softmax_Z = ad.log(softmax_Z)
    term = y_one_hot * log_softmax_Z
    sumed_term = ad.sum_op(node_A=term, dim=(0, 1), keepdim=False)
    loss = (-1 / batch_size) * sumed_term
    return loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in tqdm(range(num_batches), desc="Training"):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        # TODO: Your code here
        # we should change X_batch shape to B, L, D
        B, H, W = X_batch.shape
        # X_batch = X_batch.view(B, H * W, 1)
        y_pred_value, loss_value, *grad_values = f_run_model(X_batch, y_batch, model_weights)

        
        # Update weights and biases
        # TODO: Your code here
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)

        for j, weight in enumerate(model_weights):
            weight -= lr * grad_values[j].sum(dim=0)

        # Accumulate the loss
        # TODO: Your code here
        total_loss += loss_value.item() # a scalar
        # set description for tqdm of loss
        if (i + 1) % LOG_LOSS_INTERVAL == 0:
            tqdm.write(f"Batch {i + 1}/{num_batches}, Loss: {total_loss / (i + 1)}")


    # Compute the average loss
    
    # average_loss = total_loss / num_examples
    average_loss = total_loss / num_batches
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.01

    # TODO: Define the forward graph.
    wp = ad.Variable(name="wp")
    wq = ad.Variable(name="wq")
    wk = ad.Variable(name="wk")
    wv = ad.Variable(name="wv")
    wo = ad.Variable(name="wo")
    w1 = ad.Variable(name="w1")
    w2 = ad.Variable(name="w2")
    wc = ad.Variable(name="wc")
    X = ad.Variable(name='X')

    nodes = [wp, wq, wk, wv, wo, w1, w2, wc]


    y_predict: ad.Node = transformer(X=X, nodes=nodes, D=model_dim, L=seq_length, eps=eps, B=batch_size, num_classes=num_classes)
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # TODO: Construct the backward graph.

    # TODO: Create the evaluator.
    grads: List[ad.Node] = ad.gradients(output_node=loss, nodes=nodes) # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    # I do not use bias, as I see no benefits here
    W_P_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_Q_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_C_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))


    def f_run_model(X_run, label,  model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.

        By default the X is a minibatch
        """
        result = evaluator.run(
            input_values={
                # TODO: Fill in the mapping from variable to tensor
                X : X_run,
                y_groundtruth: label,
                wp : model_weights[0],
                wq : model_weights[1],
                wk : model_weights[2],
                wv : model_weights[3],
                wo : model_weights[4],
                w1 : model_weights[5],
                w2 : model_weights[6],
                wc : model_weights[7],
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in tqdm(range(num_batches), desc="Eval"):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            # X_batch = X_batch.view(X_batch.shape[0], X_batch.shape[1] * X_batch.shape[1], 1)

            logits = test_evaluator.run({
                # TODO: Fill in the mapping from variable to tensor
                X : X_batch,
                wp : model_weights[0],
                wq : model_weights[1],
                wk : model_weights[2],
                wv : model_weights[3],
                wo : model_weights[4],
                w1 : model_weights[5],
                w2 : model_weights[6],
                wc : model_weights[7],
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        W_P_val,
        W_Q_val,
        W_K_val,
        W_V_val,
        W_O_val,
        W_1_val,
        W_2_val,
        W_C_val
    ] # TODO: Initialize the model weights here

    # change all values into tensor
    model_weights = [torch.from_numpy(w) if isinstance(w, np.ndarray) else w for w in model_weights]

    # print number of parameters
    num_params = sum(w.numel() for w in model_weights)
    print(f"Number of parameters in the model: {num_params}")

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")

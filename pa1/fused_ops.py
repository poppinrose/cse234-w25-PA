from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2

        normalized_shape = node.normalized_shape
        eps = node.eps
        shape1 = input_values[0].shape
        shape2 = input_values[1].shape

        if len(shape1) == 2 and len(shape2) == 2:
            if shape2[0] != shape1[1]:
                input_values[1] = input_values[1].transpose(1, 0)
        elif len(shape1) == 3 and len(shape2) == 3:
            # we ignore th batch dimension
            if shape2[1] != shape1[2]:
                input_values[1] = input_values[1].transpose(2, 1)
        elif len(shape1) == 3 and len(shape2) == 2:
            if shape2[0] != shape1[2]:
                input_values[1] = input_values[1].transpose(1, 0)

        matmul_value = input_values[0] @ input_values[1] 
        mean = matmul_value.mean(dim=tuple(range(-len(normalized_shape), 0)), keepdim=True)
        var = torch.mean((matmul_value - mean) ** 2, dim=tuple(range(-len(normalized_shape), 0)), keepdim=True)
        std = torch.sqrt(var + eps)
        return (matmul_value - mean) / std
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        x = matmul(node.inputs[0], node.inputs[1]) 
        normalized_shape = node.normalized_shape
        eps = node.eps
        dims = tuple(range(-len(normalized_shape), 0))
        # calculate the mean of x
        x_mean = mean(x, dim=dims, keepdim=True) # shape of dims reduced to 1]
        x_mean = expand_as(x_mean, x)
        x_minus_mean = x - x_mean
        x_var = mean(x_minus_mean * x_minus_mean, dim=dims, keepdim=True)
        x_var = expand_as(x_var, x)
        x_std = sqrt(x_var + eps)
        term1 = output_grad
        term2 = mean(output_grad, dim=dims, keepdim=True)
        term2 = expand_as(term2, x)
        term3 = mean(output_grad * x_minus_mean, dim=dims, keepdim=True)
        term3 = expand_as(term3, x)
        term3_scalar = (x_minus_mean / (x_var + eps))
        term3 = term3_scalar * term3
        grad = (term1 - term2 - term3) / x_std

        grad_A = matmul(grad, transpose(node.inputs[1], dim0=-1, dim1=-2)) # we piece out the shape
        grad_B = matmul(transpose(node.inputs[0], dim0=-1, dim1=-2), grad) # we piece out the shape
        return [grad_A, grad_B]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        dim = node.dim
        shape1 = input_values[0].shape
        shape2 = input_values[1].shape
        if len(shape1) == 2 and len(shape2) == 2:
            if shape2[0] != shape1[1]:
                input_values[1] = input_values[1].transpose(1, 0)
        elif len(shape1) == 3 and len(shape2) == 3:
            # we ignore th batch dimension
            if shape2[1] != shape1[2]:
                input_values[1] = input_values[1].transpose(2, 1)
        elif len(shape1) == 3 and len(shape2) == 2:
            if shape2[0] != shape1[2]:
                input_values[1] = input_values[1].transpose(1, 0)

        matmul_value = input_values[0] @ input_values[1]
        max_values = torch.max(matmul_value, dim=dim, keepdim=True).values
        normalized_input = matmul_value - max_values  # for numerical stability, here occurs automatic broadcasting
        exp_value = torch.exp(normalized_input)
        denominator = torch.sum(exp_value, dim=dim, keepdim=True)
        return exp_value / denominator


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        dim = node.dim
        s = sum_op(output_grad * node, dim=dim, keepdim=True) # gradients from other indices
        s = expand_as(s, output_grad)
        grad = node * (output_grad - s)
        grad_A = matmul(grad, transpose(node.inputs[1], dim0=-1, dim1=-2)) # we piece out the shape
        grad_B = matmul(transpose(node.inputs[0], dim0=-1, dim1=-2), grad) # we piece out the shape
        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
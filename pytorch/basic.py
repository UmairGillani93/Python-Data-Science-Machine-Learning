import torch
import numpy as np

# creating tensors from data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

print(type(data))
print(type(x_data))

# creating tensor from numpy array
arr = np.array(data)
x_np = torch.from_numpy(arr)

print(type(arr))
print(type(x_np))

# converts data to ones like tensor
x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(torch.tensor(data), dtype=torch.float)
print(x_rand)

# defining Tensor with giving it's shape
SHAPE = (2,3,)

rand_tensor = torch.rand(SHAPE)
ones_tensor = torch.ones(SHAPE)
zeros_tensor = torch.zeros(SHAPE)

print(f"Random tensor: {rand_tensor} \n ")
print(f"Ones tensor: {ones_tensor} \n")
print(f"Zeros tensor: {zeros_tensor} \n")


# Attributes of a Tensor
tensor = torch.rand(3,4)
print(f"shape of tensor: {tensor.shape}")
print(f"data type of tensor: {tensor.dtype}")
print(f"device of tensor: {tensor.device}")


# define the device of tensor
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


tensor = torch.ones(4,4)
# grab first row
print("First row: ", tensor[0])
# grab all columns and first row
print("All columns and first row: ", tensor[0, :])


tensor1 = torch.ones(2,4)
print(tensor1)

print('first row: ', tensor1[0])
print('first column: ', tensor1[1])

print('first row and all columns: ', tensor1[0, :])

print('first column and all rows: ', tensor1[:, 0])

t1 = torch.cat([tensor, tensor1])

print("shape of t1: ", t1.shape)

print('\n')
print(t1)

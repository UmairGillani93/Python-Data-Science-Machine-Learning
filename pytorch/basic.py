# creating pytorch tensor from data
import torch
import numpy as np

data = [[1,2],
        [3,4]]

x_data = torch.tensor(data)
print(x_data)

# from  numpy array
arr = np.array(data)
x_np = torch.from_numpy(arr)

print(x_np)

# retains the properties of x_data
x_ones = torch.ones_like(x_data)
print(f"ones like tensor of the shape x_data: {x_ones}")

# retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"rand like tensor of the shape x_data: \n{x_rand}")


# creating tensor from shape
shape = (2,3)

print(f'Ones tensor of the shape given above: \n{torch.ones(shape)}')
print(f'Random tensor of the shape given above: \n{torch.rand(shape)}')


# ATTRIBUTES OF TENSOR
tensor = torch.rand(shape)
print(f"Shape of tensor: {tensor.shape}")
print(f'tensor size: \n{tensor.size}')
print(f'tensor device: \n{tensor.device}')

# OPERATIONS ON TENSORS
print(torch.cuda.is_available()) # cuda not available -> False

SHAPE = (4,4) # shape of 2D tensor
ones = torch.ones(SHAPE)
print(f'first row: \n{ones[0]}')
print(f'second row: \n{ones[1]}')

new_data = [[1,2,3,4],
            [5,6,7,8],
            [-1,-2,-3,-4],
            [-5,-6,-7,-8]]

new_tensor = torch.tensor(new_data)

print(f'first row of new_data: \n{new_data[0]}')
print(f'second row of new_data: \n{new_data[1]}')
print(f'third row of new_data: \n{new_data[2]}')

# # index first column
print(f'first columns new_tensor: \n{new_tensor[:, 0]}')
print(f'second columns new_tensor: \n{new_tensor[:, 1]}')
print(f'third columns new_tensor: \n{new_tensor[:, 2]}')
print(f'last columns new_tensor: \n{new_tensor[:, -1]}')
print(new_tensor.shape)

# first columns Zero
new_tensor[:, 1] = 0
print(new_tensor)

# concatenating two tensors
new_tensor1 = torch.zeros(SHAPE)
t1 = torch.cat([new_tensor, new_tensor1], dim=0)

print(t1)


# MATRIX MULTIPLICATION
t1 = torch.ones(SHAPE)
t2 = torch.rand(SHAPE)

result = t1 @ t2
print(result)


# ELEMENT-WIISE MULTIPLICATION
result_element_wise = t1 * t2
print(result_element_wise)

SHAPE = (1,4)

tensor_sum = torch.ones(SHAPE).sum()
print(tensor_sum)

tensor_item = tensor_sum.item()
print(type(tensor_item))

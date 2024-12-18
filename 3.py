import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# Print the arrays and perform operations
print("Array a:", a)
print("Array b:", b)

print("Sum of arrays a and b:", np.add(a, b))
print("Difference of arrays a and b:", np.subtract(a, b))
print("Product of arrays a and b:", np.multiply(a, b))
print("Division of arrays a and b:", np.divide(a, b))

# Other operations
print("Square root of array a:", np.sqrt(a))
print("Exponential of array a:", np.exp(a))
print("Minimum value of array a:", np.min(a))
print("Maximum value of array b:", np.max(b))
print("Mean of array a:", np.mean(a))
print("Standard deviation of array b:", np.std(b))
print("Sum of elements in array a:", np.sum(a))

# Reshape array a into a 2x3 array
c = np.array([1, 2, 3, 4, 5, 6])
print("Array c:", c)
print("Reshaped array c into 2x3:", np.reshape(c, (2, 3)))

# Example of transpose
d = np.array([[1, 2], [3, 4], [5, 6]])
print("Array d:")
print(d)
print("Transposed array d:")
print(np.transpose(d))

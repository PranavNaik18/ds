import matplotlib.pyplot as plt
import numpy as np

# Generate some data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Line Chart
plt.figure()
plt.plot(x, y)
plt.title("Line Chart")

# Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [20, 35, 30, 25]
plt.figure()
plt.bar(categories, values)
plt.title("Bar Chart")

# Scatter Plot
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 100 * np.random.rand(100)
plt.figure()
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.title("Scatter Plot")

# Pie Chart
sizes = [30, 20, 25, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
plt.figure()
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Pie Chart")

plt.show()

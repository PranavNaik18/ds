import pandas as pd

# Load the Excel file
file_path = 'data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Display the contents of the DataFrame
print("Contents of the Excel file:")
print(df)

import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 30, 22],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
output_file = 'output.xlsx'
df.to_excel(output_file, index=False)

print(f'Data has been written to {output_file}')

import pandas as pd

# Load the existing Excel file
file_path = 'output.xlsx'  # Replace with your file path
existing_df = pd.read_excel(file_path)

# New data to append
new_data = {
    'Name': ['David', 'Eva'],
    'Age': [28, 26],
    'City': ['Houston', 'Phoenix']
}
new_df = pd.DataFrame(new_data)

# Append new data to the existing DataFrame
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Write the updated DataFrame back to the Excel file
combined_df.to_excel(file_path, index=False)

print(f'New data has been appended to {file_path}')

import pandas as pd

# Load the Excel file
file_path = 'data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Filter rows where Age is greater than 25
filtered_df = df[df['Age'] > 25]

# Display the filtered DataFrame
print("Filtered data (Age > 25):")
print(filtered_df)

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Create a bar chart
df.plot(kind='bar', x='Name', y='Age', title='Age of Individuals', legend=False)
plt.ylabel('Age')
plt.show()

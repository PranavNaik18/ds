import pandas as pd

# Create a DataFrame
data = {
    'Name': ['John', 'Emma', 'Sophia', 'Michael', 'James'],
    'Age': [28, 34, 29, 42, 55],
    'Country': ['USA', 'Canada', 'India', 'UK', 'Australia'],
    'Salary': [50000, 60000, 55000, 70000, 80000],
    'Experience': [5, 8, 6, 15, 20]
}
df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Selecting specific columns
name_age = df[['Name', 'Age']]
print("\nSelected 'Name' and 'Age' columns:")
print(name_age)

# Sorting the DataFrame by 'Salary' in descending order
sorted_df = df.sort_values("Salary", ascending=False)
print("\nDataFrame sorted by salary in descending order:")
print(sorted_df)

# Updating salary for Emma
df.loc[df['Name'] == 'Emma', 'Salary'] = 65000
print("\nDataFrame after updating Emma's salary:")
print(df)

# Dropping the 'Experience' column
df = df.drop(columns=['Experience'])
print("\nDataFrame after deleting the 'Experience' column:")
print(df)

# Handling Missing Data (if any)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())  # Replace NaN with the mean salary

# Merging two DataFrames
df2 = pd.DataFrame({
    'Name': ['John', 'Sophia'],
    'City': ['New York', 'Mumbai']
})
merged_df = pd.merge(df, df2, on='Name', how='left')
print("\nMerged DataFrame:")
print(merged_df)

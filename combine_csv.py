import pandas as pd

file1 = "/u/ywagh/scheduling_results_sameDuration_0_500.csv"
file2 = "/u/ywagh/scheduling_results_sameDuration_500_1000.csv"
output = "/u/ywagh/scheduling_results_sameDuration_mega.csv"

# Read the two CSV files into dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Concatenate the two dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("/u/ywagh/scheduling_results_sameDuration_mega.csv", index=False)

file_1 = "/u/ywagh/scheduling_results_sameDuration_0_500.csv"
file_2 = "/u/ywagh/scheduling_results_sameDuration_500_1000.csv"
output = "/u/ywagh/scheduling_results_sameDuration_mega.csv"


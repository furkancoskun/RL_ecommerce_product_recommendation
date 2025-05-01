import pandas as pd

input_filename = 'ecommerce_clickstream_transactions.csv'
output_filename = 'ecommerce_clickstream_transactions_filtered.csv'
events_to_remove = ['page_view', 'login', 'logout', 'click']

try:
    # 1. Read the input CSV file into a pandas DataFrame
    print(f"Reading data from '{input_filename}'...")
    df = pd.read_csv(input_filename)
    print(f"Original number of rows: {len(df)}")

    # 2. Filter the DataFrame
    # Keep rows where 'EventType' is NOT in the events_to_remove list
    print(f"Removing rows with EventType: {', '.join(events_to_remove)}...")
    filtered_df = df[~df['EventType'].isin(events_to_remove)]
    print(f"Number of rows after filtering: {len(filtered_df)}")

    # 3. Save the filtered DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    print(f"Saving filtered data to '{output_filename}'...")
    filtered_df.to_csv(output_filename, index=False)

    print("Processing complete. Filtered data saved successfully.")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
    print("Please make sure the file exists in the same directory as the script, or provide the correct path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
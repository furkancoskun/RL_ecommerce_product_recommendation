import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filtered_filename = 'ecommerce_clickstream_transactions_filtered.csv'
top_n_products = 20 # How many top products to show in the plot

try:
    # 1. Load the Filtered Data
    print(f"Reading filtered data from '{filtered_filename}'...")
    df = pd.read_csv(filtered_filename)
    print(f"Loaded {len(df)} rows.")

    # --- Basic Data Overview ---
    print("\n--- Basic Data Info ---")
    df.info()

    # Re-check missing values (especially for ProductID which should mostly be present now)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Handle potential residual missing ProductIDs if necessary (optional, depends on data quality)
    # df.dropna(subset=['ProductID'], inplace=True) # Option: Drop rows where ProductID is still missing

    # Ensure Amount is numeric (important for describe and purchase analysis)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)


    print("\n--- Descriptive Statistics (Amount) ---")
    # Only describe 'Amount' where it's > 0 (i.e., for actual purchases)
    print(df[df['Amount'] > 0]['Amount'].describe())


    print("\n--- General Statistics ---")
    unique_users = df['UserID'].nunique()
    unique_sessions = df.groupby('UserID')['SessionID'].nunique().sum() # More accurate count
    # Alternative session count (simpler, assumes UserID+SessionID is unique combo)
    # unique_sessions_simple = df[['UserID', 'SessionID']].drop_duplicates().shape[0]
    unique_products = df['ProductID'].nunique()

    print(f"Number of Unique Users: {unique_users}")
    print(f"Number of Unique Sessions: {unique_sessions}")
    print(f"Number of Unique Products Mentioned: {unique_products}")

    print("\n--- Event Type Distribution (Filtered Data) ---")
    print(df['EventType'].value_counts())


    # --- Analysis: Most Frequently Sold Products ---
    print(f"\n--- Top {top_n_products} Most Frequently Sold Products ---")
    purchase_events = df[df['EventType'] == 'purchase'].copy() # Work on a copy

    if not purchase_events.empty:
        # Make sure ProductID is not null for purchases
        purchase_events.dropna(subset=['ProductID'], inplace=True)

        sold_product_counts = purchase_events['ProductID'].value_counts()

        print(sold_product_counts.head(top_n_products))

        # --- Visualization: Top Sold Products ---
        plt.figure(figsize=(12, 7))
        sns.barplot(x=sold_product_counts.head(top_n_products).index,
                    y=sold_product_counts.head(top_n_products).values,
                    palette='viridis')
        plt.title(f'Top {top_n_products} Most Frequently Sold Products')
        plt.xlabel('Product ID')
        plt.ylabel('Number of Times Sold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    else:
        print("No 'purchase' events found in the filtered data.")


    # --- Analysis: Products Mentioned but Not Sold ---
    print("\n--- Products Mentioned But Never Sold ---")

    # Get all unique products mentioned in *any* event type in the filtered data
    # Dropna is crucial here in case any NaNs slipped through
    all_mentioned_products = set(df['ProductID'].dropna().unique())
    print(f"Total unique products mentioned in filtered data: {len(all_mentioned_products)}")


    if not purchase_events.empty:
        # Get unique products that were actually sold
        sold_products = set(sold_product_counts.index) # Use index from value_counts
        print(f"Total unique products sold: {len(sold_products)}")

        # Find the difference
        products_not_sold = all_mentioned_products - sold_products
        count_not_sold = len(products_not_sold)

        print(f"Number of products mentioned but never sold: {count_not_sold}")

        # Show a few examples if needed (optional)
        if count_not_sold > 0 and count_not_sold < 50: # Don't print too many
             print("Examples of products not sold:")
             print(list(products_not_sold)[:10]) # Print first 10 examples
        elif count_not_sold == 0:
             print("All mentioned products have at least one sale record.")

    else:
         print("Cannot determine unsold products as no purchase events were found.")


except FileNotFoundError:
    print(f"Error: Filtered file '{filtered_filename}' not found.")
    print("Please make sure the file exists (run the previous filtering step first).")
except Exception as e:
    print(f"An unexpected error occurred during EDA: {e}")
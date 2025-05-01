import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filtered_filename = 'data/ecommerce_clickstream_transactions_filtered.csv'
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

    # Handle potential residual missing ProductIDs if necessary
    # Optional: Consider if rows without ProductID should be dropped entirely earlier
    df.dropna(subset=['ProductID'], inplace=True) # Drop rows missing ProductID as they are less useful now
    print(f"Rows after dropping missing ProductID: {len(df)}")

    # Ensure Amount is numeric (important for describe and purchase analysis)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)


    print("\n--- Descriptive Statistics (Amount for Purchases) ---")
    # Only describe 'Amount' where it's > 0 (i.e., for actual purchases)
    purchase_amounts = df[df['EventType'] == 'purchase']['Amount']
    if not purchase_amounts.empty:
        print(purchase_amounts.describe())
    else:
        print("No purchase events with Amount > 0 found.")


    print("\n--- General Statistics ---")
    unique_users = df['UserID'].nunique()
    unique_sessions = df.groupby('UserID')['SessionID'].nunique().sum()
    unique_products = df['ProductID'].nunique() # Recalculate after dropna

    print(f"Number of Unique Users: {unique_users}")
    print(f"Number of Unique Sessions: {unique_sessions}")
    print(f"Number of Unique Products Mentioned: {unique_products}")

    print("\n--- Event Type Distribution (Filtered Data) ---")
    event_type_counts = df['EventType'].value_counts()
    print(event_type_counts)


    # --- Analysis: Most Frequently Sold Products ---
    print(f"\n--- Top {top_n_products} Most Frequently Sold Products ---")
    purchase_events = df[df['EventType'] == 'purchase'].copy() # Work on a copy

    if not purchase_events.empty:
        sold_product_counts = purchase_events['ProductID'].value_counts()
        print(sold_product_counts.head(top_n_products))

        # Visualization: Top Sold Products
        plt.figure(figsize=(12, 7))
        sns.barplot(x=sold_product_counts.head(top_n_products).index,
                    y=sold_product_counts.head(top_n_products).values,
                    palette='viridis')
        plt.title(f'Top {top_n_products} Most Frequently Sold Products')
        plt.xlabel('Product ID')
        plt.ylabel('Number of Times Sold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show() # Keep plt.show() until the end

    else:
        print("No 'purchase' events found in the filtered data.")
        sold_product_counts = pd.Series(dtype=int) # Create empty series for later steps

    # --- Analysis: Most Frequently Added-to-Cart Products ---
    print(f"\n--- Top {top_n_products} Most Frequently Added-to-Cart Products ---")
    cart_events = df[df['EventType'] == 'add_to_cart'].copy()

    if not cart_events.empty:
        cart_product_counts = cart_events['ProductID'].value_counts()
        print(cart_product_counts.head(top_n_products))

        # Visualization: Top Add-to-Cart Products
        plt.figure(figsize=(12, 7))
        sns.barplot(x=cart_product_counts.head(top_n_products).index,
                    y=cart_product_counts.head(top_n_products).values,
                    palette='magma') # Different palette
        plt.title(f'Top {top_n_products} Most Frequently Added-to-Cart Products')
        plt.xlabel('Product ID')
        plt.ylabel('Number of Times Added to Cart')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()

    else:
        print("No 'add_to_cart' events found in the filtered data.")

    # --- Analysis: Most Frequently Viewed Products ---
    print(f"\n--- Top {top_n_products} Most Frequently Viewed Products ---")
    view_events = df[df['EventType'] == 'product_view'].copy()

    if not view_events.empty:
        view_product_counts = view_events['ProductID'].value_counts()
        print(view_product_counts.head(top_n_products))

        # Visualization: Top Viewed Products
        plt.figure(figsize=(12, 7))
        sns.barplot(x=view_product_counts.head(top_n_products).index,
                    y=view_product_counts.head(top_n_products).values,
                    palette='plasma') # Yet another palette
        plt.title(f'Top {top_n_products} Most Frequently Viewed Products')
        plt.xlabel('Product ID')
        plt.ylabel('Number of Times Viewed')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()

    else:
        print("No 'product_view' events found in the filtered data.")


    # --- Analysis: Products Mentioned but Not Sold ---
    print("\n--- Products Mentioned But Never Sold ---")

    # Get all unique products mentioned (already filtered for non-null ProductID)
    all_mentioned_products = set(df['ProductID'].unique())
    print(f"Total unique products mentioned in filtered data: {len(all_mentioned_products)}")

    if not sold_product_counts.empty:
        # Get unique products that were actually sold
        sold_products = set(sold_product_counts.index)
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
         # If there were no purchases, all mentioned products were technically not sold
         print(f"Number of products mentioned but never sold: {len(all_mentioned_products)} (as no purchase events exist)")


    # --- Display all plots at the end ---
    print("\nDisplaying plots...")
    plt.show()


except FileNotFoundError:
    print(f"Error: Filtered file '{filtered_filename}' not found.")
    print("Please make sure the file exists (run the previous filtering step first).")
except Exception as e:
    print(f"An unexpected error occurred during EDA: {e}")
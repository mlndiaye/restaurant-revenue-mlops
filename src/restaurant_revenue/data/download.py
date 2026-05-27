"""
Download script for DVC pipeline.
Triggers dataset download and raw data setup.
"""

from restaurant_revenue.data.load import download_dataset

if __name__ == "__main__":
    print("Starting dataset download...")
    download_dataset()
    print("Dataset download completed successfully.")

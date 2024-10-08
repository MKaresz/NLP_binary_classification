import os
import csv
import pandas as pd


def write_to_csv(data, export_name):
    # Create a DataFrame
    df = pd.DataFrame(data).astype(str)

    # Write the DataFrame to a CSV file without row names
    df.to_csv(export_name, sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)

def create_labelled_dataset(file_name, export_data_path):
    with open(file_name, 'r', encoding='utf-8') as file:
        labelled_dataset = {
                'review': [],
                'sentiment': []
                }
        while True:
            line = file.readline()
            if not line:
                break
            label = 'negative' if line[-2:-1] == "0" else 'positive'
            labelled_dataset['review'].append(line[:-3]) #i.e.: \t 0 \n
            labelled_dataset['sentiment'].append(label)

    write_to_csv( labelled_dataset, export_data_path)


def main():
    file_name = "data\yelp_labelled.txt"
    export_data_path = "yelp_test_data.csv"
    # build dataset for test
    create_labelled_dataset(file_name, export_data_path)


if __name__ == "__main__":
	main()

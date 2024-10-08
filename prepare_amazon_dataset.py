import os
import csv
import pandas as pd


def write_to_csv(data, export_name):
    # Create a DataFrame
    df = pd.DataFrame(data).astype(str)

    # Write the DataFrame to a CSV file
    df.to_csv(export_name, sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)

def read_dataset(file_name, export_data_path):
    with open(file_name, 'r', encoding='utf-8') as file:
        data_set = {
                'review': [],
                'sentiment': []
                }
        while True:
            line = file.readline()
            if not line:
                break

            data_set['review'].append(line[:-3]) #i.e.: \t 0 \n
            if line[-2:-1] == "0":
                data_set['sentiment'].append('negative')
            else:
                data_set['sentiment'].append('positive')
    write_to_csv( data_set, export_data_path)


def main():
    file_name = "data\\amazon_cells_labelled.txt"
    export_data_path = "amazon_test_data.csv"
    # build dataset for test
    read_dataset(file_name, export_data_path)


if __name__ == "__main__":
	main()

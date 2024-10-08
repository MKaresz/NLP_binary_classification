import os
import csv
import random

def combine_files_from_folder(paths, export_name):
    data = []
    for folder_path in paths:
        if 'neg' in folder_path:
            label = 'negative'
        else:
            label = 'positive'
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    data.append((content, label))
    random.shuffle(data)
    write_to_csv(data, export_name)

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['review', 'sentiment'])
        writer.writerows(data)

def main():
    test_folder_paths = [('./test/pos', './test/neg'),"test_reviews_IMDB_25K.csv"]
    train_folder_paths = [('./train/pos', './train/neg'),"train_reviews_IMDB_25K.csv"]
    
    # build dataset for test and train
    combine_files_from_folder(test_folder_paths[0], test_folder_paths[1])
    combine_files_from_folder(train_folder_paths[0], train_folder_paths[1])


if __name__ == "__main__":
	main()
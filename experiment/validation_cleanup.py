import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

column_name_for_smiles = 'target'
log_file = './train_test/1k/validation_remove_duplicate_multi_thread.log'

num_threads = 20  # Define the number of threads

def remove_duplicates(file_a, file_b, file_c, num_threads):
    # Configure logging to write to a log file and stream to the terminal
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # Read lines from file A
    with open(file_a, 'r') as f:
        lines_a = set(f.readlines())
    logging.info(f"Number of lines in file {file_a}: {len(lines_a)}")

    # Read lines from file B
    with open(file_b, 'r') as f:
        lines_b = f.readlines()
    logging.info(f"Number of lines in file {file_b}: {len(lines_b)}")

    # Remove duplicated lines from file B using multiple threads
    unique_lines = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for line in lines_b:
            futures.append(executor.submit(check_duplicate, line, lines_a, lines_b, unique_lines))
        for future in futures:
            future.result()

    num_duplicates_removed = len(lines_b) - len(unique_lines)
    logging.info(f"Number of duplicates removed: {num_duplicates_removed}")

    # Write unique lines to file C
    with open(file_c, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([column_name_for_smiles])  # Write column name
        writer.writerows([[line.strip()] for line in unique_lines])  # Write each line as a row in the 'target' column

    logging.info(f"Number of lines in file {file_c}: {len(unique_lines)}")

def check_duplicate(line, lines_a, lines_b, unique_lines):
    if line.strip().lower() not in [l.strip().lower() for l in lines_a]:
        unique_lines.append(line)
        logging.info(f"Unique line found: {line.strip()} progress: {len(unique_lines)} out of {len(lines_b)}")
    else:
        logging.info(f"Duplicate found: {line.strip()}")

# Usage example
file_a = './train_test/50ktest2/50ktest2_multiclass_test.csv'
file_b = './train_test/1k/1k_multiclass_train.csv'
file_c = './train_test/1k/1k_unique_train.csv'


remove_duplicates(file_a, file_b, file_c, num_threads)

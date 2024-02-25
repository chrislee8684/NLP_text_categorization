import random

# Turn labeled docs list to list of tuples
def read_labels(filename):
    with open(filename, 'r') as file:
        # Split each line by space and convert to tuple, for each line in the file
        labels_list = [tuple(line.strip().split(' ')) for line in file.readlines()]
    return labels_list

def split_labels(labels, test_size_ratio):
    # Ensure the labels list is shuffled to randomize the split
    random.shuffle(labels)

    # Calculate the split index
    split_index = int(len(labels) * (1 - test_size_ratio)) # turning ratio into respective index value

    # Split the labels into training and test sets
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    return train_labels, test_labels

def write_tuples_to_file(tuples, filename):
    with open(filename, 'w') as file:
        for tup in tuples:
            file.write(' '.join(map(str, tup)) + '\n') # join the tuple elements with a space and write to the file followed by a newline

def write_paths_to_file(tuples, filename):
    with open(filename, 'w') as file:
        for tup in tuples:
            file.write(str(tup[0]) + '\n') # write only the first element of the tuple to the file followed by a newline

# Ask for input labeled training file
labeled_training_docs_file = input("Please enter the name of the file containing list of labeled training documents: ")

# Ask for split ratio
while True:  # Ensure split is a fraction between 0 and 1
    split_ratio = float(input("Please enter desired test size as a fraction between 0 and 1 (ex: 0.2): "))
    if 0 < split_ratio < 1:
        print(f"Proceeding with a test size of {split_ratio}")
        break
    else:
        print("Invalid input. Please enter a fraction between 0 and 1.")

# Ask for names of output files
labeled_training_docs_subset_file = input("Please enter the name of the file for the smaller subset of labeled training documents: ")
labeled_valid_docs_file = input("Please enter the name of the file for the labeled validation documents: ")
unlabeled_valid_docs_file = input("Please enter the name of the file for the unlabeled validation documents: ")

# Split labeled training set into labeled training and validation sets
training_subset_labels_list = read_labels(labeled_training_docs_file) # get all rows from labeled training docs file (doc paths, labels)
labeled_training_subset_list, labeled_validation_list = split_labels(training_subset_labels_list, split_ratio)

# Write to output files
write_tuples_to_file(labeled_training_subset_list, labeled_training_docs_subset_file)
write_tuples_to_file(labeled_validation_list, labeled_valid_docs_file)
write_paths_to_file(labeled_validation_list, unlabeled_valid_docs_file) # for unlabeled validation docs, only want doc paths





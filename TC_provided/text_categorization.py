# Import necessary libraries
import subprocess
import math
import nltk
nltk.download('punkt')
import numpy as np

# Functions
def get_doc_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        doc_text = file.read()
    return doc_text.lower() # make all text lowercase

def compute_tf(text):
    tokens = nltk.word_tokenize(text)
    tf_dict = {}

    for token in tokens:
        token = token.lower()
        if token in tf_dict:
            tf_dict[token] += 1
        else:
            tf_dict[token] = 1

    return tf_dict

def compute_tf_list(file, has_labels):
    tf_list = []

    with open(file, 'r') as file:
        if has_labels:
            for line in file.readlines():
                doc_path, label = line.strip().split(' ')
                doc_text = get_doc_text(doc_path)
                tf_list.append((label, compute_tf(doc_text)))
        else: #unlabeled
            for line in file.readlines():
                doc_path = line.strip()
                doc_text = get_doc_text(doc_path)
                tf_list.append((doc_path, compute_tf(doc_text)))

    return tf_list

def compute_df(file): # df = document freq.
    df_dict = {}

    with open(file, 'r') as file:
        for line in file.readlines():
            doc_path = line.strip().split(' ')[0]
            doc_text = get_doc_text(doc_path)
            tokens = nltk.word_tokenize(doc_text)
            unique_words = set(tokens)

            for word in unique_words:
                df_dict[word] = df_dict.get(word, 0) + 1

    return df_dict

def get_file_length(file):
    with open(file, 'r') as file:
        lines = 0
        for line in file:
            lines += 1
    return lines

def compute_idf(file):
    df_dict = compute_df(file)
    n = get_file_length(file) # total number of documents
    idf_dict = {}

    for word, df in df_dict.items():
        idf_dict[word] = math.log(n / float(df))

    return idf_dict

def compute_tf_idf(tf_list, idf):
    for label, tf_dict in tf_list:
        for word in list(tf_dict.keys()):
            tf_dict[word] = tf_dict[word] * idf.get(word, 0)  # Use .get to avoid KeyError, defaulting to 0 if not found; test docs may have words not in training vocab

def compute_vector_list(tf_idf_list, idf):
    new_list = []
    idf_list = list(idf.keys())
    for label, tf_idf_dict in tf_idf_list:
        vec = [tf_idf_dict.get(word, 0) for word in idf_list]  # Generate vector based on IDF index
        new_list.append((label, vec))
    return new_list

# Euclidean distance method
# def calculate_all_distances(test_vector, train_vectors):
#     # Calculate the distances from the test vector to all training vectors in a batch
#     # Ensure test_vector and train_vectors are numpy arrays for efficient computation
#     distances = np.linalg.norm(train_vectors - test_vector, axis=1)
#     return distances
#
# def predict_label(test_vector, train_vectors, labels, k):
#     # Calculate distances from the test vector to all training vectors
#     distances = calculate_all_distances(test_vector, train_vectors)
#
#     # Get indices of the k smallest distances
#     nearest_indices = np.argpartition(distances, k)[:k]
#
#     # Calculate similarity scores (avoid division by zero)
#     epsilon = 1e-10
#     similarity_scores = 1 / (distances[nearest_indices] + epsilon)
#
#     # Aggregate scores by label to find the most common label among the k nearest neighbors
#     unique_knn_sums = {}
#     for index, score in zip(nearest_indices, similarity_scores):
#         label = labels[index]
#         if label in unique_knn_sums:
#             unique_knn_sums[label] += score
#         else:
#             unique_knn_sums[label] = score
#
#     # Return the label with the highest aggregated similarity score
#     return max(unique_knn_sums, key=unique_knn_sums.get)

# Cos similarity method
def calculate_all_similarities(test_vector, train_vectors):
    # Normalize the test vector and train_vectors to unit vectors
    test_vector_norm = test_vector / np.linalg.norm(test_vector)
    train_vectors_norm = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)

    # Calculate cosine similarities as dot products of normalized vectors
    similarities = np.dot(train_vectors_norm, test_vector_norm)

    return similarities


def predict_label(test_vector, train_vectors, labels, k):
    # Calculate cosine similarities from the test vector to all training vectors
    similarities = calculate_all_similarities(test_vector, train_vectors)

    # Get indices of the k highest similarities
    nearest_indices = np.argsort(similarities)[-k:]

    # Retrieve the actual similarity scores for the nearest neighbors
    nearest_similarities = similarities[nearest_indices]

    # Aggregate scores by label to find the most common label among the k nearest neighbors
    unique_knn_sums = {}
    for index, similarity in zip(nearest_indices, nearest_similarities):
        label = labels[index]
        if label in unique_knn_sums:
            unique_knn_sums[label] += similarity
        else:
            unique_knn_sums[label] = similarity

    # Return the label with the highest aggregated similarity score
    return max(unique_knn_sums, key=unique_knn_sums.get)

def make_predictions(train_list, test_list, k):
    # Extract labels and vectors from the training list
    labels = [label for label, _ in train_list]
    train_vectors = np.array([vector for _, vector in train_list])

    predicted_list = []
    for file_path, test_vector in test_list:
        # Convert test_vector to a numpy array if not already
        test_vector_np = np.array(test_vector)
        predicted_label = predict_label(test_vector_np, train_vectors, labels, k)
        predicted_list.append((file_path, predicted_label))

    return predicted_list



# Step 1: Ask for input documents
training_docs_file = input("Please enter the name of the file containing the list of labeled training documents: ")
test_docs_file = input("Please enter the name of the file containing the list of unlabeled test documents: ")

# Step 2: Compute tf*idf for training and testing sets
# Compute tf lists
train_list = compute_tf_list(training_docs_file, True)
test_list = compute_tf_list(test_docs_file, False)

# Compute idf dict (based on training set)
idf = compute_idf(training_docs_file)

# Compute tf*idf lists (altering existing tf lists)
compute_tf_idf(train_list, idf)
compute_tf_idf(test_list, idf)

# Step 3: Create vector representations of tf*idf/doc's (altering existing tf*idf lists)
train_vector_list = compute_vector_list(train_list, idf)
test_vector_list = compute_vector_list(test_list, idf)

# Step 4: Make predictions
k = 0

while k < 0 or k == 0:
    k = int(input("Please enter the K value: ")) # assuming 1 works for current scenario (mutually exclusive and exhaustive)

predictions_list = make_predictions(train_vector_list, test_vector_list, k)

# ask for output file name
output_file = input("Please enter the name of the output file: ") # Contains predictions for test set docs

with open(output_file, 'w') as file:
    # Iterate through the list of tuples
    for item in predictions_list:
        # Format each tuple as a string with a space between elements
        tuple_str = ' '.join(map(str, item))
        # Write the formatted string to the file, followed by a newline character
        file.write(tuple_str + '\n')

# Step 5: Evaluate
labeled_test_docs_file = input("Please enter the name of the file containing the list of labeled test documents: ")
print("The predicted labels and provided labels of the test documents will be compared using a perl script. ")

# Run Perl script for evaluation of predictions
command = ['perl', 'analyze.pl', output_file, labeled_test_docs_file]
process = subprocess.run(command, capture_output=True, text=True)

# Check if the process was successful
if process.returncode == 0:
    print('Perl script output:', process.stdout)
else:
    print('Error running Perl script:', process.stderr)


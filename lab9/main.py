import csv
import os
import random
import math

def euclidean(point, data):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(point, data)))


import random
import math


class MyKMeans:
    def __init__(self, n_clusters=8, max_iter=750):
        # Initialize KMeans with number of clusters and maximum iterations
        self.centroids = None  # Centroids of clusters
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum number of iterations

    def fit(self, X):
        # Fit KMeans to the data X

        # Initialize centroids with random distinct points from X
        self.centroids = random.sample(list(X), min(self.n_clusters, len(X)))

        iteration = 0  # Counter for iterations
        prev_centroids = None  # Previous centroids for convergence check
        # Loop until convergence or maximum iterations reached
        while tuple(self.centroids) != prev_centroids and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]  # Lists to store points for each cluster
            # Assign each point to the nearest centroid
            for x in X:
                dists = [euclidean(x, centroid) for centroid in self.centroids]  # Calculate distances to centroids
                centroid_index = dists.index(min(dists))  # Find index of the nearest centroid
                if centroid_index < len(sorted_points):  # Check if centroid_index is within the range
                    sorted_points[centroid_index].append(x)  # Append point to the corresponding cluster list
            prev_centroids = self.centroids  # Update previous centroids
            # Update centroids by calculating the mean of points in each cluster
            self.centroids = [tuple(sum(d) / len(d) for d in zip(*cluster)) for cluster in sorted_points]

            # Handle NaN values in centroids by reverting to previous centroids
            for i, centroid in enumerate(self.centroids):
                if any(math.isnan(coord) for coord in centroid):
                    self.centroids[i] = prev_centroids[i]
            iteration += 1  # Increment iteration counter

    def evaluate(self, X):
        # Evaluate KMeans on the data X

        centroids = []  # List to store the centroids of points
        centroid_indexes = []  # List to store the index of the centroid for each point
        # Assign each point to the nearest centroid and collect centroid information
        for x in X:
            dists = [euclidean(x, centroid) for centroid in self.centroids]  # Calculate distances to centroids
            centroid_index = dists.index(min(dists))  # Find index of the nearest centroid
            centroids.append(self.centroids[centroid_index])  # Append centroid to centroids list
            centroid_indexes.append(centroid_index)  # Append centroid index to centroid_indexes list
        return centroids, centroid_indexes


#load data
crtDir = os.getcwd()
fileName = os.path.join(crtDir, 'reviews_mixed.csv')
data = []
with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            dataNames = row
        else:
            data.append(row)
        line_count += 1

inputs = [data[i][0] for i in range(len(data))][:100]
outputs = [data[i][1] for i in range(len(data))][:100]
labelNames = list(set(outputs))

print(inputs[:2])
print(labelNames[:2])

#split data for training and testing
import numpy as np

np.random.seed(5)
noSamples = len(inputs)
indexes = [i for i in range(noSamples)]
trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace = False)
testSample = [i for i in indexes  if not i in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]

print(trainInputs[:3])

#extract features (bag of words)
print("\nExtracting features using CountVectorizer")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=50)

trainFeatures = vectorizer.fit_transform(trainInputs).toarray()
testFeatures = vectorizer.transform(testInputs).toarray()

# vocabulary size
print("vocab size: ", len(vectorizer.vocabulary_),  " words")
#no of reviews
print("traindata size: ", len(trainInputs), " reviews")
# shape of feature matrix
print("trainFeatures shape: ", trainFeatures.shape)

# vocabbulary from the train data
print('some words of the vocab: ', vectorizer.get_feature_names_out()[-20:])
# extracted features
print('some features: ', trainFeatures[:3])

#extract features using
def average_word_length(text):
    words = text.split()
    if(len(words) == 0):
        return 0
    return sum(len(word) for word in words) / len(words)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

def extract_features(data):
    #numar cuvinte
    noWords = len(word_tokenize(data))
    #numar propozitii
    noSentences = len(sent_tokenize(data))
    #numar cuvinte unice
    noUniqueWords = len(set(word_tokenize(data)))
    #numar cuvinte cu majuscule
    noUpperCaseWords = len([word for word in word_tokenize(data) if word[0].isupper()])
    #numar cuvinte cu lungime medie
    avgWordLength = average_word_length(data)
    #numar adjective
    noAdjectives = len([word for word, pos in pos_tag(word_tokenize(data)) if pos == 'JJ'])
    #numar verbe
    noVerbs = len([word for word, pos in pos_tag(word_tokenize(data)) if pos == 'VB'])
    #numar pronume
    noPronouns = len([word for word, pos in pos_tag(word_tokenize(data)) if pos == 'PRP'])

    return [noWords, noSentences, noUniqueWords, noUpperCaseWords, avgWordLength, noAdjectives, noVerbs, noPronouns]

noWords, noSentences, noUniqueWords, noUpperCaseWords, avgWordLength, noAdjectives, noVerbs, noPronouns = extract_features(trainInputs[0])
print("\nFeatures for the first review: ")
print("review: ", trainInputs[0])
print("noWords: ", noWords)
print("noSentences: ", noSentences)
print("noUniqueWords: ", noUniqueWords)
print("noUpperCaseWords: ", noUpperCaseWords)
print("avgWordLength: ", avgWordLength)
print("noAdjectives: ", noAdjectives)
print("noVerbs: ", noVerbs)
print("noPronouns: ", noPronouns)

#label texts with emotions using KMeans (tool)
print("\nLabeling texts with emotions using KMeans (tool)")
from sklearn.cluster import KMeans
unsupervisedClassifier = KMeans(n_clusters = 2, random_state = 0)
unsupervisedClassifier.fit(trainFeatures)

computedTestIndexes = unsupervisedClassifier.predict(testFeatures)
computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
for i in range(0, len(testInputs)):
      print(testInputs[i], " -> ", computedTestOutputs[i])

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(testOutputs, computedTestOutputs))

#use the classifier to classify the given text
text = ["By choosing a bike over a car, I’m reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I’m proud to be part of that movement."]
text_features = vectorizer.transform(text).toarray()

text_cluster_index = unsupervisedClassifier.predict(text_features)[0]
computedTestOutputs = labelNames[text_cluster_index]
print("Prediction for given input text: ", computedTestOutputs)

true_label = "positive"
print("acc: ", accuracy_score([true_label], [computedTestOutputs]))

#label texts with emotions using KMeans (my implementation)
print("\nLabeling texts with emotions using KMeans (my implementation)")
myUnsupervisedClassifier = MyKMeans(n_clusters = 2)
myUnsupervisedClassifier.fit(trainFeatures)

computedTestIndexes = myUnsupervisedClassifier.evaluate(testFeatures)[1]    #return the indexes of the centroids
computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
for i in range(0, len(testInputs)):
        print(testInputs[i], " -> ", computedTestOutputs[i])

print("Accuracy: ", accuracy_score(testOutputs, computedTestOutputs))

#use the classifier to classify the given text
text = ["By choosing a bike over a car, I’m reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I’m proud to be part of that movement."]
text_features = vectorizer.transform(text).toarray()

text_cluster_index = myUnsupervisedClassifier.evaluate(text_features)[1][0]
computedTestOutputs = labelNames[text_cluster_index]
print("Prediction for given input text: ", computedTestOutputs)

true_label = "positive"
print("acc: ", accuracy_score([true_label], [computedTestOutputs]))

#alternative to k-means and performance analysis - 100 points
from sklearn.cluster import DBSCAN, AgglomerativeClustering

#DBSCAN
dbscan = DBSCAN(eps = 0.5, min_samples = 5)
dbscan_labels = dbscan.fit_predict(trainFeatures)

#Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters = 2)
agglo_labels = agglo.fit_predict(trainFeatures)

#Test performance DBSCAN
computedTestIndexes = dbscan.fit_predict(testFeatures)
computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
print("\nTest performance DBSCAN")
for i in range(0, len(testInputs)):
    print(testInputs[i], " -> ", computedTestOutputs[i])
print("Accuracy DBSCAN: ", accuracy_score(testOutputs, computedTestOutputs))

#Test performance Agglomerative Clustering
computedTestIndexes = agglo.fit_predict(testFeatures)
computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
print("\nTest performance Agglomerative Clustering")
for i in range(0, len(testInputs)):
    print(testInputs[i], " -> ", computedTestOutputs[i])
print("Accuracy Agglomerative Clustering: ", accuracy_score(testOutputs, computedTestOutputs))


print("\nAzure Text Analytics")
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

subscription_key = "abc"
endpoint = "abc"

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(subscription_key))

textToAnalize = ["By choosing a bike over a car, I’m reducing my environmental footprint. Cycling promotes eco-friendly"
                 " transportation, and I’m proud to be part of that movement.."]

result = client.analyze_sentiment(textToAnalize, show_opinion_mining=True)
docs = [doc for doc in result if not doc.is_error]

print("Let's visualize the sentiment of this textToAnalize")
for idx, doc in enumerate(docs):
    print(f"Document text: {textToAnalize[idx]}")
    print(f"Overall sentiment: {doc.sentiment}")
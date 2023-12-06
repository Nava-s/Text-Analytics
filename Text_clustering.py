import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import xlsxwriter
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import json

#__________________________________________________________________________________________________

# Download NLTK stopwords
nltk.download('stopwords')
custom_stopwords_file = 'custom_stopwords.txt'

#__________________________________________________________________________________________________

#load new stopwords to extend the stopwords vocabulary
try:
    with open(custom_stopwords_file, 'r', encoding='utf-8') as file:
        custom_stopwords = [line.strip() for line in file.readlines()]
except FileNotFoundError:
    print(f"Custom stopwords file '{custom_stopwords_file}' not found. Using default stopwords.")

#__________________________________________________________________________________________________

# Load cluster parameters from JSON file
try:
    with open('config_file.json', 'r') as json_file:
        params = json.load(json_file)
except FileNotFoundError:
    print("Cluster parameters JSON file not found. Using default values.")
    params = {"kmeans_clusters": 20, "agglomerative_clusters": 20, "language": 'english'}

#__________________________________________________________________________________________________

# Extract cluster parameters
kmeans_clusters = params["kmeans_clusters"]
agglomerative_clusters = params["agglomerative_clusters"]

#__________________________________________________________________________________________________

#Extract language
language = params["language"]

#__________________________________________________________________________________________________

# Initialize NLTK stopwords and stemmer
sw = stopwords.words(language)
stemmer = SnowballStemmer(language)
sw.extend(custom_stopwords)

#__________________________________________________________________________________________________

# Load data from Excel
try:
    df = pd.read_csv('demo.csv', error_bad_lines=False)  # Ignore lines with parsing errors
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")

#__________________________________________________________________________________________________

# Extract contents from the first column
mylist = df['Contenuti'].tolist()

#__________________________________________________________________________________________________

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words=sw, tokenizer=lambda x: [stemmer.stem(i) for i in x.split()])
tfidf_matrix = tfidf.fit_transform(mylist)

#__________________________________________________________________________________________________

# K-Means clustering
kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
cluster_kmeans = kmeans.fit_predict(tfidf_matrix.toarray())

# Agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=agglomerative_clusters)
cluster_agg = agglomerative.fit_predict(tfidf_matrix.toarray())

# Affinity Propagation clustering
affinity_propagation = AffinityPropagation()
cluster_affinity = affinity_propagation.fit_predict(tfidf_matrix.toarray())

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#__________________________________________________________________________________________________

# Add clustering results to the DataFrame
df["cluster_kmeans"] = cluster_kmeans
df["cluster_agg"] = cluster_agg
df["cluster_affinity"] = cluster_affinity
df["cosine_similarity"] = list(cosine_sim)

# Write result to csv
df.to_csv('output.csv', index=False)

#__________________________________________________________________________________________________

#Print the 10 most frequent values for each cluster
for column in ["cluster_kmeans", "cluster_agg", "cluster_affinity", "cosine_similarity"]:
    print(f"\nMost frequent values in {column}:")
    if column == "cosine_similarity":
        # For cosine similarity, print the values directly
        print(df[column])
    else:
        # For clustering columns, print the most frequent values within each cluster
        for cluster_id in df[column].unique():
            cluster_data = df[df[column] == cluster_id]['Contenuti']
            all_text = ' '.join(cluster_data)
            
            # Filter out stopwords
            words = [word for word in all_text.split() if word.lower() not in sw]
            
            # Count word occurrences
            common_words = pd.Series(words).value_counts()[:10]
            print(f"Cluster {cluster_id}:\n{common_words}\n")

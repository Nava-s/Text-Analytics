import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
import matplotlib.pyplot as plt
import xlsxwriter
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize NLTK stopwords and stemmer
sw = stopwords.words('english')
stemmer = SnowballStemmer("english")

# Load data from Excel
try:
    df = pd.read_csv('demo.csv', error_bad_lines=False)  # Ignore lines with parsing errors
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")

# Extract contents from the first column
mylist = df['Contenuti'].tolist()

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words=sw, tokenizer=lambda x: [stemmer.stem(i) for i in x.split()])
tfidf_matrix = tfidf.fit_transform(mylist)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_kmeans = kmeans.fit_predict(tfidf_matrix.toarray())

# Agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=5)
cluster_agg = agglomerative.fit_predict(tfidf_matrix.toarray())

# Affinity Propagation clustering
affinity_propagation = AffinityPropagation()
cluster_affinity = affinity_propagation.fit_predict(tfidf_matrix.toarray())

# Add clustering results to the DataFrame
df["cluster_kmeans"] = cluster_kmeans
df["cluster_agg"] = cluster_agg
df["cluster_affinity"] = cluster_affinity


# Writing the headers
df["cluster_kmeans"] = cluster_kmeans
df["cluster_agg"] = cluster_agg
df["cluster_affinity"] = cluster_affinity
# Write result to csv
df.to_csv('output.csv', index=False)

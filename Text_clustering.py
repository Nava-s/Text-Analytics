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
df = pd.read_excel('categorie_paghe.xlsx', sheet_name=0)

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

# Write results to Excel
workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

# Writing the headers
worksheet.write(0, 0, 'K-Means Cluster')
worksheet.write(0, 1, 'Agglomerative Cluster')
worksheet.write(0, 2, 'Affinity Propagation Cluster')
worksheet.write(0, 3, 'Text')

# Writing the clusters and text on each row
for i in range(len(df)):
    worksheet.write(i + 1, 0, df["cluster_kmeans"][i])
    worksheet.write(i + 1, 1, df["cluster_agg"][i])
    worksheet.write(i + 1, 2, df["cluster_affinity"][i])
    worksheet.write(i + 1, 3, df["Contenuti"][i])

workbook.close()

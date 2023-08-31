import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import nltk
import xlsxwriter
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("italian")
sw = stopwords.words('italian')

#altre parole da rimuovere
otherwords = ['grazie','info','richiesta','buongiorno','buonasera',
              'arrivederci','cordialmente','cordiali','saluti','buon','pomeriggio','grazie',
              '?','volevo','avere','chiedo','cid','ciao','cortesia',
              'cortesemente','così','devo','deve','detto','info']

#for i in otherwords:
#  sw.append(i)

#script e file excel nella stessa cartella
#il primo argomento della funzione pd.read_excel deve essere il nome del file excel in cui sono contenuti i testi
df = pd.read_excel('Categorie_presenze.xlsx', sheet_name = 0) # can also index sheet by name or fetch all sheets

#Contenuti è l'intestazione della prima colonna
mylist = df['Contenuti'].tolist()

tfidf = TfidfVectorizer(stop_words=sw,tokenizer=lambda x: [stemmer.stem(i) for i in x.split()])
x = tfidf.fit_transform(mylist)
# 25 is the maximum number of clusters
c = cluster.AgglomerativeClustering(20)
a = c.fit_predict(x.toarray())

df["cluster"] = a

#grouping the data by cluster
grouped_df = df.groupby("cluster")
#getting the most common elements for each cluster
cluster_elements = grouped_df.agg(lambda x:x.value_counts().index[0])

#printing the string that best represents the cluster
print(cluster_elements["Contenuti"])

workbook = xlsxwriter.Workbook('cluster_results.xlsx')
worksheet = workbook.add_worksheet()

#Writing the headers
worksheet.write(0, 0, 'Cluster')
worksheet.write(0, 1, 'Text')

row = 1
column = 0

#Writing the cluster and text on each row
for i in range(len(df)):
    worksheet.write(row,column, df["cluster"][i])
    worksheet.write(row,column + 1, df["Contenuti"][i])
    row +=1
workbook.close()

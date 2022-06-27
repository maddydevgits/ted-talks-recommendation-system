import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
data = pd.read_csv('tedx_dataset.csv')

stopwords_list = stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word')
tfidf_des = vectorizer.fit_transform(data['details'])

from sklearn.metrics.pairwise import linear_kernel
# comping cosine similarity matrix using linear_kernal of sklearn
cosine_sim_des = linear_kernel(tfidf_des, tfidf_des)
data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])

def recommend_cosine(isbn):
  id = indices[isbn]
  # Get the pairwise similarity scores of all tedx_talks compared    that tedx_talk,
  # sorting them and getting top 5
  similarity_scores = list(enumerate(cosine_sim_des[id]))
  similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
  similarity_scores = similarity_scores[1:6]
  #Get the tedx_talk index
  ted_index = [i[0] for i in similarity_scores]
  #Return the top 5 most similar tedx_talks using integar-location based indexing (iloc)
  return data.iloc[ted_index]

print(recommend_cosine("Simplicity sells"))

from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(tfidf_des)
def recommend_euclidean_distance(isbn):
    ind = indices[isbn]
    distance = list(enumerate(D[ind]))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[1:6] 
    #Get the tedx_talk index
    ted_index = [i[0] for i in distance]
    #Return the top 5 most similar tedx_talks using integar-location based indexing (iloc)
    return data.iloc[ted_index]

print(recommend_euclidean_distance("10 top time-saving tech tips"))

from scipy.stats import pearsonr
tfidf_des_array = tfidf_des.toarray()
def recommend_pearson(isbn):
    ind = indices[isbn]
    correlation = []
    for i in range(len(tfidf_des_array)):
      correlation.append(pearsonr(tfidf_des_array[ind],   tfidf_des_array[i])[0])
    correlation = list(enumerate(correlation))
    sorted_corr = sorted(correlation, reverse=True, key=lambda x: x[1])[1:6]
    ted_index = [i[0] for i in sorted_corr]
    return data.iloc[ted_index]

print(recommend_pearson('What will future jobs look like?'))
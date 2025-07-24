import pandas as pd

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction

# Embeddings
english_embeddings = pd.read_csv("truthfulqa_embeddings_eng.csv")
filipino_embeddings = pd.read_csv("truthfulqa_embeddings_fil.csv")

# Dimensionality Reduction
umap_model_5d = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    low_memory=False,
    random_state=0
)

umap_model_2d = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.0,
    metric='cosine',
    low_memory=False,
    random_state=0
)

# empty_dimensionality_model = BaseDimensionalityReduction()

# english_embeddings_5d = pd.read_csv("truthfulqa_embeddings_5d_eng.csv")
# english_embeddings_2d = pd.read_csv("truthfulqa_embeddings_2d_eng.csv")
# filipino_embeddings_5d = pd.read_csv("truthfulqa_embeddings_5d_fil.csv")
# filipino_embeddings_2d = pd.read_csv("truthfulqa_embeddings_2d_fil.csv")

# Clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    approx_min_span_tree=False,
    core_dist_n_jobs=1,
)

# Count Vectorizing
vectorizer_model_english = CountVectorizer(stop_words='english')

with open("stopwords-tl.txt", encoding="utf-8") as f:
    filipino_stopwords = [line.strip() for line in f if line.strip()]

vectorizer_model_filipino = CountVectorizer(stop_words=filipino_stopwords)

# c-TF-IDF
ctfidf_model = ClassTfidfTransformer()

# BERTopic
topic_model_english = BERTopic(
    umap_model=umap_model_5d,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model_english,
    ctfidf_model=ctfidf_model,
)

topic_model_filipino = BERTopic(
    umap_model=umap_model_5d,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model_filipino,
    ctfidf_model=ctfidf_model,
)

# Fit Transform
english_topics, english_probs = topic_model_english.fit_transform(
    documents=english_embeddings['question'],
    embeddings=english_embeddings.drop(columns=['question']).to_numpy()
)

filipino_topics, filipino_probs = topic_model_filipino.fit_transform(
    documents=filipino_embeddings['question'],
    embeddings=filipino_embeddings.drop(columns=['question']).to_numpy()
)

# Get Topic Info
english_topic_info = topic_model_english.get_topic_info()

filipino_topic_info = topic_model_filipino.get_topic_info()

# To CSV
english_embeddings['Topic'] = english_topics
english_embeddings = pd.merge(english_embeddings, english_topic_info, on='Topic', how='left')
# english_embeddings.to_csv('truthfulqa_topics_english.csv', index=False)

filipino_embeddings['Topic'] = filipino_topics
filipino_embeddings = pd.merge(filipino_embeddings, filipino_topic_info, on='Topic', how='left')
# filipino_embeddings.to_csv('truthfulqa_topics_filipino.csv', index=False)
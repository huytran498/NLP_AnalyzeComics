import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    return ''

def perform_nlp_analysis(df):
    # Clean descriptions
    df['CleanDescription'] = df['Description'].apply(clean_text)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['CleanDescription'])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Get top terms for each cluster
    def get_top_terms(cluster, n_terms):
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf.get_feature_names()
        return [terms[i] for i in order_centroids[cluster, :n_terms]]

    top_terms = [get_top_terms(i, 5) for i in range(5)]

    # Visualize the clusters
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('K-means Clustering of Descriptions')
    plt.savefig('description_clusters.png')
    plt.close()

    return clusters, top_terms

if __name__ == "__main__":
    print("This script is intended to be imported, not run directly.")
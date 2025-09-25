import pickle
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def train_model(X_scaled, algo="kmeans", n_clusters=5):
    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algo == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algo == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Unsupported algorithm")
    model.fit(X_scaled)
    return model

def save_model(model, scaler, model_path=None, scaler_path=None):
    """
    Save model and scaler using pickle in the project root 'models/' folder
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(project_root, exist_ok=True)  # create folder if missing

    if model_path is None:
        model_path = os.path.join(project_root, "clustering_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(project_root, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

def load_model(model_path=None, scaler_path=None):
    """
    Load model and scaler from 'models/' folder
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    if model_path is None:
        model_path = os.path.join(project_root, "clustering_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(project_root, "scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

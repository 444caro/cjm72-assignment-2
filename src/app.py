from flask import Flask, request, jsonify, send_file
import numpy as np
from kmeans import KMeans
from visualizations.plot import plot_clusters
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    items = ["K-means", "Clustering", "Visualization" ]
    return render_template('index.html', items=items)

@app.route('/kmeans', methods=['POST'])
def run_kmeans(): 
    data = request.get_json()
    X = np.array(data['X'])
    n_clusters = data['n_clusters']
    max_iter = data.get('max_iter', 300)
    tol = data.get('tol', 1e-4)
    init_method = data.get['init_method']
    
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, init_method=init_method)
    y = model.fit_predict(X)
    centroids = model.centroids
    image = plot_clusters(X, y, centroids)
    
    im_io = BytesIO()
    image.save(im_io, 'PNG')
    im_io.seek(0)
    
    return send_file(im_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
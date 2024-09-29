from flask import Flask, request, jsonify, send_file, render_template
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
    init_method = data.get('init_method')
    
    initial_centroids = data.get('initial_centroids', None)
    
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, init_method=init_method, initial_centroids=initial_centroids)
    y = model.fit_predict(X)
    centroids = model.centroids
    image = plot_clusters(X, y, centroids)
    
    im_io = BytesIO()
    image.save(im_io, 'PNG')
    im_io.seek(0)
    
    return send_file(im_io, mimetype='image/png')

@app.route('/get_data_bounds', methods=['GET'])
def get_data_bounds():
    # Assuming fixed bounds for simplicity
    minX, maxX = 0, 10
    minY, maxY = 0, 10
    return jsonify({
        'minX': minX, 'maxX': maxX,
        'minY': minY, 'maxY': maxY
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
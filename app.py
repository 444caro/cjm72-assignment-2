from flask import Flask, request, jsonify, send_file, render_template, json
import numpy as np
from sklearn.datasets import make_blobs
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

kmeans = None
centroids_init = False
init_plot = None

class KMeans:
    def __init__(self, data, n_clusters, initialization):
        self.data = data
        self.n_clusters = n_clusters
        self.initialization = initialization
        self.centers = np.zeros((self.n_clusters, self.data.shape[1]))
        self.assign = [-1] * len(self.data)
        self.steps = []
        
    def initialize_centers(self):
        if self.initialization == 'random':
            self.centers = self.data[np.random.choice(len(self.data), size = self.n_clusters, replace=False)]
        elif self.initialization == 'farthest':
            self.centers = [self.data[np.random.choice(len(self.data))]]
            for _ in range(1, self.n_clusters):
                distances = np.min(np.linalg.norm(self.data[:, np.newaxis] - self.centers, axis=2), axis=0)
                next_center = self.data[np.argmax(distances)]
                self.centers.append(next_center)
        elif self.initialization == 'kmeans++':
            self.centers = [self.data[np.random.choice(len(self.data))]]
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centers]) for x in self.data])
                probabilities = distances / distances.sum()
                next_center = self.data[np.random.choice(len(self.data), p=probabilities)]
                self.centers.append(next_center)
        elif self.initialization == 'manual':
            pass
        else:
            raise ValueError('Invalid initialization method')
        
    def assign_clusters(self):
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.centers, axis=2) 
        self.assign = np.argmin(distances, axis=1)
    
    def update_centers(self):
        new_centers = np.zeros_like(self.centers)
        for i in range(self.n_clusters):
            points = self.data[self.assign == i]
            if len(points) > 0:
                new_centers[i] = points.mean(axis=0)
        return new_centers
    
    def step(self):
        global centroids_init
        if not centroids_init:
            self.initialize_centers()
            centroids_init = True
        else:
            self.assign_clusters()
            new_centers = self.update_centers()
            if np.allclose(self.centers, new_centers):
                return False
            self.centers = new_centers
        return True
    
def generate_rand_data(n = 300):
    return np.random.randn(n, 2)

def cluster_plot(data, centers, assign):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', marker=dict(color='blue', showscale = True), name = 'Data'))
    if centers is not None:
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers', marker=dict(color='red', symbol = 'x', size=10), name = 'Centroids'))
    fig.update_layout(xaxis_title = "X Axis", yaxis_title = "Y Axis", width=600, height=600)
    plot_html = pio.to_html(fig, full_html=False)
    return plot_html




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global kmeans, centroids_init, init_plot
    try:
        num_clusters = int(request.form.get('num_clusters'))
        init_method = request.form.get('init_method')
        centroids_init = False
        data = generate_rand_data(n = 300)
        kmeans = KMeans(data, num_clusters, init_method)
        fig = cluster_plot(data, None, [-1] * len(kmeans.data))
        init_plot = fig
        return jsonify({'plot': fig})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/step', methods=['POST'])
def step():
    global kmeans
    if kmeans is None:
        return jsonify({'error': 'Initialize'})
    try:
        converged = not kmeans.step()
        fig = cluster_plot(kmeans.data, kmeans.centers, kmeans.assign)
        return jsonify({'plot': fig, 'converged': converged})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/run', methods=['POST'])
def run():
    global kmeans 
    if kmeans is None:
        return jsonify({'error': 'Initialize'})
    try:
        converged = False
        while not converged:
            converged = not kmeans.step()
        fig = cluster_plot(kmeans.data, kmeans.centers, kmeans.assign)
        return jsonify({'plot': fig, 'converged': True})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/reset', methods=['POST'])
def reset():
    global kmeans, centroids_init, init_plot
    if kmeans and init_plot:
        centroids_init = False
        kmeans.assign = [-1] * len(kmeans.data)
        kmeans.centers = np.zeros((kmeans.n_clusters, kmeans.data.shape[1]))
        return jsonify({'success': True, 'plot': init_plot, 'clear plot': True})
    else:
        return jsonify({'success': False})
    
@app.route('/manual_centroids', methods=['POST'])
def manual_centroids():
    global kmeans
    if kmeans is None:
        return jsonify({'error': 'manual_centroids'})
    try:
        centroids = json.loads(request.form['centroids'])
        manual_centers = np.array([[c['x'], c['y']] for c in centroids])    
        if len(manual_centers) > kmeans.n_clusters:
            manual_centers = manual_centers[:kmeans.n_clusters]
        manual_centers[:, 0] = manual_centers[:,0] * (kmeans.data[:, 0].max()/600)
        manual_centers[:, 1] = manual_centers[:,1] * (kmeans.data[:, 1].max()/600)
        kmeans.centers = manual_centers
        kmeans.assign_clusters()
        fig = cluster_plot(kmeans.data, kmeans.centers, kmeans.assign)  
        return jsonify({'plot': fig})    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_data', methods=['GET'])
def get_data():
    global kmeans 
    if kmeans is None:
        return jsonify({'error': 'get_data'})
    fig = cluster_plot(kmeans.data, kmeans.centers, kmeans.assign)
    return jsonify(pio.to_json(fig))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
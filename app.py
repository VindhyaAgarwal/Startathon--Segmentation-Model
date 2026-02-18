from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment_image():
   
    response = {
        'seg_image_url': 'http://127.0.0.1:5000/static/segmented_sample.png',
        'dominant_class': 'Sky',
        'confidence': 0.94,
        'classes_detected': 7,
        'mean_iou': 0.75,
        'base_iou': 0.26,
        'improvement': 0.49,
        'inference_time_ms': 47
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

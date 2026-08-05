import requests
from flask import *
from flask import Flask
from flask import request
from src.server.pre_annotate import predict_yaw
from src.server.active_learning import pvrcnn_inference
from src.od3d import calculate_projected_bounding_boxes, project_points
from os import path
import time

app = Flask(__name__)


@app.route("/project_bounding_box", methods=['POST'])
def project_bounding_box():
    """
    Batched: projects one or more 3D boxes into ALL camera channels in a
    single request.

    Preferred payload:
    {
        "boxes": [
            {"x":.., "y":.., "z":.., "length":.., "width":.., "height":.., "yaw":..},
            ...
        ],
        "coordinateSystem": {"x-axis":.., "y-axis":.., "z-axis":..},
        "channels": [
            {"channel": "CAM_FRONT", "zoomFactor": 1.0, ...},
            ...
        ]
    }

    Legacy single-box payload still accepted:
    { "box": {...}, "coordinateSystem": {...}, "channels": [...] }

    Returns:
        { "CAM_FRONT": [ [[x,y],...8], [[x,y],...8], ... ],  # one list per box
          "CAM_BACK":  [ ... ],
          ... }
    Always a list per channel (even when only 1 box is sent).
    """
    data = request.json
    coordinate_system = data['coordinateSystem']

    # Prefer multi-box; fall back to legacy single "box"
    if 'boxes' in data and data['boxes'] is not None:
        boxes = data['boxes']
        if isinstance(boxes, dict):
            boxes = [boxes]
    elif 'box' in data and data['box'] is not None:
        boxes = [data['box']]
    else:
        return jsonify({}), 400
    
    result = {}
    for ch in data['channels']:
        channel_name = ch['channel']
        zoom_factor = ch.get('zoomFactor', 1.0)
        # Always returns list of length len(boxes)
        result[channel_name] = calculate_projected_bounding_boxes(
            boxes=boxes,
            coordinate_system=coordinate_system,
            zoom_factor=zoom_factor,
            channel_name=channel_name,
        )
    
    response = jsonify(result)

    return response


@app.route("/project_points", methods=['POST'])
def project_points_route():
    """
    Expects: {"points3D": [[x,y,z], ...], "projectionMatrix": [[..],[..],[..]], "scalingFactor": 1.0}
    Returns: {"points2D": [...], "points3D": [...], "distances": [...]}
    """
    data = request.json
    result = project_points(
        data['points3D'],
        data['projectionMatrix'],
        data['scalingFactor'],
    )
    return jsonify(result)
 

@app.route("/save_annotations", methods=['POST'])
def save_annotations():
    data = request.json
    
    for i in range(len(data['annotationFiles'])):        
        filePath = path.join('input', data['dataset'], data['sequence'], 'annotations', data['fileNames'][i])
        with open(filePath, 'w') as f:
            f.write(data['annotationFiles'][i])

    return {
        'status': 'success'
    }

@app.route("/save_detections", methods=['POST'])
def save_detections():
    data = request.json

    for i in range(len(data['annotationFiles'])):
        filePath = path.join('input', data['dataset'], data['sequence'], 'annotations', data['fileNames'][i])
        with open(filePath, 'w') as f:
            f.write(data['annotationFiles'][i])

    return {
        'status': 'success'
    }


@app.route("/connect-to-workstation", methods=['POST'])
def connect_to_workstation():
    data = request.json
    if (data['mode'] == 'AL'):
        data_json = {
            'mode': data['mode'],
            'op': data['op'],
            'N_select': data['N_select'],
            'query': data['query'],
        }
        print(data_json)

    elif (data['mode'] == 'inference'):
        data_json = {
            'mode': data['mode'],
            'op': data['op'],
            'frame_ids': data['filenames'],
        }

    elif (data['mode'] == 'evaluation'):
        data_json = {
            'mode': data['mode'],
            'op': data['op'],
            'frame_ids': data['filenames'],
        }

    url = 'http://172.29.0.8:5000/run-docker-script'

    try:
        print("sending data to workstation")
        response = requests.post(url, json=data_json)
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:

        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return "Gunicorn Flask server is running"

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

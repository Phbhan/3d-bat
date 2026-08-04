import requests
from flask import *
from flask import Flask
from flask import request
from src.server.pre_annotate import predict_yaw
from src.server.active_learning import pvrcnn_inference
from src.od3d import calculate_projected_bounding_box, project_points
from os import path

app = Flask(__name__)

 
@app.route("/project_bounding_box", methods=['POST'])
def project_bounding_box():
    """
    Batched: projects ONE box into MANY camera channels in a single request,
    so the client only needs one HTTP round-trip per box update (not one per
    channel, not one per corner).
 
    Expects:
    {
        "box": {"x":.., "y":.., "z":.., "length":.., "width":.., "height":.., "yaw":..},
        "coordinateSystem": {"x-axis":.., "y-axis":.., "z-axis":..},
        "channels": [
            {
                "channel": "CAM_FRONT",
                "projectionMatrix": [[..],[..],[..]],
                "zoomFactor": 1.0,
                "infraTransformMatrix": null  // optional, 4x4, only for vehicle_camera_basler_16mm
            },
            ...
        ]
    }
    Returns: { "CAM_FRONT": [[x,y], ...] or [], "CAM_BACK": [...], ... }
    """
    data = request.json
    box = data['box']
    coordinate_system = data['coordinateSystem']
 
    result = {}
    for ch in data['channels']:
        result[ch['channel']] = calculate_projected_bounding_box(
            box['x'], box['y'], box['z'],
            box['length'], box['width'], box['height'],
            box['yaw'],
            coordinate_system,
            ch.get('zoomFactor', 1.0),
            ch['channel']
        )
    return jsonify(result)
 
 
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


if __name__ == "__main__":
    app.run(debug=True)


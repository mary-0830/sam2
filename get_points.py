from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


points = []
example_images = [
    '00.png',
    '05.png',
    '10.png',
    '15.png'
]  # 示例图片列表

POINTS_DIR = os.path.join(os.path.dirname(__file__), 'points')
os.makedirs(POINTS_DIR, exist_ok=True)

def save_points_to_file(username, image_name):
    """将点保存到文件，文件名包含用户名和图片名称"""
    base_image_name = os.path.splitext(image_name)[0]  # 去掉图片扩展名
    filename = f"{username}_{base_image_name}.json"
    file_path = os.path.join(POINTS_DIR, filename)
    with open(file_path, 'w') as f:
        json.dump(points, f)

def load_points_from_file(username, image_name):
    """从文件加载点"""
    global points
    points.clear()  # 清空当前点列表
    base_image_name = os.path.splitext(image_name)[0]  # 去掉图片扩展名
    filename = f"{username}_{base_image_name}.json"
    file_path = os.path.join(POINTS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            points.extend(json.load(f))

@app.route('/')
def index():
    return render_template('index.html', example_images=example_images)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'image_path': f'/static/uploads/{filename}'})
    return redirect(url_for('index'))

@app.route('/get_image/<image_name>')
def get_image(image_name):
    if image_name in example_images:
        return jsonify({'image_path': f'/static/{image_name}'})  # 确保路径正确
    return jsonify({'error': 'Image not found'}), 404

@app.route('/add_point', methods=['POST'])
def add_point():
    global points
    data = request.json
    x, y = data['x'], data['y']
    username = data.get('username', 'unknown_user')  # 获取用户名，默认为 unknown_user
    image_name = data.get('image_name')  # 获取图片名称
    if not image_name:
        return jsonify({'status': 'error', 'message': 'Image name is required'}), 400
    points.append((x, y))
    save_points_to_file(username, image_name)  # 每次添加点后保存到文件
    return jsonify({'status': 'success', 'points': points})

@app.route('/get_points', methods=['POST'])
def get_points():
    """获取指定用户和图片的点"""
    global points
    data = request.json
    username = data.get('username', 'unknown_user')  # 获取用户名，默认为 unknown_user
    image_name = data.get('image_name')  # 获取图片名称
    if not image_name:
        return jsonify({'status': 'error', 'message': 'Image name is required'}), 400
    load_points_from_file(username, image_name)  # 加载对应的点
    return jsonify({'points': points})

if __name__ == "__main__":
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)

    # 将示例图片从原始路径复制到 static 文件夹
    source_dir = '/data3/ljj/proj/vggt/examples/kitchen/images'
    for image_name in example_images:
        source_path = os.path.join(source_dir, image_name)
        target_path = os.path.join(static_dir, image_name)
        if not os.path.exists(target_path):
            if os.path.exists(source_path):
                import shutil
                shutil.copy(source_path, target_path)
            else:
                raise FileNotFoundError(f"示例图片 {source_path} 不存在")

    app.run(host='0.0.0.0', port=5000)

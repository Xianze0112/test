import atexit
import base64
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from multiprocessing import Pipe
from multiprocessing.shared_memory import SharedMemory
import os.path
import signal
import sys
from threading import Lock, Thread
import time
import traceback
import zipfile
import zlib

from flask import Flask, request, send_file, make_response, jsonify

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import io
from GeoTr import GeoTr

app = Flask(__name__)


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        print(os.getcwd())
        print(os.path.exists(path))
        print("absolute_path",  os.path.abspath(path))
        pretrained_dict = torch.load(path, map_location='cuda:0') # , map_location='cuda:0' torch.device('cpu'))
        # print(len(pretrained_dict.keys()))
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


class GeoTrP(nn.Module):
    def __init__(self):
        super(GeoTrP, self).__init__()
        self.GeoTr = GeoTr()

    def forward(self, x):
        bm = self.GeoTr(x)  # [0]
        bm = 2 * (bm / 288) - 1
        bm = (bm + 1) / 2 * 2560
        bm = F.interpolate(bm, size=(2560, 2560), mode='bilinear', align_corners=True)
        return bm



model = GeoTrP()
model = model.cuda()
# reload geometric unwarping model

os.chdir(os.path.dirname(os.path.abspath(__file__)))
reload_model(model.GeoTr, './model_save/model.pt')
# To eval mode
model.eval()


def pack_data(*args):
    packed = io.BytesIO()
    for data in args:
        length = len(data)
        packed.write(length.to_bytes(4, 'big'))  # 写入长度信息
        packed.write(data)  # 写入数据
    return packed.getvalue()


@app.route('/predictOne', methods=['POST'])
def predictOne():
    if request.method == 'POST':
        # 从请求中获取文件
        start_time = time.time()
        file = request.files['file']
        if not file:
            return {'error': 'No file provided'}, 400

        # 读取图片
        img_array = np.fromfile(file, np.uint8)
        im_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR) / 255.
        h_, w_, c_ = im_ori.shape
        im_ori = cv2.resize(im_ori, (2560, 2560))

        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        print("sucess p1", time.time() - start_time)

        with torch.no_grad():
            # 进行推理
            start_time = time.time()
            bm = model(im.cuda())
            print("sucess p2", time.time() - start_time)

            start_time = time.time()
            bm = bm.cuda().numpy()[0]
            bm0 = bm[0, :, :]
            bm1 = bm[1, :, :]
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))

            img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR) * 255
            img_geo = cv2.resize(img_geo, (w_, h_))

            # cv2.imwrite('./rectified/' + 'a' + '_geo' + '.png', img_geo.astype(np.uint8))  # save
            # 将处理后的图片转换为字节流
            _, img_encoded = cv2.imencode('.png', img_geo)
            img_bytes = img_encoded.tobytes()

            # return send_file(io.BytesIO(img_bytes),mimetype='image/png')

            # method 2: 打包为zip传输
            x_indices = np.arange(2560)
            y_indices = np.arange(2560).reshape(-1, 1)
            transformed_bm0 = np.round((bm0 - x_indices) / 10)
            bm0_int = np.clip(transformed_bm0, -128, 127).astype(np.int8)
            transformed_bm1 = np.round((bm1 - y_indices) / 10)
            bm1_int = np.clip(transformed_bm1, -128, 127).astype(np.int8)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('image.bin', img_bytes)
                zip_file.writestr('matrix0.bin', bm0_int.tobytes())
                zip_file.writestr('matrix1.bin', bm1_int.tobytes())
            zip_buffer.seek(0)

            print("sucess p3", time.time() - start_time)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='data.zip')

            # method 3 : 将串联后的数据压缩
            packed_data = pack_data(bm0_int.tobytes(), bm1_int.tobytes(), img_bytes)
            compressed_data = zlib.compress(packed_data)
            return send_file(io.BytesIO(compressed_data), mimetype='application/octet-stream', as_attachment=True, download_name='data.zlib')

            # method 1 : json传输
            response_data = {
                'bm0': compress_matrix(bm0),
                'bm1': compress_matrix(bm1),
                'img_bytes':img_bytes.decode('latin1'),
                'bm0-shape': bm0.shape,
                'bm1-shape': bm1.shape,
            }

            return jsonify(response_data)


''' 批处理  '''
def process_image(file):
    # 读取图片
    img_array = np.frombuffer(file, np.uint8)
    im_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR) / 255.0
    h_, w_, c_ = im_ori.shape
    img_param = (w_, h_)
    
    # 预处理和调整大小
    im_ori = cv2.resize(im_ori, (2560, 2560))
    
    im = cv2.resize(im_ori, (288, 288))
    
    # 转置并转换为 PyTorch 张量
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float()
    
    return im, im_ori, img_param

def process_result(im_ori, bm, im_param, index):
    bm = bm.cuda().numpy()

    bm0 = bm[0, :, :]
    bm1 = bm[1, :, :]

    # 对图像进行处理
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))

    img_geo = cv2.remap(im_ori, bm0, bm1, cv2.INTER_LINEAR) * 255
    img_geo = cv2.resize(img_geo, im_param)

    # 将处理后的图片转换为字节流
    img_encoded = cv2.imencode('.png', img_geo)
    img_bytes = img_encoded[1].tobytes()

    # 数据转换
    x_indices = np.arange(2560)
    y_indices = np.arange(2560).reshape(-1, 1)
    transformed_bm0 = np.round((bm0 - x_indices) / 10)
    bm0_int = np.clip(transformed_bm0, -128, 127).astype(np.int8)
    transformed_bm1 = np.round((bm1 - y_indices) / 10)
    bm1_int = np.clip(transformed_bm1, -128, 127).astype(np.int8)

    return index, img_bytes, bm0_int.tobytes(), bm1_int.tobytes()


def extract_images_from_zip(zip_bytes: bytes) -> list:
    zip_buffer = io.BytesIO(zip_bytes)
    img_byte_arr_list = []
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
        for file_name in sorted(zip_file.namelist()):
            with zip_file.open(file_name) as file:
                img_byte_arr = file.read()
                img_byte_arr_list.append(img_byte_arr)
    return img_byte_arr_list

@app.route('/predictBatch', methods=['POST'])
def predictBatch():
    if request.method == 'POST':
        # 从请求中获取文件
        file = request.files['file']
        if not file:
            return {'error': 'No file provided'}, 400
        
        zip_bytes = file.read()
        img_byte_arr_list = extract_images_from_zip(zip_bytes)

        # 预处理图片
        batch_images = []
        img_ori_list = []
        img_param_list = []
        start_time = time.time()
        # for i, img_byte_arr in enumerate(img_byte_arr_list):
        #     # 读取图片
        #     print("img", i)
        #     img_array = np.frombuffer(img_byte_arr, np.uint8)
        #     im_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR) / 255.0
        #     h_, w_, c_ = im_ori.shape
        #     img_param = (w_, h_)
        #     img_param_list.append(img_param)
            
        #     # 预处理和调整大小
        #     im_ori = cv2.resize(im_ori, (2560, 2560))
        #     img_ori_list.append(im_ori)
            
        #     im = cv2.resize(im_ori, (288, 288))
            
        #     # 转置并转换为 PyTorch 张量
        #     im = im.transpose(2, 0, 1)
        #     im = torch.from_numpy(im).float()
        #     batch_images.append(im)
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_image, img_byte_arr_list)
        for im, im_ori, img_param in results:
            batch_images.append(im)
            img_ori_list.append(im_ori)
            img_param_list.append(img_param)
        print("sucess p1", time.time() - start_time)

        # 批处理，矫正
        start_time = time.time()
        batch_tensor = torch.stack(batch_images)
        print(batch_tensor.size())
        try:
            with torch.no_grad():
                batch_bm = model(batch_tensor.cuda())
        except Exception as e:
            print("ERROR", e)
            raise e
        
        print("sucess p2", time.time() - start_time)

        # 保存结果
        futures = []
        zip_buffer = io.BytesIO()
        start_time = time.time()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, bm in enumerate(batch_bm):
                    bm = bm.cuda().numpy()
                    bm0 = bm[0, :, :]
                    bm1 = bm[1, :, :]

                    # 对图像进行处理
                    bm0 = cv2.blur(bm0, (3, 3))
                    bm1 = cv2.blur(bm1, (3, 3))

                    img_geo = cv2.remap(img_ori_list[i], bm0, bm1, cv2.INTER_LINEAR) * 255
                    img_geo = cv2.resize(img_geo, img_param_list[i])

                    # 将处理后的图片转换为字节流
                    img_encoded = cv2.imencode('.png', img_geo)
                    img_bytes = img_encoded[1].tobytes()

                    # 数据转换
                    x_indices = np.arange(2560)
                    y_indices = np.arange(2560).reshape(-1, 1)
                    transformed_bm0 = np.round((bm0 - x_indices) / 10)
                    bm0_int = np.clip(transformed_bm0, -128, 127).astype(np.int8)
                    transformed_bm1 = np.round((bm1 - y_indices) / 10)
                    bm1_int = np.clip(transformed_bm1, -128, 127).astype(np.int8)
                    
                    zip_file.writestr(f'image_{i}.png', img_bytes)
                    zip_file.writestr(f'matrix0_{i}.bin', bm0_int.tobytes())
                    zip_file.writestr(f'matrix1_{i}.bin', bm1_int.tobytes())
        
        except Exception as e:
            print("ERROR", e)
            raise e

        # zip_buffer = io.BytesIO()
        # with ProcessPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(process_result, img_ori_list[i], bm, img_param_list[i], i) 
        #         for i, bm in enumerate(batch_bm)
        #     ]
        #     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        #         for future in futures:
        #             index, img_bytes, bm0_bytes, bm1_bytes = future.result()
        #             # 将处理结果写入ZIP文件
        #             zip_file.writestr(f'image_{index}.png', img_bytes)
        #             zip_file.writestr(f'matrix0_{index}.bin', bm0_bytes)
        #             zip_file.writestr(f'matrix1_{index}.bin', bm1_bytes)
        
        zip_buffer.seek(0)
        print("sucess p3", time.time() - start_time)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='data.zip')


''' 共享内存 '''
class SharedMemoryManager:
    def __init__(self):
        self.shared_memory_references = {}  # key: 第一个引用的名称    value: (引用list，超时时间)
        self.lock = Lock()

    def record_shared_memory(self, shm_list, timeout=300):
        if len(shm_list) <= 0 :
            return
        memory_name = shm_list[0].name
        with self.lock:
            # 记录创建时间和超时时间
            self.shared_memory_references[memory_name] = (shm_list, time.time() + timeout)
        return memory_name
    
    def delete_shared_memory(self, memory_name):
        print("delete_shared_memory", memory_name)
        with self.lock:
            if memory_name in self.shared_memory_references:
                shm_list, _ = self.shared_memory_references[memory_name]
                for shm in shm_list:
                    shm.close()
                    shm.unlink()
                del self.shared_memory_references[memory_name]

    def cleanup_expired_references(self):
        while True:
            with self.lock:
                to_delete = []
                
                # 遍历所有的共享内存引用
                for name, (shm_list, expiry) in self.shared_memory_references.items():
                    current_time = time.time()
                    if current_time > expiry:
                        # 如果已超时，添加到删除列表
                        to_delete.append(name)
                
                # 删除超时的共享内存
                for name in to_delete:
                    shm_list, _ = self.shared_memory_references[name]
                    for shm in shm_list:
                        shm.close()
                        shm.unlink()
                    del self.shared_memory_references[name]
            
            # 定时执行，避免频繁检查，减轻系统负担
            time.sleep(600)  # 每10分钟检查一次
    
    def shutdown(self):
        with self.lock:
            # 清理所有共享内存引用，无论是否过期
            for name, (references, _) in self.shared_memory_references.items():
                for ref in references:
                    ref.close()
                    ref.unlink()
            self.shared_memory_references.clear()
            print("All shared memory references have been cleaned up.")

@app.route('/delete_shared_memory', methods=['POST'])
def handle_client_notification():
    try:
        # 从请求的 JSON 数据中提取 memory_name
        data = request.json
        memory_name = data.get('memory_name')
        
        if not memory_name:
            return jsonify({"error": "memory_name is required"}), 400
        
        # 删除共享内存
        with manager.lock:
            if memory_name in manager.shared_memory_references:
                shm_list, _ = manager.shared_memory_references[memory_name]
                for shm in shm_list:
                    shm.close()
                    shm.unlink()
                del manager.shared_memory_references[memory_name]
            else:
                return jsonify({"error": "memory_name not found"}), 404
        
        return jsonify({"status": "success", "message": f"Shared memory {memory_name} deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def cleanup_shared_names(shared_names):
    for shm_name in shared_names:
        try:
            print("clean", shm_name)
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except Exception as e:
            print(f"Error cleaning up shared memory {shm_name}: {e}")

def cleanup_shared_memory(shared_memories):
    for shm in shared_memories:
        try:
            print("clean", shm.name)
            shm.close()
            shm.unlink()
        except Exception as e:
            print(f"Error cleaning up shared memory {shm.name}: {e}")

def process_single_image(idx, image_name, image_size):
    print(f"image_{idx} name =", image_name)
    existing_shm = SharedMemory(name=image_name)
    img_bytes = bytes(existing_shm.buf[:image_size])

    # 将字节流转换为 NumPy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    
    # 预处理图片
    im_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR) / 255.0
    h_, w_, c_ = im_ori.shape
    img_param = (w_, h_)
    
    # 预处理和调整大小
    im_ori = cv2.resize(im_ori, (2560, 2560))
    im = cv2.resize(im_ori, (288, 288))
    
    # 转置并转换为 PyTorch 张量
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float()
    
    # 关闭和释放共享内存
    existing_shm.close()
    existing_shm.unlink()

    print("end", idx, image_name)

    return im, im_ori, img_param

def process_single_result(i, bm, img_ori, img_param, conn):
    try:
        # print("start ", i)
        bm = bm.cuda().numpy()
        bm0 = bm[0, :, :]
        bm1 = bm[1, :, :]

        # 对图像进行处理
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        img_geo = cv2.remap(img_ori, bm0, bm1, cv2.INTER_LINEAR) * 255
        img_geo = cv2.resize(img_geo, img_param)

        # 将处理后的图像编码为字节流(PNG格式)
        img_encoded = cv2.imencode('.png', img_geo)
        img_bytes = img_encoded[1].tobytes()

        # # 将字节流保存为文件
        # with open(f'output_image_{i}.png', 'wb') as f:
        #     f.write(img_bytes)

        # 创建共享内存并存储字节流
        geo_shm = SharedMemory(create=True, size=len(img_bytes))
        geo_shm.buf[:len(img_bytes)] = img_bytes

        # 数据转换
        x_indices = np.arange(2560)
        y_indices = np.arange(2560).reshape(-1, 1)
        transformed_bm0 = np.round((bm0 - x_indices) / 10)
        bm0_int = np.clip(transformed_bm0, -128, 127).astype(np.int8)
        transformed_bm1 = np.round((bm1 - y_indices) / 10)
        bm1_int = np.clip(transformed_bm1, -128, 127).astype(np.int8)

        # 保存矫正矩阵信息到共享内存
        bm0_shm = SharedMemory(create=True, size=bm0_int.nbytes)
        bm1_shm = SharedMemory(create=True, size=bm1_int.nbytes)

        bm0_array = np.ndarray(bm0_int.shape, dtype=bm0_int.dtype, buffer=bm0_shm.buf)
        bm1_array = np.ndarray(bm1_int.shape, dtype=bm1_int.dtype, buffer=bm1_shm.buf)

        np.copyto(bm0_array, bm0_int)
        np.copyto(bm1_array, bm1_int)

        # print("wait ", i)
        conn.send((geo_shm.name, len(img_bytes), bm0_shm.name, bm1_shm.name))
        # 等待主进程确认收到
        conn.recv()  # 等待主进程发送确认消息
        # print("end ", i)
        return

    except Exception as e:
        print(f"Error processing image {i}: {str(e)}")
        return None


@app.route('/predictShare', methods=['POST'])
def predictShare():
    # 0. 获取共享内存名称和大小
    image_names = request.json['image_names']
    image_sizes = request.json['image_sizes']
    
    # 1. 从共享内存中读取所有图片
    start_time = time.time()
    try:
        batch_images = []
        img_ori_list = []
        img_param_list = []

        # 并行处理图像
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_image, idx, img_name, img_size) 
                       for idx, (img_name, img_size) in enumerate(zip(image_names, image_sizes))]

        # 收集结果
        for future in futures:
            im_tensor, im_ori_resized, img_param = future.result()
            batch_images.append(im_tensor)
            img_ori_list.append(im_ori_resized)
            img_param_list.append(img_param)
                
        
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}, 500
    finally:
        cleanup_shared_names(image_names)
    print("sucess p1", time.time() - start_time)
    

    # 2. 批处理，矫正
    start_time = time.time()
    batch_tensor = torch.stack(batch_images)
    print(batch_tensor.size())
    try:
        with torch.no_grad():
            batch_bm = model(batch_tensor.cuda())
    except Exception as e:
        return {'error': str(e)}, 500
    print("sucess p2", time.time() - start_time)
    

    # 3. 处理结果并写入共享内存
    geo_names = []
    geo_sizes = []
    bm0_names = []
    bm1_names = []
    share_memories = []
    first_shm_name = ''
    start_time = time.time()
    try:
        with ProcessPoolExecutor() as executor:
            pipes = [Pipe() for _ in range(len(img_param_list))]
            futures = [
                executor.submit(process_single_result, i, bm, img_ori, img_param, pipes[i][1])
                for i, (bm, img_ori, img_param) in enumerate(zip(batch_bm, img_ori_list, img_param_list))
            ]

            # 获取共享内存引用并操作
            for parent_conn, _ in pipes:
                geo_name, geo_size, bm0_name, bm1_name = parent_conn.recv()  # 从子进程接收共享内存名称
                geo_names.append(geo_name)
                geo_sizes.append(geo_size)
                bm0_names.append(bm0_name)
                bm1_names.append(bm1_name)
                share_memories.append(SharedMemory(name=geo_name))
                share_memories.append(SharedMemory(name=bm0_name))
                share_memories.append(SharedMemory(name=bm1_name))
                parent_conn.send("done")  # 通知子进程可以结束了

            for future in futures:
                future.result()
        first_shm_name = manager.record_shared_memory(share_memories)
    except Exception as e:
        cleanup_shared_memory(share_memories)
        manager.delete_shared_memory(first_shm_name)
        print("ERROR IN predict p3" , e)
        return {'error': str(e)}, 500
    print("sucess p3", time.time() - start_time)
    
    return jsonify({'status': 'success', 'first_shm_name': first_shm_name, 
                    'geo_names': geo_names, 'geo_sizes': geo_sizes, 'bm0_names': bm0_names, 'bm1_names': bm1_names})


def cleanup_shared_memory():
    # 确保所有共享内存被清理
    manager.shutdown()

def signal_handler(sig, frame):
    print('Signal received, cleaning up shared memory...')
    cleanup_shared_memory()
    sys.exit(0)

if __name__ == '__main__':
    manager = SharedMemoryManager()
    cleanup_thread = Thread(target=manager.cleanup_expired_references, daemon=True)
    cleanup_thread.start()
    # 注册共享内存清理函数
    atexit.register(cleanup_shared_memory)
    # 捕获常见终止信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.run(host='0.0.0.0', debug=True, port=8080)

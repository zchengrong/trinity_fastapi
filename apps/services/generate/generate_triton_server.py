#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：trinity_client
@File    ：service.py
@Author  ：周成融
@Date    ：2023/7/26 12:01:05
@detail  ：
"""
import asyncio
import io
import json
import logging

import numpy as np
import random
import redis
import tritonclient
import tritonclient.grpc as grpc_client
from minio import Minio
import cv2
from PIL import Image
import time

from apps.apis.generate.model import GenerateImageModel
from apps.services.generate.remove_background import remove_background
from config.minio_env import *
from config.redis_env import *
from config.triton_env import *
from utils.generate_uuid import generate_uuid
from utils.runtime import RunTime


class GenerateImage:
    def __init__(self, request_data):
        self.tasks_id = request_data.tasks_id
        self.image_url = request_data.image_url
        self.user_id = request_data.user_id
        self.content = request_data.content
        self.category = request_data.category
        self.mode = request_data.mode
        self.version = request_data.version

        self.model_name = f"{self.category}_stable_diffusion"

        self.triton_client = grpc_client.InferenceServerClient(url=f"{GENERATE_TRITON_IP}:{GENERATE_TRITON_PORT_gPRC}")
        self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.minio_client = Minio(f"{MINIO_IP}:{MINIO_PORT}", access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)
        self.samples = 4  # no.of images to generate
        self.steps = 24
        self.guidance_scale = 7
        self.seed = random.randint(0, 2000000000)
        self.batch_size = 1
        self.generate_data = json.dumps({'status': 'PENDING', 'message': "pending", 'data': ''})
        self.redis_client.set(self.tasks_id, self.generate_data)
        self.triton_client.get_model_metadata(model_name=self.model_name, model_version=self.version)
        self.triton_client.get_model_config(model_name=self.model_name, model_version=self.version)
        self.image = self.get_image()

    def get_image(self):
        # Get data of an object.
        # Read data from response.
        try:
            response = self.minio_client.get_object(self.image_url.split('/')[0], self.image_url[self.image_url.find('/') + 1:])
            img = np.frombuffer(response.data, np.uint8)  # 转成8位无符号整型
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 解码
            img = self.preprocess_image(img, self.category)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.random.randn(512, 512, 3)
        return img

    def callback(self, result, error):
        if error:
            generate_data = json.dumps({'status': 'FAILURE', 'message': f"{error}", 'data': f"{error}"})
            self.redis_client.set(self.tasks_id, generate_data)
        else:
            images = result.as_numpy("IMAGES")
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]

            # for i in range(len(pil_images)):
            #     pil = pil_images[i]
            #     pil.save(f'./temp_i2_{i}.png')
            # self.image_grid(pil_images, rows, cols)
            url_list = []
            for i, image in enumerate(pil_images):

                if self.category == "sketch":
                    image = remove_background(np.asarray(image))
                image_url = self.upload_png(image, user_id=self.user_id, category=f"{self.category}", object_name=f"{generate_uuid()}_{i}.png", )
                url_list.append(image_url)
            generate_data = json.dumps({'status': 'SUCCESS', 'message': 'success', 'data': f'{url_list}'})

            self.redis_client.set(self.tasks_id, generate_data)

    def read_tasks_status(self):
        status_data = json.loads(self.redis_client.get(self.tasks_id))
        logging.info(f"{self.tasks_id} ===> {status_data}")
        return status_data

    def upload_png(self, image, user_id, category, object_name):
        try:
            image_data = io.BytesIO()
            image.save(image_data, format='PNG')
            image_data.seek(0)
            image_bytes = image_data.read()
            image_url = f"aida-users/{self.minio_client.put_object(f'aida-users', f'{user_id}/{category}/{object_name}', io.BytesIO(image_bytes), len(image_bytes), content_type='image/png').object_name}"

            return image_url
        except Exception as e:
            logging.warning(f"upload_png_mask runtime exception : {e}")

    @RunTime
    async def get_result(self):
        # Input placeholder
        prompt_in = tritonclient.grpc.InferInput(name="PROMPT", shape=(self.batch_size,), datatype="BYTES")
        samples_in = tritonclient.grpc.InferInput("SAMPLES", (self.batch_size,), "INT32")
        steps_in = tritonclient.grpc.InferInput("STEPS", (self.batch_size,), "INT32")
        guidance_scale_in = tritonclient.grpc.InferInput("GUIDANCE_SCALE", (self.batch_size,), "FP32")
        seed_in = tritonclient.grpc.InferInput("SEED", (self.batch_size,), "INT64")
        input_images_in = tritonclient.grpc.InferInput("INPUT_IMAGES", self.image.shape, "FP16")
        images = tritonclient.grpc.InferRequestedOutput(name="IMAGES",
                                                        # binary_data=False # grpc not binary_data
                                                        )
        mode_in = tritonclient.grpc.InferInput("MODE", (self.batch_size,), "INT32")

        # Setting inputs
        prompt_in.set_data_from_numpy(np.asarray([self.content] * self.batch_size, dtype=object))
        samples_in.set_data_from_numpy(np.asarray([self.samples], dtype=np.int32))
        steps_in.set_data_from_numpy(np.asarray([self.steps], dtype=np.int32))
        guidance_scale_in.set_data_from_numpy(np.asarray([self.guidance_scale], dtype=np.float32))
        seed_in.set_data_from_numpy(np.asarray([self.seed], dtype=np.int64))
        input_images_in.set_data_from_numpy(self.image.astype(np.float16))
        mode_in.set_data_from_numpy(np.asarray([self.mode], dtype=np.int32))

        ctx = self.triton_client.async_infer(
            model_name=self.model_name,
            model_version=self.version,
            inputs=[prompt_in, samples_in, steps_in, guidance_scale_in, seed_in, input_images_in, mode_in],
            outputs=[images],
            callback=self.callback
        )
        await asyncio.sleep(4)
        time_out = 120
        while self.read_tasks_status()['status'] == "PENDING" and time_out > 0:
            if self.read_tasks_status()['status'] == "REVOKED":
                ctx.cancel()
            time_out -= 1
            await asyncio.sleep(0.5)
        return self.read_tasks_status()


if __name__ == '__main__':
    class Dict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__  # dict.k  ==>  dict[k]
        # __getattr__ = dict.get  # dict.k  ==>  dict.get(k)
        # __getattr__ = lambda d, k: d.get(k, '')  # dict.k  ==>  dict.get(k,default)


    request_data = Dict({
        "user_id": 78,
        "image_url": "123_123.png",
        "category": "print",
        "mode": 1,
        "content": "a simple print",
        "version": "1",
        "tasks_id": "123456"
    })

    server = GenerateImage(request_data)
    print(server.get_result())

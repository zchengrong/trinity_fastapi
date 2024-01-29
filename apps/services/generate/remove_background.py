import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from config.triton_env import *
import tritonclient.http as httpclient


def remove_background(image):
    image_obj, mask = get_mask(image)
    seg_result = seg_infer_image(image_obj)

    temp_front = seg_result == 1
    front_mask = (mask * (temp_front + 0).astype(np.uint8))
    temp_back = seg_result == 2
    back_mask = (mask * (temp_back + 0).astype(np.uint8))

    if len(front_mask.shape) > 2:
        front_mask = front_mask[0]
    else:
        front_mask = front_mask

    if len(back_mask.shape) > 2:
        back_mask = back_mask[0]
    else:
        back_mask = back_mask

    result_mask = front_mask + back_mask
    white_background = np.ones_like(image_obj) * 255
    result_image = np.where(result_mask[:, :, None].astype(bool), image_obj, white_background)

    return Image.fromarray(result_image)


def get_mask(image_obj):
    pre_mask = None
    if len(image_obj.shape) == 2:
        image_obj = cv2.cvtColor(image_obj, cv2.COLOR_GRAY2RGB)
    if image_obj.shape[2] == 4:  # 如果是四通道 mask
        pre_mask = image_obj[:, :, 3]
        image_obj = image_obj[:, :, :3]

    Contour = get_contours(image_obj)
    Mask = np.zeros(image_obj.shape[:2], np.uint8)
    if len(Contour):
        Max_contour = Contour[0]
        Epsilon = 0.001 * cv2.arcLength(Max_contour, True)
        Approx = cv2.approxPolyDP(Max_contour, Epsilon, True)
        cv2.drawContours(Mask, [Approx], -1, 255, -1)
    else:
        Mask = np.ones(image_obj.shape[:2], np.uint8) * 255

    if pre_mask is None:
        mask = Mask
    else:
        mask = cv2.bitwise_and(Mask, pre_mask)
    return image_obj, mask


def get_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Edge = cv2.Canny(gray, 10, 150)
    kernel = np.ones((5, 5), np.uint8)
    Edge = cv2.dilate(Edge, kernel=kernel, iterations=1)
    Edge = cv2.erode(Edge, kernel=kernel, iterations=1)
    Contour, _ = cv2.findContours(Edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Contour = sorted(Contour, key=cv2.contourArea, reverse=True)
    return Contour


def seg_infer_image(image_obj):
    image, ori_shape = seg_preprocess(image_obj)
    client = httpclient.InferenceServerClient(url=f"{DESIGN_TRITON_IP}:{DESIGN_TRITON_PORT}")
    transformed_img = image.astype(np.float32)
    # 输入集
    inputs = [
        httpclient.InferInput(SEGMENTATION['input'], transformed_img.shape, datatype="FP32")
    ]
    inputs[0].set_data_from_numpy(transformed_img, binary_data=True)
    # 输出集
    outputs = [
        httpclient.InferRequestedOutput(SEGMENTATION['output'], binary_data=True),
    ]
    results = client.infer(model_name=SEGMENTATION['name'], inputs=inputs, outputs=outputs)
    # 推理
    # 取结果
    inference_output1 = torch.from_numpy(results.as_numpy(SEGMENTATION['output']))
    seg_result = seg_postprocess(inference_output1, ori_shape)
    return seg_result


def seg_preprocess(img_path):
    img = mmcv.imread(img_path)
    ori_shape = img.shape[:2]
    img_scale = (224, 224)
    scale_factor = []
    img, x, y = mmcv.imresize(img, img_scale, return_scale=True)
    scale_factor.append(x)
    scale_factor.append(y)
    img = mmcv.imnormalize(img, mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375]), to_rgb=True)
    preprocessed_img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    return preprocessed_img, ori_shape


def seg_postprocess(output, ori_shape):
    seg_logit = F.interpolate(output, size=ori_shape, scale_factor=None, mode='bilinear', align_corners=False)
    seg_logit = F.softmax(seg_logit, dim=1)
    seg_pred = seg_logit.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    return seg_pred

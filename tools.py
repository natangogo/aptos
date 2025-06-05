import matplotlib.pyplot as plt
import numpy as np
from setting import *
import cv2
import torch
import os

def plot_loss_curve(losses, title="Loss per Epoch"):
    """
    損失の推移を折れ線グラフで表示する関数。
    
    Parameters:
        losses (list of float): エポックごとの損失値のリスト
        title (str): グラフのタイトル
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def cal_metrics(out, gt):
    over = out + gt
    intersection = len(np.flatnonzero(over == 2))
    un_intersection = len(np.flatnonzero(over == 1))
    out_num = len(np.flatnonzero(out == 1))
    gt_num = len(np.flatnonzero(gt == 1))
    IoU = intersection / (intersection + un_intersection + 1e-9)
    DICE = (2 * intersection) / (out_num + gt_num + 1e-9)

    return IoU, DICE


def save_image(path, image, out, gt, IoU, name):
    case_name = name[0].split('.')[0]
    image_uint8 = np.array(image * 255, dtype=np.uint8)
    image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    out_binary_uint8 = np.array(out * 255, dtype=np.uint8)
    gt_binary_uint8 = np.array(gt * 255, dtype=np.uint8)

    contours_gt, _ = cv2.findContours(gt_binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_out, _ = cv2.findContours(out_binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours_gt)):
        image_uint8 = cv2.drawContours(image_uint8, contours_gt, i, (0, 0, 255), 1)
    for j in range(len(contours_out)):
        image_uint8 = cv2.drawContours(image_uint8, contours_out, j, (0, 255, 0), 1)
    
    image_name = case_name + '_' + str(IoU)+ '.png'
    out_name = case_name + '_' + 'OUT' + '.png'
    image_path = os.path.join(path, image_name)
    out_path = os.path.join(path, out_name)

    cv2.imwrite(image_path, image_uint8)
    cv2.imwrite(out_path, out_binary_uint8)

    return out_path
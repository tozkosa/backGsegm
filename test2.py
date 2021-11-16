import cv2
import math
import os
import glob
from IPython.display import display, Image
"""
path1 = "/home/nakagawa/ダウンロード/zero-DCE用データセット集/test_data/real"
files1 = glob.glob(path1 + '/*')
path2 = "/home/nakagawa/ダウンロード/zero-DCE用データセット集/aipho_test_groundtruth"
files2 = glob.glob(path2 + '/*')
files1.sort()
files2.sort()
for f1, f2 in zip(files1, files2):
"""
f1 = "180606-047.JPG"
f2 = "190307-122.JPG"

    # 画像読込
img1 = cv2.imread(f1)
filename = os.path.splitext(os.path.basename(f1))[0]
img2 = cv2.imread(f2)
print(f1)
print(img1.shape)
print(img2.shape)
# A-KAZE検出器の生成
detector = cv2.AKAZE_create()
# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)
# Brute-Force Matcherの生成
bf = cv2.BFMatcher()
# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)
# データを間引く
ratio = 0.2
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])
# 特徴量をマッチング状況に応じてソート
good = sorted(matches, key = lambda x : x[1].distance)
# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)
cv2.imwrite(filename + '.jpg', img3)
# 特徴量データを取得
q_kp = []
t_kp = []
for p in good[:2]:
    for px in p:
        q_kp.append(kp1[px.queryIdx])
        t_kp.append(kp2[px.trainIdx])
# 加工対象の画像から特徴点間の角度と距離を計算
q_x1, q_y1 = q_kp[0].pt
q_x2, q_y2 = q_kp[-1].pt
q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)
# テンプレート画像から特徴点間の角度と距離を計算
t_x1, t_y1 = t_kp[0].pt
t_x2, t_y2 = t_kp[-1].pt
t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)
# 切出し位置の計算
x1 = q_x1 - t_x1 * (q_len / t_len)
x2 = x1 + img2.shape[1] * (q_len / t_len)
y1 = q_y1 - t_y1 * (q_len / t_len)
y2 = y1 + img2.shape[0] * (q_len / t_len)
# 画像サイズ
x, y, c = img1.shape
size = (x+300, y+300)
# 回転の中心位置
center = (q_x1, q_y1)
# 回転角度
angle = q_deg - t_deg
# サイズ比率
scale = 1.0
# 回転変換行列の算出
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
# アフィン変換
img_rot = cv2.warpAffine(img1, rotation_matrix, size, flags=cv2.INTER_CUBIC)
# 画像の切出し
img_rot = img_rot[int(y1):int(y2), int(x1):int(x2)]
print(img_rot.shape)

# 縮尺調整
x, y, c = img2.shape
img_rot = cv2.resize(img_rot, (600, 600))

cv2.imwrite(filename + '.jpg', img_rot)
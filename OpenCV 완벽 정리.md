# OpenCV 완벽 정리

## 목차
1. [OpenCV 개요](#opencv-개요)
2. [설치 및 설정](#설치-및-설정)
3. [기본 이미지 처리](#기본-이미지-처리)
4. [이미지 변환](#이미지-변환)
5. [필터링과 블러링](#필터링과-블러링)
6. [엣지 검출](#엣지-검출)
7. [컨투어 및 형태 검출](#컨투어-및-형태-검출)
8. [색상 공간 변환](#색상-공간-변환)
9. [특징점 검출](#특징점-검출)
10. [비디오 처리](#비디오-처리)
11. [객체 추적](#객체-추적)
12. [머신러닝 및 딥러닝](#머신러닝-및-딥러닝)

---

## OpenCV 개요

**OpenCV**(Open Source Computer Vision Library)는 컴퓨터 비전 및 머신러닝을 위한 오픈소스 라이브러리입니다.

### 주요 특징
- C++, Python, Java 등 다양한 언어 지원
- 2500개 이상의 최적화된 알고리즘 제공
- 실시간 컴퓨터 비전 애플리케이션 개발 가능
- 크로스 플랫폼 지원 (Windows, Linux, macOS, Android, iOS)

### 주요 용도
- 얼굴 인식 및 검출
- 객체 인식 및 추적
- 이미지 및 비디오 처리
- 증강 현실
- 로봇 비전
- 의료 영상 분석

---

## 설치 및 설정

### Python 설치

```bash
# 기본 설치
pip install opencv-python

# 전체 패키지 설치 (contrib 모듈 포함)
pip install opencv-contrib-python
```

### 기본 임포트

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### 버전 확인

```python
print(cv2.__version__)
```

---

## 기본 이미지 처리

### 이미지 읽기, 표시, 저장

```python
# 이미지 읽기
img = cv2.imread('image.jpg')  # BGR 형식으로 읽음
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 읽기

# 이미지 표시
cv2.imshow('Image', img)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()

# 이미지 저장
cv2.imwrite('output.jpg', img)
```

### 이미지 속성 확인

```python
# 이미지 크기 (높이, 너비, 채널)
print(img.shape)

# 픽셀 수
print(img.size)

# 데이터 타입
print(img.dtype)
```

### 이미지 크기 조정

```python
# 특정 크기로 조정
resized = cv2.resize(img, (width, height))

# 비율로 조정
resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# 보간법 지정
resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
# INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA
```

### 이미지 자르기

```python
# [y:y+h, x:x+w] 형식
cropped = img[100:300, 200:400]
```

### 이미지 회전

```python
# 중심점 기준 회전
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

---

## 이미지 변환

### 색상 공간 변환

```python
# BGR을 그레이스케일로
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR을 RGB로
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR을 HSV로
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

### 이진화 (Thresholding)

```python
# 단순 이진화
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 적응형 이진화
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Otsu의 이진화
ret, otsu = cv2.threshold(gray, 0, 255, 
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 기하학적 변환

```python
# 평행 이동
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])  # x방향 100, y방향 50 이동
translated = cv2.warpAffine(img, M, (cols, rows))

# 어핀 변환
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (cols, rows))

# 원근 변환
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (300, 300))
```

---

## 필터링과 블러링

### 블러링 기법

```python
# 평균 블러
blur = cv2.blur(img, (5, 5))

# 가우시안 블러
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 미디언 블러 (소금-후추 노이즈 제거에 효과적)
median = cv2.medianBlur(img, 5)

# 양방향 필터 (엣지 보존하면서 블러)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### 모폴로지 연산

```python
# 커널 생성
kernel = np.ones((5, 5), np.uint8)

# 침식 (Erosion)
erosion = cv2.erode(img, kernel, iterations=1)

# 팽창 (Dilation)
dilation = cv2.dilate(img, kernel, iterations=1)

# 열림 (Opening) - 침식 후 팽창
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 닫힘 (Closing) - 팽창 후 침식
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 그래디언트
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

---

## 엣지 검출

### Canny 엣지 검출

```python
edges = cv2.Canny(img, threshold1=100, threshold2=200)
```

### Sobel 엣지 검출

```python
# X 방향 엣지
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# Y 방향 엣지
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# 결합
sobel = cv2.magnitude(sobelx, sobely)
```

### Laplacian 엣지 검출

```python
laplacian = cv2.Laplacian(img, cv2.CV_64F)
```

---

## 컨투어 및 형태 검출

### 컨투어 찾기

```python
# 이진 이미지 생성
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 컨투어 찾기
contours, hierarchy = cv2.findContours(thresh, 
                                       cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 그리기
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 모든 컨투어
cv2.drawContours(img, contours, 0, (0, 255, 0), 3)   # 첫 번째 컨투어만
```

### 컨투어 특징

```python
# 첫 번째 컨투어
cnt = contours[0]

# 면적
area = cv2.contourArea(cnt)

# 둘레
perimeter = cv2.arcLength(cnt, True)

# 근사 컨투어
epsilon = 0.01 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)

# 볼록 껍질
hull = cv2.convexHull(cnt)

# 바운딩 박스
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 최소 외접 원
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, center, radius, (0, 255, 0), 2)
```

### 도형 검출

```python
# 직선 검출 (Hough Transform)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                        minLineLength=100, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 원 검출 (Hough Circle Transform)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                          param1=50, param2=30, 
                          minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
```

---

## 색상 공간 변환

### HSV 색상 범위 추출

```python
# BGR을 HSV로 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 파란색 범위 정의
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# 마스크 생성
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 마스크 적용
result = cv2.bitwise_and(img, img, mask=mask)
```

### 히스토그램

```python
# 그레이스케일 히스토그램
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 컬러 히스토그램
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

# 히스토그램 평활화
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)
```

---

## 특징점 검출

### 코너 검출

```python
# Harris Corner Detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
```

### SIFT (Scale-Invariant Feature Transform)

```python
# SIFT 생성
sift = cv2.SIFT_create()

# 키포인트와 디스크립터 검출
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 키포인트 그리기
img_kp = cv2.drawKeypoints(img, keypoints, None, 
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

### ORB (Oriented FAST and Rotated BRIEF)

```python
# ORB 생성
orb = cv2.ORB_create()

# 키포인트와 디스크립터 검출
keypoints, descriptors = orb.detectAndCompute(gray, None)

# 키포인트 그리기
img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
```

### 특징점 매칭

```python
# 두 이미지의 특징점 검출
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFMatcher (Brute-Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과 그리기
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], 
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

---

## 비디오 처리

### 비디오 읽기

```python
# 비디오 파일 열기
cap = cv2.VideoCapture('video.mp4')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 프레임 단위 처리
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 프레임 표시
    cv2.imshow('Frame', gray)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 비디오 저장

```python
cap = cv2.VideoCapture(0)

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# VideoWriter 객체 생성
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        # 프레임 처리
        out.write(frame)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

### 비디오 속성

```python
# 프레임 수
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# 프레임 크기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 특정 프레임으로 이동
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
```

---

## 객체 추적

### 광학 흐름 (Optical Flow)

```python
# Lucas-Kanade 방법
# 추적할 특징점 검출
feature_params = dict(maxCorners=100, qualityLevel=0.3, 
                     minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Lucas-Kanade 파라미터
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 광학 흐름 계산
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

# Dense Optical Flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 
                                    0.5, 3, 15, 3, 5, 1.2, 0)
```

### 객체 추적기

```python
# 다양한 추적 알고리즘
tracker = cv2.TrackerKCF_create()  # KCF
# tracker = cv2.TrackerCSRT_create()  # CSRT
# tracker = cv2.TrackerMOSSE_create()  # MOSSE

# 첫 프레임에서 ROI 선택
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# 추적
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    success, bbox = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

### 배경 차분 (Background Subtraction)

```python
# MOG2 배경 차분
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 전경 마스크 생성
    fg_mask = bg_subtractor.apply(frame)
    
    cv2.imshow('Foreground', fg_mask)
```

---

## 머신러닝 및 딥러닝

### 얼굴 검출 (Haar Cascade)

```python
# Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# 얼굴 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    # 눈 검출
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```

### DNN 모듈 (딥러닝)

```python
# 사전 학습된 모델 로드
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# 이미지 전처리
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

# 순전파
net.setInput(blob)
detections = net.forward()

# 결과 처리
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
```

### YOLO 객체 검출

```python
# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 클래스 이름 로드
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 전처리
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 결과 처리
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            # 바운딩 박스 좌표
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 2)
```

---

## 유용한 팁과 Best Practices

### 성능 최적화

```python
# NumPy 연산 활용
# OpenCV는 내부적으로 NumPy 배열을 사용하므로 NumPy 연산이 빠름

# 이미지 타입 변환
img = img.astype(np.float32)

# ROI(Region of Interest) 활용
roi = img[y1:y2, x1:x2]

# 불필요한 복사 피하기
result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 새 배열 생성
```

### 이미지 표시 (Matplotlib 활용)

```python
import matplotlib.pyplot as plt

# BGR을 RGB로 변환 후 표시
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.axis('off')
plt.show()
```

### 텍스트 및 도형 그리기

```python
# 텍스트 그리기
cv2.putText(img, 'OpenCV', (50, 50), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 선 그리기
cv2.line(img, (0, 0), (100, 100), (0, 255, 0), 2)

# 사각형 그리기
cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 2)

# 원 그리기
cv2.circle(img, (100, 100), 50, (255, 255, 0), -1)  # -1은 채우기

# 타원 그리기
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1)

# 다각형 그리기
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255), 3)
```

### 마우스 이벤트 처리

```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click at ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right click at ({x}, {y})")

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
```

---

## 자주 사용하는 코드 스니펫

### 웹캠에서 실시간 얼굴 검출

```python
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 이미지에서 특정 색상 추출

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 빨간색 범위 (HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

result = cv2.bitwise_and(img, img, mask=mask)
```

### 이미지 비교 (템플릿 매칭)

```python
# 템플릿 매칭
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 매칭 위치 표시
top_left = max_loc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
```

---

## 참고 자료

### 공식 문서
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### 유용한 링크
- [OpenCV GitHub](https://github.com/opencv/opencv)
- [OpenCV 포럼](https://forum.opencv.org/)

### 추가 학습 주제
- 카메라 캘리브레이션
- 3D 재구성
- 스테레오 비전
- 파노라마 스티칭
- HDR 이미징
- 이미지 분할 (Image Segmentation)
- 객체 인식 (Object Recognition)

---

**작성 날짜**: 2024년  
**OpenCV 버전**: 4.x 기준

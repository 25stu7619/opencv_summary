# Image Classification vs Regression 완벽 비교

## 📋 목차
1. [핵심 차이점 요약](#핵심-차이점-요약)
2. [Image Classification 상세](#image-classification-상세)
3. [Regression 상세](#regression-상세)
4. [비교표](#비교표)
5. [실전 예제 코드](#실전-예제-코드)
6. [언제 무엇을 사용할까](#언제-무엇을-사용할까)

---

## 핵심 차이점 요약

### 🎯 한 문장 정리

```
Classification: "이것은 무엇인가?" (카테고리 분류)
Regression:     "이것은 얼마인가?" (수치 예측)
```

### 📊 시각적 비교

```
┌─────────────────────────────────────────────────────────┐
│              Image Classification                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  입력: 사진 🖼️                                          │
│         ↓                                               │
│  출력: [개, 고양이, 새] 중 하나                          │
│                                                         │
│  예시:                                                  │
│  🐕 사진 → "개" (카테고리)                              │
│  🐱 사진 → "고양이" (카테고리)                          │
│  🐦 사진 → "새" (카테고리)                              │
│                                                         │
│  출력 형태: 이산적 (Discrete)                           │
│  └─ 정해진 카테고리 중 하나를 선택                       │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Regression                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  입력: 사진 🖼️                                          │
│         ↓                                               │
│  출력: 연속적인 숫자 값                                  │
│                                                         │
│  예시:                                                  │
│  🏠 집 사진 → "3억 5천만원" (가격)                      │
│  👤 얼굴 사진 → "28.5세" (나이)                         │
│  📏 물체 사진 → "x=120.5, y=85.3" (위치)                │
│                                                         │
│  출력 형태: 연속적 (Continuous)                         │
│  └─ 무한히 많은 가능한 값                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Image Classification 상세

### 🔍 정의

**Image Classification**은 입력 이미지를 **미리 정의된 카테고리(클래스)** 중 하나로 분류하는 작업입니다.

### 📝 특징

```
1️⃣ 출력: 카테고리 (클래스 레이블)
   - 개, 고양이, 개 중 하나
   - 0, 1, 2 같은 정수
   - One-hot encoding: [1, 0, 0]

2️⃣ 출력 개수: 고정됨
   - 3개 클래스 → 3개 출력 뉴런
   - 1000개 클래스 → 1000개 출력 뉴런

3️⃣ 활성화 함수: Softmax
   - 확률 분포로 변환
   - 합이 1이 되도록

4️⃣ 손실 함수: Cross-Entropy
   - 분류 성능 측정
```

### 🏗️ 모델 구조

```python
# Classification 모델 예시

입력 이미지 (224, 224, 3)
    ↓
┌─────────────────────┐
│   CNN Backbone      │
│   (특징 추출)        │
│   - Conv layers     │
│   - Pooling         │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Flatten           │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Dense(512)        │
│   ReLU              │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Dense(3)          │  ← 클래스 개수만큼
│   Softmax           │  ← 확률로 변환
└─────────────────────┘
    ↓
출력: [0.8, 0.15, 0.05]
      개   고양이  새
      
→ 예측: "개" (가장 높은 확률)
```

### 💻 코드 예시

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Classification 모델
def build_classification_model(num_classes=3):
    model = models.Sequential([
        # CNN 백본
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 분류기
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        
        # 출력층
        layers.Dense(num_classes, activation='softmax')  # ← Softmax!
    ])
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # ← Cross-Entropy!
        metrics=['accuracy']
    )
    
    return model

# 사용 예시
model = build_classification_model(num_classes=3)

# 예측
predictions = model.predict(image)
# 출력: [[0.8, 0.15, 0.05]]
#        개   고양이  새

predicted_class = np.argmax(predictions[0])
# 출력: 0 (개)
```

### 📊 출력 해석

```python
# Classification 출력 예시
predictions = [0.8, 0.15, 0.05]

해석:
- 개일 확률: 80%
- 고양이일 확률: 15%
- 새일 확률: 5%
- 합계: 100%

최종 예측: "개" (가장 높은 확률)
```

### 🎯 실제 응용 사례

```
1. 동물 분류
   입력: 동물 사진
   출력: [개, 고양이, 새, 말, 소]

2. 의료 영상 진단
   입력: X-ray 이미지
   출력: [정상, 폐렴, 결핵, 암]

3. 감정 인식
   입력: 얼굴 사진
   출력: [행복, 슬픔, 화남, 놀람, 중립]

4. 제품 분류
   입력: 제품 사진
   출력: [의류, 전자제품, 식품, 가구, 책]

5. 필기 인식
   입력: 손글씨 이미지
   출력: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## Regression 상세

### 🔍 정의

**Regression**은 입력 이미지로부터 **연속적인 수치 값**을 예측하는 작업입니다.

### 📝 특징

```
1️⃣ 출력: 연속적인 실수 값
   - 나이: 28.5세
   - 가격: 3,500,000원
   - 좌표: (x=120.5, y=85.3)

2️⃣ 출력 개수: 예측할 값의 개수
   - 1개 값 → 1개 출력 뉴런
   - 10개 좌표 → 10개 출력 뉴런

3️⃣ 활성화 함수: Linear (또는 없음)
   - 실수 범위 전체 사용
   - 제한 없음

4️⃣ 손실 함수: MSE, MAE
   - 예측값과 실제값의 차이
```

### 🏗️ 모델 구조

```python
# Regression 모델 예시

입력 이미지 (224, 224, 3)
    ↓
┌─────────────────────┐
│   CNN Backbone      │
│   (특징 추출)        │
│   - Conv layers     │
│   - Pooling         │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Flatten           │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Dense(512)        │
│   ReLU              │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Dense(1)          │  ← 예측할 값의 개수
│   Linear            │  ← 활성화 함수 없음
└─────────────────────┘
    ↓
출력: [28.5]
      나이(세)
      
→ 예측: 28.5세
```

### 💻 코드 예시

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Regression 모델
def build_regression_model(num_outputs=1):
    model = models.Sequential([
        # CNN 백본
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 회귀기
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        
        # 출력층
        layers.Dense(num_outputs, activation='linear')  # ← Linear!
        # 또는 activation=None
    ])
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='mse',  # ← MSE (Mean Squared Error)!
        metrics=['mae']  # MAE (Mean Absolute Error)
    )
    
    return model

# 사용 예시
model = build_regression_model(num_outputs=1)

# 예측
predictions = model.predict(image)
# 출력: [[28.5]]
#        나이

predicted_age = predictions[0][0]
# 출력: 28.5
```

### 📊 출력 해석

```python
# Regression 출력 예시

# 1) 단일 값 예측 (나이)
prediction = [28.5]
해석: 예측 나이 = 28.5세

# 2) 다중 값 예측 (얼굴 랜드마크)
prediction = [120.5, 85.3, 200.1, 90.5, 160.7, 120.0]
             [x1,    y1,   x2,    y2,   x3,    y3]
해석: 
  - 좌측 눈 위치: (120.5, 85.3)
  - 우측 눈 위치: (200.1, 90.5)
  - 코 위치: (160.7, 120.0)

# 3) 집 가격 예측
prediction = [350000000]
해석: 예측 가격 = 3억 5천만원
```

### 🎯 실제 응용 사례

```
1. 나이 예측
   입력: 얼굴 사진
   출력: 28.5세

2. 집 가격 예측
   입력: 집 사진
   출력: 3억 5천만원

3. 얼굴 랜드마크 검출
   입력: 얼굴 사진
   출력: (x1, y1, x2, y2, ...) 좌표들

4. 깊이 추정
   입력: 일반 사진
   출력: 각 픽셀의 깊이 값

5. 자세 추정
   입력: 사람 사진
   출력: 관절 위치 좌표들

6. 객체 크기 측정
   입력: 물체 사진
   출력: 길이(cm), 넓이(cm²)
```

---

## 비교표

### 📋 상세 비교

| 항목 | Classification | Regression |
|------|----------------|------------|
| **출력 타입** | 카테고리 (이산적) | 숫자 (연속적) |
| **출력 예시** | "개", "고양이", "새" | 28.5, 120.3, [x, y] |
| **출력 뉴런 수** | 클래스 개수 (예: 3개) | 예측할 값 개수 (예: 1개) |
| **활성화 함수** | **Softmax** | **Linear** (또는 없음) |
| **손실 함수** | **Cross-Entropy** | **MSE, MAE** |
| **평가 지표** | Accuracy, F1-Score | MSE, MAE, R² |
| **출력 범위** | [0, 1] (확률) | (-∞, +∞) (실수) |
| **출력 합** | 1.0 (100%) | 제약 없음 |
| **예측 방법** | argmax (최대 확률) | 직접 사용 |

### 📊 수식 비교

#### Classification

```
출력층:
y = Softmax(Wx + b)

Softmax:
y_i = exp(x_i) / Σ exp(x_j)

손실 함수 (Cross-Entropy):
Loss = -Σ y_true * log(y_pred)

예시:
실제: [1, 0, 0] (개)
예측: [0.8, 0.15, 0.05]
Loss = -(1*log(0.8) + 0*log(0.15) + 0*log(0.05))
     = -log(0.8)
     ≈ 0.223
```

#### Regression

```
출력층:
y = Wx + b  (Linear)

손실 함수 (MSE):
Loss = (1/n) * Σ(y_true - y_pred)²

예시:
실제: 30.0세
예측: 28.5세
Loss = (30.0 - 28.5)²
     = 2.25
```

---

## 실전 예제 코드

### 🔴 Classification 전체 예제

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# 1. 데이터 준비 (CIFAR-10 예시)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 정규화
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("학습 데이터:", X_train.shape)  # (50000, 32, 32, 3)
print("학습 레이블:", y_train.shape)  # (50000, 10)

# 2. Classification 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # 출력층: 10개 클래스
    layers.Dense(10, activation='softmax')  # ← Softmax
])

# 3. 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # ← Cross-Entropy
    metrics=['accuracy']
)

# 4. 학습
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# 5. 예측
sample_image = X_test[0:1]
prediction = model.predict(sample_image)

print("\n예측 결과:")
print("확률:", prediction[0])
print("예측 클래스:", np.argmax(prediction[0]))
print("실제 클래스:", np.argmax(y_test[0]))

"""
출력 예시:
확률: [0.05, 0.02, 0.01, 0.8, 0.03, 0.01, 0.02, 0.01, 0.03, 0.02]
예측 클래스: 3
실제 클래스: 3
"""
```

### 🔵 Regression 전체 예제

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 데이터 준비 (나이 예측 예시)
# 가상 데이터 생성
np.random.seed(42)
n_samples = 1000

# 이미지 데이터 (64x64 RGB)
X_train = np.random.rand(n_samples, 64, 64, 3).astype('float32')

# 나이 레이블 (연속값)
y_train = np.random.uniform(18, 80, size=(n_samples, 1)).astype('float32')

print("학습 데이터:", X_train.shape)  # (1000, 64, 64, 3)
print("학습 레이블:", y_train.shape)  # (1000, 1)
print("레이블 샘플:", y_train[:5])    # [[32.5], [45.2], [28.1], ...]

# 2. Regression 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # 출력층: 1개 값 (나이)
    layers.Dense(1, activation='linear')  # ← Linear
])

# 3. 컴파일
model.compile(
    optimizer='adam',
    loss='mse',     # ← MSE
    metrics=['mae']  # MAE
)

# 4. 학습
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 5. 예측
sample_image = X_train[0:1]
prediction = model.predict(sample_image)

print("\n예측 결과:")
print("예측 나이:", prediction[0][0], "세")
print("실제 나이:", y_train[0][0], "세")
print("오차:", abs(prediction[0][0] - y_train[0][0]), "세")

"""
출력 예시:
예측 나이: 32.8 세
실제 나이: 32.5 세
오차: 0.3 세
"""
```

### 🟣 얼굴 랜드마크 검출 (Regression 응용)

```python
# 얼굴 랜드마크: 여러 개의 연속 값 예측

# 1. 데이터 준비
n_samples = 1000
X_train = np.random.rand(n_samples, 128, 128, 3).astype('float32')

# 5개 랜드마크의 (x, y) 좌표 = 10개 값
y_train = np.random.rand(n_samples, 10).astype('float32')
# [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]

print("학습 레이블:", y_train.shape)  # (1000, 10)
print("레이블 샘플:", y_train[0])     # [0.2, 0.3, 0.8, 0.3, ...]

# 2. 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    
    # 출력층: 10개 값 (5개 점의 x, y 좌표)
    layers.Dense(10, activation='linear')  # ← 10개 출력
])

# 3. 컴파일
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 4. 예측
prediction = model.predict(X_train[0:1])
print("\n예측 랜드마크:", prediction[0])
# [0.18, 0.32, 0.82, 0.28, 0.51, 0.49, ...]

# 좌표 쌍으로 변환
landmarks = prediction[0].reshape(-1, 2)
print("좌표 형식:")
for i, (x, y) in enumerate(landmarks):
    print(f"  점 {i+1}: ({x:.2f}, {y:.2f})")
"""
출력:
  점 1: (0.18, 0.32)
  점 2: (0.82, 0.28)
  점 3: (0.51, 0.49)
  점 4: (0.25, 0.75)
  점 5: (0.78, 0.73)
"""
```

---

## 언제 무엇을 사용할까

### 🤔 의사결정 플로우

```
질문: "예측하려는 것이 무엇인가?"
    ↓
┌───────────────────────────────┐
│ "카테고리인가, 숫자인가?"       │
└───────────────────────────────┘
    ↓                    ↓
카테고리              연속 숫자
    ↓                    ↓
Classification      Regression
```

### ✅ Classification을 사용하는 경우

```
1. 출력이 정해진 카테고리 중 하나
   ✓ 동물 종류 (개, 고양이, 새)
   ✓ 질병 유무 (정상, 질병)
   ✓ 감정 (행복, 슬픔, 화남)

2. "이것은 무엇인가?" 질문에 답할 때
   ✓ 이 사진은 개인가, 고양이인가?
   ✓ 이 환자는 건강한가, 아픈가?
   ✓ 이 얼굴은 웃는가, 우는가?

3. 확률이 필요할 때
   ✓ 개일 확률: 80%
   ✓ 고양이일 확률: 15%
   ✓ 새일 확률: 5%
```

### ✅ Regression을 사용하는 경우

```
1. 출력이 연속적인 숫자
   ✓ 나이 (28.5세)
   ✓ 가격 (3,500,000원)
   ✓ 좌표 (x=120.5, y=85.3)

2. "이것은 얼마인가?" 질문에 답할 때
   ✓ 이 사람은 몇 살인가?
   ✓ 이 집은 얼마인가?
   ✓ 이 물체는 어디에 있는가?

3. 정밀한 수치가 필요할 때
   ✓ 정확한 위치
   ✓ 정확한 측정값
   ✓ 정확한 예측값
```

### 🔄 혼합 사용 예시

```python
# 예시: 얼굴 분석 시스템

# 1. Classification: 성별 분류
gender_model = build_classification_model(num_classes=2)
gender = gender_model.predict(face_image)
# 출력: [0.9, 0.1] → 남성

# 2. Regression: 나이 예측
age_model = build_regression_model(num_outputs=1)
age = age_model.predict(face_image)
# 출력: [28.5] → 28.5세

# 3. Regression: 얼굴 랜드마크
landmark_model = build_regression_model(num_outputs=10)
landmarks = landmark_model.predict(face_image)
# 출력: [120, 85, 200, 90, ...] → 좌표들
```

---

## 핵심 요약

### 🎯 기억해야 할 5가지

```
1️⃣ Classification = 카테고리 분류
   출력: [개, 고양이, 새] 중 하나
   활성화: Softmax
   손실: Cross-Entropy

2️⃣ Regression = 숫자 예측
   출력: 연속적인 실수 값
   활성화: Linear
   손실: MSE/MAE

3️⃣ Classification 출력은 확률
   합이 1.0 (100%)
   [0.8, 0.15, 0.05]

4️⃣ Regression 출력은 실수
   제약 없음
   28.5, 120.3, -15.2

5️⃣ 문제에 따라 선택
   "무엇?" → Classification
   "얼마?" → Regression
```

### 📊 한눈에 보는 비교

```
┌────────────────────┬─────────────────┬─────────────────┐
│      항목          │ Classification  │   Regression    │
├────────────────────┼─────────────────┼─────────────────┤
│ 질문              │ "무엇인가?"      │ "얼마인가?"      │
│ 출력              │ 카테고리         │ 숫자             │
│ 활성화 함수       │ Softmax          │ Linear           │
│ 손실 함수         │ Cross-Entropy    │ MSE/MAE          │
│ 출력 예시         │ [0.8, 0.15, 0.05]│ 28.5             │
└────────────────────┴─────────────────┴─────────────────┘
```

이제 Classification과 Regression의 차이를 완벽히 이해하셨습니다! 🎉

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


# Load model đã huấn luyện
from tensorflow.keras.models import load_model
# Tải mô hình
import tensorflow as tf

model = tf.keras.models.load_model("D:/ky 1 nam 3/ML/module/trashclassificationmodel.keras")


# Thông số ảnh
IMG_SIZE = 150
# Tạo val_generator nếu cần
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'trash-Splitted/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Lấy nhãn lớp
classes = list(val_generator.class_indices.keys())
print(f"Classes: {classes}")

# Bắt đầu camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

        
    # Resize và tiền xử lý frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Dự đoán
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    label = f"{classes[class_id]}: {predictions[0][class_id]*100:.2f}%"

    # Hiển thị kết quả
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Trash Classification', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

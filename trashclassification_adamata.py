import requests
import zipfile
import io
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout

output_dir = "./data"
url = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

# Mendownload file ZIP (mungkin bisa dihindari jika sudah tersedia secara lokal)
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(output_dir)

data_dir = './data/dataset-resized'
garbage_types = os.listdir(data_dir)

# Menggunakan pandas untuk memanipulasi file path gambar
data = []
for garbage_type in garbage_types:
    garbage_type_path = os.path.join(data_dir, garbage_type)
    if os.path.isdir(garbage_type_path):
        for file in os.listdir(garbage_type_path):
            data.append((os.path.join(garbage_type_path, file), garbage_type))

df = pd.DataFrame(data, columns=['filepath', 'label'])

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Augmentasi dan generator data
train_datagen = ImageDataGenerator(
    rotation_range=60,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.05,
    brightness_range=[0.9, 1.1],
    channel_shift_range=10,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(384, 384),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="label",
    target_size=(384, 384),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_labels = train_df['label'].unique()
class_labels

train_generator.class_indices

weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_df['label'])

class_weights = dict(zip(train_generator.class_indices.values(), weights))


# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(384, 384, 3))

for layer in base_model.layers[:143]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(filepath="best_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

callbacks = [reduce_lr, early_stopping, model_checkpoint]

# Model training
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

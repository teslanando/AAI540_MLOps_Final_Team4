import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model(test_path, model_path):
    model = tf.keras.models.load_model(model_path)
    test_df = pd.read_csv(test_path)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col='resized_path',
        y_col='label',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='binary',
        batch_size=32,
        shuffle=False
    )

    predictions = model.predict(test_generator, steps=np.ceil(test_generator.samples/test_generator.batch_size))
    predicted_classes = np.where(predictions > 0.5, 1, 0)
    true_classes = test_generator.classes

    metrics = classification_report(true_classes, predicted_classes, output_dict=True)
    print(metrics)

    # Save the report
    with open('/opt/ml/processing/evaluation/evaluation.json', 'w') as f:
        f.write(json.dumps(metrics))

evaluate_model('/opt/ml/processing/test/test.csv', '/opt/ml/model/model.h5')

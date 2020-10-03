import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from arguments import make_argparser
from output import print_predictions
from preprocessing import process_image


def load_model(location):
    return tf.keras.models.load_model(
        location, custom_objects={"KerasLayer": hub.KerasLayer}
    )


def predict(processed_img, model, top_k):
    image = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(image)
    return tf.math.top_k(prediction, k=top_k, sorted=True)


if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    pil_img = Image.open(args.input)
    numpy_img = np.asarray(pil_img)
    processed_test_image = process_image(numpy_img)

    loaded_model = load_model(args.model)

    probs, classes = predict(processed_test_image, loaded_model, args.top_k)

    print_predictions(probs, classes)

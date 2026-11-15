"""
EfficientNetV2B0 256x256 CLAHE classifier: preprocessing + Grad-CAM.

Extracted from the training/XAI notebook session (make_clahe, load256, gradcam).
Model file: EfficientNetV2B0_clahe_256.keras
  input  (None, 256, 256, 3) float32, RAW 0-255  (rescaling + ImageNet normalization are INSIDE the model)
  output (None, 4) softmax
"""
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # index == label_id used in training
INPUT_SIZE = (256, 256)
GRADCAM_LAYER = "top_activation"  # 8x8x1280 at 256px input


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Uploaded file bytes -> BGR uint8 (same channel order as cv2.imread used in training)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def clahe_gray3(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Grayscale -> CLAHE -> replicate to 3 channels (BGR). Exactly what make_clahe() did in training."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def preprocess(img_bgr: np.ndarray, simulate_training_jpeg: bool = False) -> np.ndarray:
    """
    BGR uint8 image -> float32 (256, 256, 3) RGB in 0-255, ready for model(x[None]).

    Order matters: CLAHE at the ORIGINAL resolution first, then resize (as in training).
    Do NOT divide by 255 and do NOT call preprocess_input: the model does that itself.

    simulate_training_jpeg: training wrote the CLAHE image back to a .jpg before loading it.
    Leave False in production; True is only for the parity script.
    """
    x = clahe_gray3(img_bgr)
    if simulate_training_jpeg:
        ok, buf = cv2.imencode(".jpg", x)  # OpenCV default quality 95, same as cv2.imwrite
        x = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("float32")
    return tf.image.resize(rgb, INPUT_SIZE).numpy().astype(np.float32)  # bilinear, no antialias (as training)


def predict(model, x: np.ndarray) -> np.ndarray:
    """x: (256,256,3) from preprocess(). Returns probabilities (4,)."""
    return model(x[None], training=False).numpy()[0]


def feature_map_shape(model, layer_name: str = GRADCAM_LAYER):
    """(h, w, channels) of the Grad-CAM layer, for the UI label (e.g. 8 x 8 x 1280)."""
    return tuple(int(d) for d in model.get_layer(layer_name).output.shape[1:])


def gradcam(model, x: np.ndarray, layer_name: str = GRADCAM_LAYER, class_index=None, out_size=None):
    """
    Grad-CAM for the classifier (same maths as the notebook's gradcam()).
    Returns (cam, class_index, probs): cam float32 (H, W) in [0, 1], H,W = out_size or x's size.
    The model is a flat functional graph (no nested backbone), so get_layer() works directly.
    """
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    xt = tf.convert_to_tensor(x[None], tf.float32)
    with tf.GradientTape() as tape:
        conv, preds = grad_model(xt, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    out_h, out_w = out_size or x.shape[:2]
    cam = cv2.resize(cam, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    cam = cam / (cam.max() + 1e-8)
    return cam.astype(np.float32), class_index, preds[0].numpy()


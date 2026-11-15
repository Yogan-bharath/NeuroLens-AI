"""
Seg-Grad-CAM for the 2D DeepLabV3+ segmentation model (deeplabv3plus.keras).
Extracted from the tumor-detection notebook (make_gradcam_heatmap).

Model: input (None,256,256,3) float32 RGB scaled to [0,1] (img/255), output (None,256,256,1) sigmoid.
Reuse the app's EXISTING segmentation preprocessing; this module only adds the explanation.
"""
import cv2
import numpy as np
import tensorflow as tf

# Auto-generated layer names from the notebook session; they may not exist in the saved artifact.
NOTEBOOK_LAYERS = ("conv2d_20", "up_sampling2d_2")


def _last_activation_with_shape(model, shape):
    found = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            try:
                s = tuple(int(d) for d in layer.output.shape[1:])
            except Exception:
                continue
            if s == shape:
                found = layer.name
    return found


def resolve_layers(model, img_size: int = 256) -> dict:
    """
    Pick the layers to explain.
    1) If the notebook's two layer names exist in the loaded model, use them (source='notebook').
    2) Otherwise choose by ROLE and shape (source='role'):
         deep    = ASPP fused output activation      (img/16, img/16, 256)
         decoder = last decoder activation before the final x4 upsampling (img/4, img/4, 256)
    Always log/print the result so a human can confirm it.
    """
    names = {layer.name for layer in model.layers}
    if all(n in names for n in NOTEBOOK_LAYERS):
        return {"source": "notebook", "layers": list(NOTEBOOK_LAYERS)}
    deep = _last_activation_with_shape(model, (img_size // 16, img_size // 16, 256))
    decoder = _last_activation_with_shape(model, (img_size // 4, img_size // 4, 256))
    return {"source": "role", "deep": deep, "decoder": decoder, "layers": [l for l in (deep, decoder) if l]}


def feature_map_shape(model, layer_name: str):
    return tuple(int(d) for d in model.get_layer(layer_name).output.shape[1:])


def seg_gradcam(model, x_batch: np.ndarray, layer_name: str, out_size=(256, 256)) -> np.ndarray:
    """
    x_batch: (1, 256, 256, 3) float32 in [0,1] (same preprocessing the segmentation endpoint already uses).
    Target score = SUM of predicted tumor probability over all pixels
    ("how much tumor did the model predict overall"). Returns float32 (H, W) in [0, 1].
    """
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    xt = tf.convert_to_tensor(x_batch, tf.float32)
    with tf.GradientTape() as tape:
        conv, pred_mask = grad_model(xt, training=False)
        target = tf.reduce_sum(pred_mask)
    grads = tape.gradient(target, conv)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv0 = conv[0]
    heat = tf.squeeze(conv0 @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy()
    out_h, out_w = out_size
    return cv2.resize(heat, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

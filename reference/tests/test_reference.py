"""
Tests for the reference modules. No real weights are needed: stand-in models are built here,
and the 3D U-Net / DeepLabV3+ builders are copied from the notebooks so layer names and shapes are real.
Run:  python -m pytest tests -q
"""
import os, sys, tempfile
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier_efficientnet256 as clf
import seg2d_xai as seg
import unet3d as u3


# ---------------------------------------------------------------- builders
def tiny_classifier():
    inp = layers.Input((256, 256, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    x = layers.Conv2D(8, 3, strides=4, padding="same", activation="relu")(x)
    x = layers.Conv2D(8, 3, strides=8, padding="same")(x)
    x = layers.Activation("relu", name="top_activation")(x)          # (8, 8, 8)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(4, activation="softmax")(x)
    return Model(inp, out)


def conv3d_block(x, filters, name):  # copied from the notebook
    x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_3d_unet(patch=(96, 96, 64), base=32):  # copied from the notebook (base filters configurable)
    def enc(x, f, name):
        feat = conv3d_block(x, f, name)
        return feat, layers.MaxPooling3D(2, name=f"{name}_pool")(feat)

    def dec(x, skip, f, name):
        x = layers.Conv3DTranspose(f, 2, strides=2, padding="same", name=f"{name}_up")(x)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
        return conv3d_block(x, f, name)

    inp = layers.Input((*patch, 4), name="mri_3d_input")
    s1, p1 = enc(inp, base, "enc1")
    s2, p2 = enc(p1, base * 2, "enc2")
    s3, p3 = enc(p2, base * 4, "enc3")
    b = conv3d_block(p3, base * 8, "bottleneck")
    d3 = dec(b, s3, base * 4, "dec3")
    d2 = dec(d3, s2, base * 2, "dec2")
    d1 = dec(d2, s1, base, "dec1")
    out = layers.Conv3D(4, 1, activation="softmax", name="segmentation_output")(d1)
    return Model(inp, out, name="3D_UNet_BraTS")


def build_deeplabv3plus(input_shape=(256, 256, 3)):  # copied from the notebook, weights=None (no download)
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, AveragePooling2D,
                                         UpSampling2D, Concatenate, Input, SpatialDropout2D)
    l2reg = 3e-4

    def ASPP(x, filters=256):
        shape = x.shape

        def cba(t, k, d=1):
            t = Conv2D(filters, k, dilation_rate=d, padding="same", use_bias=False,
                       kernel_regularizer=regularizers.l2(l2reg))(t)
            t = BatchNormalization(momentum=0.9)(t)
            return Activation("relu")(t)

        b0, b1, b2, b3 = cba(x, 1), cba(x, 3, 6), cba(x, 3, 12), cba(x, 3, 18)
        b4 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
        b4 = Conv2D(filters, 1, padding="same", use_bias=False)(b4)
        b4 = BatchNormalization(momentum=0.9)(b4)
        b4 = Activation("relu")(b4)
        b4 = UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(b4)
        x = Concatenate()([b0, b1, b2, b3, b4])
        x = SpatialDropout2D(0.15)(x)
        x = Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        return Activation("relu")(x)

    inputs = Input(shape=input_shape)
    backbone = ResNet50(weights=None, include_top=False, input_tensor=inputs)
    low = backbone.get_layer("conv2_block3_out").output
    high = backbone.get_layer("conv4_block6_out").output
    x = ASPP(high, 256)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    low = Conv2D(48, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(l2reg))(low)
    low = BatchNormalization(momentum=0.9)(low)
    low = Activation("relu")(low)
    x = Concatenate()([x, low])
    for _ in range(2):
        x = Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(l2reg))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation("relu")(x)
        if _ == 0:
            x = SpatialDropout2D(0.2)(x)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    return Model(inputs, Conv2D(1, 1, activation="sigmoid")(x), name="DeepLabV3Plus")


# ---------------------------------------------------------------- A: classifier
def test_preprocess_shape_dtype_range_and_gray_channels():
    img = np.random.default_rng(0).integers(0, 256, (300, 340, 3), dtype=np.uint8)
    x = clf.preprocess(img)
    assert x.shape == (256, 256, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 255.0
    assert np.allclose(x[..., 0], x[..., 1]) and np.allclose(x[..., 1], x[..., 2])  # CLAHE is grayscale


def test_preprocess_matches_training_pipeline():
    """Reference = exactly what training did: imread -> CLAHE -> imwrite(.jpg) -> tf decode -> tf resize."""
    rng = np.random.default_rng(1)
    base = (rng.normal(100, 40, (220, 200)).clip(0, 255)).astype(np.uint8)
    img = cv2.cvtColor(cv2.GaussianBlur(base, (0, 0), 3), cv2.COLOR_GRAY2BGR)
    with tempfile.TemporaryDirectory() as d:
        src, mid = os.path.join(d, "a.jpg"), os.path.join(d, "b.jpg")
        cv2.imwrite(src, img)
        im = cv2.imread(src)
        cv2.imwrite(mid, clf.clahe_gray3(im))
        raw = tf.io.read_file(mid)
        ref = tf.image.decode_image(raw, channels=3, expand_animations=False)
        ref = tf.image.resize(ref, (256, 256)).numpy()
        ours = clf.preprocess(im, simulate_training_jpeg=True)
        no_jpeg = clf.preprocess(im, simulate_training_jpeg=False)
    d1, d2 = np.abs(ours - ref), np.abs(no_jpeg - ref)
    print(f"jpeg round-trip: mean {d1.mean():.3f} max {d1.max():.2f} | production (no re-encode): mean {d2.mean():.3f} max {d2.max():.2f}")
    # Measured: decoding the SAME jpeg bytes with OpenCV vs TensorFlow already differs by mean ~0.56, max 3
    # grey levels (of 255). So this path must stay within that decoder noise.
    assert d1.mean() <= 0.8 and d1.max() <= 4.0
    assert d2.mean() <= 1.5 and d2.max() <= 8.0       # production path skips the re-encode; still close


def test_classifier_gradcam_shape_range_determinism():
    model = tiny_classifier()
    x = np.random.default_rng(2).uniform(0, 255, (256, 256, 3)).astype(np.float32)
    cam1, cls1, probs = clf.gradcam(model, x)
    cam2, cls2, _ = clf.gradcam(model, x)
    assert cam1.shape == (256, 256) and cam1.dtype == np.float32
    assert 0.0 <= cam1.min() and cam1.max() <= 1.0 and not np.isnan(cam1).any()
    assert np.array_equal(cam1, cam2) and cls1 == cls2 and np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert clf.feature_map_shape(model) == (8, 8, 8)
    cam3, cls3, _ = clf.gradcam(model, x, class_index=(cls1 + 1) % 4, out_size=(128, 128))
    assert cam3.shape == (128, 128) and cls3 == (cls1 + 1) % 4


# ---------------------------------------------------------------- B: 2D seg XAI (real architecture, random weights)
def test_deeplab_layer_resolution_and_seg_gradcam():
    model = build_deeplabv3plus()
    info = seg.resolve_layers(model)
    print("resolved:", info)
    assert info["source"] == "role"                     # notebook names are session-specific
    deep, dec = info["deep"], info["decoder"]
    assert deep and dec
    assert not deep.startswith("conv") and not dec.startswith("conv")  # not ResNet backbone layers
    assert seg.feature_map_shape(model, deep) == (16, 16, 256)
    assert seg.feature_map_shape(model, dec) == (64, 64, 256)
    x = np.random.default_rng(3).uniform(0, 1, (1, 256, 256, 3)).astype(np.float32)
    for name in (deep, dec):
        cam = seg.seg_gradcam(model, x, name)
        assert cam.shape == (256, 256) and cam.dtype == np.float32
        assert 0.0 <= cam.min() and cam.max() <= 1.0 and not np.isnan(cam).any()


def test_notebook_layer_names_are_used_when_present():
    inp = layers.Input((256, 256, 3))
    x = layers.Conv2D(4, 3, padding="same", name="conv2d_20")(inp)
    x = layers.UpSampling2D(1, name="up_sampling2d_2")(x)
    m = Model(inp, layers.Conv2D(1, 1, activation="sigmoid")(x))
    assert seg.resolve_layers(m) == {"source": "notebook", "layers": ["conv2d_20", "up_sampling2d_2"]}


# ---------------------------------------------------------------- C: 3D U-Net
def test_normalize_modality():
    v = np.zeros((10, 10, 10), np.float32)
    v[2:8, 2:8, 2:8] = np.random.default_rng(4).normal(300, 50, (6, 6, 6))
    out = u3.normalize_modality(v)
    assert (out[v == 0] == 0).all()
    nz = out[v != 0]
    assert abs(nz.mean()) < 1e-4 and abs(nz.std() - 1) < 1e-3
    assert np.array_equal(u3.normalize_modality(np.zeros((4, 4, 4), np.float32)), np.zeros((4, 4, 4)))


def test_prepare_volume_checks_and_order():
    rng = np.random.default_rng(5)
    vols = [rng.uniform(1, 10, (12, 12, 8)).astype(np.float32) for _ in range(4)]
    out = u3.prepare_volume(*vols)
    assert out.shape == (12, 12, 8, 4) and out.dtype == np.float32
    for i in range(4):
        assert np.allclose(out[..., i], u3.normalize_modality(vols[i]))
    try:
        u3.prepare_volume(vols[0], vols[1], vols[2], vols[3][:, :, :5])
        assert False
    except ValueError:
        pass


def test_make_starts_matches_notebook():
    assert u3.make_starts(155, 64, 32) == [0, 32, 64, 91]
    assert u3.make_starts(240, 96, 64) == [0, 64, 128, 144]
    assert u3.make_starts(50, 64, 32) == [0]
    n = len(u3.make_starts(240, 96, 64)) ** 2 * len(u3.make_starts(155, 64, 32))
    assert n == 64                                      # windows for a 240x240x155 volume


def _pointwise_model(shape, w):
    inp = layers.Input(shape)
    conv = layers.Conv3D(4, 1, activation="softmax")
    out = conv(inp)
    m = Model(inp, out)
    conv.set_weights(w)
    return m


def test_sliding_window_equals_full_volume_for_pointwise_model():
    """A 1x1x1 conv has no spatial context, so sliding-window output must equal full-volume output exactly."""
    patch = (32, 32, 16)
    rng = np.random.default_rng(6)
    probe = layers.Conv3D(4, 1, activation="softmax")
    probe.build((None, 4, 4, 4, 4))
    w = [rng.normal(0, 2, a.shape).astype(np.float32) for a in probe.get_weights()]
    patch_model = _pointwise_model((*patch, 4), w)
    for vol_shape in [(50, 45, 30), (40, 40, 10), (32, 32, 16)]:   # last two need padding / exact fit
        img = rng.normal(size=(*vol_shape, 4)).astype(np.float32)
        full_model = _pointwise_model((*vol_shape, 4), w)
        expected = full_model(img[None], training=False).numpy()[0]
        got = u3.sliding_window_predict(patch_model, img, patch_shape=patch, stride=(16, 16, 8), batch_size=3)
        assert got.shape == expected.shape
        assert np.allclose(got, expected, atol=1e-5), vol_shape
        assert np.allclose(got.sum(-1), 1.0, atol=1e-4)


def test_sliding_window_progress_callback():
    patch = (16, 16, 8)
    m = _pointwise_model((*patch, 4), [np.zeros((1, 1, 1, 4, 4), np.float32), np.zeros(4, np.float32)])
    seen = []
    u3.sliding_window_predict(m, np.zeros((24, 24, 12, 4), np.float32), patch_shape=patch, stride=(8, 8, 4),
                              batch_size=4, progress_cb=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1] and len(seen) >= 2


def legacy_notebook_resize(heatmap, patch_shape):
    """The notebook's resize, kept ONLY to document the bug."""
    h = tf.image.resize(heatmap[..., None], patch_shape[:2]).numpy()[..., 0]
    if h.shape[2] != patch_shape[2]:
        h = np.stack([tf.image.resize(h[:, :, z][..., None], patch_shape[:2]).numpy()[..., 0]
                      for z in range(h.shape[2])], axis=2)
    return h


def test_resize_fixes_the_notebook_depth_bug():
    hm = np.zeros((12, 12, 8), np.float32)
    hm[6, 6, 7] = 1.0                                   # hot voxel in the LAST bottleneck depth slice
    old = legacy_notebook_resize(hm, (96, 96, 64))
    new = u3.resize_trilinear(hm, (96, 96, 64))
    assert old.shape == (96, 96, 96)                    # the bug: depth 96, not 64
    assert new.shape == (96, 96, 64)
    depth_profile = new.sum(axis=(0, 1))
    assert int(depth_profile.argmax()) >= 64 - 8        # peak lands in the last ~8 patch slices
    old_peak = int(old.sum(axis=(0, 1)).argmax())
    print("legacy peak depth index:", old_peak, "(patch has only 64 slices); fixed:", int(depth_profile.argmax()))


def test_resize_matches_tf_bilinear_in_plane():
    a = np.random.default_rng(7).uniform(0, 1, (12, 12, 3)).astype(np.float32)
    ref = tf.image.resize(a, (96, 96)).numpy()
    assert np.allclose(u3.resize_trilinear(a, (96, 96, 3)), ref, atol=1e-5)


def test_gradcam3d_on_real_architecture():
    model = build_3d_unet(patch=(96, 96, 64), base=4)
    assert tuple(model.get_layer(u3.GRADCAM_LAYER).output.shape[1:]) == (12, 12, 8, 32)
    patch = np.random.default_rng(8).normal(size=(96, 96, 64, 4)).astype(np.float32)
    maps = u3.gradcam3d(model, patch)
    assert set(maps) == {1, 2, 3}
    for c, m in maps.items():
        assert m.shape == (96, 96, 64) and m.dtype == np.float32      # NOT (96, 96, 96)
        assert 0.0 <= m.min() and m.max() <= 1.0 and not np.isnan(m).any()
    again = u3.gradcam3d(model, patch, class_indices=(2,))
    assert np.array_equal(again[2], maps[2])


def test_entropy():
    uniform = np.full((5, 5, 5, 4), 0.25, np.float32)
    onehot = np.zeros((5, 5, 5, 4), np.float32)
    onehot[..., 0] = 1.0
    assert np.allclose(u3.prediction_entropy(uniform), 1.0, atol=1e-5)
    assert u3.prediction_entropy(onehot).max() < 1e-3


def test_tumor_center_patch_embed_and_empty_case():
    pred = np.zeros((120, 130, 80), np.uint8)
    pred[60:70, 40:50, 30:40] = 2
    start = u3.tumor_center_start(pred, (96, 96, 64))
    assert start is not None and all(s >= 0 for s in start)
    assert all(s + p <= v for s, p, v in zip(start, (96, 96, 64), pred.shape))   # patch stays inside the volume
    img = np.random.default_rng(9).normal(size=(*pred.shape, 4)).astype(np.float32)
    patch = u3.extract_patch(img, start)
    assert patch.shape == (96, 96, 64, 4)
    pm = np.ones((96, 96, 64), np.float32)
    full = u3.embed_patch_map(pm, start, pred.shape)
    assert full.shape == pred.shape and full.sum() == pm.size
    assert u3.tumor_center_start(np.zeros((120, 130, 80), np.uint8)) is None


def test_pad_to_min_roundtrip():
    img = np.ones((50, 60, 20, 4), np.float32)
    padded, orig = u3.pad_to_min(img, (96, 96, 64))
    assert padded.shape == (96, 96, 64, 4) and orig == (50, 60, 20)
    assert padded[:50, :60, :20].min() == 1.0 and padded[50:].max() == 0.0

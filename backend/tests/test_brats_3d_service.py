import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from app.core.exceptions import InvalidConfigurationException
from app.services.brats_3d_service import (
    evaluate_segmentation,
    map_brats_ground_truth,
    prepare_multimodal_volumes,
    serialize_segmentation_nifti,
    sliding_window_predict,
)


def _nifti_bytes(data: np.ndarray, affine: np.ndarray) -> bytes:
    return nib.Nifti1Image(data, affine).to_bytes()


def test_prepare_multimodal_volumes_uses_training_order_and_normalization():
    shape = (96, 96, 64)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    coords = np.indices(shape, dtype=np.float32)
    source_volumes = [
        (coords[0] + 2).astype(np.float32),
        (coords[1] + 4).astype(np.float32),
        (coords[2] + 8).astype(np.float32),
        (coords[0] + coords[1] * 2 + coords[2] * 3).astype(np.float32),
    ]
    files = [
        (name, _nifti_bytes(volume, affine), f"sample_{name}.nii")
        for name, volume in zip(("flair", "t1", "t1ce", "t2"), source_volumes)
    ]

    stacked, reference = prepare_multimodal_volumes(files)

    assert stacked.shape == (*shape, 4)
    assert stacked.dtype == np.float32
    for channel, source in enumerate(source_volumes):
        nonzero = source != 0
        expected = np.zeros_like(source, dtype=np.float32)
        expected[nonzero] = (source[nonzero] - source[nonzero].mean()) / source[nonzero].std()
        np.testing.assert_allclose(stacked[..., channel], expected, atol=1e-5)
    assert not np.allclose(stacked[..., 0], stacked[..., 1])
    assert reference.shape == shape


def test_prepare_multimodal_volumes_rejects_affine_mismatch():
    shape = (96, 96, 64)
    data = np.ones(shape, dtype=np.float32)
    affine = np.eye(4)
    shifted_affine = affine.copy()
    shifted_affine[0, 3] = 2
    files = [
        ("flair", _nifti_bytes(data, affine), "flair.nii"),
        ("t1", _nifti_bytes(data, shifted_affine), "t1.nii"),
        ("t1ce", _nifti_bytes(data, affine), "t1ce.nii"),
        ("t2", _nifti_bytes(data, affine), "t2.nii"),
    ]

    with pytest.raises(InvalidConfigurationException, match="affine"):
        prepare_multimodal_volumes(files)


def test_serialized_prediction_preserves_nifti_spatial_metadata():
    shape = (4, 5, 6)
    affine = np.array([
        [-1, 0, 0, 10],
        [0, -1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    prediction = np.ones(shape, dtype=np.uint8)

    result_bytes = serialize_segmentation_nifti(prediction, reference)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_file.write(result_bytes)
        temp_path = temp_file.name
    try:
        result = nib.load(temp_path)
        result_data = np.asarray(result.dataobj)
    finally:
        os.unlink(temp_path)

    assert result.shape == shape
    np.testing.assert_allclose(result.affine, affine)
    assert result.header.get_zooms()[:3] == reference.header.get_zooms()[:3]
    assert result_data.dtype == np.uint8


def test_sliding_window_predict_aggregates_overlapping_patch_outputs():
    class ConstantModel:
        def __call__(self, batch, training=False):
            output = np.zeros((1, 96, 96, 64, 4), dtype=np.float32)
            output[..., 2] = 1.0
            return output

    image = np.zeros((96, 96, 64, 4), dtype=np.float32)
    probabilities = sliding_window_predict(ConstantModel(), image)

    assert probabilities.shape == (96, 96, 64, 4)
    assert np.all(np.argmax(probabilities, axis=-1) == 2)


def test_ground_truth_mapping_and_absent_class_metrics_follow_policy():
    ground_truth = np.array([0, 1, 2, 4], dtype=np.uint8)
    prediction = np.array([0, 1, 2, 3], dtype=np.uint8)

    np.testing.assert_array_equal(map_brats_ground_truth(ground_truth), [0, 1, 2, 3])
    metrics = evaluate_segmentation(ground_truth, prediction)

    assert metrics["per_class"]["Necrotic / non-enhancing tumor"]["dice"] == 1.0
    assert metrics["per_class"]["Peritumoral edema"]["dice"] == 1.0
    assert metrics["per_class"]["Enhancing tumor"]["dice"] == 1.0
    assert metrics["valid_patient_count"] == 1
    assert metrics["valid_class_count"] == 3
    assert metrics["mean_foreground_dice"] == 1.0

    absent = evaluate_segmentation(
        np.zeros((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2, 2), dtype=np.uint8),
    )
    assert absent["valid_patient_count"] == 0
    assert absent["valid_class_count"] == 0
    assert np.isnan(absent["per_class"]["Enhancing tumor"]["dice"])


def test_ground_truth_mapping_rejects_unexpected_braTS_labels():
    with pytest.raises(InvalidConfigurationException, match="Unexpected BraTS"):
        map_brats_ground_truth(np.array([0, 3], dtype=np.uint8))


def test_exported_segmentation_uses_original_braTS_enhancing_label():
    shape = (4, 5, 6)
    reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    prediction = np.full(shape, 3, dtype=np.uint8)
    result_bytes = serialize_segmentation_nifti(
        np.where(prediction == 3, 4, prediction).astype(np.uint8),
        reference,
    )
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_file.write(result_bytes)
        temp_path = temp_file.name
    try:
        exported = np.asarray(nib.load(temp_path).dataobj)
    finally:
        os.unlink(temp_path)
    assert np.all(exported == 4)

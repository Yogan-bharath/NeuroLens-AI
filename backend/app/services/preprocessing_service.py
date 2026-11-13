import io
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from pathlib import Path
from app.core.config import settings
from app.core.logging import logger


class PreprocessingService:
    @staticmethod
    def load_pil_rgb(image_source: Union[bytes, str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Accept a filepath, bytes, PIL image, or NumPy array; return an RGB PIL image.
        Matches exact notebook _load_pil_rgb implementation.
        """
        if isinstance(image_source, bytes):
            image_source = Image.open(io.BytesIO(image_source))

        if isinstance(image_source, (str, Path)):
            with Image.open(image_source) as im:
                return im.convert("RGB").copy()

        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB").copy()

        arr = np.asarray(image_source)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError(f"Expected grayscale/RGB/RGBA image, got shape {arr.shape}")
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr).astype(np.float32)
            if arr.size and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def apply_clahe_rgb(pil_rgb: Image.Image) -> Image.Image:
        """
        Optional grayscale CLAHE; matches notebook _apply_clahe_rgb.
        Converts RGB to grayscale, applies CLAHE (clipLimit=2.0, tileGridSize=(8, 8)),
        and repeats across 3 channels.
        """
        rgb = np.asarray(pil_rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(np.repeat(enhanced[..., None], 3, axis=-1), mode="RGB")

    @classmethod
    def preprocess_for_classifier(
        cls,
        image_source: Union[bytes, Image.Image, np.ndarray],
        use_clahe: Optional[bool] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess MRI scan for EfficientNetV2B0 classifier.
        Matches notebook preprocess_for_classifier:
        RGB -> (optional CLAHE) -> resize 128x128 BILINEAR -> np.float32 / 255.0.
        Returns tensor of shape (1, 128, 128, 3).
        """
        pil_rgb = cls.load_pil_rgb(image_source)
        clahe_enabled = use_clahe if use_clahe is not None else settings.CLASSIFIER_USE_CLAHE
        source = cls.apply_clahe_rgb(pil_rgb) if clahe_enabled else pil_rgb

        size = target_size or (settings.IMG_SIZE, settings.IMG_SIZE)
        resized = source.resize(size, Image.Resampling.BILINEAR)
        normalized = np.asarray(resized, dtype=np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    @classmethod
    def preprocess_for_swin_transformer(
        cls, image_source: Union[bytes, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Swin training pipeline: RGB, 224×224 bilinear resize, float32 [0, 1]."""
        pil_rgb = cls.load_pil_rgb(image_source)
        resized = pil_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        return np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)

    @classmethod
    def preprocess_for_segmenter(
        cls,
        image_source: Union[bytes, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess MRI scan for BRISC Binary U-Net segmenter.
        Matches notebook preprocess_for_segmenter:
        RGB -> resize 128x128 BILINEAR -> np.float32 / 255.0.
        Returns tensor of shape (1, 128, 128, 3).
        """
        pil_rgb = cls.load_pil_rgb(image_source)
        size = target_size or (settings.IMG_SIZE, settings.IMG_SIZE)
        resized = pil_rgb.resize(size, Image.Resampling.BILINEAR)
        normalized = np.asarray(resized, dtype=np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    @classmethod
    def preprocess_for_deeplabv3plus(cls, image_source: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """DeepLabV3+ RGB 256×256 float32 input; normalization is deployment-configurable."""
        pil_rgb = cls.load_pil_rgb(image_source)
        resized = pil_rgb.resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        if settings.DEEPLAB_INPUT_NORMALIZATION.lower() == "zero_one":
            array /= 255.0
        elif settings.DEEPLAB_INPUT_NORMALIZATION.lower() != "none":
            raise ValueError("DEEPLAB_INPUT_NORMALIZATION must be 'zero_one' or 'none'.")
        return np.expand_dims(array, axis=0)

    @classmethod
    def overlay_heatmap_on_image(
        cls,
        image_source: Union[bytes, Image.Image],
        heatmap_2d: np.ndarray,
        alpha: float = 0.40
    ) -> Tuple[bytes, bytes]:
        """
        Matches notebook _overlay_heatmap and _resize_heatmap.
        Resizes heatmap to original dimensions via BILINEAR, applies jet colormap,
        and alpha blends with original RGB.
        Returns (raw_heatmap_bytes, overlay_bytes).
        """
        pil_rgb = cls.load_pil_rgb(image_source)
        w, h = pil_rgb.size
        rgb = np.asarray(pil_rgb, dtype=np.float32) / 255.0

        # Resize heatmap to match original image dimensions using BILINEAR
        heat = (np.clip(heatmap_2d, 0, 1) * 255).astype(np.uint8)
        heat_resized = np.asarray(
            Image.fromarray(heat).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32
        ) / 255.0

        # Apply Jet colormap
        colored_heat = plt.get_cmap("jet")(heat_resized)[..., :3]

        # Blend
        overlay = np.clip((1.0 - alpha) * rgb + alpha * colored_heat, 0.0, 1.0)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        raw_heat_uint8 = (colored_heat * 255).astype(np.uint8)

        # Encode PNGs
        heat_buf = io.BytesIO()
        Image.fromarray(raw_heat_uint8).save(heat_buf, format="PNG")

        overlay_buf = io.BytesIO()
        Image.fromarray(overlay_uint8).save(overlay_buf, format="PNG")

        return heat_buf.getvalue(), overlay_buf.getvalue()

    @classmethod
    def overlay_mask_on_image(
        cls,
        image_source: Union[bytes, Image.Image],
        mask_2d: np.ndarray,
        alpha: float = 0.45
    ) -> Tuple[bytes, bytes]:
        """
        Matches notebook _overlay_mask:
        Binary mask resized to original image dimensions using NEAREST neighbor interpolation.
        Applies red overlay on lesion areas.
        Returns (binary_mask_bytes, overlay_bytes).
        """
        pil_rgb = cls.load_pil_rgb(image_source)
        w, h = pil_rgb.size
        rgb = np.asarray(pil_rgb, dtype=np.float32) / 255.0

        mask_bool = np.asarray(mask_2d).squeeze().astype(bool)

        # Resize mask to match original image dimensions using NEAREST
        if mask_bool.shape != (h, w):
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
            mask_resized = mask_img.resize((w, h), Image.Resampling.NEAREST)
            mask_bool = np.asarray(mask_resized) > 0
        else:
            mask_resized = Image.fromarray((mask_bool.astype(np.uint8) * 255))

        out = rgb.copy()
        # Red highlight: increase red channel, attenuate green and blue
        out[mask_bool, 0] = (1.0 - alpha) * out[mask_bool, 0] + alpha
        out[mask_bool, 1] *= (1.0 - alpha)
        out[mask_bool, 2] *= (1.0 - alpha)
        out = np.clip(out, 0.0, 1.0)

        overlay_uint8 = (out * 255).astype(np.uint8)

        mask_buf = io.BytesIO()
        mask_resized.save(mask_buf, format="PNG")

        overlay_buf = io.BytesIO()
        Image.fromarray(overlay_uint8).save(overlay_buf, format="PNG")

        return mask_buf.getvalue(), overlay_buf.getvalue()


preprocessing_service = PreprocessingService()

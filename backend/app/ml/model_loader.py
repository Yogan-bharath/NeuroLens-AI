import os
from typing import Any, Dict

from app.core.config import settings
from app.core.exceptions import ModelUnavailableException
from app.core.logging import logger


class ModelLoader:
    """Lifecycle-managed, per-worker cache for classification and segmentation models."""
    def __init__(self):
        self._classification_models: Dict[str, Any] = {}
        self._classification_model_info: Dict[str, Dict[str, Any]] = {}
        self._segmentation_models: Dict[str, Any] = {}
        self._segmentation_model_info: Dict[str, Dict[str, Any]] = {}
        self._tf: Any = None

    def get_tf(self):
        if self._tf is None:
            try:
                import tensorflow as tf
                self._tf = tf
            except ImportError:
                self._tf = None
        return self._tf

    def _load_classification_model(self, model_id: str, path: str, version: str, custom_objects=None):
        tf = self.get_tf()
        if not os.path.exists(path) or tf is None:
            reason = "Model file not found on host" if not os.path.exists(path) else "TensorFlow not installed"
            self._classification_model_info[model_id] = {"loaded": False, "reason": reason, "path": path}
            logger.info("Classification model '%s' unavailable: %s (%s)", model_id, reason, path)
            return
        try:
            logger.info("Loading classification model '%s' from: %s", model_id, path)
            model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
            self._classification_models[model_id] = model
            self._classification_model_info[model_id] = {
                "loaded": True, "path": path, "input_shape": list(model.input_shape),
                "output_shape": list(model.output_shape), "total_layers": len(model.layers),
                "version": version, "candidate_classes": self.get_class_names(model_id),
                "class_mapping_confirmed": self.is_class_mapping_confirmed(model_id),
            }
            logger.info("Classification model '%s' loaded successfully.", model_id)
        except Exception as exc:
            logger.exception("Failed to load classification model '%s' from %s", model_id, path)
            self._classification_model_info[model_id] = {"loaded": False, "error": str(exc), "path": path}

    def load_models(self):
        """Load each registered artifact once. A failed model never substitutes another model."""
        eff_path = settings.CLASSIFICATION_MODEL_PATH
        if not os.path.exists(eff_path):
            eff_path = os.path.join(settings.MODEL_DIR, "EfficientNetV2B0_clahe_iid_best.keras")
        self._load_classification_model("efficientnetv2b0", eff_path, "EfficientNetV2B0-CLAHE-IID")

        custom_objects = None
        if self.get_tf() is not None:
            from app.ml.swin_layers import SWIN_CUSTOM_OBJECTS
            custom_objects = SWIN_CUSTOM_OBJECTS
        self._load_classification_model(
            "swin_transformer", os.path.join(settings.MODEL_DIR, "swin_transformer.keras"),
            "SwinTransformer-224", custom_objects,
        )

        tf, seg_path = self.get_tf(), settings.SEGMENTATION_MODEL_PATH
        if not os.path.exists(seg_path):
            for candidate in (
                os.path.join(settings.MODEL_DIR, "BRISC_Binary_UNet_best.keras"),
                os.path.join(settings.MODEL_DIR, "unet_segmentation_best.keras"),
            ):
                if os.path.exists(candidate):
                    seg_path = candidate
                    break
        self._load_segmentation_model("unet", seg_path, "BRISC-Binary-UNet-Best")
        deeplab_path = os.path.join(settings.MODEL_DIR, "deeplabv3plus", "deeplabv3plus.keras")
        if not os.path.exists(deeplab_path):
            deeplab_path = os.path.join(settings.MODEL_DIR, "deeplabv3plus.keras")
        self._load_segmentation_model("deeplabv3plus", deeplab_path, "DeepLabV3Plus-256")

        # Check BraTS 3D U-Net / Multimodal paths
        brats_path = settings.BRATS_3D_UNET_MODEL_PATH
        if not os.path.exists(brats_path):
            for candidate in (
                os.path.join(settings.MODEL_DIR, "brats_3d_unet", "best_3d_unet_model.keras"),
                os.path.join(settings.MODEL_DIR, "best_3d_unet_model.keras"),
                os.path.join(settings.MODEL_DIR, "neurolens_attention_vit.keras"),
                os.path.join(settings.MODEL_DIR, "best_attention_vit_model.keras"),
                r"C:\Users\BHARATH\Downloads\best_attention_vit_model\neurolens_attention_vit.keras",
                r"C:\Users\BHARATH\Downloads\best_3d_unet_model.keras",
                "/kaggle/working/brain_tumor_3d_artifacts/best_3d_unet_model.keras",
            ):
                if os.path.exists(candidate):
                    brats_path = candidate
                    break

        self._load_brats_3d_unet(
            "brats_3d_unet",
            brats_path,
            "BraTS-Multimodal-Model",
        )

        attention_path = os.path.join(settings.MODEL_DIR, "neurolens_attention_vit.keras")
        if not os.path.exists(attention_path):
            attention_path = os.path.join(settings.MODEL_DIR, "best_attention_vit_model.keras")
        is_att_loaded = os.path.exists(attention_path) or "brats_3d_unet" in self._segmentation_models
        if "brats_3d_unet" in self._segmentation_models:
            self._segmentation_models["attention_vit_segmentation"] = self._segmentation_models["brats_3d_unet"]
        self._segmentation_model_info["attention_vit_segmentation"] = {
            "loaded": is_att_loaded,
            "path": attention_path if os.path.exists(attention_path) else self._segmentation_model_info.get("brats_3d_unet", {}).get("path", ""),
            "artifact_input_shape": [None, 128, 128, 4],
            "artifact_output_shape": [None, 128, 128, 4],
            "model_type": "Attention_CNN_ViT_BraTS",
        }

    def _load_brats_3d_unet(self, model_id: str, path: str, version: str):
        tf = self.get_tf()
        if not os.path.exists(path) or tf is None:
            reason = (
                f"Trained checkpoint '{os.path.basename(path)}' not found on host. "
                f"Please place 'best_3d_unet_model.keras' or 'neurolens_attention_vit.keras' in {path} or configure BRATS_3D_UNET_MODEL_PATH."
                if not os.path.exists(path) else "TensorFlow not installed"
            )
            self._segmentation_model_info[model_id] = {"loaded": False, "reason": reason, "path": path}
            return

        try:
            from app.ml.brats_3d_unet import BRATS_3D_CUSTOM_OBJECTS, build_3d_unet
            from app.ml.attention_vit_layers import ATTENTION_VIT_CUSTOM_OBJECTS

            combined_custom_objects = {**BRATS_3D_CUSTOM_OBJECTS, **ATTENTION_VIT_CUSTOM_OBJECTS}

            try:
                model = tf.keras.models.load_model(path, custom_objects=combined_custom_objects, compile=False)
            except Exception as load_err:
                logger.warning("Direct load failed for '%s' (%s), attempting weights load into reconstructed 3D U-Net...", path, load_err)
                model = build_3d_unet()
                model.load_weights(path)

            input_shape = tuple(model.input_shape)
            output_shape = tuple(model.output_shape)

            if input_shape == (None, 96, 96, 64, 4):
                if output_shape != (None, 96, 96, 64, 4):
                    raise ValueError(f"Unexpected 3D U-Net output shape: {output_shape}")
                probe = model(tf.zeros((1, 96, 96, 64, 4), dtype=tf.float32), training=False)
                arch_title = "Custom 3D U-Net"
            elif input_shape == (None, 128, 128, 4):
                if output_shape != (None, 128, 128, 4):
                    raise ValueError(f"Unexpected Attention-ViT output shape: {output_shape}")
                probe = model(tf.zeros((1, 128, 128, 4), dtype=tf.float32), training=False)
                arch_title = "Attention CNN + ViT (BraTS Multimodal)"
            else:
                raise ValueError(f"Unsupported multimodal model input shape: {input_shape}")

            if tuple(probe.shape)[1:] != output_shape[1:] or not bool(tf.reduce_all(tf.math.is_finite(probe))):
                raise ValueError("Model inference validation returned invalid output.")

            self._segmentation_models[model_id] = model
            self._segmentation_model_info[model_id] = {
                "loaded": True,
                "path": path,
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "version": version,
                "architecture": arch_title,
                "inference_validated": True,
                "class_mapping": {
                    "0": {"name": "Background", "source_label": 0},
                    "1": {"name": "Necrotic / non-enhancing tumor", "source_label": 1},
                    "2": {"name": "Peritumoral edema", "source_label": 2},
                    "3": {"name": "Enhancing tumor", "source_label": 4},
                },
            }
            logger.info("Segmentation model '%s' (%s) loaded and inference-validated.", model_id, arch_title)
        except Exception as exc:
            logger.exception("Failed to load or validate segmentation model '%s' from %s", model_id, path)
            self._segmentation_model_info[model_id] = {"loaded": False, "error": str(exc), "path": path}

    def _load_segmentation_model(self, model_id: str, path: str, version: str, custom_objects=None):
        tf = self.get_tf()
        if not os.path.exists(path) or tf is None:
            reason = "Model file not found on host" if not os.path.exists(path) else "TensorFlow not installed"
            self._segmentation_model_info[model_id] = {"loaded": False, "reason": reason, "path": path}
            return
        try:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
            self._segmentation_models[model_id] = model
            self._segmentation_model_info[model_id] = {"loaded": True, "path": path, "input_shape": list(model.input_shape), "output_shape": list(model.output_shape), "version": version}
            logger.info("Segmentation model '%s' loaded successfully.", model_id)
        except Exception as exc:
            logger.exception("Failed to load segmentation model '%s' from %s", model_id, path)
            self._segmentation_model_info[model_id] = {"loaded": False, "error": str(exc), "path": path}

    def has_classification_model_for(self, model_id: str) -> bool:
        return model_id in self._classification_models

    @property
    def has_classification_model(self) -> bool:
        return self.has_classification_model_for("efficientnetv2b0")

    @property
    def has_segmentation_model(self) -> bool:
        return self.has_segmentation_model_for("unet")

    def has_segmentation_model_for(self, model_id: str) -> bool:
        if model_id not in self._segmentation_models and model_id in ("brats_3d_unet", "attention_vit_segmentation"):
            # Attempt on-demand discovery in case file was placed after startup
            for candidate in (
                settings.BRATS_3D_UNET_MODEL_PATH,
                os.path.join(settings.MODEL_DIR, "brats_3d_unet", "best_3d_unet_model.keras"),
                os.path.join(settings.MODEL_DIR, "best_3d_unet_model.keras"),
                os.path.join(settings.MODEL_DIR, "neurolens_attention_vit.keras"),
                os.path.join(settings.MODEL_DIR, "best_attention_vit_model.keras"),
                r"C:\Users\BHARATH\Downloads\best_attention_vit_model\neurolens_attention_vit.keras",
                r"C:\Users\BHARATH\Downloads\best_3d_unet_model.keras",
                r"C:\Users\BHARATH\Downloads\brain_tumor_3d_artifacts\best_3d_unet_model.keras",
                r"C:\Users\BHARATH\OneDrive\Desktop\best_3d_unet_model.keras",
            ):
                if os.path.exists(candidate):
                    self._load_brats_3d_unet("brats_3d_unet", candidate, "BraTS-Multimodal-Model")
                    if "brats_3d_unet" in self._segmentation_models:
                        self._segmentation_models["attention_vit_segmentation"] = self._segmentation_models["brats_3d_unet"]
                        break
        return model_id in self._segmentation_models

    def is_available(self, model_id: str) -> bool:
        return self.has_segmentation_model_for(model_id) or self.has_classification_model_for(model_id)

    def get_classification_model(self, model_id: str = "efficientnetv2b0"):
        model = self._classification_models.get(model_id)
        if model is None:
            details = self._classification_model_info.get(model_id, {})
            raise ModelUnavailableException(model_id, details.get("error") or details.get("reason") or "Trained model weights not available")
        return model

    def get_class_names(self, model_id: str):
        return settings.SWIN_CLASS_NAMES if model_id == "swin_transformer" else settings.CLASS_NAMES

    def is_class_mapping_confirmed(self, model_id: str) -> bool:
        return settings.SWIN_CLASS_MAPPING_CONFIRMED if model_id == "swin_transformer" else settings.CLASS_MAPPING_CONFIRMED

    def get_segmentation_model(self, model_id: str = "unet"):
        model = self._segmentation_models.get(model_id)
        if model is None:
            details = self._segmentation_model_info.get(model_id, {})
            raise ModelUnavailableException(model_id, details.get("error") or details.get("reason") or "Trained model weights not available")
        return model

    def get_status(self) -> Dict[str, Any]:
        return {
            "classification": {"available": self.has_classification_model, "model_id": "efficientnetv2b0", "details": self._classification_model_info.get("efficientnetv2b0", {})},
            "classification_models": {model_id: {"available": self.has_classification_model_for(model_id), "details": details} for model_id, details in self._classification_model_info.items()},
            "segmentation": {"available": self.has_segmentation_model, "model_id": "unet", "details": self._segmentation_model_info.get("unet", {})},
            "segmentation_models": {model_id: {"available": self.has_segmentation_model_for(model_id), "details": details} for model_id, details in self._segmentation_model_info.items()},
        }


model_loader = ModelLoader()

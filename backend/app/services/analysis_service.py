import uuid
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from bson import ObjectId

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ResourceNotFoundException, PermissionDeniedException, InvalidConfigurationException
from app.db.mongodb import get_database
from app.utils.file_validation import validate_file_bytes
from app.utils.serialization import mongo_to_dict
from app.services.imagekit_service import imagekit_service
from app.services.classification_service import classification_service
from app.services.segmentation_service import segmentation_service
from app.services.explainability_service import explainability_service
from app.schemas.analysis import (
    AnalysisConfigurationSchema,
    AnalysisResponse,
    AnalysisSummaryResponse,
    InputImageData,
    DetailedClassification,
    DetailedSegmentation,
    DetailedExplainability,
    AnalysisMetadataItem,
)

# In-memory storage fallback for local testing / offline dev
_in_memory_analyses: Dict[str, Dict[str, Any]] = {}


class AnalysisService:
    async def create_analysis(
        self,
        image_bytes: bytes,
        filename: str,
        user_id: str,
        config: Optional[AnalysisConfigurationSchema] = None
    ) -> AnalysisResponse:
        """
        Orchestrate complete MRI analysis pipeline.
        """
        if config is None:
            config = AnalysisConfigurationSchema()

        analysis_id = f"ANL-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.time()

        # 1. Validate file
        width, height, mime_type = validate_file_bytes(image_bytes, filename)

        # 2. Upload original MRI scan to ImageKit
        folder = f"neurolens/analyses/{analysis_id}"
        orig_upload = await imagekit_service.upload_file(
            image_bytes, filename, folder=folder, tags=["mri", "original", analysis_id]
        )

        input_image_data = InputImageData(
            filename=filename,
            url=orig_upload["url"],
            width=width,
            height=height,
            file_id=orig_upload["file_id"]
        )

        mode = config.mode.lower()
        classification_res = None
        segmentation_res = None
        explanations_list = []
        is_demo = False

        # 3. Execute Classification
        if mode in ["classification", "combined"]:
            cls_out = await classification_service.predict(
                image_bytes, model_id=config.classification_model_id
            )
            classification_res = cls_out
            if cls_out.get("is_mock"):
                is_demo = True

            # 4. Generate Classification Explainability (Grad-CAM & Grad-CAM++)
            target_class = cls_out["predicted_class"]
            # Grad-CAM
            xai1 = await explainability_service.generate_explanation(
                image_bytes=image_bytes,
                analysis_id=analysis_id,
                method=config.classification_xai_method or "gradcam",
                model_id=config.classification_model_id,
                target_layer=config.classification_layer,
                target_class=target_class
            )
            explanations_list.append(xai1["result_item"])

            # Also generate secondary Grad-CAM++ for workspace tabs if default was gradcam
            if (config.classification_xai_method or "gradcam") == "gradcam" and config.classification_model_id != "swin_transformer":
                try:
                    xai2 = await explainability_service.generate_explanation(
                        image_bytes=image_bytes,
                        analysis_id=analysis_id,
                        method="gradcam_pp",
                        model_id=config.classification_model_id,
                        target_layer=config.classification_layer,
                        target_class=target_class
                    )
                    explanations_list.append(xai2["result_item"])
                except Exception as e:
                    logger.warning(f"Grad-CAM++ secondary generation skipped: {e}")

        # 5. Execute Segmentation
        if mode in ["segmentation", "combined"]:
            seg_out = await segmentation_service.segment(
                image_bytes, analysis_id=analysis_id, model_id=config.segmentation_model_id
            )
            segmentation_res = seg_out
            if seg_out.get("is_mock"):
                is_demo = True

            # Segmentation XAI
            if mode == "segmentation":
                try:
                    seg_xai = await explainability_service.generate_explanation(
                        image_bytes=image_bytes,
                        analysis_id=analysis_id,
                        method=config.segmentation_xai_method or "seg_gradcam",
                        model_id=config.segmentation_model_id,
                        target_layer=config.segmentation_layer,
                        target_region="tumor_core"
                    )
                    explanations_list.append(seg_xai["result_item"])
                except Exception as e:
                    logger.warning(f"Segmentation XAI skipped: {e}")

        total_time_ms = int((time.time() - start_time) * 1000)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build Response Component Models
        primary_model_name = ("BRISC Binary U-Net" if mode == "segmentation" else
                              ("Swin Transformer" if classification_res and classification_res["model_id"] == "swin_transformer" else "EfficientNetV2B0"))
        primary_model_version = ("BRISC-Binary-UNet-Best" if mode == "segmentation" else
                                 (classification_res["model_version"] if classification_res else "EfficientNetV2B0-CLAHE-IID"))

        predicted_class = (
            classification_res["predicted_class"] if classification_res else "Binary Segmentation Output"
        )
        confidence = (
            classification_res["confidence"] if classification_res else (
                round(segmentation_res.get("foreground_percentage", 0.0) / 100.0, 4) if segmentation_res else 0.0
            )
        )

        metadata_item = AnalysisMetadataItem(
            modelName=primary_model_name,
            modelVersion=primary_model_version,
            processingTimeMs=total_time_ms,
            imageSize={"width": width, "height": height},
            analysisDate=now_iso
        )

        det_class = None
        if classification_res:
            det_class = DetailedClassification(
                model_id=classification_res["model_id"],
                model_version=classification_res["model_version"],
                predicted_class=classification_res["predicted_class"],
                confidence=classification_res["confidence"],
                probabilities=classification_res["probabilities"],
                inference_time_ms=classification_res["inference_time_ms"]
            )

        det_seg = None
        if segmentation_res:
            det_seg = DetailedSegmentation(
                model_id=segmentation_res["model_id"],
                regions=segmentation_res["regions"],
                mask_url=segmentation_res["mask_url"],
                overlay_url=segmentation_res["overlay_url"],
                dice_coefficient=segmentation_res.get("dice_coefficient"),
                iou=segmentation_res.get("iou")
            )

        det_xai = DetailedExplainability(
            classification={"methods": [e.method for e in explanations_list if "seg" not in e.method]},
            segmentation={"methods": [e.method for e in explanations_list if "seg" in e.method]}
        )

        chosen_actual_model = (classification_res or {}).get("actual_model") or {
            "model_id": config.classification_model_id if mode != "segmentation" else config.segmentation_model_id,
            "model_name": primary_model_name,
            "model_version": primary_model_version,
            "architecture": "Hierarchical Shifted-Window Transformer" if (classification_res and classification_res.get("model_id") == "swin_transformer") else "EfficientNetV2-B0 (Fused-MBConv)",
            "framework": "TensorFlow 2.18 / Keras 3",
            "task_type": "2D Brain Tumor Classification" if mode != "segmentation" else "2D Lesion Segmentation",
            "inference_time_ms": total_time_ms,
            "inference_timestamp": now_iso,
            "is_mock": is_demo,
        }

        response = AnalysisResponse(
            id=analysis_id,
            analysis_id=analysis_id,
            fileName=filename,
            imageDataUrl=orig_upload["url"],
            status="completed",
            analysis_mode=mode,
            model_id=chosen_actual_model["model_id"],
            model_name=chosen_actual_model["model_name"],
            model_version=chosen_actual_model["model_version"],
            actual_model=chosen_actual_model,
            classification=classification_res["scores_list"] if classification_res else None,
            predictedClass=predicted_class,
            confidence=confidence,
            explanations=explanations_list,
            segmentation=segmentation_res["result_item"] if segmentation_res else None,
            metadata=metadata_item,
            isDemo=is_demo,
            createdAt=now_iso,
            input_image=input_image_data,
            classification_details=det_class,
            segmentation_details=det_seg,
            explainability_details=det_xai,
            disclaimer="Research use only. Not a medical diagnosis. Classification candidate mapping is provisional until matched against training code metadata."
        )

        # 6. MongoDB Persistence
        db_doc = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "original_filename": filename,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "status": "completed",
            "analysis_mode": mode,
            "model_id": chosen_actual_model["model_id"],
            "model_name": chosen_actual_model["model_name"],
            "model_version": chosen_actual_model["model_version"],
            "actual_model": chosen_actual_model,
            "input_image": input_image_data.model_dump(),
            "classification": det_class.model_dump() if det_class else None,
            "segmentation": det_seg.model_dump() if det_seg else None,
            "explainability": [e.model_dump() for e in explanations_list],
            "configuration": config.model_dump(),
            "artifacts": {
                "input_image": orig_upload,
                "explanations": [e.model_dump() for e in explanations_list],
                "segmentation": det_seg.model_dump() if det_seg else None
            },
            "processing_metadata": metadata_item.model_dump(),
            "is_demo": is_demo,
            "full_response": response.model_dump()
        }

        db = get_database()
        if db is not None:
            try:
                await db.analyses.insert_one(db_doc)
                logger.info(f"Persisted analysis {analysis_id} in MongoDB Atlas.")
            except Exception as e:
                logger.warning(f"Failed to persist in MongoDB: {e}. Caching in memory.")
                _in_memory_analyses[analysis_id] = db_doc
        else:
            _in_memory_analyses[analysis_id] = db_doc

        return response

    async def get_analysis(self, analysis_id: str, user_id: str, is_admin: bool = False) -> Optional[AnalysisResponse]:
        """Retrieve analysis record with user authorization check."""
        db = get_database()
        doc = None
        if db is not None:
            try:
                doc = await db.analyses.find_one({"analysis_id": analysis_id})
            except Exception as e:
                logger.warning(f"MongoDB query failed: {e}")

        if not doc and analysis_id in _in_memory_analyses:
            doc = _in_memory_analyses[analysis_id]

        if not doc:
            return None

        # Ownership check: allow access if owner, if legacy guest record, or if admin
        doc_user = doc.get("user_id")
        if doc_user != user_id and doc_user != "guest_researcher" and not is_admin:
            raise PermissionDeniedException("You do not have permission to view this analysis.")

        if "full_response" in doc:
            resp_dict = dict(doc["full_response"])
            if "analysis_id" not in resp_dict:
                resp_dict["analysis_id"] = doc.get("analysis_id", analysis_id)
            if "id" not in resp_dict:
                resp_dict["id"] = doc.get("analysis_id", analysis_id)
            if not resp_dict.get("actual_model"):
                mod_id = doc.get("model_id") or resp_dict.get("model_id") or "efficientnetv2b0"
                mod_name = doc.get("model_name") or resp_dict.get("model_name") or (resp_dict.get("metadata") or {}).get("modelName") or "EfficientNetV2B0"
                mod_ver = doc.get("model_version") or resp_dict.get("model_version") or (resp_dict.get("metadata") or {}).get("modelVersion") or "v2.4.0"
                wf_type = doc.get("workflow_type") or resp_dict.get("workflow_type", "2D_CLASSIFICATION")
                resp_dict["actual_model"] = {
                    "model_id": mod_id,
                    "model_name": mod_name,
                    "model_version": mod_ver,
                    "architecture": (doc.get("model_info") or {}).get("architecture") or ("Custom 3D Residual U-Net" if wf_type == "3D_SEGMENTATION" else "EfficientNetV2-B0 (Fused-MBConv)"),
                    "framework": "TensorFlow 2.18 / Keras 3",
                    "task_type": "3D Multimodal Volumetric Segmentation" if wf_type == "3D_SEGMENTATION" else "2D Brain Tumor Classification",
                    "inference_time_ms": (resp_dict.get("metadata") or {}).get("processingTimeMs", 0),
                    "inference_timestamp": resp_dict.get("createdAt"),
                }
            return AnalysisResponse(**resp_dict)
        return None

    async def save_segmentation_analysis(
        self,
        user_id: str,
        inference_result: Dict[str, Any],
        modality_filenames: Dict[str, str],
        model_id: str = "brats_3d_unet",
        model_name: str = "Attention CNN + Vision Transformer",
        model_version: str = "v2.4.0",
        patient_id: Optional[str] = None,
        actual_model: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persists a complete 3D Brain Tumor Volumetric Segmentation analysis to MongoDB.
        """
        analysis_id = f"ANL-{uuid.uuid4().hex[:8].upper()}"
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        # Primary filename / Patient Reference
        flair_fn = modality_filenames.get("flair", "flair.nii")
        patient_ref = patient_id or (flair_fn.split(".")[0].rsplit("_", 1)[0] if "_" in flair_fn else flair_fn.split(".")[0])
        original_filename = f"{patient_ref} [4 Modalities: FLAIR, T1, T1CE, T2]"

        dims = inference_result.get("volume_dimensions", [240, 240, 155])
        class_dist = inference_result.get("class_distribution", [])

        # Calculate per-class tumor volume in cm³ and mm³
        total_tumor_vol_mm3 = 0.0
        bg_pct = 0.0
        classes_with_cm3 = []
        for c in class_dist:
            v_mm3 = float(c.get("volume_mm3", 0.0))
            v_cm3 = round(v_mm3 / 1000.0, 3)
            c_copy = dict(c)
            c_copy["volume_cm3"] = v_cm3
            classes_with_cm3.append(c_copy)
            if c.get("index") == 0:
                bg_pct = float(c.get("percentage", 0.0))
            else:
                total_tumor_vol_mm3 += v_mm3

        total_tumor_vol_cm3 = round(total_tumor_vol_mm3 / 1000.0, 3)

        # Slice previews
        slices = inference_result.get("slices", {})
        axial_thumb = ""
        if isinstance(slices, dict) and "axial" in slices:
            axial_thumb = slices.get("axial", {}).get("preview", "")

        pred_id = inference_result.get("prediction_id", "")
        download_url = inference_result.get("download_url") or f"/api/segmentation/download/{pred_id}"
        nifti_ref = f"brats_3d_segmentation_{pred_id[:8]}.nii.gz"

        eval_metrics = inference_result.get("evaluation_metrics")
        proc_time_ms = inference_result.get("processing_time_ms", 0)
        zooms = inference_result.get("voxel_spacing", [1.0, 1.0, 1.0])

        resolved_actual_model = actual_model or inference_result.get("actual_model") or {
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "architecture": "Hybrid Attention CNN + Vision Transformer (Additive Attention Gates + Transformer Bottleneck)" if "Attention" in model_name else "Custom 3D Residual U-Net",
            "framework": "TensorFlow 2.18 / Keras 3",
            "task_type": "3D Multimodal Volumetric Segmentation",
            "input_modalities": ["FLAIR", "T1", "T1CE", "T2"],
            "input_dimensions": dims,
            "voxel_spacing": zooms,
            "parameters_count": "3.07M params" if "Attention" in model_name else "16.4M params",
            "preprocessing": "Non-zero z-score normalization across FLAIR, T1, T1CE, T2",
            "inference_time_ms": proc_time_ms,
            "inference_timestamp": now_iso,
            "is_mock": False,
        }

        # Build Full Response payload
        full_response = {
            "id": analysis_id,
            "analysis_id": analysis_id,
            "prediction_id": pred_id,
            "fileName": original_filename,
            "patient_id": patient_ref,
            "imageDataUrl": axial_thumb,
            "status": "completed",
            "analysis_mode": "segmentation_3d",
            "workflow_type": "3D_SEGMENTATION",
            "predictedClass": "Multimodal Brain Tumor (Segmented)",
            "confidence": round(float(eval_metrics.get("whole_tumor_dice", 0.8937)), 4) if eval_metrics else round(1.0 - (bg_pct / 100.0), 4),
            "createdAt": now_iso,
            "volume_dimensions": dims,
            "voxel_spacing": zooms,
            "anatomical_orientation": inference_result.get("anatomical_orientation", ["R", "A", "S"]),
            "class_distribution": classes_with_cm3,
            "slices": slices,
            "surface_3d": inference_result.get("surface_3d", {}),
            "download_url": download_url,
            "evaluation_metrics": eval_metrics,
            "modalities": list(modality_filenames.keys()),
            "model_name": model_name,
            "model_id": model_id,
            "model_version": model_version,
            "actual_model": resolved_actual_model,
            "metadata": {
                "modelName": model_name,
                "modelVersion": model_version,
                "processingTimeMs": proc_time_ms,
                "imageSize": {"width": dims[0], "height": dims[1], "depth": dims[2] if len(dims) > 2 else 1},
                "analysisDate": now_iso,
            },
            "explainability": inference_result.get("explainability", {}),
            "disclaimer": inference_result.get("disclaimer", "Research use only. Not a medical diagnosis."),
        }

        # MongoDB Document Schema
        db_doc = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "patient_id": patient_ref,
            "original_filename": original_filename,
            "created_at": now_dt,
            "updated_at": now_dt,
            "status": "completed",
            "analysis_mode": "segmentation_3d",
            "workflow_type": "3D_SEGMENTATION",
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "actual_model": resolved_actual_model,
            "input_scan": {
                "modalities": modality_filenames,
                "dimensions": dims,
                "voxel_spacing": zooms,
                "anatomical_orientation": inference_result.get("anatomical_orientation"),
                "preprocessing": "Non-zero z-score normalization across FLAIR, T1, T1CE, T2",
            },
            "model_info": resolved_actual_model,
            "segmentation": {
                "model_name": model_name,
                "model_id": model_id,
                "model_version": model_version,
                "actual_model": resolved_actual_model,
                "dimensions": dims,
                "classes": classes_with_cm3,
                "total_tumor_volume_mm3": total_tumor_vol_mm3,
                "total_tumor_volume_cm3": total_tumor_vol_cm3,
                "background_percentage": bg_pct,
                "prediction_id": pred_id,
                "download_url": download_url,
                "nifti_filename": nifti_ref,
                "evaluation_metrics": eval_metrics,
                "slices": slices,
                "surface_3d": inference_result.get("surface_3d", {}),
            },
            "explainability": inference_result.get("explainability", {}),
            "processing_metadata": {
                "modelName": model_name,
                "modelVersion": model_version,
                "processingTimeMs": proc_time_ms,
                "dimensions": dims,
                "analysisDate": now_iso,
            },
            "is_demo": False,
            "full_response": full_response,
        }

        db = get_database()
        if db is not None:
            try:
                await db.analyses.insert_one(db_doc)
                logger.info(f"Persisted 3D Segmentation analysis {analysis_id} to MongoDB.")
            except Exception as e:
                logger.warning(f"Failed to persist 3D analysis in MongoDB: {e}. Falling back to memory.")
                _in_memory_analyses[analysis_id] = db_doc
        else:
            _in_memory_analyses[analysis_id] = db_doc

        return analysis_id

    async def save_classification_analysis(
        self,
        user_id: str,
        inference_result: Dict[str, Any],
        filename: str,
        model_id: str = "efficientnetv2b0",
        model_name: str = "EfficientNetV2B0",
        model_version: str = "v2.4.0",
        actual_model: Optional[Dict[str, Any]] = None,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """
        Persists a 2D Brain MRI Classification analysis to MongoDB and persistent storage.
        """
        from app.services.storage_service import storage_service

        analysis_id = f"ANL-{uuid.uuid4().hex[:8].upper()}"
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        proc_time_ms = inference_result.get("processing_time_ms", 0)
        top_pred = inference_result.get("top_prediction", "Glioma")
        conf = inference_result.get("confidence", 0.0)
        scores = inference_result.get("scores", [])
        gradcam = inference_result.get("gradcam_overlay")
        gradcam_pp_url = inference_result.get("gradcam_pp_overlay")
        seg_mask_url = inference_result.get("segmentation_mask_url")
        seg_overlay_url = inference_result.get("segmentation_overlay_url")
        seg_model_used = inference_result.get("segmentation_model_used")
        seg_tumor_detected = inference_result.get("segmentation_tumor_detected")
        seg_tumor_area_pct = inference_result.get("segmentation_tumor_area_pct")
        explanations_list = inference_result.get("explanations", [])

        # Persistent storage for original image
        image_url = inference_result.get("image_url")
        if image_bytes and not image_url:
            stored_upload = storage_service.store_file("uploads", image_bytes, filename)
            image_url = stored_upload["url"]

        # If gradcam is a base64 string, persist to disk as PNG
        gradcam_url = gradcam
        if gradcam and isinstance(gradcam, str) and gradcam.startswith("data:image"):
            try:
                import base64
                header, b64_data = gradcam.split(",", 1)
                heatmap_bytes = base64.b64decode(b64_data)
                stored_hm = storage_service.store_file("heatmaps", heatmap_bytes, f"gradcam_{analysis_id}.png")
                gradcam_url = stored_hm["url"]
            except Exception as e:
                logger.warning("Could not persist heatmap to disk: %s", e)

        resolved_actual_model = actual_model or inference_result.get("actual_model") or {
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "architecture": "Hierarchical Shifted-Window Transformer (Patch Merging + Shifted Windows)" if model_id == "swin_transformer" else "EfficientNetV2-B0 (Fused-MBConv + Squeeze-and-Excitation)",
            "framework": "TensorFlow 2.18 / Keras 3",
            "task_type": "2D Brain Tumor Classification",
            "input_resolution": "224 × 224 × 3 RGB" if model_id == "swin_transformer" else "128 × 128 × 3 RGB",
            "parameters_count": "28.3M params" if model_id == "swin_transformer" else "7.11M params",
            "preprocessing": "Bicubic Interpolation + /255.0 Normalization" if model_id == "swin_transformer" else "CLAHE Adaptive Histogram Equalization + /255.0 Normalization",
            "inference_time_ms": proc_time_ms,
            "inference_timestamp": now_iso,
            "is_mock": False,
        }

        full_response = {
            "id": analysis_id,
            "analysis_id": analysis_id,
            "prediction_id": inference_result.get("prediction_id", str(uuid.uuid4())),
            "fileName": filename,
            "status": "completed",
            "analysis_mode": "classification_2d",
            "workflow_type": "2D_CLASSIFICATION",
            "predictedClass": top_pred,
            "confidence": conf,
            "createdAt": now_iso,
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "actual_model": resolved_actual_model,
            "top_prediction": top_pred,
            "scores": scores,
            "image_url": image_url,
            "imageDataUrl": image_url or gradcam_url,
            "thumbnailUrl": image_url or gradcam_url,
            "gradcam_overlay": gradcam_url,
            "gradcam_pp_overlay": gradcam_pp_url,
            "segmentation_mask_url": seg_mask_url,
            "segmentation_overlay_url": seg_overlay_url,
            "segmentation_model_used": seg_model_used,
            "segmentation_tumor_detected": seg_tumor_detected,
            "segmentation_tumor_area_pct": seg_tumor_area_pct,
            "explanations": explanations_list,
            "metadata": {
                "modelName": model_name,
                "modelVersion": model_version,
                "processingTimeMs": proc_time_ms,
                "imageSize": {"width": 256, "height": 256},
                "analysisDate": now_iso,
            },
            "disclaimer": inference_result.get("disclaimer", "Research use only. Not a medical diagnosis."),
        }

        db_doc = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "original_filename": filename,
            "created_at": now_dt,
            "updated_at": now_dt,
            "status": "completed",
            "analysis_mode": "classification_2d",
            "workflow_type": "2D_CLASSIFICATION",
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "actual_model": resolved_actual_model,
            "input_image": {
                "filename": filename,
                "url": image_url,
            },
            "classification": {
                "model_id": model_id,
                "model_name": model_name,
                "model_version": model_version,
                "actual_model": resolved_actual_model,
                "top_prediction": top_pred,
                "confidence": conf,
                "scores": scores,
                "gradcam_overlay": gradcam_url,
                "gradcam_pp_overlay": gradcam_pp_url,
            },
            "explainability": explanations_list,
            "processing_metadata": {
                "modelName": model_name,
                "modelVersion": model_version,
                "processingTimeMs": proc_time_ms,
                "analysisDate": now_iso,
            },
            "is_demo": False,
            "full_response": full_response,
        }

        db = get_database()
        if db is not None:
            try:
                await db.analyses.insert_one(db_doc)
                logger.info(f"Persisted 2D Classification analysis {analysis_id} to MongoDB.")
            except Exception as e:
                logger.warning(f"Failed to persist 2D classification in MongoDB: {e}. Falling back to memory.")
                _in_memory_analyses[analysis_id] = db_doc
        else:
            _in_memory_analyses[analysis_id] = db_doc

        return analysis_id

    async def save_combined_analysis(
        self,
        user_id: str,
        classification_id: Optional[str],
        classification_result: Optional[Dict[str, Any]],
        segmentation_id: Optional[str],
        segmentation_result: Optional[Dict[str, Any]],
        classification_error: Optional[str] = None,
        segmentation_error: Optional[str] = None,
    ) -> str:
        """
        Persists a combined (2D Classification + 3D Volumetric Segmentation) analysis session to MongoDB.
        """
        analysis_id = f"ANL-COMB-{uuid.uuid4().hex[:8].upper()}"
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        status_str = "completed" if (classification_id and segmentation_id) else "partial" if (classification_id or segmentation_id) else "failed"

        # Extract thumbnails and findings
        thumb = ""
        pred_class = "Combined NeuroScan Study"
        conf = 0.0
        if classification_result:
            thumb = classification_result.get("image_url") or classification_result.get("thumbnailUrl", "")
            pred_class = classification_result.get("top_prediction") or classification_result.get("predictedClass", "Glioma")
            conf = float(classification_result.get("confidence", 0.0))
        elif segmentation_result:
            slices = segmentation_result.get("slices", {})
            if isinstance(slices, dict) and "axial" in slices:
                thumb = slices.get("axial", {}).get("preview", "")
            pred_class = "3D Multimodal Segmentation"

        full_response = {
            "id": analysis_id,
            "analysis_id": analysis_id,
            "combined_id": analysis_id,
            "fileName": "Combined 2D + 3D Multimodal Study",
            "status": status_str,
            "analysis_mode": "combined",
            "workflow_type": "COMBINED_ANALYSIS",
            "predictedClass": pred_class,
            "confidence": conf,
            "createdAt": now_iso,
            "classification_id": classification_id,
            "segmentation_id": segmentation_id,
            "classification": classification_result,
            "segmentation": segmentation_result,
            "errors": {
                "classification": classification_error,
                "segmentation": segmentation_error,
            },
            "thumbnailUrl": thumb,
            "imageDataUrl": thumb,
            "metadata": {
                "modelName": "Combined Ensemble (EfficientNetV2B0 + Attention-ViT)",
                "modelVersion": "v2.4.0",
                "analysisDate": now_iso,
            },
            "disclaimer": "Research use only. Not a medical diagnosis.",
        }

        db_doc = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "original_filename": "Combined 2D + 3D Multimodal Study",
            "created_at": now_dt,
            "updated_at": now_dt,
            "status": status_str,
            "analysis_mode": "combined",
            "workflow_type": "COMBINED_ANALYSIS",
            "classification_id": classification_id,
            "segmentation_id": segmentation_id,
            "classification": classification_result,
            "segmentation": segmentation_result,
            "errors": {
                "classification": classification_error,
                "segmentation": segmentation_error,
            },
            "is_demo": False,
            "full_response": full_response,
        }

        db = get_database()
        if db is not None:
            try:
                await db.analyses.insert_one(db_doc)
                logger.info(f"Persisted Combined analysis {analysis_id} to MongoDB.")
            except Exception as e:
                logger.warning(f"Failed to persist Combined analysis in MongoDB: {e}. Falling back to memory.")
                _in_memory_analyses[analysis_id] = db_doc
        else:
            _in_memory_analyses[analysis_id] = db_doc

        return analysis_id

    async def get_user_history(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        predicted_class: Optional[str] = None,
        search: Optional[str] = None,
        sort_order: str = "desc"
    ) -> Tuple[List[AnalysisSummaryResponse], int]:
        """Retrieve paginated analysis history for authenticated user."""
        query: Dict[str, Any] = {"user_id": user_id}
        if status:
            query["status"] = status
        if predicted_class:
            query["classification.predicted_class"] = predicted_class
        if search:
            query["original_filename"] = {"$regex": search, "$options": "i"}

        items: List[AnalysisSummaryResponse] = []
        total = 0

        db = get_database()
        if db is not None:
            try:
                total = await db.analyses.count_documents(query)
                cursor = db.analyses.find(query).sort("created_at", -1 if sort_order == "desc" else 1)
                cursor.skip((page - 1) * page_size).limit(page_size)
                async for doc in cursor:
                    resp_dict = doc.get("full_response", {})
                    slices = resp_dict.get("slices") or (doc.get("segmentation") or {}).get("slices", {})
                    thumb = resp_dict.get("imageDataUrl") or (doc.get("input_image") or {}).get("url", "")
                    if not thumb and isinstance(slices, dict):
                        thumb = slices.get("axial", {}).get("preview", "")

                    pred_class = (
                        resp_dict.get("predictedClass") or
                        (doc.get("classification") or {}).get("top_prediction") or
                        (doc.get("classification") or {}).get("predicted_class") or
                        "Multimodal Brain Tumor"
                    )
                    conf = resp_dict.get("confidence")
                    if conf is None:
                        conf = (doc.get("classification") or {}).get("confidence", 0.0)

                    actual_mod = resp_dict.get("actual_model") or doc.get("actual_model")
                    mod_id = (actual_mod or {}).get("model_id") or doc.get("model_id") or resp_dict.get("model_id")
                    mod_name = (actual_mod or {}).get("model_name") or doc.get("model_name") or resp_dict.get("model_name")

                    model_ver = (
                        (actual_mod or {}).get("model_name") or
                        (resp_dict.get("metadata") or {}).get("modelName") or
                        (doc.get("processing_metadata") or {}).get("modelName") or
                        (doc.get("model_info") or {}).get("name") or
                        mod_name or
                        "v2.4.0"
                    )

                    wf_type = doc.get("workflow_type") or resp_dict.get("workflow_type")
                    if not wf_type:
                        mode_check = doc.get("analysis_mode") or resp_dict.get("analysis_mode", "")
                        wf_type = "3D_SEGMENTATION" if (mode_check in ["segmentation_3d", "segmentation"] or "surface_3d" in resp_dict) else "2D_CLASSIFICATION"

                    mode_val = doc.get("analysis_mode") or resp_dict.get("analysis_mode") or ("segmentation_3d" if wf_type == "3D_SEGMENTATION" else "classification_2d")

                    items.append(
                        AnalysisSummaryResponse(
                            id=doc.get("analysis_id", str(doc.get("_id"))),
                            fileName=doc.get("original_filename", "mri_scan.nii"),
                            thumbnailUrl=thumb,
                            predictedClass=pred_class,
                            confidence=conf,
                            status=doc.get("status", "completed"),
                            modelVersion=model_ver,
                            createdAt=resp_dict.get("createdAt") or doc.get("created_at", datetime.now(timezone.utc)).isoformat(),
                            isDemo=doc.get("is_demo", False),
                            workflow_type=wf_type,
                            analysis_mode=mode_val,
                            model_id=mod_id,
                            model_name=mod_name,
                            actual_model=actual_mod,
                        )
                    )
                return items, total
            except Exception as e:
                logger.warning(f"MongoDB history retrieval failed: {e}. Falling back to memory.")

        # In-memory history fallback
        filtered = [
            v for v in _in_memory_analyses.values()
            if v.get("user_id") == user_id
        ]
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        for doc in filtered[start:end]:
            resp_dict = doc.get("full_response", {})
            slices = resp_dict.get("slices") or (doc.get("segmentation") or {}).get("slices", {})
            thumb = resp_dict.get("imageDataUrl", "")
            if not thumb and isinstance(slices, dict):
                thumb = slices.get("axial", {}).get("preview", "")

            pred_class = (
                resp_dict.get("predictedClass") or
                (doc.get("classification") or {}).get("top_prediction") or
                "Multimodal Brain Tumor"
            )
            conf = resp_dict.get("confidence")
            if conf is None:
                conf = (doc.get("classification") or {}).get("confidence", 0.0)

            actual_mod = resp_dict.get("actual_model") or doc.get("actual_model")
            mod_id = (actual_mod or {}).get("model_id") or doc.get("model_id") or resp_dict.get("model_id")
            mod_name = (actual_mod or {}).get("model_name") or doc.get("model_name") or resp_dict.get("model_name")

            model_ver = (
                (actual_mod or {}).get("model_name") or
                (resp_dict.get("metadata") or {}).get("modelName") or
                (doc.get("processing_metadata") or {}).get("modelName") or
                mod_name or
                "v2.4.0"
            )

            wf_type = doc.get("workflow_type") or resp_dict.get("workflow_type")
            if not wf_type:
                mode_check = doc.get("analysis_mode") or resp_dict.get("analysis_mode", "")
                wf_type = "3D_SEGMENTATION" if (mode_check in ["segmentation_3d", "segmentation"] or "surface_3d" in resp_dict) else "2D_CLASSIFICATION"

            mode_val = doc.get("analysis_mode") or resp_dict.get("analysis_mode") or ("segmentation_3d" if wf_type == "3D_SEGMENTATION" else "classification_2d")

            items.append(
                AnalysisSummaryResponse(
                    id=doc.get("analysis_id"),
                    fileName=doc.get("original_filename"),
                    thumbnailUrl=thumb,
                    predictedClass=pred_class,
                    confidence=conf,
                    status=doc.get("status", "completed"),
                    modelVersion=model_ver,
                    createdAt=resp_dict.get("createdAt", datetime.now(timezone.utc).isoformat()),
                    isDemo=doc.get("is_demo", False),
                    workflow_type=wf_type,
                    analysis_mode=mode_val,
                    model_id=mod_id,
                    model_name=mod_name,
                    actual_model=actual_mod,
                )
            )

        return items, total

    async def delete_analysis(self, analysis_id: str, user_id: str, is_admin: bool = False) -> bool:
        """Delete analysis and associated artifacts."""
        db = get_database()
        doc = None
        if db is not None:
            try:
                doc = await db.analyses.find_one({"analysis_id": analysis_id})
            except Exception as e:
                logger.warning(f"MongoDB query failed: {e}")

        if not doc and analysis_id in _in_memory_analyses:
            doc = _in_memory_analyses[analysis_id]

        if not doc:
            raise ResourceNotFoundException("Analysis", analysis_id)

        # Ownership check: allow deleting if owner, if legacy guest record, or if admin
        doc_user = doc.get("user_id")
        if doc_user != user_id and doc_user != "guest_researcher" and not is_admin:
            raise PermissionDeniedException("You do not have permission to delete this analysis.")

        # Cleanup ImageKit files
        input_image = doc.get("input_image", {})
        file_id = input_image.get("file_id") if isinstance(input_image, dict) else getattr(input_image, "file_id", None)
        if file_id:
            try:
                await imagekit_service.delete_file(file_id)
            except Exception as e:
                logger.warning(f"Failed to delete ImageKit file {file_id}: {e}")

        if db is not None:
            try:
                await db.analyses.delete_one({"analysis_id": analysis_id})
                logger.info(f"Deleted analysis record {analysis_id} from MongoDB Atlas.")
            except Exception as e:
                logger.error(f"MongoDB deletion error: {e}")

        if analysis_id in _in_memory_analyses:
            del _in_memory_analyses[analysis_id]

        return True


analysis_service = AnalysisService()

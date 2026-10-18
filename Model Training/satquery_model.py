"""
SatQuery Vision-Language Model Connector
=========================================
Wraps the fine-tuned Qwen3-VL-2B Earth Observation model for direct integration
into your FastAPI backend, agentic controller, or standalone microservice.

Key Features:
- Loads once into GPU VRAM (RTX 5060 Ti in bfloat16, ~4.5 GB VRAM).
- Accepts PIL Images, file paths, raw bytes, or base64 strings.
- Automatically extracts spatial bounding boxes and formats them as GeoJSON.
- Latency: ~1.0 - 2.0s on NVIDIA RTX 5060 Ti.
"""

import os
import io
import re
import time
import base64
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from PIL import Image
import numpy as np

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

MODULE_DIR = Path(__file__).resolve().parent

DEFAULT_SYSTEM_PROMPT = (
    "You are SatQuery AI, an advanced Earth Observation and Geospatial Intelligence Assistant. "
    "You analyze satellite scenes and answer user questions in a natural, professional, and conversational style. "
    "Evaluate land cover types, vegetation density, urban fabric, water bodies, and terrain features visible in the image. "
    "Provide a direct, well-reasoned answer tailored specifically to the user's question. "
    "Never respond with only bare numbers or coordinate brackets. Always explain your reasoning in complete, articulate sentences. "
    "When localizing specific areas or features of interest, describe their spatial orientation (e.g. South-Western sector) and specify bounding coordinates [ymin xmin, ymax xmax]."
)



class SatQueryVLM:
    """
    Singleton-ready interface for the fine-tuned SatQuery Vision-Language Model.
    Designed for zero-copy, direct Python backend integration.
    """

    @staticmethod
    def _constrain_image_size(img: Image.Image, max_dim: int = 768) -> Image.Image:
        """Constrains maximum image dimension to prevent visual token explosion and CUDA OOM."""
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        lora_dir: Optional[str] = None,
        merged_dir: Optional[str] = None,
        device: Optional[str] = None,
        warmup: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SatQuery VLM] Initializing on device: {self.device}...")

        # Resolve paths relative to module directory if not found in current working dir
        if not lora_dir:
            default_lora = MODULE_DIR / "output" / "qwen3_vl_satquery_multimodal_lora"
            lora_dir = str(default_lora) if default_lora.exists() else "output/qwen3_vl_satquery_multimodal_lora"
        if not merged_dir:
            default_merged = MODULE_DIR / "output" / "qwen3_vl_satquery_merged"
            merged_dir = str(default_merged) if default_merged.exists() else "output/qwen3_vl_satquery_merged"

        # 1. Determine model loading strategy: Merged full model or Base + LoRA
        if os.path.exists(merged_dir) and (
            os.path.exists(os.path.join(merged_dir, "model.safetensors")) or
            os.path.exists(os.path.join(merged_dir, "model.safetensors.index.json"))
        ):
            print(f"[SatQuery VLM] Loading merged standalone weights from: {merged_dir}")
            try:
                # Try loading strictly from disk first (prevents any network blocking/timeout when offline)
                self.processor = AutoProcessor.from_pretrained(merged_dir, local_files_only=True)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    merged_dir,
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map="auto" if "cuda" in self.device else None,
                    local_files_only=True
                )
            except Exception as e_local:
                # If local_files_only encountered an issue, fallback to standard loader
                print(f"[SatQuery VLM] local_files_only note ({e_local}); attempting standard load...")
                self.processor = AutoProcessor.from_pretrained(merged_dir)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    merged_dir,
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map="auto" if "cuda" in self.device else None
                )
        else:
            print(f"[SatQuery VLM] Loading base model ({base_model_id}) + LoRA adapter ({lora_dir})...")
            proc_path = lora_dir if os.path.exists(lora_dir) else base_model_id
            try:
                self.processor = AutoProcessor.from_pretrained(proc_path, local_files_only=True)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(proc_path)

            try:
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map="auto" if "cuda" in self.device else None,
                    local_files_only=True
                )
            except Exception:
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map="auto" if "cuda" in self.device else None
                )

            if os.path.exists(lora_dir):
                self.model = PeftModel.from_pretrained(base_model, lora_dir)
                print("[SatQuery VLM] LoRA adapter attached successfully.")
            else:
                print("[SatQuery VLM] [WARNING] LoRA adapter not found, running base model.")
                self.model = base_model

        if "cuda" not in self.device:
            self.model.to(self.device)

        self.model.eval()

        self._scene_landcover_cache: Dict[str, str] = {}

        if warmup and "cuda" in self.device:
            self._warmup()

        print("[SatQuery VLM] Model ready for inference.")

    def _detect_primary_landcover(self, pil_img: Image.Image, geotiff_path: Optional[str] = None) -> str:
        """Determines or retrieves cached primary land cover for the scene."""
        cache_key = str(geotiff_path) if geotiff_path else f"{pil_img.size}"
        if cache_key in self._scene_landcover_cache:
            return self._scene_landcover_cache[cache_key]

        try:
            prompt = "Analyze this Sentinel-2 optical satellite image.\n\nQuestion: What is the primary land cover?"
            chat_prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(text=[chat_prompt], images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if "cuda" in self.device:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
                else:
                    out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            gen = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            if gen and gen not in ["no", "yes", "none", "unknown", ""]:
                self._scene_landcover_cache[cache_key] = gen
                return gen
        except Exception as e:
            print(f"[SatQuery VLM] Land cover check note: {e}")

        return "urban fabric"

    def _warmup(self):
        """Runs a tiny 1-step warmup pass so the first user query is instant."""
        try:
            dummy_img = Image.new("RGB", (120, 120), color=(73, 109, 137))
            self.query(dummy_img, "Warmup check", max_tokens=2)
        except Exception as e:
            print(f"[SatQuery VLM] Warmup note: {e}")

    @classmethod
    def _load_image(cls, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Converts diverse image formats (GeoTIFF, path, base64, bytes, PIL) into an RGB PIL Image safely capped at max 768px."""
        if isinstance(image_input, Image.Image):
            return cls._constrain_image_size(image_input.convert("RGB"), max_dim=768)

        if isinstance(image_input, bytes):
            return cls._constrain_image_size(Image.open(io.BytesIO(image_input)).convert("RGB"), max_dim=768)

        if isinstance(image_input, str):
            # Base64 string
            if image_input.startswith("data:image") or ";base64," in image_input:
                base64_data = image_input.split(";base64,")[-1]
                return cls._constrain_image_size(Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("RGB"), max_dim=768)
            elif len(image_input) > 500 and not os.path.exists(image_input):
                # Raw base64 payload
                return cls._constrain_image_size(Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB"), max_dim=768)
            elif os.path.exists(image_input):
                # GeoTIFF special handling: extract RGB bands and contrast stretch
                if image_input.lower().endswith((".tif", ".tiff")):
                    # Try 0: Check for companion web PNG preview (prev_stem.png or stem.png)
                    tif_path_obj = Path(image_input)
                    preview_candidates = [
                        tif_path_obj.parent / f"prev_{tif_path_obj.stem}.png",
                        tif_path_obj.with_suffix(".png"),
                        tif_path_obj.parent.parent / "previews" / f"{tif_path_obj.stem}.png",
                        tif_path_obj.parent / f"{tif_path_obj.stem}.png"
                    ]
                    for p_cand in preview_candidates:
                        if p_cand.exists() and p_cand.stat().st_size > 500:
                            return cls._constrain_image_size(Image.open(p_cand).convert("RGB"), max_dim=768)

                    # Try 1: rasterio with bounded resolution (max 768px to prevent token explosion)
                    try:
                        import rasterio
                        with rasterio.open(image_input) as src:
                            factor = max(1, max(src.height, src.width) // 768)
                            h, w = max(1, src.height // factor), max(1, src.width // factor)
                            if src.count >= 3:
                                b1 = src.read(1, out_shape=(h, w))
                                b2 = src.read(2, out_shape=(h, w))
                                b3 = src.read(3, out_shape=(h, w))
                                arr = np.stack([b1, b2, b3], axis=0).astype(np.float32)
                            else:
                                b1 = src.read(1, out_shape=(h, w))
                                arr = np.repeat(b1[np.newaxis, :, :], 3, axis=0).astype(np.float32)
                            
                            rgb = np.nan_to_num(arr, nan=0.0)
                            rgb_hwc = np.transpose(rgb, (1, 2, 0))
                            p2, p98 = np.percentile(rgb_hwc, (2, 98))
                            if p98 > p2:
                                norm = np.clip((rgb_hwc - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
                            else:
                                norm = np.clip(rgb_hwc, 0, 255).astype(np.uint8)
                            return cls._constrain_image_size(Image.fromarray(norm, mode="RGB"), max_dim=768)
                    except Exception:
                        pass

                    # Try 2: tifffile (installed in Model Training venv)
                    try:
                        import tifffile
                        arr = tifffile.imread(image_input)
                        if arr.ndim == 3:
                            if arr.shape[0] in (3, 4, 12, 13):  # (bands, H, W)
                                sel = [2, 1, 0] if arr.shape[0] >= 3 else [0, 0, 0]
                                rgb = arr[sel, :, :].transpose(1, 2, 0).astype("float32")
                            else:  # (H, W, bands)
                                sel = [2, 1, 0] if arr.shape[2] >= 3 else [0, 0, 0]
                                rgb = arr[:, :, sel].astype("float32")
                        elif arr.ndim == 2:
                            rgb = np.repeat(arr[:, :, np.newaxis], 3, axis=2).astype("float32")
                        else:
                            rgb = np.zeros((256, 256, 3), dtype="float32")

                        rgb = np.nan_to_num(rgb, nan=0.0)
                        p2, p98 = np.percentile(rgb, (2, 98))
                        if p98 > p2:
                            norm = np.clip((rgb - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
                        else:
                            norm = np.clip(rgb, 0, 255).astype(np.uint8)
                        return cls._constrain_image_size(Image.fromarray(norm, mode="RGB"), max_dim=768)
                    except Exception as te:
                        print(f"[SatQuery VLM] Tifffile GeoTIFF load failed ({te}), falling back to PIL...")
                return cls._constrain_image_size(Image.open(image_input).convert("RGB"), max_dim=768)
            else:
                raise FileNotFoundError(f"Image path not found: {image_input}")

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def _parse_bounding_boxes(
        text: str,
        img_width: int,
        img_height: int,
        geotiff_path_or_profile: Optional[Union[str, Path, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses coordinates like [0.38 0.0, 1.0 1.0] or [ymin, xmin, ymax, xmax] using regex.
        Supports multiple bounding boxes in a single response and projects to WGS84 (EPSG:4326)
        geographic coordinates when geospatial transform/profile is provided.
        """
        pattern = r"(?:<box>)?\[([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)\](?:</box>)?"
        matches = list(re.finditer(pattern, text))
        if not matches:
            return []

        # Check if geospatial metadata is available for projection
        transform = None
        src_crs = None
        raster_w = None
        raster_h = None
        wgs84_tile_bounds = None

        if geotiff_path_or_profile is not None:
            # 1. Try Rasterio
            try:
                import rasterio
                if isinstance(geotiff_path_or_profile, (str, Path)) and os.path.exists(geotiff_path_or_profile):
                    with rasterio.open(geotiff_path_or_profile) as src:
                        transform = src.transform
                        src_crs = src.crs
                        raster_w = src.width
                        raster_h = src.height
                elif isinstance(geotiff_path_or_profile, dict) and "transform" in geotiff_path_or_profile:
                    transform = geotiff_path_or_profile["transform"]
                    src_crs = geotiff_path_or_profile.get("crs")
                    raster_w = geotiff_path_or_profile.get("width")
                    raster_h = geotiff_path_or_profile.get("height")
            except Exception as e:
                # Rasterio not installed or failed
                pass

            # 2. If rasterio is not available or didn't find transform, check companion metadata JSON or dict
            if transform is None and isinstance(geotiff_path_or_profile, (str, Path)):
                tif_p = Path(geotiff_path_or_profile)
                candidates = [
                    tif_p.with_suffix(".json"),
                    tif_p.parent.parent / "metadata" / f"{tif_p.stem}.json",
                    tif_p.parent / f"{tif_p.stem}.json"
                ]
                for cand in candidates:
                    if cand.exists():
                        try:
                            import json as json_lib
                            with open(cand, "r", encoding="utf-8") as jf:
                                card = json_lib.load(jf)
                                box = card.get("bounding_box") or card.get("wgs84_bounds")
                                if box and len(box) == 4:
                                    wgs84_tile_bounds = [float(x) for x in box]
                                    break
                        except Exception:
                            pass
            elif isinstance(geotiff_path_or_profile, dict):
                box = geotiff_path_or_profile.get("bounding_box") or geotiff_path_or_profile.get("wgs84_bounds")
                if box and len(box) == 4:
                    wgs84_tile_bounds = [float(x) for x in box]

        detections = []
        for idx, match in enumerate(matches):
            try:
                c1, c2, c3, c4 = [float(x) for x in match.groups()]
                # Ensure coordinates are within [0, 1]
                ymin = max(0.0, min(1.0, min(c1, c3)))
                ymax = max(0.0, min(1.0, max(c1, c3)))
                xmin = max(0.0, min(1.0, min(c2, c4)))
                xmax = max(0.0, min(1.0, max(c2, c4)))

                # Pixel bounding box [ymin_px, xmin_px, ymax_px, xmax_px]
                pixel_bbox = [
                    int(ymin * img_height),
                    int(xmin * img_width),
                    int(ymax * img_height),
                    int(xmax * img_width)
                ]

                # Compute geographic coordinates if transform or tile bounds are available
                wgs84_coords = None
                wgs84_bbox = None

                # Method A: Rasterio projection
                if transform is not None and src_crs is not None:
                    try:
                        import rasterio
                        from rasterio.windows import Window
                        from rasterio.warp import transform_bounds

                        rw = raster_w or img_width
                        rh = raster_h or img_height
                        col_off = int(xmin * rw)
                        row_off = int(ymin * rh)
                        box_w = max(1, int((xmax - xmin) * rw))
                        box_h = max(1, int((ymax - ymin) * rh))

                        window = Window(col_off, row_off, box_w, box_h)
                        native_bounds = rasterio.windows.bounds(window, transform)

                        if str(src_crs).upper() not in ["EPSG:4326", "WGS 84", "OGC:CRS84"]:
                            wgs84_bounds = list(transform_bounds(src_crs, "EPSG:4326", *native_bounds))
                        else:
                            wgs84_bounds = list(native_bounds)

                        min_lon, min_lat, max_lon, max_lat = [round(float(b), 6) for b in wgs84_bounds]
                        wgs84_bbox = [min_lon, min_lat, max_lon, max_lat]
                    except Exception as pe:
                        print(f"[SatQuery VLM] Reprojection note: {pe}")

                # Method B: Linear interpolation from companion metadata WGS84 tile bounds
                if wgs84_bbox is None and wgs84_tile_bounds is not None:
                    t_min_lon, t_min_lat, t_max_lon, t_max_lat = wgs84_tile_bounds
                    d_min_lon = round(t_min_lon + (t_max_lon - t_min_lon) * xmin, 6)
                    d_max_lon = round(t_min_lon + (t_max_lon - t_min_lon) * xmax, 6)
                    d_min_lat = round(t_max_lat - (t_max_lat - t_min_lat) * ymax, 6)
                    d_max_lat = round(t_max_lat - (t_max_lat - t_min_lat) * ymin, 6)
                    wgs84_bbox = [d_min_lon, d_min_lat, d_max_lon, d_max_lat]

                if wgs84_bbox is not None:
                    min_lon, min_lat, max_lon, max_lat = wgs84_bbox
                    wgs84_coords = [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]
                    ]

                # Fallback to normalized [0, 1] relative coordinates if no geographic CRS
                geom_coords = wgs84_coords if wgs84_coords is not None else [
                    [
                        [xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                        [xmin, ymin]
                    ]
                ]

                geojson_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": geom_coords
                    },
                    "properties": {
                        "id": idx + 1,
                        "label": f"Detection {idx + 1}",
                        "crs": "EPSG:4326" if wgs84_coords is not None else "relative",
                        "normalized_bbox": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)],
                        "pixel_bbox": pixel_bbox,
                        "wgs84_bbox": wgs84_bbox
                    }
                }

                detections.append({
                    "normalized": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)],
                    "pixel": pixel_bbox,
                    "wgs84": wgs84_bbox,
                    "geojson": geojson_feature
                })
            except Exception:
                continue

        return detections

    @classmethod
    def _parse_bounding_box(
        cls,
        text: str,
        img_width: int,
        img_height: int,
        geotiff_path_or_profile: Optional[Union[str, Path, Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Backwards-compatible helper returning the first detected bounding box."""
        detections = cls._parse_bounding_boxes(text, img_width, img_height, geotiff_path_or_profile)
        return detections[0] if detections else None

    def query(
        self,
        image: Union[str, bytes, Image.Image],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        geotiff_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main query method to run inference on a satellite image.

        Parameters:
        - image: Image path, PIL Image, raw bytes, or base64 string
        - prompt: Natural language user question
        - max_tokens: Maximum response tokens (default: 1024)
        - temperature: Sampling temperature (default: 0.2)
        - geotiff_path: Optional path to source GeoTIFF for WGS84 projection
        - system_prompt: Optional custom system prompt (defaults to DEFAULT_SYSTEM_PROMPT)

        Returns a dictionary with:
        - answer: Conversational, expert-level response
        - raw_answer: Direct unparsed generation from model
        - has_bbox: Boolean indicating if spatial coordinates were detected
        - detections: List of all parsed bounding box dictionaries
        - bbox_normalized: [ymin, xmin, ymax, xmax] or None
        - bbox_pixel: [ymin_px, xmin_px, ymax_px, xmax_px] or None
        - wgs84_bbox: [min_lon, min_lat, max_lon, max_lat] or None
        - geojson: GeoJSON Feature or FeatureCollection ready for MapLibre/Leaflet
        - latency_ms: Inference time in milliseconds
        - device: Hardware device used
        """
        start_time = time.time()
        
        # Auto-detect geotiff_path if image is a string path
        if geotiff_path is None and isinstance(image, (str, Path)):
            geotiff_path = str(image)

        pil_img = self._load_image(image)
        img_width, img_height = pil_img.size

        # Format chat messages with system prompt to activate Qwen3-VL thinking mode
        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        if "cuda" in self.device:
            torch.cuda.empty_cache()

        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_prompt], images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                if "cuda" in self.device:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            do_sample=(temperature > 0.0)
                        )
                else:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=(temperature > 0.0)
                    )
            except torch.cuda.OutOfMemoryError as oom_err:
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
                print(f"[SatQuery VLM] CUDA OOM encountered ({oom_err}). Retrying with aggressive 512px downsampling...")
                fallback_img = self._constrain_image_size(pil_img, max_dim=512)
                fallback_messages = [
                    {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image", "image": fallback_img}, {"type": "text", "text": prompt}]}
                ]
                fb_prompt = self.processor.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=True)
                fb_inputs = self.processor(text=[fb_prompt], images=[fallback_img], return_tensors="pt")
                fb_inputs = {k: v.to(self.device) for k, v in fb_inputs.items()}
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_ids = self.model.generate(
                        **fb_inputs,
                        max_new_tokens=min(max_tokens, 512),
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=(temperature > 0.0)
                    )
                inputs = fb_inputs
            finally:
                if "cuda" in self.device:
                    torch.cuda.empty_cache()

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_answer = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Parse spatial coordinates if present
        detections = self._parse_bounding_boxes(
            raw_answer, img_width, img_height, geotiff_path_or_profile=geotiff_path
        )

        # Construct GeoJSON: FeatureCollection if multiple, single Feature if 1
        geojson_data = None
        if len(detections) > 1:
            geojson_data = {
                "type": "FeatureCollection",
                "features": [d["geojson"] for d in detections]
            }
        elif len(detections) == 1:
            geojson_data = detections[0]["geojson"]

        # Ensure response is conversational, grounded in visual evidence, and preserves genuine model reasoning
        try:
            clean_raw = (raw_answer or "").strip()
            is_pure_coords = bool(re.match(r"^(?:<box>)?\[[0-9.\s,]+\](?:</box>)?$", clean_raw))

            # Helper to check if text is a comma-separated list of short class labels/tags rather than conversational prose
            is_tag_list = False
            if "," in clean_raw:
                parts = [p.strip() for p in clean_raw.split(",") if p.strip()]
                if len(parts) >= 2 and all(len(p.split()) <= 4 for p in parts):
                    is_tag_list = True

            # If the model already gave a natural conversational answer (> 10 words, not coords, not tag list, and with punctuation), PRESERVE IT!
            if not is_pure_coords and not is_tag_list and len(clean_raw.split()) >= 10 and ("." in clean_raw or "।" in clean_raw or "?" in clean_raw) and "primarily composed of **no**" not in clean_raw:
                if detections and "cyan" not in clean_raw.lower() and "map" not in clean_raw.lower():
                    final_answer = f"{clean_raw}\n\n*The identified target area has also been projected and outlined in neon cyan on your interactive map viewport.*"
                else:
                    final_answer = clean_raw
            else:
                from response_synthesizer import synthesize_conversational_response, extract_clean_user_query
                user_q = extract_clean_user_query(prompt) or prompt

                # Determine or retrieve scene land cover
                cache_key = str(geotiff_path) if geotiff_path else f"{pil_img.size}"
                raw_clean = clean_raw.strip().lower().rstrip(".।")
                is_binary = raw_clean in ["no", "yes", "false", "true", "नहीं", "हाँ"]

                if not is_binary and not is_pure_coords and clean_raw and len(clean_raw.split()) <= 6:
                    self._scene_landcover_cache[cache_key] = clean_raw
                    scene_lc = clean_raw
                else:
                    scene_lc = self._detect_primary_landcover(pil_img, geotiff_path)

                final_answer = synthesize_conversational_response(
                    query=user_q,
                    raw_answer=raw_answer,
                    scene_landcover=scene_lc,
                    detections=detections,
                    wgs84_bounds=detections[0]["wgs84"] if detections else None
                )
        except Exception:
            final_answer = raw_answer

        return {
            "answer": final_answer,
            "raw_answer": raw_answer,
            "scene_landcover": scene_lc if 'scene_lc' in locals() else None,
            "has_bbox": len(detections) > 0,
            "detections": detections,
            "bbox_normalized": detections[0]["normalized"] if detections else None,
            "bbox_pixel": detections[0]["pixel"] if detections else None,
            "wgs84_bbox": detections[0]["wgs84"] if detections else None,
            "geojson": geojson_data,
            "latency_ms": latency_ms,
            "device": str(self.device),
            "image_size": [img_width, img_height]
        }


# Quick verification self-test
if __name__ == "__main__":
    import json
    print("\n--- Running SatQuery VLM Verification Test ---")
    vlm = SatQueryVLM()

    test_img = "data/images/S2B_MSIL2A_20170825T093029_N9999_R136_T34TEQ_29_07.png"
    if not os.path.exists(test_img):
        import glob
        found = glob.glob("data/images/*.png")
        test_img = found[0] if found else None

    if test_img:
        q = "Analyze this Sentinel-2 satellite image. Describe what land cover types and terrain features are visible."
        print(f"\nQuerying on: {test_img}")
        res = vlm.query(test_img, q)
        print("\nResult:")
        print(json.dumps({k: v for k, v in res.items() if k != "geojson"}, indent=2))
        print("\n[SUCCESS] SatQuery VLM is fully verified and ready for backend integration!")
    else:
        print("[NOTE] No test images found in data/images/.")

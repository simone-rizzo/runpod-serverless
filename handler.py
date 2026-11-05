"""Runpod Serverless worker: instance segmentation of litter with RF-DETR-Seg (TACO, 10 classes).

The weights are not in the image: they are read from the network volume, which Runpod mounts
at /runpod-volume on Serverless workers (the same volume is at /workspace on a Pod).
"""
import base64
import binascii
import io
import logging
import os
import time

import numpy as np
import requests
import runpod
from PIL import Image, ImageOps, UnidentifiedImageError
from pycocotools import mask as mask_utils
from rfdetr import RFDETRSegMedium

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("taco-worker")

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/runpod-volume/taco_segmentation/medium_12ep/checkpoint_best_total.pth"
)
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.4"))
# The training images were downscaled to this longest side. predict() then resizes to the model
# resolution without antialiasing, so feeding it a full-size photo would alias badly.
MAX_SIDE = int(os.environ.get("MAX_SIDE", "640"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_MB", "25")) * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 20

CLASS_NAMES = [
    "Bottle",
    "Bottle cap",
    "Can",
    "Cigarette",
    "Cup",
    "Lid",
    "Other",
    "Plastic bag + wrapper",
    "Pop tab",
    "Straw",
]


class BadInput(ValueError):
    """The request is wrong; reported to the caller instead of failing the worker."""


def load_model():
    if not os.path.isfile(MODEL_PATH):
        volume = "/runpod-volume"
        mounted = os.path.isdir(volume)
        raise FileNotFoundError(
            f"Weights not found at {MODEL_PATH}. "
            + ("Check MODEL_PATH against the volume contents." if mounted else
               f"{volume} is not mounted: attach the network volume to this endpoint.")
        )
    started = time.perf_counter()
    model = RFDETRSegMedium(pretrain_weights=MODEL_PATH, num_classes=len(CLASS_NAMES))
    log.info("Loaded %s in %.1fs on %s", MODEL_PATH, time.perf_counter() - started, model.model.device)
    return model


def download(url):
    if not url.lower().startswith(("http://", "https://")):
        raise BadInput("image_url must be http or https")
    try:
        # A browser-less default User-Agent is refused by some image hosts.
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S,
                          headers={"User-Agent": "taco-segmentation-worker/1.0"}) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                data += chunk
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise BadInput(f"Image is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
    except requests.RequestException as exc:
        raise BadInput(f"Could not download image_url: {exc}") from exc
    return bytes(data)


def decode_base64(value):
    # Accepts both raw base64 and data URIs (data:image/jpeg;base64,...).
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadInput("image is not valid base64") from exc


def read_image(job_input):
    has_image, has_url = bool(job_input.get("image")), bool(job_input.get("image_url"))
    if has_image == has_url:
        raise BadInput("Provide exactly one of 'image' (base64) or 'image_url'")
    if has_image and not isinstance(job_input["image"], str):
        raise BadInput("image must be a base64 string")
    if has_url and not isinstance(job_input["image_url"], str):
        raise BadInput("image_url must be a string")
    data = decode_base64(job_input["image"]) if has_image else download(job_input["image_url"])
    try:
        with Image.open(io.BytesIO(data)) as source:
            # Same order as prepare_dataset.py: phone photos are stored rotated plus an EXIF flag.
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise BadInput(f"Could not decode image: {exc}") from exc


def read_threshold(job_input):
    value = job_input.get("threshold", DEFAULT_THRESHOLD)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise BadInput("threshold must be a number between 0 and 1")
    return float(value)


def encode_mask(mask, size):
    """Binary mask -> COCO compressed RLE, at the size of the image the client sent."""
    if mask.shape[::-1] != size:
        resized = Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST)
        mask = np.asarray(resized)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def segment(image, threshold, return_masks):
    """Returns detections in the pixel coordinates of `image`, EXIF rotation applied."""
    width, height = image.size
    scale = min(1.0, MAX_SIDE / max(width, height))
    resized = image
    if scale < 1.0:
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                               Image.LANCZOS)

    result = MODEL.predict(resized, threshold=threshold, include_source_image=False)

    # The head has num_classes + 1 outputs; the last one is the background slot, not a class.
    real = result.class_id < len(CLASS_NAMES)
    scale_x, scale_y = width / resized.width, height / resized.height
    order = np.argsort(-result.confidence[real])
    boxes, scores, labels = result.xyxy[real][order], result.confidence[real][order], result.class_id[real][order]
    masks = result.mask[real][order] if return_masks else [None] * len(order)

    detections = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        x1, y1, x2, y2 = (float(v) for v in box)
        detection = {
            "class_id": int(label),
            "class_name": CLASS_NAMES[int(label)],
            "confidence": round(float(score), 4),
            "bbox_xyxy": [round(x1 * scale_x, 1), round(y1 * scale_y, 1),
                          round(x2 * scale_x, 1), round(y2 * scale_y, 1)],
        }
        if return_masks:
            detection["mask_rle"] = encode_mask(mask, (width, height))
        detections.append(detection)
    return detections


def handler(job):
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        return {"error": "input must be a JSON object"}
    try:
        image = read_image(job_input)
        threshold = read_threshold(job_input)
        return_masks = job_input.get("return_masks", True)
        if not isinstance(return_masks, bool):
            raise BadInput("return_masks must be true or false")
    except BadInput as exc:
        return {"error": str(exc)}

    started = time.perf_counter()
    detections = segment(image, threshold, return_masks)
    return {
        "image_width": image.width,
        "image_height": image.height,
        "threshold": threshold,
        "inference_ms": round((time.perf_counter() - started) * 1000),
        "detections": detections,
    }


# Loaded once per worker, outside the handler, so only the cold start pays for it.
MODEL = load_model()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

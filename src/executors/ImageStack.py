from __future__ import annotations

import base64
from collections import deque
import threading
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

_CURRENT_FILE = Path(__file__).resolve()
_COMPONENT_ROOT = _CURRENT_FILE.parents[2]
_PROJECT_ROOT = _CURRENT_FILE.parents[4] if len(_CURRENT_FILE.parents) > 4 else _COMPONENT_ROOT
for _path in (_COMPONENT_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

try:
    from sdks.novavision.src.base.component import Component as NovaVisionComponent  # type: ignore
except ImportError:
    NovaVisionComponent = None

NovaVisionScriptExecutor = None
NovaVisionImageHelper = None


def _load_image_helper():
    global NovaVisionImageHelper
    if NovaVisionImageHelper is None:
        try:
            from sdks.novavision.src.media.image import Image as helper  # type: ignore
        except Exception:
            return None
        NovaVisionImageHelper = helper
    return NovaVisionImageHelper

try:
    from components.ImageStack.src.models.PackageModel import (  # type: ignore
        ImageStackOutputs,
        ImageStackResponse,
        PackageModel,
        NovaVisionImage,
    )
    from components.ImageStack.src.utils.response import build_response  # type: ignore
except ImportError:
    try:
        from src.models.PackageModel import (  # type: ignore
            ImageStackOutputs,
            ImageStackResponse,
            PackageModel,
            NovaVisionImage,
        )
        from src.utils.response import build_response
    except ImportError:
        from novavision.image_stack.models.PackageModel import (  # type: ignore
            ImageStackOutputs,
            ImageStackResponse,
            PackageModel,
            NovaVisionImage,
        )
        from novavision.image_stack.utils.response import build_response  # type: ignore


if NovaVisionComponent is None:
    class _ComponentBase:
        def __init__(self, request, bootstrap):
            self.request = request
            self.bootstrap = bootstrap
            self.redis_db = bootstrap.get("redis_db") if isinstance(bootstrap, dict) else None
            self.flowUID = None
            self.matchedID = None
            self.uID = None

else:
    _ComponentBase = NovaVisionComponent


MAX_STACK_SIZE = 64
MAX_RESOLUTION_WIDTH = 1920
MAX_RESOLUTION_HEIGHT = 1080
JPEG_QUALITY = 75


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(by_alias=True)
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        try:
            return value.dict(by_alias=True)
        except TypeError:
            return value.dict()
    return value


def _plain_value(value: Any) -> Any:
    value = _model_to_dict(value)
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _unwrap_value(value: Any) -> Any:
    while isinstance(value, dict) and "value" in value:
        value = value["value"]
    return value


def _search_param(data: Any, name: str) -> Any:
    data = _model_to_dict(data)
    if isinstance(data, dict):
        if name in data:
            return _unwrap_value(data[name])
        if data.get("name") == name and "value" in data:
            return data["value"]
        for item in data.values():
            result = _search_param(item, name)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _search_param(item, name)
            if result is not None:
                return result
    return None


def _coerce_int(value: Any, default: int) -> int:
    value = _unwrap_value(value)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Image Stack configuration must be an integer, got {value!r}.") from error


def _coerce_bool(value: Any, default: bool = False) -> bool:
    value = _unwrap_value(value)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _shape_from_key(shape_key: Any) -> Optional[Tuple[int, ...]]:
    if shape_key is None or shape_key == "":
        return None
    if isinstance(shape_key, str):
        try:
            shape_key = base64.b64decode(shape_key)
        except Exception:
            return None
    if isinstance(shape_key, (bytes, bytearray, memoryview)):
        try:
            return tuple(int(item) for item in np.frombuffer(shape_key, dtype=np.int64))
        except (TypeError, ValueError):
            return None
    try:
        return tuple(int(item) for item in np.asarray(shape_key).tolist())
    except (TypeError, ValueError):
        return None


class ImageStack(_ComponentBase):
    """Accumulate JPEG-compressed frames in a per-node in-memory FIFO."""

    @staticmethod
    def bootstrap(config: Optional[dict] = None) -> dict:
        return {
            "status": "ready",
            "image_stack_buffers": {},
            "image_stack_clear_seen": {},
            "image_stack_lock": threading.RLock(),
        }

    def __init__(self, request, bootstrap):
        if NovaVisionComponent is not None and hasattr(request, "data"):
            super().__init__(request, bootstrap)
            self.request.model = PackageModel(**self.request.data)
        else:
            self.request = request
            self.bootstrap = bootstrap
            self.redis_db = bootstrap.get("redis_db") if isinstance(bootstrap, dict) else None
            self.flowUID = None
            self.matchedID = None
            self.uID = None

        self._bootstrap = bootstrap if isinstance(bootstrap, dict) else {}
        self._buffers = self._bootstrap.setdefault("image_stack_buffers", {})
        self._clear_seen = self._bootstrap.setdefault("image_stack_clear_seen", {})
        self._lock = self._bootstrap.setdefault("image_stack_lock", threading.RLock())

    def _request_data(self) -> Any:
        if hasattr(self.request, "data"):
            return self.request.data
        return _model_to_dict(self.request)

    def _get_param(self, name: str) -> Any:
        if hasattr(self.request, "get_param"):
            try:
                value = self.request.get_param(name)
                if value is not None:
                    return value
            except Exception:
                pass
        return _search_param(self._request_data(), name)

    def _state_key(self) -> str:
        data = self._request_data()
        if isinstance(data, dict):
            matched_id = data.get("matchedID") or data.get("matchedId")
            node_uid = data.get("uID") or data.get("uid")
        else:
            matched_id = node_uid = None

        # NovaVision uses flowUID as a per-execution correlation ID, so it can
        # change for every incoming frame. The package node identity is stable
        # for the lifetime of the subscriber and is the correct buffer scope.
        node_uid = matched_id or node_uid or getattr(self, "uID", None) or "local-node"
        return str(node_uid)

    @staticmethod
    def _validate_limits(stack_size: int, width: int, height: int) -> None:
        if not 1 <= stack_size <= MAX_STACK_SIZE:
            raise ValueError("StackSize must be between 1 and 64.")
        if not 64 <= width <= MAX_RESOLUTION_WIDTH:
            raise ValueError("ResolutionWidth must be between 64 and 1920.")
        if not 64 <= height <= MAX_RESOLUTION_HEIGHT:
            raise ValueError("ResolutionHeight must be between 64 and 1080.")

    @staticmethod
    def _decode_local_image(image: Dict[str, Any]) -> np.ndarray:
        value = image.get("value")
        encoding = str(image.get("encoding", "base64")).lower()
        shape = _shape_from_key(image.get("shape_key"))

        if isinstance(value, np.ndarray):
            return value

        if encoding == "bytes" and isinstance(value, (bytes, bytearray, memoryview)) and shape:
            try:
                return np.frombuffer(value, dtype=np.uint8).reshape(shape)
            except ValueError as error:
                raise ValueError("The input image bytes do not match shape_key.") from error

        if isinstance(value, str):
            try:
                value = base64.b64decode(value, validate=True)
            except Exception as error:
                raise ValueError("The input image value is not valid base64 data.") from error

        if isinstance(value, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(value, dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if decoded is not None:
                return decoded
            if shape:
                try:
                    return np.frombuffer(value, dtype=np.uint8).reshape(shape)
                except ValueError as error:
                    raise ValueError("The input image bytes could not be decoded.") from error

        raise ValueError("inputImage must contain a valid encoded image or ndarray.")

    def _decode_image(self, image: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        image = _plain_value(image)
        if isinstance(image, list):
            if not image:
                raise ValueError("inputImage is empty.")
            image = image[0]
        if isinstance(image, np.ndarray):
            return image, {}
        if not isinstance(image, dict):
            raise ValueError("inputImage must be a NovaVision image object or image list.")

        image = dict(image)
        image_helper = _load_image_helper()
        if image_helper is not None:
            try:
                image.setdefault("r_key", "")
                decoded = image_helper.get_frame(img=image, redis_db=getattr(self, "redis_db", None))
                if decoded is not None and hasattr(decoded, "value") and isinstance(decoded.value, np.ndarray):
                    return decoded.value, image
            except Exception:
                if image.get("r_key"):
                    raise ValueError("The input image could not be loaded from NovaVision Redis storage.")

        return self._decode_local_image(image), image

    @staticmethod
    def _compress_frame(frame: np.ndarray, max_width: int, max_height: int) -> Tuple[bytes, Tuple[int, ...]]:
        frame = np.asarray(frame)
        if frame.size == 0 or frame.ndim not in (2, 3):
            raise ValueError("inputImage must be a non-empty 2D or 3D image array.")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not success:
            raise ValueError("OpenCV could not JPEG-encode inputImage.")
        return encoded.tobytes(), tuple(int(item) for item in frame.shape)

    @staticmethod
    def _source_uid(image: Dict[str, Any]) -> str:
        return str(image.get("uID") or image.get("uid") or uuid4())

    @staticmethod
    def _source_metadata(image: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {"name", "uID", "uid", "mimeType", "encoding", "value", "r_key", "shape_key", "type"}
        return {key: value for key, value in image.items() if key not in reserved}

    @classmethod
    def _output_metadata(cls, image: Dict[str, Any], shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Return source metadata with dimensions matching the encoded output."""

        metadata = cls._source_metadata(image)
        output_height, output_width = shape[:2]

        nested_metadata = metadata.get("metadata")
        if isinstance(nested_metadata, dict) and ({"width", "height"} & nested_metadata.keys()):
            nested_metadata = dict(nested_metadata)
            source_width = nested_metadata.get("width")
            source_height = nested_metadata.get("height")
            if source_width is not None and source_width != output_width:
                nested_metadata.setdefault("source_width", source_width)
            if source_height is not None and source_height != output_height:
                nested_metadata.setdefault("source_height", source_height)
            nested_metadata["width"] = output_width
            nested_metadata["height"] = output_height
            metadata["metadata"] = nested_metadata

        if "width" in metadata or "height" in metadata:
            source_width = metadata.get("width")
            source_height = metadata.get("height")
            if source_width is not None and source_width != output_width:
                metadata.setdefault("source_width", source_width)
            if source_height is not None and source_height != output_height:
                metadata.setdefault("source_height", source_height)
            metadata["width"] = output_width
            metadata["height"] = output_height

        return metadata

    @staticmethod
    def _frame_image(frame: Dict[str, Any], *, name: str = "outputImages") -> Any:
        shape_bytes = np.asarray(frame["shape"], dtype=np.int64).tobytes()
        payload = {
            "name": name,
            "uID": frame["uID"],
            "mimeType": "image/jpg",
            "encoding": "base64",
            "value": base64.b64encode(frame["jpeg"]).decode("ascii"),
            "shape_key": base64.b64encode(shape_bytes).decode("ascii"),
            "r_key": "",
            "type": "Image",
        }
        payload.update(frame["metadata"])
        payload["name"] = name
        payload["uID"] = frame["uID"]
        payload["mimeType"] = "image/jpg"
        payload["encoding"] = "base64"
        payload["value"] = base64.b64encode(frame["jpeg"]).decode("ascii")
        payload["shape_key"] = base64.b64encode(shape_bytes).decode("ascii")
        payload["r_key"] = ""
        payload["type"] = "Image"

        return NovaVisionImage(**payload)

    @staticmethod
    def _decode_stored_frame(frame: Dict[str, Any]) -> np.ndarray:
        encoded = np.frombuffer(frame["jpeg"], dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Image Stack could not decode a stored frame.")
        return decoded

    @classmethod
    def _build_preview_frame(
        cls,
        frames: Iterable[Dict[str, Any]],
        max_width: int,
        max_height: int,
    ) -> Dict[str, Any]:
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("Image Stack cannot build a preview without frames.")

        decoded_frames = [(frame, cls._decode_stored_frame(frame)) for frame in frame_list]
        first_image = decoded_frames[0][1]
        source_height, source_width = first_image.shape[:2]
        columns = min(4, len(decoded_frames))
        rows = (len(decoded_frames) + columns - 1) // columns
        row_height = max(1, max_height // rows)
        label_height = min(28, max(18, row_height // 5))
        image_height = max(1, row_height - label_height)
        aspect_ratio = source_width / max(1, source_height)
        cell_width = min(
            max(1, max_width // columns),
            source_width,
            max(1, int(round(image_height * aspect_ratio))),
        )
        cell_height = max(1, int(round(cell_width / max(aspect_ratio, 1e-6))))
        canvas_width = cell_width * columns
        canvas_height = (cell_height + label_height) * rows
        canvas = np.full((canvas_height, canvas_width, 3), 32, dtype=np.uint8)

        for index, (frame, image) in enumerate(decoded_frames):
            image_height_source, image_width_source = image.shape[:2]
            scale = min(
                1.0,
                cell_width / max(1, image_width_source),
                cell_height / max(1, image_height_source),
            )
            thumbnail_width = max(1, int(round(image_width_source * scale)))
            thumbnail_height = max(1, int(round(image_height_source * scale)))
            thumbnail = cv2.resize(
                image,
                (thumbnail_width, thumbnail_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST,
            )
            row, column = divmod(index, columns)
            x = column * cell_width + (cell_width - thumbnail_width) // 2
            y = row * (cell_height + label_height) + (cell_height - thumbnail_height) // 2
            canvas[y : y + thumbnail_height, x : x + thumbnail_width] = thumbnail
            cv2.rectangle(
                canvas,
                (column * cell_width, row * (cell_height + label_height)),
                (column * cell_width + cell_width - 1, row * (cell_height + label_height) + cell_height - 1),
                (180, 180, 180),
                1,
            )
            frame_index = frame["metadata"].get("frame_index")
            label = f"{index + 1}: frame {frame_index}" if frame_index is not None else f"{index + 1}"
            cv2.putText(
                canvas,
                label,
                (column * cell_width + 6, row * (cell_height + label_height) + cell_height + label_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

        success, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not success:
            raise ValueError("OpenCV could not JPEG-encode the Image Stack preview.")
        return {
            "jpeg": encoded.tobytes(),
            "shape": tuple(int(item) for item in canvas.shape),
            "uID": f"stack-preview-{frame_list[0]['uID']}",
            "metadata": {
                "stack_count": len(frame_list),
                "preview_order": "newest-first",
            },
        }

    def _build_response(
        self,
        frames: Iterable[Dict[str, Any]],
        max_width: int,
        max_height: int,
    ) -> Any:
        frame_list = list(frames)
        images = [self._frame_image(frame) for frame in frame_list]
        preview = self._frame_image(
            self._build_preview_frame(frame_list, max_width, max_height),
            name="outputPreview",
        )
        return ImageStackResponse(
            outputs=ImageStackOutputs(
                outputImages={"value": images},
                outputPreview={"value": preview},
                outputData={"value": len(images)},
            )
        )

    def run(self) -> Any:
        input_image = self._get_param("inputImage")
        if input_image is None or (isinstance(input_image, list) and not input_image):
            raise ValueError("inputImage is required.")

        stack_size = _coerce_int(self._get_param("StackSize"), 10)
        width = _coerce_int(self._get_param("ResolutionWidth"), 1920)
        height = _coerce_int(self._get_param("ResolutionHeight"), 1080)
        clear = _coerce_bool(self._get_param("ClearBuffer"), False)
        self._validate_limits(stack_size, width, height)

        frame, source_image = self._decode_image(input_image)
        jpeg, shape = self._compress_frame(frame, width, height)
        stored_frame = {
            "jpeg": jpeg,
            "shape": shape,
            "uID": self._source_uid(source_image),
            "metadata": self._output_metadata(source_image, shape),
        }

        state_key = self._state_key()
        with self._lock:
            buffer = self._buffers.get(state_key)
            if buffer is None:
                buffer = deque(maxlen=stack_size)
                self._buffers[state_key] = buffer
            elif buffer.maxlen != stack_size:
                buffer = deque(list(buffer)[:stack_size], maxlen=stack_size)
                self._buffers[state_key] = buffer

            previous_clear = bool(self._clear_seen.get(state_key, False))
            if clear and not previous_clear:
                buffer.clear()
            self._clear_seen[state_key] = clear
            buffer.appendleft(stored_frame)
            frames = list(buffer)

        self.response = self._build_response(frames, width, height)
        if hasattr(self.request, "data") or isinstance(self.request, dict):
            return build_response(context=self)
        return self.response


if __name__ == "__main__":
    try:
        from sdks.novavision.src.helper.executor import Executor as NovaVisionScriptExecutor  # type: ignore
    except Exception as error:
        raise RuntimeError("NovaVision SDK Executor is required when running ImageStack.py as a script.") from error
    if NovaVisionScriptExecutor is None:
        raise RuntimeError("NovaVision SDK Executor is required when running ImageStack.py as a script.")
    NovaVisionScriptExecutor(sys.argv[1]).run()

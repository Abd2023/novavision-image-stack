from __future__ import annotations

from base64 import b64encode
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executors.ImageStack import ImageStack
from src.models.PackageModel import ImageStackRequest


def image_request(frame_index: int, stack_size: int = 3) -> ImageStackRequest:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = 30 + frame_index * 40
    frame[:, :, 1] = 80 + frame_index * 20
    frame[:, :, 2] = 150
    return ImageStackRequest(
        inputs={
            "inputImage": {
                "value": [
                    {
                        "name": "inputImage",
                        "uID": f"frame-{frame_index}",
                        "mimeType": "image/jpg",
                        "encoding": "bytes",
                        "value": frame.tobytes(),
                        "shape_key": b64encode(np.asarray(frame.shape, dtype=np.int64).tobytes()).decode("ascii"),
                        "r_key": "",
                        "type": "Image",
                    }
                ]
            }
        },
        configs={"StackSize": {"value": stack_size}},
    )


def main() -> None:
    bootstrap = ImageStack.bootstrap()
    for frame_index in range(1, 5):
        response = ImageStack(image_request(frame_index), bootstrap).run()
        output_images = response.outputs.outputImages.value
        print(
            f"frame={frame_index} count={response.outputs.outputData.value} "
            f"order={[image.uID for image in output_images]}"
        )


if __name__ == "__main__":
    main()


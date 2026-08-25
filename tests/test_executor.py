from __future__ import annotations

import base64
from copy import deepcopy

import cv2
import numpy as np
import pytest

from src.executors.ImageStack import ImageStack
from src.models.PackageModel import ImageStackRequest


def encoded_image(index: int, *, shape=(120, 160, 3), uid: str | None = None, metadata=None):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :, 0] = (20 + index * 31) % 255
    frame[:, :, 1] = (80 + index * 17) % 255
    frame[:, :, 2] = 200
    shape_key = base64.b64encode(np.asarray(frame.shape, dtype=np.int64).tobytes()).decode("ascii")
    image = {
        "name": "inputImage",
        "uID": uid or f"frame-{index}",
        "mimeType": "image/jpg",
        "encoding": "bytes",
        "value": frame.tobytes(),
        "shape_key": shape_key,
        "r_key": "",
        "type": "Image",
    }
    if metadata:
        image.update(metadata)
    return frame, image


def request_for(index: int, *, stack_size=3, width=1920, height=1080, clear=False, uid=None, metadata=None):
    _, image = encoded_image(index, uid=uid, metadata=metadata)
    clear_value = {
        "name": "True" if clear else "False",
        "value": clear,
        "type": "bool",
        "field": "option",
    }
    return ImageStackRequest(
        inputs={"inputImage": {"value": [image]}},
        configs={
            "StackSize": {"value": stack_size},
            "ResolutionWidth": {"value": width},
            "ResolutionHeight": {"value": height},
            "ClearBuffer": {"value": clear_value},
        },
    )


def run_local(request, bootstrap=None):
    return ImageStack(request, bootstrap if bootstrap is not None else ImageStack.bootstrap()).run()


def output_ids(response):
    return [image.uID for image in response.outputs.outputImages.value]


def output_payload(response):
    return response.outputs.outputImages.value


def decode_output(image):
    encoded = base64.b64decode(image.value)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    return decoded


def test_first_frame_and_newest_first_order():
    bootstrap = ImageStack.bootstrap()

    first = run_local(request_for(1), bootstrap)
    second = run_local(request_for(2), bootstrap)
    third = run_local(request_for(3), bootstrap)

    assert first.outputs.outputData.value == 1
    assert output_ids(second) == ["frame-2", "frame-1"]
    assert output_ids(third) == ["frame-3", "frame-2", "frame-1"]


def test_fifo_eviction_keeps_the_latest_stack_size_frames():
    bootstrap = ImageStack.bootstrap()
    for index in range(1, 5):
        response = run_local(request_for(index, stack_size=3), bootstrap)

    assert response.outputs.outputData.value == 3
    assert output_ids(response) == ["frame-4", "frame-3", "frame-2"]


def test_stack_preview_is_a_single_image_for_image_view():
    bootstrap = ImageStack.bootstrap()
    for index in range(1, 5):
        response = run_local(request_for(index, stack_size=4), bootstrap)

    preview = response.outputs.outputPreview.value
    assert preview.name == "outputPreview"
    assert preview.type == "Image"
    assert preview.stack_count == 4
    assert preview.preview_order == "newest-first"
    assert decode_output(preview).size > 0


def test_stack_growth_and_shrink_preserve_newest_existing_frames():
    bootstrap = ImageStack.bootstrap()
    run_local(request_for(1, stack_size=2), bootstrap)
    run_local(request_for(2, stack_size=2), bootstrap)
    grown = run_local(request_for(3, stack_size=4), bootstrap)
    shrunk = run_local(request_for(4, stack_size=2), bootstrap)

    assert output_ids(grown) == ["frame-3", "frame-2", "frame-1"]
    assert output_ids(shrunk) == ["frame-4", "frame-3"]


def test_resize_preserves_aspect_ratio_and_never_upscales():
    _, image = encoded_image(1, shape=(200, 400, 3))
    request = ImageStackRequest(
        inputs={"inputImage": {"value": [image]}},
        configs={"ResolutionWidth": {"value": 100}, "ResolutionHeight": {"value": 100}},
    )
    response = run_local(request)
    resized = decode_output(response.outputs.outputImages.value[0])

    assert resized.shape[:2] == (50, 100)

    _, small_image = encoded_image(2, shape=(40, 50, 3))
    no_upscale = run_local(ImageStackRequest(inputs={"inputImage": {"value": [small_image]}}))
    assert decode_output(no_upscale.outputs.outputImages.value[0]).shape[:2] == (40, 50)


def test_every_output_is_valid_jpeg_with_base64_safe_image_fields():
    _, image = encoded_image(1, metadata={"timestamp": 123, "metadata": {"camera": "cam-1"}})
    response = run_local(ImageStackRequest(inputs={"inputImage": {"value": [image]}}))
    output = output_payload(response)[0]

    assert output.mimeType == "image/jpg"
    assert output.encoding == "base64"
    assert output.r_key == ""
    assert output.uID == "frame-1"
    assert output.timestamp == 123
    assert output.metadata == {"camera": "cam-1"}
    assert decode_output(output).size > 0


def test_clear_buffer_uses_a_rising_edge():
    bootstrap = ImageStack.bootstrap()
    run_local(request_for(1, stack_size=4), bootstrap)
    run_local(request_for(2, stack_size=4), bootstrap)
    cleared = run_local(request_for(3, stack_size=4, clear=True), bootstrap)
    held = run_local(request_for(4, stack_size=4, clear=True), bootstrap)
    released = run_local(request_for(5, stack_size=4, clear=False), bootstrap)
    cleared_again = run_local(request_for(6, stack_size=4, clear=True), bootstrap)

    assert output_ids(cleared) == ["frame-3"]
    assert output_ids(held) == ["frame-4", "frame-3"]
    assert output_ids(released) == ["frame-5", "frame-4", "frame-3"]
    assert output_ids(cleared_again) == ["frame-6"]


def test_state_is_independent_for_different_node_keys_and_persists_between_instances():
    bootstrap = ImageStack.bootstrap()
    first_request = suite_request("node-a", 1)
    second_request = suite_request("node-b", 2)

    first = ImageStack(first_request, bootstrap).run()
    second = ImageStack(second_request, bootstrap).run()
    first_again = ImageStack(first_request, bootstrap).run()

    assert package_count(first) == 1
    assert package_count(second) == 1
    assert package_count(first_again) == 2

    fresh = ImageStack(suite_request("node-a", 3), ImageStack.bootstrap()).run()
    assert package_count(fresh) == 1


def test_empty_and_invalid_images_raise_clear_errors():
    with pytest.raises(ValueError, match="inputImage is required"):
        run_local(ImageStackRequest())

    invalid = ImageStackRequest(
        inputs={
            "inputImage": {
                "value": [
                    {
                        "name": "inputImage",
                        "uID": "invalid",
                        "mimeType": "image/jpg",
                        "encoding": "base64",
                        "value": "not-base64",
                        "type": "Image",
                    }
                ]
            }
        }
    )
    with pytest.raises(ValueError, match="valid base64"):
        run_local(invalid)


class FakeSuiteRequest:
    def __init__(self, data):
        self.data = data


def suite_request(node_uid: str, frame_index: int, *, flow_uid: str = "flow-1") -> FakeSuiteRequest:
    _, image = encoded_image(frame_index)
    return FakeSuiteRequest(
        {
            "type": "component",
            "name": "ImageStack",
            "uID": node_uid,
            "flowUID": flow_uid,
            "matchedID": None,
            "debug": "False",
            "api": "False",
            "configs": {
                "executor": {
                    "name": "ConfigExecutor",
                    "value": {
                        "name": "ImageStack",
                        "value": {
                            "name": "ImageStack",
                            "type": "Request",
                            "inputs": {"inputImage": {"name": "inputImage", "value": [image], "type": "list"}},
                            "configs": {
                                "StackSize": {"name": "StackSize", "value": 3, "type": "number", "field": "textInput"},
                                "ResolutionWidth": {"name": "ResolutionWidth", "value": 1920, "type": "number", "field": "textInput"},
                                "ResolutionHeight": {"name": "ResolutionHeight", "value": 1080, "type": "number", "field": "textInput"},
                                "ClearBuffer": {"name": "ClearBuffer", "value": {"name": "False", "value": False, "type": "bool", "field": "option"}, "type": "object", "field": "dropdownlist"},
                            },
                        },
                        "type": "object",
                        "field": "option",
                    },
                    "type": "executor",
                    "field": "dependentDropdownlist",
                }
            },
        }
    )


def package_count(package):
    return package.configs.executor.value.value.outputs.outputData.value


def test_suite_flow_uid_changes_do_not_reset_the_node_buffer():
    bootstrap = ImageStack.bootstrap()

    first = ImageStack(suite_request("node-suite", 1, flow_uid="execution-1"), bootstrap).run()
    second = ImageStack(suite_request("node-suite", 2, flow_uid="execution-2"), bootstrap).run()
    third = ImageStack(suite_request("node-suite", 3, flow_uid="execution-3"), bootstrap).run()

    assert package_count(first) == 1
    assert package_count(second) == 2
    assert package_count(third) == 3


def test_suite_request_returns_nested_package_response_with_runtime_metadata():
    package = ImageStack(suite_request("node-suite", 1), ImageStack.bootstrap()).run()
    payload = package.model_dump(by_alias=True) if hasattr(package, "model_dump") else package.dict(by_alias=True)
    nested = payload["configs"]["executor"]["value"]["value"]

    assert payload["uID"] == "node-suite"
    assert payload["flowUID"] == "flow-1"
    assert nested["name"] == "ImageStack"
    assert set(nested["outputs"]) == {"outputImages", "outputPreview", "outputData"}
    assert nested["outputs"]["outputData"]["value"] == 1
    assert nested["outputs"]["outputPreview"]["type"] == "object"


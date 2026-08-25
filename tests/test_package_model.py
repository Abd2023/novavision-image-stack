from __future__ import annotations

from typing import get_args

import pytest

from src.models.PackageModel import (
    ClearBuffer,
    ConfigExecutor,
    ImageStackRequest,
    ImageStackResponse,
    InputImage,
    OutputData,
    OutputImages,
    OutputPreview,
    PackageModel,
    ResolutionHeight,
    ResolutionWidth,
    StackSize,
    NovaVisionOutput,
)


def dump(model, *, by_alias: bool = False):
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


def schema(model):
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def test_package_model_has_component_type_and_configs_executor_traversal():
    package = PackageModel()

    assert package.name == "ImageStack"
    assert package.type == "component"
    assert package.configs.executor.name == "ConfigExecutor"
    assert package.configs.executor.type == "executor"
    assert package.configs.executor.value.name == "ImageStack"
    assert package.executor is package.configs.executor


def test_package_model_schema_exposes_expected_sockets_and_task_title():
    package_payload = dump(PackageModel(), by_alias=True)
    request_payload = dump(ImageStackRequest(), by_alias=True)
    response_payload = dump(ImageStackResponse(), by_alias=True)
    executor_schema = schema(ConfigExecutor)

    assert set(request_payload["inputs"]) == {"inputImage"}
    assert set(request_payload["configs"]) == {
        "StackSize",
        "ResolutionWidth",
        "ResolutionHeight",
        "ClearBuffer",
    }
    assert set(response_payload["outputs"]) == {"outputImages", "outputPreview", "outputData"}
    assert package_payload["configs"]["executor"]["value"]["name"] == "ImageStack"
    assert schema(InputImage)["title"] == "Image"
    assert schema(OutputImages)["title"] == "Images"
    assert schema(OutputPreview)["title"] == "Stack Preview"
    assert schema(OutputData)["title"] == "Frame Count"
    assert response_payload["outputs"]["outputPreview"]["type"] == "object"
    assert response_payload["outputs"]["outputPreview"]["value"] is None
    assert "Image Stack" in str(executor_schema)


def test_defaults_and_config_ranges_match_reference_behavior():
    request = ImageStackRequest()
    configs = request.configs

    assert configs.stack_size.value == 10
    assert configs.resolution_width.value == 1920
    assert configs.resolution_height.value == 1080
    assert configs.clear_buffer.value.value is False
    assert request.inputs.inputImage.type == "list"

    with pytest.raises(ValueError):
        StackSize(value=65)
    with pytest.raises(ValueError):
        ResolutionWidth(value=63)
    with pytest.raises(ValueError):
        ResolutionHeight(value=1081)


def test_clear_buffer_is_a_boolean_dropdown_and_output_models_use_sdk_bases():
    assert ClearBuffer().field == "dropdownlist"
    assert ClearBuffer().type == "object"
    assert issubclass(OutputImages, NovaVisionOutput)
    assert issubclass(OutputPreview, NovaVisionOutput)
    assert issubclass(OutputData, NovaVisionOutput)


def test_config_executor_has_one_image_stack_option():
    fields = getattr(ConfigExecutor, "model_fields", None) or getattr(ConfigExecutor, "__fields__", {})
    assert "value" in fields
    option_annotation = fields["value"].annotation
    assert get_args(option_annotation) == () or "ImageStackExecutor" in str(option_annotation)

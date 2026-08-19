from __future__ import annotations

import base64
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, root_validator

try:
    from sdks.novavision.src.base.model import (
        Config as NovaVisionConfig,
        Configs as NovaVisionConfigs,
        Image as NovaVisionImage,
        Input as NovaVisionInput,
        Inputs as NovaVisionInputs,
        Output as NovaVisionOutput,
        Outputs as NovaVisionOutputs,
        Package as NovaVisionPackage,
        Param as NovaVisionParam,
        Request as NovaVisionRequest,
        Response as NovaVisionResponse,
    )
except ImportError:
    from pydantic import BaseModel

    class NovaVisionModel(BaseModel):
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"
            allow_population_by_field_name = True
            populate_by_name = True

    class NovaVisionParam(NovaVisionModel):
        name: str = ""
        value: Any = None
        type: str = "object"

    class NovaVisionConfig(NovaVisionParam):
        field: str = "textInput"

    class NovaVisionInput(NovaVisionParam):
        pass

    class NovaVisionOutput(NovaVisionParam):
        listen: str = "continuous"
        branch: str = "forward"

    class NovaVisionInputs(NovaVisionModel):
        pass

    class NovaVisionOutputs(NovaVisionModel):
        pass

    class NovaVisionConfigs(NovaVisionModel):
        pass

    class NovaVisionRequest(NovaVisionModel):
        pass

    class NovaVisionResponse(NovaVisionModel):
        pass

    class NovaVisionImage(NovaVisionParam):
        uID: str = ""
        mimeType: str = "image/jpg"
        encoding: str = "base64"
        value: Any = ""
        r_key: Optional[str] = ""
        shape_key: Any = None

    class NovaVisionPackage(NovaVisionModel):
        type: str = "component"
        name: str = "ImageStack"
        configs: Any = None
        uID: str = "local-node"
        flowUID: Optional[str] = None
        matchedID: Optional[str] = None
        debug: str = "False"
        api: str = "False"


class InputImage(NovaVisionInput):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[NovaVisionImage], NovaVisionImage] = Field(default_factory=list)
    type: str = "list"

    @root_validator(pre=True)
    def set_type_from_value(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        image_value = values.get("value", [])
        if isinstance(image_value, list):
            normalized_images = []
            for image in image_value:
                if isinstance(image, dict) and isinstance(image.get("value"), (bytes, bytearray, memoryview)):
                    image = dict(image)
                    image["value"] = base64.b64encode(bytes(image["value"])).decode("ascii")
                normalized_images.append(image)
            values["value"] = normalized_images
        values["type"] = "list" if isinstance(values.get("value", []), list) else "object"
        return values

    class Config:
        title = "Image"


class OutputImages(NovaVisionOutput):
    name: Literal["outputImages"] = "outputImages"
    value: List[NovaVisionImage] = Field(default_factory=list)
    type: Literal["list"] = "list"

    class Config:
        title = "Images"


class OutputPreview(NovaVisionOutput):
    name: Literal["outputPreview"] = "outputPreview"
    value: NovaVisionImage = Field(default_factory=NovaVisionImage)
    type: Literal["Image"] = "Image"

    class Config:
        title = "Stack Preview"


class OutputData(NovaVisionOutput):
    name: Literal["outputData"] = "outputData"
    value: int = 0
    type: Literal["number"] = "number"

    class Config:
        title = "Frame Count"


class StackSize(NovaVisionConfig):
    name: Literal["StackSize"] = "StackSize"
    value: int = Field(default=10, ge=1, le=64)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Stack Size"
        schema_extra = {"shortDescription": "Maximum number of frames to retain (1-64)."}
        json_schema_extra = schema_extra


class ResolutionWidth(NovaVisionConfig):
    name: Literal["ResolutionWidth"] = "ResolutionWidth"
    value: int = Field(default=1920, ge=64, le=1920)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Resolution Width"
        schema_extra = {"shortDescription": "Maximum stored frame width (64-1920)."}
        json_schema_extra = schema_extra


class ResolutionHeight(NovaVisionConfig):
    name: Literal["ResolutionHeight"] = "ResolutionHeight"
    value: int = Field(default=1080, ge=64, le=1080)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Resolution Height"
        schema_extra = {"shortDescription": "Maximum stored frame height (64-1080)."}
        json_schema_extra = schema_extra


class ClearBufferFalse(NovaVisionConfig):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "False"


class ClearBufferTrue(NovaVisionConfig):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "True"


class ClearBuffer(NovaVisionConfig):
    name: Literal["ClearBuffer"] = "ClearBuffer"
    value: Union[ClearBufferFalse, ClearBufferTrue] = Field(default_factory=ClearBufferFalse)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Clear Buffer"
        schema_extra = {"shortDescription": "Clear once when switched from False to True."}
        json_schema_extra = schema_extra


class ImageStackInputs(NovaVisionInputs):
    inputImage: InputImage = Field(default_factory=InputImage)


class ImageStackConfigs(NovaVisionConfigs):
    stack_size: StackSize = Field(default_factory=StackSize, alias="StackSize")
    resolution_width: ResolutionWidth = Field(default_factory=ResolutionWidth, alias="ResolutionWidth")
    resolution_height: ResolutionHeight = Field(default_factory=ResolutionHeight, alias="ResolutionHeight")
    clear_buffer: ClearBuffer = Field(default_factory=ClearBuffer, alias="ClearBuffer")

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class ImageStackOutputs(NovaVisionOutputs):
    outputImages: OutputImages = Field(default_factory=OutputImages)
    outputPreview: OutputPreview = Field(default_factory=OutputPreview)
    outputData: OutputData = Field(default_factory=OutputData)


class ImageStackRequest(NovaVisionRequest):
    name: Literal["ImageStack"] = "ImageStack"
    type: Literal["Request"] = "Request"
    inputs: Optional[ImageStackInputs] = Field(default_factory=ImageStackInputs)
    configs: ImageStackConfigs = Field(default_factory=ImageStackConfigs)

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Image Stack Request"
        schema_extra = {"target": "configs"}
        json_schema_extra = schema_extra


class ImageStackResponse(NovaVisionResponse):
    name: Literal["ImageStack"] = "ImageStack"
    type: Literal["Response"] = "Response"
    outputs: ImageStackOutputs = Field(default_factory=ImageStackOutputs)

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Image Stack Response"


class ImageStackExecutor(NovaVisionConfig):
    name: Literal["ImageStack"] = "ImageStack"
    value: Union[ImageStackRequest, ImageStackResponse] = Field(default_factory=ImageStackRequest)
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Image Stack"
        schema_extra = {"target": {"value": 0}}
        json_schema_extra = schema_extra


class ConfigExecutor(NovaVisionConfig):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: ImageStackExecutor = Field(default_factory=ImageStackExecutor)
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Task"
        schema_extra = {"shortDescription": "Accumulate the latest video frames."}
        json_schema_extra = schema_extra


class PackageConfigs(NovaVisionConfigs):
    executor: ConfigExecutor = Field(default_factory=ConfigExecutor)

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True


class PackageModel(NovaVisionPackage):
    configs: PackageConfigs = Field(default_factory=PackageConfigs)
    type: Literal["component"] = "component"
    name: Literal["ImageStack"] = "ImageStack"
    uID: str = "local-node"
    flowUID: Optional[str] = None
    matchedID: Optional[str] = None
    debug: str = "False"
    api: str = "False"

    @root_validator(pre=True)
    def map_legacy_executor_location(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)
        if "executor" in values or "Executor" in values:
            configs = values.get("configs") if isinstance(values.get("configs"), dict) else {}
            if "executor" not in configs:
                configs["executor"] = values.get("executor", values.get("Executor"))
            values["configs"] = configs
        values.pop("executor", None)
        values.pop("Executor", None)
        values.pop("field", None)
        return values

    @property
    def executor(self) -> ConfigExecutor:
        return self.configs.executor

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
        populate_by_name = True
        title = "Image Stack"


__all__ = [
    "ClearBuffer",
    "ClearBufferFalse",
    "ClearBufferTrue",
    "ConfigExecutor",
    "ImageStackConfigs",
    "ImageStackExecutor",
    "ImageStackInputs",
    "ImageStackOutputs",
    "ImageStackRequest",
    "ImageStackResponse",
    "InputImage",
    "OutputData",
    "OutputImages",
    "OutputPreview",
    "PackageConfigs",
    "PackageModel",
    "ResolutionHeight",
    "ResolutionWidth",
    "StackSize",
]

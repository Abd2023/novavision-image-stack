from __future__ import annotations

from typing import Any

PackageHelper = None


def _load_package_helper():
    global PackageHelper
    if PackageHelper is not None:
        return PackageHelper
    try:
        from sdks.novavision.src.helper.package import PackageHelper as helper  # type: ignore
    except Exception:
        return None
    PackageHelper = helper
    return PackageHelper

try:
    from components.ImageStack.src.models.PackageModel import (  # type: ignore
        ConfigExecutor,
        ImageStackExecutor,
        PackageConfigs,
        PackageModel,
    )
except ImportError:
    try:
        from src.models.PackageModel import (
            ConfigExecutor,
            ImageStackExecutor,
            PackageConfigs,
            PackageModel,
        )
    except ImportError:
        from novavision.image_stack.models.PackageModel import (  # type: ignore
            ConfigExecutor,
            ImageStackExecutor,
            PackageConfigs,
            PackageModel,
        )


def _request_data(context: Any) -> dict[str, Any]:
    request = getattr(context, "request", None)
    data = getattr(request, "data", None)
    return data if isinstance(data, dict) else {}


def _build_without_sdk_helper(context: Any, package_configs: PackageConfigs) -> PackageModel:
    data = _request_data(context)
    package_fields = getattr(PackageModel, "model_fields", None) or getattr(PackageModel, "__fields__", {})
    defaults = {
        "uID": data.get("uID", getattr(context, "uID", None)),
        "flowUID": data.get("flowUID", getattr(context, "flowUID", None)),
        "matchedID": data.get("matchedID", getattr(context, "matchedID", None)),
        "debug": data.get("debug", getattr(context, "debug", "False")),
        "api": data.get("api", getattr(context, "api", "False")),
    }
    metadata = {
        name: value
        for name, value in defaults.items()
        if name in package_fields and value is not None
    }
    return PackageModel(configs=package_configs, **metadata)


def build_response(context: Any) -> PackageModel:
    """Build the standard NovaVision package response for Image Stack."""

    executor_response = ImageStackExecutor(value=context.response)
    package_configs = PackageConfigs(executor=ConfigExecutor(value=executor_response))

    package_helper = _load_package_helper()
    if package_helper is not None:
        package = package_helper(packageModel=PackageModel, packageConfigs=package_configs)
        return package.build_model(context)

    return _build_without_sdk_helper(context, package_configs)

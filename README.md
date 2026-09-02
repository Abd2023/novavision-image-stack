# NovaVision Image Stack

NovaVision component that keeps the most recent video frames in a bounded,
per-node in-memory FIFO buffer keyed by the stable package node identity. Every
frame is downsampled when necessary and
JPEG-compressed at quality 75 before it is stored. The component returns the
newest frame first, exposes the current frame count, and generates a
single-image contact-sheet preview for Suite's Image View widget.

The implementation is behavior-equivalent to the Roboflow Image Stack v1
workflow block.

## Repository status

This repository follows NovaVision's Package template. `setup.py` declares the
installable package, `src/executors/ImageStack.py` contains the component,
`src/models/PackageModel.py` defines the Suite schema, and
`src/utils/response.py` builds responses through NovaVision's `PackageHelper`.
Image-runtime files such as `service.py`, Dockerfiles, and image-level
requirements files intentionally belong to the parent NovaVision image and are
not part of this package repository.

The repository contains no AI model, dataset, or model weights.

## Validation

```powershell
python setup.py --name
python -m compileall -q src setup.py
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

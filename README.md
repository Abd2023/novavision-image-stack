# NovaVision Image Stack

NovaVision component that keeps the most recent video frames in a bounded,
per-node in-memory FIFO buffer keyed by the stable package node identity. Every
frame is downsampled when necessary and
JPEG-compressed at quality 75 before it is stored. The component returns the
newest frame first, exposes the current frame count, and generates a
single-image contact-sheet preview for Suite's Image View widget.

The implementation is behavior-equivalent to the Roboflow Image Stack v1
workflow block. See [NOTICE](NOTICE) for attribution and reference links.

## Repository status

This repository contains the NovaVision OpenCV component implementation. It
does not contain an AI model, dataset, or model weights.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

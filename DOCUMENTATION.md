# NovaVision Image Stack Documentation

## Overview

Image Stack is a NovaVision OpenCV component for temporal frame buffering. It
accepts the current video frame on every workflow cycle, compresses it to JPEG,
and keeps a bounded per-node FIFO history in process memory.

The component is a utility block. It does not train or run an AI model, and it
does not require datasets or weights.

## Interface

| Kind | Name | Description |
| --- | --- | --- |
| Input | `inputImage` | One image or image list; the first image is the current frame. |
| Config | `StackSize` | Maximum history size, 1-64, default 10. |
| Config | `ResolutionWidth` | Maximum stored width, 64-1920, default 1920. |
| Config | `ResolutionHeight` | Maximum stored height, 64-1080, default 1080. |
| Config | `ClearBuffer` | Clears on a False-to-True transition, default False. |
| Output | `outputImages` | Base64 JPEG NovaVision image list, newest first. |
| Output | `outputData` | Current frame count. |

## Algorithm

1. Read the first `inputImage` for the current workflow cycle.
2. Decode it with the NovaVision Redis/image helper when running in Suite.
3. Downsample oversized frames without changing aspect ratio or upscaling.
4. JPEG-encode the frame at quality 75.
5. Append the compressed frame to the front of a per-flow/per-node deque.
6. Evict the oldest frame automatically when the configured limit is reached.
7. Return the deque contents newest first and the current count.

The buffer and clear-toggle state are held in the executor bootstrap dictionary.
They survive new executor instances in the same process and reset after a
service restart or redeploy.

## Clear behavior

`ClearBuffer=True` clears the buffer before adding the current frame only when
the previous cycle used `False`. While the setting remains `True`, new frames
continue accumulating. Set it back to `False` before enabling it again for a
second reset.

## Local validation

```powershell
python -m pytest
python service.py
python apps\run_sample_client.py
python -m compileall -q src apps service.py
```

## Suite validation

Import the package into the NovaVision OpenCV image, connect Video Feed
`outputImage` to Image Stack `inputImage`, and connect `outputImages` to a
downstream image consumer. The count should grow until `StackSize`; the oldest
frame should then be evicted. A rising-edge Clear Buffer action should return a
one-frame stack, and a process restart should reset the in-memory history.


# API specification

The authoritative declarations are `include/ai_engine.h`:60 public functions,
existing signatures and x86 stdcall aliases. Current native version0.14.6,
deliveryv23.6, Worker protocol26. Interface details are generated in
[offline HTML](易语言_DLL_API_说明.html) and [module declarations](../examples/e_language_onnx_module.txt).

| Parameter | Contract |
| --- | --- |
| YOLO device | 0 AUTO,1 DirectML,2 CPU,3 TensorRT |
| OCR device | 0 AUTO,1 DirectML,2 CPU |
| device_id | Nonnegative; DXGI index for DirectML, CUDA index for TensorRT; CPU callers supply0 |
| session_count | Positive execution-slot count, independent of caller/internal thread counts |

AUTO applies `device_id` separately to each backend's index space. DXGI0 may be
an integrated GPU; CUDA0 identifies the first CUDA device. Confirm the selected
adapter/backend through the model runtime-status JSON. Explicit backend failure
returns an error; AUTO alone may choose another backend.

Example parameters: CPU `(2,0,5)`, DirectML `(1,DXGI_index,5)`, TensorRT
`(3,CUDA_index,5)` in device/device_id/session_count positions of YOLO loading.
Choose slot count using the business calibration tool, not by increasing it blindly.

Five caller threads may share one model handle. Release handles only after their
users finish. Worker shutdown is a coordinated last-client operation; one host
must not terminate a Worker still serving another business process.

See [易语言 examples](../examples/e_language.md) for loading, QPC full-call timing,
runtime-state inspection and error handling. Governance does not add public APIs.

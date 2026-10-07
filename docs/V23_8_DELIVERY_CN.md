# v23.8 / 0.14.8 交付与验证

本轮候选修复YOLO预处理和AUTO校准身份，工具支持显式P95门槛，公开60签名、stdcall别名及Worker协议26保持兼容。用户本轮接受每路完整调用P95≤20ms；YOLO26和V8的正式采样、输入、二进制SHA、配置及证据边界见[本机报告](YOLO26_LOCAL_20MS_CN.md)。

`output/`仅有整套`CQ_X86.dll`、`CQ_AI_worker.exe`和生成的`易语言_DLL_API_说明.html`。交付以`release/current.json`选择；历史v23.5/v23.6/v23.7不可覆盖。新候选只有通过实际配对、正确性、缓存、AUTO、30分钟、文档和真实Finish gate后才事务晋升，保留旧整套回滚。

最终YOLO26配置：TensorRT FP16、2执行槽、五调用线程、Graph开启，已通过30分钟稳定性并晋升。当前可选模块在两种模式每轮均改善P95≥10%，与原Graph未通过的模块不能混用。AUTO须读取这份bus工作负载与模块对应的profile。V8配置：DirectML FP32、3槽、五调用线程，读取重新生成的V8 profile。NVIDIA模块和固定依赖独立交付，不能把可选模块塞入output；实际业务目录可把模块放在Worker旁。只更新核心三文件而没有对应依赖/profile，不能保证复现20ms。

工具包保留完整调用基准、采样CSV、设备/内存诊断、校准入口和AUTO冷缓存重启验证工具。`--p95-limit-ms`或PowerShell `-P95LimitMs`显式传20；旧调用默认30，不改变旧20/30ms业务验收。准确使用方法见[构建和验收说明](YOLO_OPTIMIZATION_CN.md)。

Python x64 Wheel另行安装，在新的虚拟环境以`python -I`运行实际CV/OCR/YOLO验收；最终结果登记到manifest证据。较宽native检查中的两个既有CV 5ms绝对期限在本E5机器上新旧版本都失败，日志保留，不能称全库全通过。

本机固定图片达标与五窗口正式发布分开登记：`local_benchmark_validation`只表示本轮范围；`target_performance_validation`仍待验，`formal_release`保持false。30分钟及最终晋升详情在本机报告更新，文件事实以当前manifest和不可变证据为准。

```powershell
python scripts/generate_e_language_api_doc.py --manifest docs/YOLO_CANDIDATE_METADATA.json --module examples/e_language_onnx_module.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1 -Metadata docs/YOLO_CANDIDATE_METADATA.json -ValidationDir outpush/v23.8-validation -StageDir outpush/new-v23.8-delivery -SkipNvidiaPacking
```

示例目录必须是新目录，实际打包还应指定已验证x86/x64和Worker构建路径。事务检查会核验真实配对二进制、原始检测JSON、所有正式单路及同二进制1800秒稳定性。占用时保留原文件，验证失败回滚，不结束用户Worker。

2026-10-08实际晋升完成：当前指针选择`release/v23.8/manifest.json`，真实output加载器125次并发调用零错误。旧三文件回滚位于`outpush/rollback/before-v23.8-20261008-011941-241/`。源码提交`5ade814`已通过SSH推送。

独立NVIDIA包为`D:/codex-delivery/CV_OCR/v23.8/CQ_AI_NVIDIA_v23.8_TRT10.13.3_CUDA12.8.zip`，大小2,269,387,626字节，SHA256 `74ba31cbb5237c7d797fd43b7b24b813e95c09aa5439165f9c4fc4449d6ce365`；ZIP CRC、全部依赖逐文件SHA及模块SHA已验证。Wheel采用模块C，SHA256 `dde232a304356eb1c903dc86dde8f4e4fe5242aca07978501081aac1bcd68e85`。配置文件在`release/v23.8/evidence/business-calibration.tsv`和`v8-business-calibration.tsv`，工具包也单独包含这些参数。

晋升后的默认严格Finish gate返回`ok=true, allowed=true`。工具包移至同一独立交付目录，SHA256 `e184fbc943734772c1004a037ad41a93c17edccadca9dce3453eceafaf30d113`。原始CSV、失败尝试及构建记录按原相对路径保存在`YOLO_v23.8_raw_benchmarks_and_build.zip`，SHA256 `a0958fc306f887bee011b81acc29c7a238c228209d2d40f8deb20e683cdfe859`；归档通过CRC，删除的CSV逐文件SHA通过比对。两份早期GPU采样的后续日志保留在`YOLO_v23.8_tail_telemetry_final.zip`。核心ZIP及Wheel仍以不可变`release/v23.8/`为准。

本次临时构建、重复交付及采样CSV已清理；49项input、历史交付和旧output回滚保护记录通过。工程检查虚拟环境移到`D:/codex-toolchains/CV_OCR/aios-venv`并逐文件验证，检查工具源码仍在`.cache/aios-source`。容量和删除审计见[存储说明](REPOSITORY_STORAGE_CN.md)及`outpush/yolo26-15ms-implementation/storage-after-cleanup-actual.json`。

# v23.8 / 0.14.8 交付与验证

本轮候选修复YOLO预处理和AUTO校准身份，工具支持显式P95门槛，公开60签名、stdcall别名及Worker协议26保持兼容。用户本轮接受每路完整调用P95≤20ms；YOLO26和V8的正式采样、输入、二进制SHA、配置及证据边界见[本机报告](YOLO26_LOCAL_20MS_CN.md)。

`output/`仅有整套`CQ_X86.dll`、`CQ_AI_worker.exe`和生成的`易语言_DLL_API_说明.html`。交付以`release/current.json`选择；历史v23.5/v23.6/v23.7不可覆盖。新候选只有通过实际配对、正确性、缓存、AUTO、30分钟、文档和真实Finish gate后才事务晋升，保留旧整套回滚。

最终候选YOLO26：TensorRT FP16、2执行槽、五调用线程、Graph开启。当前可选模块在两种模式每轮均改善P95≥10%，与原Graph未通过的模块不能混用；完整稳定性通过前仍不晋升。AUTO须读取这份bus工作负载与模块对应的profile。V8候选：DirectML FP32、3槽、五调用线程，读取重新生成的V8 profile。NVIDIA模块和固定依赖独立交付，不能把可选模块塞入output；实际业务目录可把模块放在Worker旁。只更新核心三文件而没有对应依赖/profile，不能保证复现20ms。

工具包保留完整调用基准、采样CSV、设备/内存诊断、校准入口和AUTO冷缓存重启验证工具。`--p95-limit-ms`或PowerShell `-P95LimitMs`显式传20；旧调用默认30，不改变旧20/30ms业务验收。准确使用方法见[构建和验收说明](YOLO_OPTIMIZATION_CN.md)。

Python x64 Wheel另行安装，在新的虚拟环境以`python -I`运行实际CV/OCR/YOLO验收；最终结果登记到manifest证据。较宽native检查中的两个既有CV 5ms绝对期限在本E5机器上新旧版本都失败，日志保留，不能称全库全通过。

本机固定图片达标与五窗口正式发布分开登记：`local_benchmark_validation`只表示本轮范围；`target_performance_validation`仍待验，`formal_release`保持false。30分钟及最终晋升详情在本机报告更新，文件事实以当前manifest和不可变证据为准。

```powershell
python scripts/generate_e_language_api_doc.py --manifest docs/YOLO_CANDIDATE_METADATA.json --module examples/e_language_onnx_module.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1 -Metadata docs/YOLO_CANDIDATE_METADATA.json -ValidationDir outpush/v23.8-validation -StageDir outpush/new-v23.8-delivery -SkipNvidiaPacking
```

示例目录必须是新目录，实际打包还应指定已验证x86/x64和Worker构建路径。事务检查会核验真实配对二进制、原始检测JSON、所有正式单路及同二进制1800秒稳定性。占用时保留原文件，验证失败回滚，不结束用户Worker。

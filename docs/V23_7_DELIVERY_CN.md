# v23.7 / 0.14.7 交付与验证

本版保持60个公开函数、stdcall别名、错误码、结果布局和Worker协议26，兼容V5、V5u/V8/V11及V26普通检测头、原生端到端检测头。模型要求与工具用法见[兼容说明](YOLO_MODEL_COMPATIBILITY_CN.md)。

## 成品

`output/`仅有配套`CQ_X86.dll`、`CQ_AI_worker.exe`、`易语言_DLL_API_说明.html`。当前三文件ZIP和x64 Python Wheel由`release/current.json`选择，文件SHA-256、源代码快照及不可变证据位于对应manifest；历史v23.5/v23.6包保持原哈希。三文件必须整套更新，不混用Worker。

工具包包含五路完整调用基准、目标机一键校准、模型检查/准备工具和易语言声明/QPC示例。可选TensorRT ABI3模块与基础包分开，NVIDIA固定依赖复用外部归档中的`nvidia/`目录；部署步骤见[YOLO说明](YOLO_OPTIMIZATION_CN.md)。新版不复用旧模块、engine校准结果。

## 可重复验证

参考运行库固定为ORT1.24.4/DirectML1.15.4。V5导出基线为官方v7.0，V11/V26为Ultralytics8.4.28；原文件、导出文件、运行库及成品分别留存SHA-256。参考解码使用本库的最近邻、padding114及跨类别NMS0.45，而非官方默认预处理/NMS。

- 用户V26输入为640×640，输出`[1,300,6]`；bus图、置信度0.5得到4人和1公交车。旧版把记录数误当类别数的问题已修复。
- V5 FP32、V11 FP32、V26普通头和原生端到端头、旧best/smc及动态V11：路径/内存、CPU/DirectML、五线程共享句柄与独立参考对照。旧模型检测JSON与0.14.6同后端结果完全相等。
- 格式单元测试覆盖objectness、无objectness、端到端不二次NMS、单类歧义、转置、空结果、阈值边界、非法类别、NaN/Inf及损坏元数据/嵌套NMS。
- 公开导出与独立x86加载器检查配套Worker绝对路径、协议、请求关联和并发结果；受影响的x86/x64、OCR/CV与Python安装回归保留日志。

真实模型矩阵在x64、x86分别通过14组CPU/DirectML用例；动态V11使用640输入。x64覆盖全部路径/内存加载，x86静态模型同时覆盖两种加载方式，动态V11覆盖路径加载。五个不同版本模型同时加载与200次并发调用通过，中文路径和中文标签、非法类别、NaN及歧义模型均有独立检查。

原生回归：x86的20项通过；x64的16项中首次有一项仍断言旧版本号，更新该测试后复测通过。最后的六列歧义识别收紧后，x64重测4项、x86重测3项均通过，清洁构建另通过4项。Python Wheel在新虚拟环境安装验证通过；TensorRT模块缺失、ABI2旧模块、NVIDIA运行库缺失和无NVIDIA时CPU可用这4种场景通过。

五路完整调用计时使用QPC，加载另列；每线程预热100次、每轮1000次，持续/同时发起各3轮，共30,000次正式调用、零错误。下表为本机i7-13620H CPU运行，旧best模型320输入，5个执行槽、每槽1个内部线程，各轮整体分位数，单位ms；每路P50/P95/P99、CPU、内存和吞吐保存在原始JSON/CSV。

| 模式 | 第1轮 P50 / P95 | 第2轮 P50 / P95 | 第3轮 P50 / P95 |
| --- | ---: | ---: | ---: |
| x86＋Worker持续调用 | 48.003 / 89.212 | 44.478 / 65.555 | 44.678 / 57.820 |
| x86＋Worker同时发起 | 44.260 / 57.980 | 45.079 / 57.241 | 47.317 / 65.462 |

V26的640输入变化帧测试运行1800.091秒，共65,313次调用、零错误，每次结果与对应帧的顺序基线相等。整体P50/P95/P99为135.780/172.876/242.080ms，吞吐36.283次/秒。这是不同模型、不同输入尺寸的稳定性负载，不能作为旧320模型的速度对比。预热120秒后1556次内存采样中，客户端均值无增长，Worker前后均值增加0.123MiB，未发现持续增长、死锁或请求错配。

旧模型性能对比采用已登记v23.6 Wheel中的0.14.6 DLL与0.14.7，统一基准程序和ORT1.24.4，交替测试顺序，并只固定测试进程的CPU集合以降低混合核心调度差异。三对测试的比值中位数：持续P50 −1.02%、P95 −10.61%；同时发起P50 −1.35%、P95 +0.09%，最慢单路指标也通过5%门槛。未出现超过5%的可重复回退；结论与每轮数据记录在`performance-comparison.json`。早期误用历史0.14.5比较目录产生的记录保留在`outpush/v23.7-validation`，排除出0.14.6回归验收，不作为通过证据。

离线HTML的63个章节/导航、搜索和hash跳转通过实际内联JavaScript测试。当前浏览器安全策略拒绝本地文件预览，因此未将浏览器视觉检查标为通过；离线资产与交互逻辑已经验证。

## 重建、打包与晋升

```powershell
powershell -File scripts/build_delivery.ps1 -PythonExe python -Nvidia
# 完成与最终二进制SHA绑定的验证，更新候选元数据后生成HTML。
python scripts/generate_e_language_api_doc.py --manifest docs/YOLO_CANDIDATE_METADATA.json --module examples/e_language_onnx_module.txt
powershell -File scripts/package_python_x64.ps1 -Metadata docs/YOLO_CANDIDATE_METADATA.json -SkipNativeBuild -SkipNativeTests -OutputDir outpush/python-new
powershell -File scripts/package_e_language.ps1 -Metadata docs/YOLO_CANDIDATE_METADATA.json -ValidationDir outpush/v23.7-validation -StageDir outpush/delivery-new
```

这些示例目录必须是新目录。打包先核验真实二进制、版本、哈希及新30分钟报告，再生成HTML/ZIP哈希。晋升对原output进行整套备份并事务替换；占用保留原文件，验证失败回滚，不结束业务进程。`scripts/promote_release.ps1`只登记一次，不能覆盖已注册版本。

## 目标机待验

本机i7-13620H的CPU/DirectML证据不能替代E5-2696 v4＋RTX2070、五窗口实际负载。TensorRT仅验证模块构建和依赖缺失错误；GPU执行、缓存、FP16/Graph和30分钟GPU稳定性仍待验。本轮交付状态为功能可用、目标机性能待验。

解压工具包，在目标机执行：

```powershell
powershell -File tools/run_yolo_acceptance.ps1 -RuntimeDir D:/your/application -Model D:/models/best.onnx -ImagesDir D:/businessBMP -BusinessWindowsActive
```

只有五窗口真实运行时使用`-BusinessWindowsActive`。最终标准仍是每路完整调用P50≤20ms、P95≤30ms；FP16必须数量/类别一致、坐标误差≤1像素、分数差≤0.02，Graph每轮P95改善≥10%。待验项不会因编译或内部GPU计时而标成通过。

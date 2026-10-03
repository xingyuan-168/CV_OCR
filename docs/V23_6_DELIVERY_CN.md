# v23.6 DLL、Worker 与离线 HTML 交付

当前源码版本为 **0.14.6**，交付标识 **v23.6**，Worker协议 **26**。本轮先交付最新可用版，目标E5-2696 v4、128GB、RTX2070五窗口性能另行验收。

## 交付边界

`output/` 仅包含以下配套文件：

```text
CQ_X86.dll
CQ_AI_worker.exe
易语言_DLL_API_说明.html
```

DLL为x86，Worker为x64。公开函数保持60个，原有签名及stdcall别名保留；正整数session_count仍表示执行槽容量。YOLO支持设备0/1/2/3，OCR和AI_InitEx仅支持0..2。协议26新管道使用`cq_ai_worker_v26_core_0146`，与之前0.14.5协议26候选及协议25成品区分。

当前三文件ZIP和Wheel已登记到release/v23.6，由release/current.json选择；可选NVIDIA ZIP仅在独立交付目录保留一份，基准/校准工具ZIP及本机日志保留在outpush。归档和恢复步骤见[仓库容量说明](REPOSITORY_STORAGE_CN.md)。历史release/v23.5的ZIP、Wheel和成员哈希不修改。易语言业务目录启用NVIDIA时，可以另行解压可选包；仓库output仍只维护基础三文件。

## 构建与验证顺序

1. 以CMake Release构建x86 DLL、x64 Worker和可选TensorRT模块，记录构建缓存、源码提交与源文件SHA。
2. 计算DLL/Worker哈希，执行版本、内嵌运行库、PE依赖、导出和绝对路径加载验收。
3. 执行受影响x86/x64的YOLO传输/生命周期、OCR/CV、ANSI路径、Python绑定回归。
4. 新版本五线程共享句柄，持续与同时发起各三轮，每路预热100次、正式1000次；独立计时，不启用逐帧诊断或ORT profiling。
5. 新版本五路变化帧运行1800秒，每个返回结果与同图顺序参考逐次比较；记录CPU、内存和稳定性。
6. 更新`docs/YOLO_CANDIDATE_METADATA.json`与本报告实测；根据二进制哈希生成HTML，再计算HTML和ZIP哈希。
7. `package_e_language.ps1`验证上述报告、对候选三文件做独立加载检查，备份旧output后整组替换，再按output绝对路径复验。占用时保留原目录，失败时恢复旧配套。

统一入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
# 已单独完成可选NVIDIA包且其模块/哈希未变时：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1 -SkipNvidiaPacking
```

重复打包选择新的`-StageDir outpush/新目录`，已有成品与报告不删除。回滚文件和原哈希放在`outpush/rollback/`。`scripts/verify_current_release.ps1 -AllowSourceCandidate`同时检查历史原始哈希和当前源码版本一致性。

## 本轮实测

最新二进制共60,000次正式调用（持续/同时发起各三轮，完整独立重复两组）零错误；1800.141秒变化帧正式调用39,708次零错误，每次结果与同图顺序参考完全相等。x86 16项与x64 12项受影响回归通过，Python Wheel离线安装后CV/OCR/多模型YOLO验收通过。x86小字号多行500次测试原300秒超时，改为900秒后387.39秒完成，结果检查未减少。

| 测量组/模式/轮 | 最差单路P50 ms | 最差单路P95 ms | 最差单路P99 ms | 吞吐 次/秒 |
|---|---:|---:|---:|---:|
| initial-measurement/continuous/1 | 200.375 | 242.701 | 271.747 | 24.784 |
| initial-measurement/continuous/2 | 202.115 | 242.406 | 277.400 | 24.617 |
| initial-measurement/continuous/3 | 202.755 | 245.485 | 275.554 | 24.454 |
| initial-measurement/simultaneous/1 | 198.314 | 228.783 | 254.079 | 23.658 |
| initial-measurement/simultaneous/2 | 184.687 | 226.131 | 260.156 | 32.918 |
| initial-measurement/simultaneous/3 | 26.640 | 41.755 | 55.377 | 153.029 |
| latest/continuous/1 | 27.529 | 76.989 | 112.998 | 145.124 |
| latest/continuous/2 | 31.134 | 83.306 | 114.898 | 130.004 |
| latest/continuous/3 | 187.850 | 238.427 | 283.243 | 31.926 |
| latest/simultaneous/1 | 199.167 | 227.843 | 263.553 | 23.609 |
| latest/simultaneous/2 | 197.337 | 230.420 | 268.223 | 23.569 |
| latest/simultaneous/3 | 199.296 | 226.631 | 259.605 | 23.620 |

30分钟变化帧最差单路P50/P95/P99为217.575/300.047/459.347ms，吞吐22.058次/秒。首组、复测及全部每路CSV均保留，**本轮CPU表现波动明显，未达到20/30ms，不把较快的一轮或上一版本约25ms作为当前保证**。

内存记录包含一次并行诊断模型加载：t234..294秒Worker私有内存升至403.945MiB峰值，释放后227.008MiB，比早期210.197MiB多保留16.811MiB。完整首尾均值警告保留在summary.json，不能称为零内存增加；最后600秒544个采样中调用方114.535MiB、Worker227.008MiB均完全稳定，未见持续增长。目标机应在实际单业务配置下复测。

阶段诊断与长期测试、另一个五槽模型、打包重叠：550个关联请求IPC P95约13.578ms，属于额外负载下诊断，不代表目标机通信成本；保留stages.json，目标机实测超过2ms后才决定共享内存。

绝对路径x86加载器验证版本0.14.6、实际配套Worker路径、线协议26、60个公开导出、中文模型路径以及125次五线程完整结果。缺少TensorRT模块/缺少NVIDIA依赖均返回具体错误，CPU无NVIDIA依赖仍可启动。工具脚本通过Windows PowerShell5.1烟测；校准记录导出及AUTO回读已烟测。HTML接口/原型校验及离线搜索、导航、深链接脚本检查通过；Codex浏览器webview两次无法挂载，因此浏览器视觉预览未验，未将其记为通过。

二进制SHA-256：

```text
CQ_X86.dll  5a6f5f04930173041c90a9938ed1a54b2b3391da6d7c140c6e9e348c90528d29
CQ_AI_worker.exe  381e4130de614c5515be70241666f9fd07a445cdebd6bbbf5ab4b0ffe0dd0f09
```

原始报告在`outpush/v23.6-validation/`，新交付清单在`outpush/v23.6-delivery/manifest.json`；旧候选数据见[历史验证报告](YOLO_VALIDATION_CN.md)，不能替代本轮。

本机为i7-13620H、Intel UHD、约16GB内存，无NVIDIA GPU。**目标20/30ms尚未验收，TensorRT执行、FP16精度、Graph收益、GPU缓存和30分钟GPU稳定性待目标机实测。**

## 目标机一键验收

基础三个文件部署到业务目录；使用设备3前解压可选NVIDIA包的`CQ_YOLO_TensorRT.dll`和`nvidia/`到Worker旁。系统NVIDIA驱动必须暴露CUDA Driver API至少12.8，nvcuda.dll由驱动提供。固定传统TensorRT10.13.3.9，CUDA runtime12.8.90、NVRTC12.8.93、cuBLAS12.8.4.1；对应许可随包交付。

解压工具ZIP，在五个业务窗口实际运行时，以64位Python3.9+执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run_yolo_acceptance.ps1 `
  -RuntimeDir D:/业务程序目录 -Model D:/模型/best.onnx `
  -ImagesDir D:/五窗口变化BMP -PythonExe python -BusinessWindowsActive
```

工具按绝对路径验证业务DLL及配套Worker，在独立报告目录复制哈希相同的运行文件；业务目录不修改。基准仅释放自己的模型，保留正在服务业务的Worker。扫描CPU/DirectML/TensorRT FP32/FP16和1/2/3/5执行槽，CPU扫描物理核预算内1/2/4/6/8内部线程。FP16每图检测数量/类别必须一致，匹配框坐标差≤1像素、分数差≤0.02；失败不发布FP16记录。Graph须成功捕获，并在两种模式的每轮P95改善至少10%。

报告包含硬件、实际版本和文件SHA、每路/整体P50/P95/P99、吞吐、CPU、内存以及nvidia-smi可用时的显存采样。新建隔离缓存进行首次构建、命中、损坏重建检查，并单独跑30分钟变化帧；正式性能与阶段诊断分开。`-CalibrateOnly`跳过长时间稳定性，`-SmokeOnly`仅检查工具，不构成性能验收。

回传整个acceptance时间戳目录，包括calibration-report.json、acceptance-summary.json、原始samples/resources CSV、gpu.csv、stages.json和日志。最终每路完整调用P50≤20ms、P95≤30ms；工具必须明确区分功能通过、测量未达标和目标机验收完成。基线fixtures不能替代实际五窗口帧。

若目标机通信、打包和收发P95仍超过2ms，下一步实施共享内存图像槽；否则按排队、预处理、执行、后处理阶段继续优化。当前本机没有证据可以宣布目标GPU性能达标。

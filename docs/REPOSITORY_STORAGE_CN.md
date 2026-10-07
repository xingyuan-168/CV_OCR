# 仓库容量与依赖恢复

2026-10-08 v23.8交付后，按登记路径清理本轮构建、重复包、隔离运行库和采样CSV。清理后、最终交付提交前，含`.git`实测1.433GiB；提交后继续核对1.5GiB上限，实际记录在`outpush/yolo26-15ms-implementation/storage-after-cleanup-actual.json`。49项input、历史release及旧output回滚保护记录通过；没有删除原有用户交付和回滚。

原始证据和构建ZIP保留在`D:/codex-delivery/CV_OCR/v23.8/`，通过CRC与CSV逐文件SHA验证，失败尝试也保留。目录及哈希见[v23.8说明](V23_8_DELIVERY_CN.md)与[本轮存储审计](evidence/STORAGE_V23_8_SUMMARY.json)。工程检查虚拟环境逐文件核对后移至`D:/codex-toolchains/CV_OCR/aios-venv`，源码留在`.cache/aios-source`，可通过下式运行真实AIOS检查：

```powershell
& 'D:/codex-toolchains/CV_OCR/aios-venv/Scripts/python.exe' -c 'from codex_ai_os.cli.app import main; main()' doctor . --json
```

清理函数修正了空目录文件集合和零字节统计，实际空目录回归通过。删除审计按文件长度统计，不把外部保留量或Git重打包减少量等同于物理磁盘释放量。

项目源码、交付、依赖和临时构建分别管理。原目录约19.79GiB，其中outpush
11.96GiB、third_party6.71GiB；原生src约0.73MiB，Git历史约0.21GiB。
重复运行库、候选及构建缓存导致了目录膨胀，源码不需要这些重复副本。

## 保留规则

- input、output三文件、Git历史、models、测试夹具、注册release和原回滚不清理。
- 当前交付由release/current.json选择。v23.5/v23.6注册包和证据保持原始哈希。
- NVIDIA可选包单独放在D:/Projects-tools/交付归档/CV_OCR/v23.6，仓库仅登记文件名、大小、SHA；换机器可用任意归档目录。
- outpush/governance-v23.6保留容量审计、执行清单和从旧候选提取的报告。独有图片、模型按SHA归档；重复图片保留来源索引。
- third_party保留编译需要的OpenCV静态库、ORT/DirectML SDK、NVIDIA/CUDA头文件和许可证。下载、完整SDK、解压和中间对象可恢复。
- 完成一次构建/验收并清理临时目录后，整个仓库包含.git的目标容量不超过1.5GiB。

## 清理与审计

```powershell
python scripts/repository_storage.py prepare --archive-dir D:/Projects-tools/交付归档/CV_OCR/v23.6
powershell -File scripts/clean_repository.ps1 -PythonExe python
powershell -File scripts/clean_repository.ps1 -PythonExe python -Apply
```

默认只预览。先验证归档的ZIP CRC和逐文件SHA、保护文件SHA及保留报告，
再按configs/storage-policy.json登记的绝对路径删除。拒绝越界、目录联接和未登记路径；
文件占用时报告失败，不终止业务进程。删除文件后仅移除空目录。
构建目录需要复建验证成功后另行生成`prepare --include-builds`清单。

审计分别记录仓库减少量、外部保留量和真正删除量。把包移出仓库不算释放磁盘。
具体执行记录位于outpush/governance-v23.6；不把容量估计当作已完成结果。

## 缓存与构建

```powershell
powershell -File scripts/prepare_dependencies.ps1 -PythonExe python
python scripts/prepare_nvidia_headers.py
powershell -File scripts/build_delivery.ps1 -PythonExe python -Nvidia
```

配置、架构和全部安装文件SHA一致时直接复用，不下载、不重新构建。
缓存缺失或损坏才恢复。OpenCV固定源提交、x86补丁、/MT和IPP；使用
`-ForceRebuild`可显式重建。安装后记录本机实际产物SHA，因此不同机器编译时间戳不会造成永远重建。
Windows依赖准备通过vswhere发现VS2022的实际安装目录，兼容BuildTools、
Community、Professional和Enterprise；VC143运行库从该安装目录提取。

NVIDIA头文件以SDK10.13.3.9实际文件SHA为准。公开源码标签头文件与SDK不完全相同，
不能互换作为“同一构建”。缺失时下载固定SHA的官方SDK到系统临时目录，仅留下所需头文件和许可证。
该SDK原文件名包含cuda-12.9；当前可选运行包使用CUDA12.8运行库，不改写来源名称。
CUDA轮子版本、URL和SHA固定在configs/dependencies.lock.json，构建不要求本机GPU。

## 统一打包

基础包不要求NVIDIA依赖或可选ZIP。复用已注册交付：

```powershell
python scripts/package_delivery.py --reuse-current --out outpush/verified-copy
```

新候选使用build/release-x86、release-worker和release-x64；必须提供与这些二进制
哈希一致的正式五路、30分钟及独立加载报告。重新编译不会自动继承旧二进制的性能证据。
`--metadata`可指定新候选的验证元数据；`--model`/`--image`明确选择测试输入。
可选包须显式指定`--nvidia-only --nvidia-deps ... --nvidia-zip ...`；已有包不覆盖。
Python轮子与诊断staging成功后自动移除，`--keep-stage`仅用于排错。
核心ZIP生成成功后也清理其staging；`--no-zip`保留三文件供人工核验。

注册新交付使用promote_release.ps1的PackageDir、PythonWheel和SelectCurrent参数。
已注册版本不能替换。配套加载、原生回归及候选加载须串行执行，避免同协议Worker共享实例影响目录核验。
当前三文件版本以release/current.json为准；v23.7仅新增小型ABI3模块更新包，复用唯一外部NVIDIA依赖归档，禁止在仓库再次展开完整SDK；目标RTX2070五路20/30ms、FP16和Graph验收继续待验。

v23.7交付后清理了本轮五个构建目录、测试运行库副本、诊断staging和已归档参考模型，共663.154MiB；清理后、提交前仓库含Git为1.402GiB。新交付、旧回滚、全部验证记录和可选包均保留，提交后容量另存`outpush/v23.7-validation/storage-final.json`并继续检查1.5GiB上限。原input及历史交付的25项保护指纹与原output回滚副本一致。

隔离导出环境、权重和缓存数据已删除；外部临时目录仅保留44字节的uv标记文件。AIOS将其中的空`.git`标记判为不可读Git引用并拒绝删除父目录，未绕过此限制。该保留项单列在`external-temp-cleanup.json`，不属于仓库或NVIDIA交付归档。临时环境文件字节统计可能含硬链接，不用它声称释放了等量物理磁盘块。

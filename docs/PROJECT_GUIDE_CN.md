# CQ_AI 0.14.5 / v23.5 项目说明

## 1. 当前基线

| 项目版本 | 交付版本 | Worker 协议 | 日期 |
| --- | --- | --- | --- |
| `0.14.5` | `v23.5` | `25` | `2026-08-28` |

v23.5 保持 60 个公开导出。`CV_FindMultiText`、
`CV_FindTransparentMultiText` 返回 `ID,x,y|...`，`OCR_FindMultiText` 返回
`ID,cx,cy|...`。九个兼容易语言接口支持 `origin_x/origin_y`，模板可从内存
ZIP 原子加载；OCR 保留滤色、阅读顺序和异常碎片联合识别。

## 2. 架构

```text
32 位易语言 -> CQ_X86.dll -> 本地静态 OpenCV
                         -> 协议 25 命名管道 -> CQ_AI_worker.exe x64
                                                  -> ORT 1.24.4
                                                  -> CPU / DirectML 1.15.4

64 位 Python -> CQ_AI_x64.dll -> ORT 1.24.4 -> CPU / DirectML 1.15.4
```

Worker 把内嵌运行库验证后释放到当前用户私有缓存。`AI_Release()`只释放模型；
宿主正常结束前调用 `AI_ShutdownWorker()`。Python x64 直接 DLL 不使用 Worker。

## 3. 仓库布局

```text
include/          公开 C ABI
src/              DLL、Worker、OpenCV、OCR、YOLO 和运行库加载实现
configs/          当前配置示例
models/           正式内嵌 OCR 模型和必要 YOLO 测试模型
examples/         易语言与 Python 示例
python/           ctypes 包装和 Wheel 元数据
tests/            自动测试及只读 fixtures
tools/            构建期运行库打包器
scripts/          当前依赖、打包、文档和验证脚本
release/v23.5/    唯一正式交付物及 manifest
```

`third_party/`、`build-*`、`outpush/` 和 `test-results/` 都是可再生本地缓存，
不得提交。

## 4. 固定依赖

- OpenCV `5.0.0`，提交 `40738fb16ceddb5fb3fea747585f7ce6abb0605b`，
  x86/x64 均为静态 `/MT`；x86 应用仓库内 AVX2 编译补丁。
- Microsoft.ML.OnnxRuntime.DirectML `1.24.4`。
- Microsoft.AI.DirectML `1.15.4`。
- Visual C++ 2022 x64 运行库，仅嵌入 Worker/Wheel 使用。

执行 `scripts/prepare_dependencies.ps1` 会下载、验签、构建并放入被忽略的
`third_party/`。下载包的 SHA-256 固定在脚本内。

## 5. 构建与测试

根目录 README 给出 x64 Worker、x86 DLL 和 Python Wheel 的完整命令。正式验收要求：

1. x86 全部 CTest 通过，包括 ABI、CV、OCR、YOLO、Worker、CPU、DirectML、AUTO。
2. x64 全部 CTest 通过，包括嵌入模型、运行时探测和业务基准。
3. `CQ_X86.dll` 仅依赖系统 DLL；Worker 不存在外部 ORT、DirectML 或 VC 运行库依赖。
4. 公开头文件和成品均为 60 个导出，stdcall 参数字节数符合当前 ABI。
5. 易语言 ZIP 恰好包含三个文件；Wheel 为 `py3-none-win_amd64` 且不含 EXE。
6. `scripts/verify_current_release.ps1` 和 `scripts/check_repository_hygiene.ps1` 通过。

## 6. 发布约束

`release/v23.5/manifest.json` 是交付事实源，记录项目版本、交付版本、协议、
两个包的大小和 SHA-256，以及易语言 ZIP 成员哈希。普通构建只写 `outpush/`；
只有正式发版才替换 `release/`，同时删除此前版本并更新清单、文档和测试。

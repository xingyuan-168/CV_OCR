# YOLO26 与 V8 本机20ms验证

用户在2026-10-07把验收门槛由15ms调整为20ms，15ms保留为进一步优化目标。本轮用E5-2696 v4、128GiB、RTX2070测试x86 DLL与配套x64 Worker的完整调用。当前最终候选正式采样与30分钟稳定性已通过20ms；交付状态以当前manifest为准。

## 固定输入与计时

- YOLO26：用户原始`input/yolov26/yolo26n.onnx`，SHA256 `355c4f26cc1d48e581aaeb4a7f837baf7d5993e6f83337d30d50d0476bc5b158`，输入640×640，输出`[1,300,6]`。原`bus.jpg`只在input外解码成810×1080的24位BMP，没有缩放。
- V8：用户原始`input/best.onnx`，SHA256 `ca5407763a638b86d73e9f17c423ebec39c9b0c8c1ee7f33de64aa32c73177b8`，配用户`input/yolo测试图.bmp`。模型原生输入320×320，输出`[1,10,2100]`，6类，dense/BCN，保留外部NMS。
- 置信度0.5、NMS0.45。五个调用线程共享句柄；每路预热100次、正式1000次，持续调用和同时发起各3轮。QPC包住`YOLO_InferJson`，包含打包、通信、排队、预处理、执行、解码和JSON返回。冷加载另列。
- 每轮每一路P95≤20ms才通过。表中每项分别取三轮最差单路，不用总体平均替代单路判定；超限比例来自完整原始样本。

## 最终候选C的正式结果

核心DLL/Worker与B逐字节相同；可选TensorRT模块SHA256 `8e45b8abc9e799f47018924e727d02ff5960e5ecb593a273827f8649c62cc7a5`，普通推理关闭阶段event，诊断模式保留。新module身份导致AUTO键变化，因此YOLO26和V8都重新生成profile并验证AUTO。

| 模型/模式 | P50 | P95 | P99 | 最大值 | 超15ms | 超20ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26/continuous | 9.139 | 10.393 | 10.956 | 14.397 | 0.0000% | 0.0000% |
| yolo26/simultaneous | 12.092 | 14.187 | 15.233 | 19.050 | 0.4000% | 0.0000% |
| yolo26_auto/continuous | 9.274 | 10.621 | 11.969 | 15.588 | 0.0133% | 0.0000% |
| yolo26_auto/simultaneous | 12.080 | 14.426 | 15.513 | 17.848 | 0.5733% | 0.0000% |
| v8/continuous | 7.557 | 16.140 | 16.830 | 20.141 | 5.3733% | 0.0067% |
| v8/simultaneous | 8.956 | 11.423 | 12.251 | 13.361 | 0.0000% | 0.0000% |
| v8_auto/continuous | 7.611 | 16.898 | 17.413 | 19.207 | 10.0800% | 0.0000% |
| v8_auto/simultaneous | 8.974 | 11.215 | 11.866 | 12.796 | 0.0000% | 0.0000% |

最终YOLO26配置 `d3-s2-t1-fp16-graph`。FP16 Graph相对普通执行的P95改善：持续各轮26.687%/24.004%/41.839%，同时14.600%/19.268%/20.373%；每轮均≥10%，所以本候选启用Graph。V8采用DirectML/3槽，重新验证moduleC对应AUTO键。旧计时模块的Graph结果继续保留，不能混用。

## 核心B与旧计时模块的正式结果（保留）

最终配对版本0.14.8、协议26。DLL SHA256 `1a59527ef0c2c2f0c7d2e70731574e8c96a2869095b0a29a9800183f4e2bef54`；Worker SHA256 `14620e2cecc8044b8465eeb93ef6e15cd5bc63df0d0f39d51a94259ae64c0a37`。下表单位ms：

| 模型/配置/模式 | P50 | P95 | P99 | 最大值 | 超15ms | 超20ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26 TRT FP16/2槽/持续 | 12.622 | 15.714 | 17.042 | 19.159 | 2.820% | 0% |
| YOLO26 TRT FP16/2槽/同时 | 13.150 | 17.504 | 18.370 | 20.738 | 18.713% | 0.0067% |
| YOLO26 AUTO/持续 | 14.187 | 17.683 | 18.621 | 21.594 | 5.613% | 0.0200% |
| YOLO26 AUTO/同时 | 13.365 | 17.474 | 18.406 | 21.055 | 19.460% | 0.0133% |
| V8 DML FP32/3槽/持续 | 7.433 | 16.455 | 16.887 | 21.150 | 11.273% | 0.0067% |
| V8 DML FP32/3槽/同时 | 8.930 | 11.658 | 12.222 | 13.273 | 0% | 0% |
| V8 AUTO/持续 | 7.432 | 16.534 | 17.045 | 19.568 | 9.247% | 0% |
| V8 AUTO/同时 | 8.706 | 11.239 | 11.760 | 12.777 | 0% | 0% |

这八组共120,000次正式调用，零推理错误。P95达标不表示每次调用都低于20ms。YOLO26仍未达到原15ms标准；V8持续模式的3槽候选也未达到15ms。

V8的1/2/3/5槽扫描最差单路P95分别为16.227、12.089、16.455、21.208ms。5槽超标；3槽和2槽都达20ms，现有校准策略在满足尾延迟后选P50较低的3槽。更看重尾延迟时可使用2槽实测结果重新校准该配置。相同DirectML/5槽条件下，新旧v23.7比较：持续P50 +0.40%、P95 −0.004%；同时P50 −0.55%、P95 −0.99%，未出现超过5%的回退。

## 修复与证据边界

原v23.7 DirectML/5槽的bus完整调用最差单路P95为持续39.785ms、同时28.685ms。现有TensorRT模块缺少`nvinfer_10.dll`；本轮按锁定SHA恢复TensorRT10.13.3.9、CUDA runtime12.8.90、NVRTC12.8.93和cuBLAS12.8.4.1，依赖独立交付。基础output仍只有三个核心文件。

初始配对二进制完成FP32/FP16×1/2/3/5槽及Graph扫描，两种模式各3轮。普通模式FP32最差单路P95依次30.710、24.760、40.722、58.380ms；FP16依次26.606、18.493、40.210、46.440ms。初始扫描明确失败15ms。随后修正AUTO校准身份并重新构建；初始扫描只用于配置筛选，最终通过结论来自上表的最终二进制重测。

`src/yolo_preprocess.h`把逐像素归一化改为256项查表，仍用原FP32除法生成表值，分离padding和有效区域循环。5种格式、正负stride、320/640尺寸、灰度256值和BMP方向的回归按位相等。

`src/yolo_pool.cpp`原校准键包含尚未解压的DirectML文件或`missing`，导致解压后重启Worker键变化。修复后使用已验证嵌入bundle哈希，算法身份升至v6。隔离的冷缓存、DirectML解压、全新Worker重启三次身份一致；三个Worker PID不同。旧键不强行迁移。

工具新增显式P95门槛：原调用默认30ms，本轮传20ms；校准、PowerShell入口、汇总、打包和生成HTML贯通同一门槛。新增回归拒绝整体平均掩盖单路超标、假loader证据、不同二进制或错模型的通过报告。

最终Graph候选最差P95为16.130ms，但未在两种模式每轮均改善至少10%，所以最终Graph关闭。没有因个别更快结果跳过原条件。

独立550请求诊断：预处理P95 1.501ms、Run 4.360ms、槽位等待8.167ms、IPC1.705ms；基线分别3.448、32.086、IPC1.905ms。IPC低于2ms触发门槛。诊断有额外开销，不能与正式采样混合，也不能把不同阶段P95相加作完整P95。

## 正确性、缓存及回归

YOLO26 FP16与CPU参考及合格TRT FP32数量/类别一致，坐标误差≤1px、分数差≤0.02；bus得到4人和1公交车。V8独立CPU参考对照通过。最终配对独立加载器125次调用通过，60个公开函数及stdcall别名、协议26保持一致。

独立x64缓存首次构建、命中、主动损坏后重建均通过，三次检测JSON完全一致。首次构建约99.44秒，不纳入推理速度。该独立进程须把NVIDIA目录添加到进程PATH，供vendor builder按名称加载资源DLL；一键入口已处理，未修改机器级PATH。缺模块、缺NVIDIA库、无NVIDIA时CPU可用的检查通过。

Python单元测试35项：34通过、1因可选ONNX工具缺失跳过。最终native受影响YOLO/ABI/transport回归通过。较宽的CV检查中，两个既有5ms绝对性能断言在E5主机失败；旧v23.7也失败，保留配对日志，不把全库测试标为全部通过。CV源码没有变更。

## 复现与部署

使用最终配对核心文件及独立NVIDIA包，在业务部署目录把可选模块放在Worker旁，NVIDIA目录通过`CQ_AI_NVIDIA_DIR`指定。`output/`不添加模块或依赖。完整环境、版本锁定和工具说明见[YOLO说明](YOLO_OPTIMIZATION_CN.md)。

本机候选配置：YOLO26设AUTO、正整数session_count=2，使用当前可选模块对应bus的business-calibration.tsv和workload；V8设AUTO、session_count=3，使用V8自己的profile/workload。模型、槽数、设备、驱动、可选模块或图片工作负载变化应重新校准，不能混用两份profile。

```powershell
python scripts/calibrate_yolo.py --benchmark build/portable-x86/Release/ai_engine_yolo_benchmark.exe --model input/yolov26/yolo26n.onnx --image outpush/yolo26-local-20261007/bus.bmp --devices 3 --sessions 1,2,3,5 --fp16 --graphs --warmup 100 --samples 1000 --rounds 3 --p95-limit-ms 20 --preserve-worker --output outpush/new-yolo26-run
python scripts/calibrate_yolo.py --benchmark build/portable-x86/Release/ai_engine_yolo_benchmark.exe --model input/best.onnx --image input/yolo测试图.bmp --devices 1 --sessions 1,2,3,5 --warmup 100 --samples 1000 --rounds 3 --p95-limit-ms 20 --preserve-worker --output outpush/new-v8-run
```

本轮进一步优化的代码依据是`src/tensorrt_module.cpp`原来每次Run都记录4个CUDA event并读取3个阶段时间，现已限定到诊断模式。现有Graph只捕获enqueueV3，H2D/D2H在Graph之外；若最终稳定性仍失败，下一项可验证固定缓冲区完整流水捕获，并对长时间尾延迟进行带绝对时间的阶段诊断。每项必须重新做正确性、两模式三轮和稳定性，不能承诺未测收益。

## 30分钟与交付状态

第一轮最终AUTO、FP16、2槽完成1800.015秒、667,007次调用，零错误。最差单路P95 17.640ms、P99 33.287ms、最大104.295ms，整体超20ms比例3.920%。客户端和Worker前后60样本均值分别增加0.012和0.362MiB，未触发2MiB增长门槛；整卡显存均值从1379.167增至1420.850MiB，增加41.683MiB，触发16MiB稳定性门槛。因此该轮`local_benchmark_accepted=false`，没有晋升。

原始文件和失败汇总按`outpush/yolo26-15ms-implementation/cohort-b-20ms/soak-observation-1/`相对路径保存在独立原始证据ZIP中，已登记的小型汇总保留在`release/v23.8/evidence/`。观察窗口内Worker进程级Dedicated/Shared/Committed读数保持158,912,512/83,886,080/242,798,592字节，模型释放后提交读数下降到741,376字节；这是辅助诊断，不能替代完整稳定性验收。

[NVIDIA NVML说明](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlProcessInfo__t.html)指出WDDM的进程显存值不可用；[Microsoft说明](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value)也记录了进程计数器可能误报的情形。本轮仍保留整卡增长门槛，没有据此自动放宽通过条件。

第二轮同核心二进制、同模型、100预热完成1800.011秒，621,628次调用零错误；最差单路P95 27.099ms、P99 40.154ms、最大113.414ms，整体超20ms比例7.573%。CPU内存和整卡显存增长检查通过，但延迟失败，该候选没有晋升。证据按`outpush/yolo26-15ms-implementation/cohort-b-20ms/`相对路径保存在独立原始证据ZIP，第一轮在其中的`soak-observation-1/`。不能只取第一轮较低的P95宣称稳定达标。

两轮按单路累计调用时长分分钟检查，前约13-15分钟P95约14-16ms，随后尾延迟明显升高。由于原始CSV不含绝对调用时间，这个分段仅用于定位，不能精确归因于某个后台事件。目前未确认根因。

候选C已将CUDA四个event的创建、记录和阶段时间读取限定到设置`CQ_AI_YOLO_TRACE_PREFIX`的诊断模式，保持H2D、enqueueV3/Graph、D2H和完成同步。新的FP16 Graph在两种模式每轮都改善P95≥10%，已通过启用条件；30分钟已通过20ms，原失败数据保留。报告仅覆盖这两张图片的五线程基准，五业务窗口变化帧验收仍待验。

## 最终候选C的30分钟结论

AUTO持续调用1800.008秒，841,787次，错误0；最差单路P95 16.503ms。`local_benchmark_accepted=true`。CPU与GPU前后均值、增长和峰值见[完整汇总](evidence/YOLO26_LOCAL_20MS_SUMMARY.json)。

只有完整通过时才晋升；本次实际状态和Finish gate在交付状态中记录。P95不能保证每次调用都小于20ms。

2026-10-08：实际output配对加载器125次并发调用零错误后，已事务晋升并登记v23.8；旧整套回滚保留。当前选择与证据为`release/current.json`和`release/v23.8/`。独立NVIDIA包、Wheel及配置位置见[v23.8交付说明](V23_8_DELIVERY_CN.md)。

本轮最差单路P99 24.641ms、最大78.648ms；整体超15ms 7.3985%、超20ms 2.2767%。客户端/Worker增长分别0.099/0.250MiB；整卡显存前后均值1357.750/1357.000MiB，增长−0.750MiB，峰值1362MiB。原15ms完整长期标准仍未达到。真实候选Finish gate和晋升后的默认严格Finish gate均返回`ok=true, allowed=true`，实际执行受影响测试命令。原始CSV及构建记录已校验归档，位置见[v23.8交付说明](V23_8_DELIVERY_CN.md)；清理缓存后重测须复建基准程序或使用归档工具。

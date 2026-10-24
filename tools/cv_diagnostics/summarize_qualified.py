"""Build candidate evidence only from completed, hash-bound local acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from release_manifest import digest
from verify_cv_acceptance import INPUT_HASHES, verify_candidate_cv, verify_timing
from summarize_yolo_acceptance import memory_summary


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def info(path):return dict(size=path.stat().st_size,sha256=digest(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation",type=Path,required=True)
    args=parser.parse_args();folder=args.validation.resolve()
    old=read(ROOT/"outpush/cv-transparent-mask-fix/build-metadata.json")
    paths=subprocess.check_output(["git","ls-files","-z","--","CMakeLists.txt","include","src"],cwd=ROOT).decode().split("\0")
    snapshot={p:hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in paths if p}
    metadata=dict(old,git_source_sha256=snapshot,build_base_revision="dd60f5ffec2afec3f0fa8b6247437ac40069f84b",
        source_revision="dd60f5f + normalized candidate source snapshot",delivery_status="qualified",
        candidate_runtime_directory=str(ROOT/"build/cv-cascade-x86-production"),
        delivery_document="CV_TRANSPARENT_5MS_RESULT_CN.md",cv_transparent_validation="passed",
        cv_transparent_p95_limit_ms=5,cv_independent_reference_validation="passed",
        local_benchmark_validation="not_run",target_performance_validation="pending",
        functional_validation="passed",formal_release=False)
    builds={arch:ROOT/f"build/cv-cascade-{arch}-production" for arch in ("x86","x64")}
    metadata["binary_fingerprints"]={name:info(builds["x86"]/name) for name in ("CQ_X86.dll","CQ_AI_worker.exe")}
    metadata["cv_x64_sha256"]=digest(builds["x64"]/"CQ_AI_x64.dll")
    metadata["functional_evidence"]=dict(file=(folder/"delivery-build.json").relative_to(ROOT/"outpush").as_posix(),**info(folder/"delivery-build.json"))
    evidence={};architectures={}
    for arch in ("x86","x64"):
        cases=[]
        expected=metadata["binary_fingerprints"]["CQ_X86.dll"]["sha256"] if arch=="x86" else metadata["cv_x64_sha256"]
        for legacy,stem,key in ((False,"cv-timing-","timing_"),(True,"cv-legacy-timing-","legacy_")):
            path=folder/f"{stem}{arch}.json";report=read(path)
            verify_timing(report,expected,arch,legacy=legacy)
            evidence[key+arch]=dict(file=path.name,**info(path))
            cases+=report["cases"]
        pressure=[]
        for threads in (1,4,16):
            path=folder/f"cv-stress-{arch}-{threads}.json";data=read(path)
            evidence[f"stress_{arch}_{threads}"]=dict(file=path.name,**info(path));pressure.append(data)
        architectures[arch]=dict(dll_sha256=expected,cases=cases,stress=pressure,
            worst_p95_ms=max(r["p95_ms"] for c in cases for r in c["rounds"]),
            worst_p99_ms=max(r["p99_ms"] for c in cases for r in c["rounds"]),
            maximum_ms=max(r["max_ms"] for c in cases for r in c["rounds"]),
            samples=sum(r["count"] for c in cases for r in c["rounds"]),
            errors=sum(r["errors"] for c in cases for r in c["rounds"]))
    metadata["cv_transparent_evidence"]=evidence
    verify_candidate_cv(metadata,folder,metadata["binary_fingerprints"])
    for name,expected in INPUT_HASHES.items():
        if digest(ROOT/"input"/name)!=expected:raise ValueError("Read-only image changed")
    for name in ("continuous.json","simultaneous.json","soak-30min.json"):
        report=read(folder/name)
        if (report["dll_sha256"]!=metadata["binary_fingerprints"]["CQ_X86.dll"]["sha256"] or
            report["worker_sha256"]!=metadata["binary_fingerprints"]["CQ_AI_worker.exe"]["sha256"] or
            report["worker_protocol"]!=26 or any(r["errors"] for r in report["rounds"])):
            raise ValueError("Paired-binary regression failed")
    soak=read(folder/"soak-30min.json")
    if len(soak["rounds"])!=1 or soak["rounds"][0]["elapsed_s"]<1800:raise ValueError("Incomplete soak")
    memory=memory_summary(folder/"soak-30min.json.resources.csv")
    if memory["continuous_growth_detected"]:raise ValueError("Paired soak sustained memory growth")
    for name in ("hooks-x86.txt","hooks-x64.txt","native-x86.txt","native-x64.txt",
                 "no-opencv-x64.txt","no-opencv-x86-proxy-retry.txt"):
        if "100% tests passed" not in (folder/name).read_text(encoding="utf-8",errors="replace"):
            raise ValueError(f"Missing completed native regression {name}")
    summary=dict(requirement_id="REQ-CV-001",project_version="0.14.9",delivery_version="v23.9",
        accepted=True,limit_ms=5,scope="Single-caller complete DLL call;20 scenarios/architecture;64warmups,1000samples,3rounds",
        source_sha256=snapshot,input_sha256=INPUT_HASHES,architectures=architectures,
        first_call_measurements=read(folder/"cv-cold-summary.json"),
        stress_environment="x86 and x64 each run1/4/16 callers sequentially; architectures and paired CPU soak run concurrently; no performance benchmark or compilation runs during formal5ms timing",
        paired_validation=dict(loader=read(folder/"delivery-build.json"),soak_seconds=soak["rounds"][0]["elapsed_s"],memory=memory,
            scope="Fresh CPU functional/ABI/protocol compatibility only; no resumed business YOLO diagnosis or20ms acceptance claim"),
        regression_setup_note="Initial no-OpenCV x86 proxy test lacked its paired Worker; copied the freshly built owned Worker and reran successfully. Initial failure log retained.",
        evidence=evidence)
    metadata["validation_summary"]=("透明找图在(596,165)正确命中95×81目标，322个有效彩色像素相同、精算1.0；"
        "x86/x64共120000个正式样本零错误，每架构20场景各三轮P95均≤5ms。"
        "六项十分钟变化帧压力和新DLL/Worker三十分钟CPU配对稳定性完成；业务YOLO问题未恢复调查。")
    metadata["functional_validation_scope"]=metadata["validation_summary"]
    (folder/"build-metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (folder/"summary.json").write_text(json.dumps(dict(functional_passed=True,memory_growth_detected=False,
        local_benchmark_accepted=False,cv_accepted_5ms=True,paired_memory=memory),indent=2)+"\n",encoding="utf-8")
    (folder/"cv-acceptance-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ROOT/"docs/evidence/CV_TRANSPARENT_5MS_20261024_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 透明找图5ms验收结果","", "2026-10-24，REQ-CV-001，0.14.9 / v23.9。当前安装交付以 `release/current.json` 和 `output/` 文件哈希为准。", "",
        "原图1.bmp/2.bmp正确命中左上角 **(596,165)**，尺寸95×81。322个有效彩色像素相同，单目标精算 **1.0**；单目标和透明多目标0.9/0.99/1、灰度0.9/0.99均通过。输入哈希保持不变。", "",
        "**x86、x64各20场景、每场景三轮完整DLL调用P95≤5ms；共120000样本零错误。** 下表取每个场景三轮最差P95，单位ms。64次预热、1000次正式采样，最近秩统计，架构/场景/轮次分别判定。", "",
        "| 场景 | x86 | x64 |","|---|---:|---:|"]
    tables={a:{c["name"]:max(r["p95_ms"] for r in c["rounds"]) for c in v["cases"]} for a,v in architectures.items()}
    for name in tables["x86"]:lines.append(f"| {name} | {tables['x86'][name]:.4f} | {tables['x64'][name]:.4f} |")
    lines += ["",f"全场景最差P95：x86 **{architectures['x86']['worst_p95_ms']:.4f}ms**，x64 **{architectures['x64']['worst_p95_ms']:.4f}ms**。",
        f"最大单次调用分别为{architectures['x86']['maximum_ms']:.4f}ms和{architectures['x64']['maximum_ms']:.4f}ms；P95通过不代表每次调用都≤5ms。完整P50/P95/P99、最大值及超5ms比例保存在[逐轮证据](evidence/CV_TRANSPARENT_5MS_20261024_SUMMARY.json)。",
        "", "透明基准只计完整DLL调用；旧普通基准额外包含逐次结果比较，因此计时略保守。文件读取和模板加载在计时外。13个掩码场景的冷调用分别用全新进程测量；7个旧普通场景记录预热前调用，前者与后者明确区分。冷调用未套用预热后的5ms门槛。", "",
        "## 原因与修复", "",
        "原漏检来自透明背景参与矩形粗筛，以及精算失败后错误抑制邻居。第一轮正确性修复又因32点方差下界过松、零方差窗口穿透、整帧FFT固定成本导致回退。现在对≤1024个有效像素使用带保守误差保护的正交像素对上界，配合平面字节AVX2扫描；最终结果仍由原整数精算确认，不限制候选数量，不降低匹配阈值。",
        "", "另对旧普通彩色多模板的独立FFT计算使用8块有重叠边缘的有界并行，保留评分定义、全局峰值/NMS与输出顺序。每工作区最多7个辅助任务，调用返回前全部结束；新增缓存计入原空闲预算。普通单目标、大模板、灰度及标准透明API语义保持兼容。", "",
        "数学依据及资源边界见[数值证明](CV_MASKED_PROJECTION_NUMERICS.md)，失败候选和判因数据见[改善方案](CV_TRANSPARENT_5MS_IMPROVEMENT_PLAN_CN.md)。", "",
        "## 稳定性与交付边界", "", "| 架构 | 调用线程 | 实际秒数 | 正确调用数 | 错误 | 后段内存增量MiB |", "|---|---:|---:|---:|---:|---:|"]
    for arch,data in architectures.items():
        for r in data["stress"]:lines.append(f"| {arch} | {r['threads']} | {r['elapsed_s']:.2f} | {r['calls']} | {r['errors']} | {r['memory_growth_bytes']/2**20:.3f} |")
    lines += ["", "压力包含7种变化帧场景（真实透明掩码和普通分块路径），x86/x64同时施压，各自顺序执行1/4/16线程；并发排队延迟另报，不代替单调用5ms验收。模板释放、异常恢复、生命周期和工作区预算回归通过。首次无OpenCV代理测试缺少配对Worker，补齐测试目录后复测通过，初始日志保留。", "",
        "新二进制另完成两种模式各三轮五路回归（共30000样本）、配对加载125次并发、30分钟多帧CPU稳定性验证，零错误；60个签名及stdcall别名、协议26、正整数session_count保持兼容。此处CPU YOLO运行用于交付兼容性，不构成业务YOLO20ms验收。OCR、业务YOLO、光效容忍、缩放/旋转和PNG导入保持独立待办。", "",
        "全部正式报告绑定实际DLL/Worker哈希；生产DLL无私有测试导出。历史交付和原图保留，晋升通过事务同时切换三文件output与当前指针；加载或真实Finish失败即恢复两者。"]
    (ROOT/"docs/CV_TRANSPARENT_5MS_RESULT_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({arch:data["worst_p95_ms"] for arch,data in architectures.items()}))


if __name__=="__main__":main()

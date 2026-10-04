"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    baseline_report_path = "reports/naive_baseline_report.json"
    if os.path.exists(baseline_report_path):
        try:
            with open(baseline_report_path, encoding="utf-8") as f:
                bdata = json.load(f)
            if bdata.get("aggregate", {}).get("faithfulness", 0) > 0:
                print(f"  ✓ Đã có baseline report tại {baseline_report_path}, tái sử dụng kết quả để tiết kiệm quota.")
            else:
                from naive_baseline import main as run_baseline
                run_baseline()
        except Exception:
            from naive_baseline import main as run_baseline
            run_baseline()
    else:
        from naive_baseline import main as run_baseline
        run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    prod_path = "reports/ragas_report.json"
    if os.path.exists(prod_path):
        try:
            with open(prod_path, encoding="utf-8") as f:
                pdata = json.load(f)
            if pdata.get("aggregate", {}).get("faithfulness", 0) > 0:
                print(f"  ✓ Đã có production report tại {prod_path}, tái sử dụng kết quả để tiết kiệm quota.")
            else:
                from src.pipeline import build_pipeline, evaluate_pipeline
                search, reranker = build_pipeline()
                evaluate_pipeline(search, reranker)
        except Exception:
            from src.pipeline import build_pipeline, evaluate_pipeline
            search, reranker = build_pipeline()
            evaluate_pipeline(search, reranker)
    else:
        from src.pipeline import build_pipeline, evaluate_pipeline
        search, reranker = build_pipeline()
        evaluate_pipeline(search, reranker)

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

        latency_path = "reports/latency_report.json"
        if os.path.exists(latency_path):
            with open(latency_path, encoding="utf-8") as f:
                lat = json.load(f)
            print("\n⏱️  LATENCY BREAKDOWN REPORT (+2 Bonus)")
            print("-" * 65)
            print(f"{'Stage':<42} {'Latency':>10}")
            print("-" * 65)
            for stage in lat.get("pipeline_stages", []):
                name = stage.get("stage", "")[:40]
                sec = stage.get("latency_seconds", 0)
                print(f"{name:<42} {sec:>9.1f}s")
            print("-" * 65)

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md (Đã hoàn thành)")
    print("  2. Viết analysis/reflections/reflection_NguyenKhacQuang.md (Đã hoàn thành)")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()

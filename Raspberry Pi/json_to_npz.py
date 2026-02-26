#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def sanitize_person_name(name: str) -> str:
    # 用于日志显示，尽量保留原样
    if name is None:
        return "UNKNOWN"
    s = str(name).strip()
    return s if s else "UNKNOWN"


def load_one_json(path: str, expected_dim: int = 512) -> Tuple[bool, Dict[str, Any], str]:
    """
    returns: (ok, data, err_msg)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, {}, f"JSON读取失败: {e}"

    if not isinstance(data, dict):
        return False, {}, "JSON根对象不是 dict"

    person_name = data.get("person_name", None)
    if person_name is None:
        # 兼容：尝试从文件名推断（调用方可覆盖）
        pass

    embs = data.get("embeddings", None)
    if embs is None:
        return False, {}, "缺少字段 embeddings"
    if not isinstance(embs, list):
        return False, {}, "embeddings 不是 list"
    if len(embs) == 0:
        return False, {}, "embeddings 为空"

    # 检查维度并转 float32
    valid_embs = []
    for i, e in enumerate(embs, start=1):
        try:
            arr = np.asarray(e, dtype=np.float32).reshape(-1)
        except Exception:
            return False, {}, f"第{i}条 embedding 无法转为 float32"
        if arr.shape[0] != expected_dim:
            return False, {}, f"第{i}条 embedding 维度异常: {arr.shape[0]} != {expected_dim}"
        # 再做一次 L2 归一化，防止录入脚本版本差异导致未归一化
        arr = l2_normalize(arr)
        valid_embs.append(arr)

    data["_valid_embeddings_np"] = valid_embs
    return True, data, ""


def collect_json_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    files = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(input_dir, fn))
    files.sort()
    return files


def build_db_from_jsons(
    json_paths: List[str],
    mode: str = "mean",
    expected_dim: int = 512,
    required_preprocess_mode: str = "",
    strict_preprocess_mode: bool = False,
) -> Dict[str, Any]:
    """
    mode:
      - mean: 每人聚合为1条 embedding（均值后L2 normalize）
      - all : 保留每人所有样本 embedding
    """

    rows_names: List[str] = []
    rows_embs: List[np.ndarray] = []

    person_sample_count: Dict[str, int] = {}
    preprocess_modes_seen: Dict[str, int] = {}

    skipped_files: List[Dict[str, str]] = []
    warnings: List[str] = []

    for path in json_paths:
        ok, data, err = load_one_json(path, expected_dim=expected_dim)
        if not ok:
            skipped_files.append({"file": path, "reason": err})
            continue

        # person name
        person_name = data.get("person_name")
        if not person_name:
            # 兼容：从文件名推断
            person_name = os.path.splitext(os.path.basename(path))[0]
        person_name = sanitize_person_name(person_name)

        # preprocess mode 检查（来自 collecting.py 的 meta）
        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
        ppm = str(meta.get("preprocess_mode", "") or "")
        if ppm:
            preprocess_modes_seen[ppm] = preprocess_modes_seen.get(ppm, 0) + 1

        if required_preprocess_mode:
            if ppm and ppm != required_preprocess_mode:
                msg = (f"文件 {os.path.basename(path)} preprocess_mode={ppm} "
                       f"与要求 {required_preprocess_mode} 不一致")
                if strict_preprocess_mode:
                    skipped_files.append({"file": path, "reason": msg})
                    continue
                else:
                    warnings.append(msg)

        embs: List[np.ndarray] = data["_valid_embeddings_np"]
        person_sample_count[person_name] = person_sample_count.get(person_name, 0) + len(embs)

        if mode == "mean":
            # 每人一个JSON通常就是一个人，但仍做鲁棒处理
            # 如果同名人有多个json（理论上不该发生），这里后续再同名合并
            mean_emb = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
            rows_names.append(person_name)
            rows_embs.append(mean_emb)

        elif mode == "all":
            for emb in embs:
                rows_names.append(person_name)
                rows_embs.append(emb)

        else:
            raise ValueError(f"unknown mode: {mode}")

    # 对 mean 模式做“同名合并”（防止同名存在多个json）
    if mode == "mean" and rows_names:
        grouped: Dict[str, List[np.ndarray]] = {}
        for n, e in zip(rows_names, rows_embs):
            grouped.setdefault(n, []).append(e)

        merged_names: List[str] = []
        merged_embs: List[np.ndarray] = []
        for person_name in sorted(grouped.keys()):
            person_embs = grouped[person_name]
            if len(person_embs) == 1:
                merged_names.append(person_name)
                merged_embs.append(person_embs[0])
            else:
                # 多个均值再均值（相当于同名多次采集融合）
                emb = l2_normalize(np.mean(np.stack(person_embs, axis=0), axis=0))
                merged_names.append(person_name)
                merged_embs.append(emb)
                warnings.append(f"同名 {person_name} 存在多个json，已自动合并为1条 embedding")

        rows_names = merged_names
        rows_embs = merged_embs

    # all 模式下按名字排序，便于可读性和稳定复现
    if mode == "all" and rows_names:
        items = sorted(zip(rows_names, rows_embs), key=lambda x: x[0])
        rows_names = [x[0] for x in items]
        rows_embs = [x[1] for x in items]

    if len(rows_embs) == 0:
        raise RuntimeError("没有可用 embedding，无法生成 NPZ。请检查输入JSON。")

    embeddings_np = np.stack(rows_embs, axis=0).astype(np.float32)
    names_np = np.asarray(rows_names, dtype=object)

    # counts：按输出库统计每个名字条数（mean模式通常每人1条；all模式可能多条）
    out_counts: Dict[str, int] = {}
    for n in rows_names:
        out_counts[n] = out_counts.get(n, 0) + 1

    result = {
        "names": names_np,                         # object array, shape [N]
        "embeddings": embeddings_np,               # float32 [N, D]
        "counts_per_name": np.asarray(
            [[k, str(v)] for k, v in sorted(out_counts.items())],
            dtype=object
        ),
        "meta_build": np.asarray([{
            "created_at": now_str(),
            "mode": mode,
            "expected_dim": expected_dim,
            "required_preprocess_mode": required_preprocess_mode,
            "strict_preprocess_mode": strict_preprocess_mode,
            "num_rows": int(embeddings_np.shape[0]),
            "num_unique_names": int(len(set(rows_names))),
            "json_files_input": int(len(json_paths)),
            "json_files_skipped": int(len(skipped_files)),
            "preprocess_modes_seen": preprocess_modes_seen,
        }], dtype=object),
        "skipped_files": np.asarray(skipped_files, dtype=object),
        "warnings": np.asarray(warnings, dtype=object),
    }
    return result


def print_summary(npz_payload: Dict[str, Any]):
    names = npz_payload["names"]
    embs = npz_payload["embeddings"]
    counts_per_name = npz_payload["counts_per_name"]
    meta_build = npz_payload["meta_build"][0]
    skipped = npz_payload["skipped_files"]
    warnings = npz_payload["warnings"]

    print("========== Build Summary ==========")
    print(f"时间                : {meta_build['created_at']}")
    print(f"模式                : {meta_build['mode']}")
    print(f"输入JSON总数         : {meta_build['json_files_input']}")
    print(f"跳过JSON数           : {meta_build['json_files_skipped']}")
    print(f"输出embedding条数    : {meta_build['num_rows']}")
    print(f"输出人员数           : {meta_build['num_unique_names']}")
    print(f"embedding维度        : {embs.shape[1]}")
    print(f"预处理模式统计        : {meta_build['preprocess_modes_seen']}")
    print("-----------------------------------")
    print("每人条数（输出库中）:")
    for row in counts_per_name.tolist():
        # row = [name, count]
        print(f"  {row[0]}: {row[1]}")
    if len(warnings) > 0:
        print("-----------------------------------")
        print("Warnings:")
        for w in warnings.tolist():
            print(f"  - {w}")
    if len(skipped) > 0:
        print("-----------------------------------")
        print("Skipped Files:")
        for item in skipped.tolist():
            # item is dict
            print(f"  - {item.get('file')}: {item.get('reason')}")
    print("===================================")


def main():
    ap = argparse.ArgumentParser(
        description="将 collecting.py 生成的 face_db_json/*.json 合并为 face_db_embeddings.npz"
    )
    ap.add_argument(
        "--input-dir",
        default="face_db_json",
        help="输入JSON目录（默认 face_db_json）"
    )
    ap.add_argument(
        "--output",
        default="face_db_embeddings.npz",
        help="输出NPZ路径（默认 face_db_embeddings.npz）"
    )
    ap.add_argument(
        "--mode",
        choices=["mean", "all"],
        default="mean",
        help="建库模式：mean=每人均值1条（推荐）；all=保留所有样本"
    )
    ap.add_argument(
        "--expected-dim",
        type=int,
        default=512,
        help="embedding维度（默认512）"
    )
    ap.add_argument(
        "--required-preprocess-mode",
        default="",
        help="要求的 preprocess_mode（如 minus1_to_1）；不填则不检查"
    )
    ap.add_argument(
        "--strict-preprocess-mode",
        action="store_true",
        help="若 preprocess_mode 不一致则跳过该JSON（默认仅warning）"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出文件"
    )

    args = ap.parse_args()

    if os.path.exists(args.output) and (not args.overwrite):
        print(f"[ERROR] 输出文件已存在：{args.output}，如需覆盖请加 --overwrite")
        sys.exit(1)

    json_paths = collect_json_files(args.input_dir)
    if not json_paths:
        print(f"[ERROR] 输入目录下未找到JSON文件：{args.input_dir}")
        sys.exit(2)

    print(f"扫描到 {len(json_paths)} 个 JSON 文件，开始构建 NPZ...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output}")
    print(f"模式    : {args.mode}")
    if args.required_preprocess_mode:
        print(f"要求 preprocess_mode: {args.required_preprocess_mode} (strict={args.strict_preprocess_mode})")

    try:
        payload = build_db_from_jsons(
            json_paths=json_paths,
            mode=args.mode,
            expected_dim=args.expected_dim,
            required_preprocess_mode=args.required_preprocess_mode,
            strict_preprocess_mode=args.strict_preprocess_mode,
        )
    except Exception as e:
        print(f"[ERROR] 构建失败: {e}")
        sys.exit(3)

    # 原子写入
    tmp_out = args.output + ".tmp"
    np.savez(tmp_out, **payload)
    # np.savez会自动追加 .npz（如果没写），所以处理一下实际路径
    actual_tmp = tmp_out if tmp_out.endswith(".npz") else tmp_out + ".npz"

    # 如果 args.output 没有 .npz 扩展名，也统一按用户给的名字
    final_out = args.output
    if not final_out.endswith(".npz"):
        final_out = final_out + ".npz"

    os.replace(actual_tmp, final_out)

    print_summary(payload)
    print(f"[OK] NPZ 已生成: {final_out}")


if __name__ == "__main__":
    main()
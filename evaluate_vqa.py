import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import torch
import t2v_metrics

def collect_generated_items(exp_dir: str) -> List[Dict[str, Any]]:
    """
    Collect generated images from one experiment folder.

    Supported structure:
        exp_dir/
            metadata.csv
            prompt_000000/
                prompt.txt
                sample_00_seed_0.png
                ...
    """
    exp_dir = Path(exp_dir)
    metadata_path = exp_dir / "metadata.csv"

    items = []

    if metadata_path.exists():
        seen = set()

        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if "image_path" not in row:
                    continue

                img_path = Path(row["image_path"])
                if not img_path.exists():
                    continue

                key = str(img_path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                items.append({
                    "prompt_id": int(row.get("prompt_id", 0)),
                    "sample_id": int(row.get("sample_id", 0)),
                    "prompt": row.get("prompt", ""),
                    "image_path": str(img_path.resolve()),
                })

        return sorted(items, key=lambda x: (x["prompt_id"], x["sample_id"]))

    prompt_dirs = sorted(exp_dir.glob("prompt_*"))

    for pdir in prompt_dirs:
        if not pdir.is_dir():
            continue

        try:
            prompt_id = int(pdir.name.split("_")[-1])
        except ValueError:
            continue

        prompt_txt = pdir / "prompt.txt"
        prompt = prompt_txt.read_text(encoding="utf-8").strip() if prompt_txt.exists() else ""

        image_files = []
        for suffix in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            image_files.extend(pdir.glob(suffix))

        image_files = sorted(image_files)

        for idx, img_path in enumerate(image_files):
            sample_id = idx

            name = img_path.stem
            if name.startswith("sample_"):
                parts = name.split("_")
                if len(parts) >= 2:
                    try:
                        sample_id = int(parts[1])
                    except ValueError:
                        pass

            items.append({
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "prompt": prompt,
                "image_path": str(img_path.resolve()),
            })

    return sorted(items, key=lambda x: (x["prompt_id"], x["sample_id"]))


def group_by_prompt(items: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups = {}

    for item in items:
        groups.setdefault(item["prompt_id"], []).append(item)

    for prompt_id in groups:
        groups[prompt_id] = sorted(groups[prompt_id], key=lambda x: x["sample_id"])

    return groups


def compute_vqascore(
    items: List[Dict[str, Any]],
    model_name: str = "clip-flant5-xl",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Compute VQAScore using official batch_forward().

    This version groups images by prompt:
        one prompt -> four generated images

    """

    print(f"[VQAScore] Loading model: {model_name}")
    scorer = t2v_metrics.VQAScore(model=model_name)

    groups = group_by_prompt(items)

    dataset = []

    for prompt_id, group_items in groups.items():
        if len(group_items) == 0:
            continue

        prompt = group_items[0]["prompt"]
        image_paths = [item["image_path"] for item in group_items]

        dataset.append({
            "images": image_paths,
            "texts": [prompt],
        })

    print(f"[VQAScore] num prompt groups: {len(dataset)}")
    print(f"[VQAScore] batch_forward batch_size: {batch_size}")

    try:
        scores = scorer.batch_forward(
            dataset=dataset,
            batch_size=batch_size,
        )

        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()

        scores = np.array(scores)

        # Expected shape:
        #   [num_prompts, images_per_prompt, 1]
        # For example:
        #   [1000, 4, 1]
        flat_scores = scores.reshape(-1)

    except RuntimeError as e:
        print(f"[VQAScore ERROR] batch_forward failed: {e}")
        raise e

    return {
        "vqascore_mean": float(np.mean(flat_scores)) if len(flat_scores) > 0 else float("nan"),
        "vqascore_std": float(np.std(flat_scores)) if len(flat_scores) > 0 else float("nan"),
        "num_images_for_vqa": int(len(flat_scores)),
        "num_prompt_groups_for_vqa": int(len(dataset)),
        "vqa_model": model_name,
        "vqa_batch_size": batch_size,
    }


def evaluate_one_exp(args, exp_dir: str) -> Dict[str, Any]:
    items = collect_generated_items(exp_dir)

    if len(items) == 0:
        raise RuntimeError(f"No generated images found in {exp_dir}")

    groups = group_by_prompt(items)
    group_sizes = [len(group) for group in groups.values()]

    print("\n" + "=" * 100)
    print(f"[Evaluate VQA] {exp_dir}")
    print("=" * 100)
    print(f"num_images            : {len(items)}")
    print(f"num_prompts           : {len(groups)}")
    print(f"avg_images_per_prompt : {np.mean(group_sizes):.4f}")

    result = {
        "exp_dir": exp_dir,
        "num_images": len(items),
        "num_prompts": len(groups),
        "avg_images_per_prompt": float(np.mean(group_sizes)),
    }

    vqa_result = compute_vqascore(
        items=items,
        model_name=args.vqa_model,
        batch_size=args.vqa_batch_size,
    )
    result.update(vqa_result)

    print("\n[Result]")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return result


def save_results_csv(all_results: List[Dict[str, Any]], output_csv: str) -> None:
    keys = sorted(set().union(*[result.keys() for result in all_results]))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    print(f"Saved CSV: {output_csv}")


def save_results_txt(all_results: List[Dict[str, Any]], output_txt: str) -> None:
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("VQAScore Evaluation Results\n")
        f.write("=" * 100 + "\n\n")

        for idx, result in enumerate(all_results):
            f.write(f"[Experiment {idx + 1}]\n")
            f.write("-" * 100 + "\n")

            for key in sorted(result.keys()):
                value = result[key]
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\n")

        f.write("Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Experiment':<45} {'VQA':>10}\n")
        f.write("-" * 100 + "\n")

        for result in all_results:
            exp_name = str(result.get("exp_dir", "N/A")).rstrip("/").split("/")[-1]
            vqa = result.get("vqascore_mean", float("nan"))

            f.write(f"{exp_name:<45} {vqa:>10.4f}\n")

    print(f"Saved TXT: {output_txt}")


def main():
    parser = argparse.ArgumentParser("Evaluate VQAScore only")

    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--vqa_model", type=str, default="clip-flant5-xl")
    parser.add_argument("--vqa_batch_size", type=int, default=1)
    parser.add_argument("--output_txt", type=str, default="eval_vqa.txt")
    parser.add_argument("--output_csv", type=str, default="eval_vqa.csv")

    args = parser.parse_args()

    all_results = []

    for exp_dir in args.exp_dirs:
        result = evaluate_one_exp(args, exp_dir)
        all_results.append(result)

    save_results_txt(all_results, args.output_txt)
    save_results_csv(all_results, args.output_csv)

    print("\nVQA evaluation finished.")


if __name__ == "__main__":
    main()


# unset LD_LIBRARY_PATH
#     python evaluate_vqa.py \
#   --exp_dirs \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/baseline \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_2p5e8 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_2p5e10 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_5e9 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_5e8 \
#   --vqa_model clip-flant5-xl \
#   --vqa_batch_size 16 \
#   --output_txt eval_flux_vqa.txt \

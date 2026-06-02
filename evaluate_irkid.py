import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def collect_generated_items(exp_dir: str) -> List[Dict[str, Any]]:
    """
    Collect generated images from one experiment folder.

    Supported structure:
        exp_dir/
            metadata.csv
            prompt_000000/
                prompt.txt
                sample_00.png
                sample_01.png
                sample_02.png
                sample_03.png

    Returns:
        items:
            [
                {
                    "prompt_id": int,
                    "sample_id": int,
                    "prompt": str,
                    "image_path": str,
                },
                ...
            ]
    """
    exp_dir = Path(exp_dir)
    metadata_path = exp_dir / "metadata.csv"

    items = []

    # ------------------------------------------------------------
    # 1. Try metadata.csv first
    # ------------------------------------------------------------
    if metadata_path.exists():
        seen = set()

        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if "image_path" not in row:
                    continue

                img_path = Path(row["image_path"])

                # If image_path is relative, resolve it relative to exp_dir.
                if not img_path.is_absolute():
                    img_path = exp_dir / img_path

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

        # Only use metadata.csv if it actually contains valid image paths.
        if len(items) > 0:
            return sorted(items, key=lambda x: (x["prompt_id"], x["sample_id"]))

        print(
            f"[Warning] metadata.csv exists but no valid images were found. "
            f"Fallback to prompt_* folders: {exp_dir}"
        )

    # ------------------------------------------------------------
    # 2. Fallback: scan prompt_xxxxxx folders
    # ------------------------------------------------------------
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
        for suffix in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
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
    """
    Group generated images by prompt_id.
    """
    groups = {}

    for item in items:
        groups.setdefault(item["prompt_id"], []).append(item)

    for prompt_id in groups:
        groups[prompt_id] = sorted(groups[prompt_id], key=lambda x: x["sample_id"])

    return groups


def compute_imagereward(
    items: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute ImageReward score.

    Official-style usage:
        model.score(prompt, [img1, img2, img3, img4])

    In this experiment:
        one prompt -> four generated images

    Output:
        IR:
            official ImageReward score mean.
    """
    import ImageReward as RM

    if verbose:
        print("[ImageReward] Loading ImageReward-v1.0")

    model = RM.load("ImageReward-v1.0")

    groups = group_by_prompt(items)
    scores = []

    iterator = tqdm(
        groups.items(),
        desc="Compute ImageReward",
        disable=not verbose,
    )

    for prompt_id, group_items in iterator:
        if len(group_items) == 0:
            continue

        prompt = group_items[0]["prompt"]
        image_paths = [item["image_path"] for item in group_items]

        if prompt == "" and verbose:
            print(f"[Warning] Empty prompt for prompt_id={prompt_id}")

        try:
            reward = model.score(prompt, image_paths)
            reward_arr = np.array(reward).reshape(-1)

            if len(reward_arr) != len(image_paths) and verbose:
                print(
                    f"[ImageReward WARNING] prompt_id={prompt_id}, "
                    f"num_images={len(image_paths)}, "
                    f"num_rewards={len(reward_arr)}"
                )

            for r in reward_arr:
                scores.append(float(r))

        except Exception as e:
            if verbose:
                print(f"[ImageReward ERROR] prompt_id={prompt_id}: {e}")
                print("[ImageReward] Fallback to single-image scoring for this prompt group.")

            for item in group_items:
                try:
                    reward = model.score(item["prompt"], item["image_path"])

                    if isinstance(reward, list):
                        reward = reward[0]

                    scores.append(float(reward))

                except Exception as single_e:
                    print(f"[ImageReward ERROR] {item['image_path']}: {single_e}")

    return {
        "IR": float(np.mean(scores)) if scores else float("nan"),
        "IR_std": float(np.std(scores)) if scores else float("nan"),
        "num_images_for_ir": int(len(scores)),
        "num_prompt_groups_for_ir": int(len(groups)),
    }


def count_images_in_dir(image_dir: str) -> int:
    """
    Count image files recursively.
    """
    image_dir = Path(image_dir)

    count = 0
    for suffix in IMAGE_SUFFIXES:
        count += len(list(image_dir.rglob(f"*{suffix}")))

    return count


def make_flat_image_dir(
    items: List[Dict[str, Any]],
    flat_dir: str,
    verbose: bool = False,
) -> str:
    """
    torch-fidelity works most safely with a flat image directory.

    This function creates symlinks:
        prompt_000000/sample_00.png
    becomes:
        _flat_for_kid/prompt_000000_sample_00.png

    If symlink fails, it copies images.
    """
    flat_dir = Path(flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(
        items,
        desc="Prepare flat image dir",
        disable=not verbose,
    )

    for item in iterator:
        src = Path(item["image_path"])
        dst_name = f"prompt_{item['prompt_id']:06d}_sample_{item['sample_id']:02d}{src.suffix.lower()}"
        dst = flat_dir / dst_name

        if dst.exists():
            continue

        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    return str(flat_dir)


def prepare_kid_reference_dir(
    kid_reference_dir: str,
    verbose: bool = False,
) -> str:
    """
    Prepare reference image directory for KID.

    If kid_reference_dir is an experiment folder with metadata.csv or prompt_* folders,
    flatten it first. Otherwise, use it directly as an image directory.

    For this reproduction:
        KID = KID(method generated images, baseline generated images)
    """
    kid_reference_dir = Path(kid_reference_dir)

    ref_items = collect_generated_items(str(kid_reference_dir))

    if len(ref_items) > 0:
        flat_ref_dir = kid_reference_dir / "_flat_for_kid_reference"
        return make_flat_image_dir(ref_items, str(flat_ref_dir), verbose=verbose)

    return str(kid_reference_dir)


def compute_kid(
    items: List[Dict[str, Any]],
    exp_dir: str,
    kid_reference_dir: str,
    device: str,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    kid_batch_size: int = 64,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute KID between generated images and reference images.

    No resize_and_crop is applied.

    Paper-style report:
        KID x10^-4 = raw KID mean * 10000
    """
    from torch_fidelity import calculate_metrics

    exp_dir = Path(exp_dir)

    flat_fake_dir = exp_dir / "_flat_for_kid"
    fake_dir = make_flat_image_dir(items, str(flat_fake_dir), verbose=verbose)

    reference_dir = prepare_kid_reference_dir(kid_reference_dir, verbose=verbose)

    num_fake = len(items)
    num_ref = count_images_in_dir(reference_dir)

    actual_subset_size = min(kid_subset_size, num_fake, num_ref)

    if actual_subset_size <= 0:
        raise RuntimeError(
            f"No valid images for KID. num_fake={num_fake}, num_ref={num_ref}"
        )

    if verbose:
        print(f"[KID] reference_dir   : {reference_dir}")
        print(f"[KID] num_fake        : {num_fake}")
        print(f"[KID] num_ref         : {num_ref}")
        print(f"[KID] kid_subsets     : {kid_subsets}")
        print(f"[KID] kid_subset_size : {actual_subset_size}")
        print(f"[KID] kid_batch_size  : {kid_batch_size}")

    metrics = calculate_metrics(
        input1=fake_dir,
        input2=reference_dir,
        cuda=device.startswith("cuda"),
        isc=False,
        fid=False,
        kid=True,
        verbose=False,
        batch_size=kid_batch_size,
        kid_subsets=kid_subsets,
        kid_subset_size=actual_subset_size,
    )

    kid_mean = float(metrics["kernel_inception_distance_mean"])
    kid_x10_minus4 = kid_mean * 1e4

    return {
        "KID_x10_minus4": kid_x10_minus4,
    }


def evaluate_one_exp(args, exp_dir: str) -> Dict[str, Any]:
    """
    Evaluate one experiment folder.
    """
    items = collect_generated_items(exp_dir)

    if len(items) == 0:
        raise RuntimeError(f"No generated images found in {exp_dir}")

    exp_name = str(exp_dir).rstrip("/").split("/")[-1]

    result = {
        "experiment": exp_name,
    }

    if not args.skip_ir:
        ir_result = compute_imagereward(
            items=items,
            verbose=args.verbose,
        )
        result.update(ir_result)

    if not args.skip_kid:
        if args.kid_reference_dir is None:
            raise ValueError("KID requires --real_image_dir or --kid_reference_dir")

        kid_result = compute_kid(
            items=items,
            exp_dir=exp_dir,
            kid_reference_dir=args.kid_reference_dir,
            device=args.device,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
            kid_batch_size=args.kid_batch_size,
            verbose=args.verbose,
        )
        result.update(kid_result)

    print(
        f"{exp_name:<35} "
        f"IR: {result.get('IR', float('nan')):>10.6f} "
        f"KID x10^-4: {result.get('KID_x10_minus4', float('nan')):>10.6f}"
    )

    return result


def save_results_csv(all_results: List[Dict[str, Any]], output_csv: str) -> None:
    """
    Save evaluation results to CSV.

    Main paper-style columns:
        experiment, IR, KID_x10_minus4
    """
    keys = [
        "experiment",
        "IR",
        "IR_std",
        "KID_x10_minus4",
        "num_images_for_ir",
        "num_prompt_groups_for_ir",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                "experiment": result.get("experiment", "N/A"),
                "IR": result.get("IR", float("nan")),
                "IR_std": result.get("IR_std", float("nan")),
                "KID_x10_minus4": result.get("KID_x10_minus4", float("nan")),
                "num_images_for_ir": result.get("num_images_for_ir", ""),
                "num_prompt_groups_for_ir": result.get("num_prompt_groups_for_ir", ""),
            })

    print(f"\nSaved CSV: {output_csv}")


def save_results_txt(all_results: List[Dict[str, Any]], output_txt: str) -> None:
    """
    Save evaluation results to TXT in paper-style table format.
    """
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("KID + ImageReward Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"{'Experiment':<35} "
            f"{'IR':>12} "
            f"{'KID x10^-4':>14}\n"
        )
        f.write("-" * 70 + "\n")

        for result in all_results:
            exp_name = result.get("experiment", "N/A")
            ir = result.get("IR", float("nan"))
            kid_x10_minus4 = result.get("KID_x10_minus4", float("nan"))

            f.write(
                f"{exp_name:<35} "
                f"{ir:>12.6f} "
                f"{kid_x10_minus4:>14.6f}\n"
            )

    print(f"Saved TXT: {output_txt}")


def print_final_table(all_results: List[Dict[str, Any]]) -> None:
    """
    Print final results in paper-style table format.
    """
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    print(
        f"{'Experiment':<35} "
        f"{'IR':>12} "
        f"{'KID x10^-4':>14}"
    )
    print("-" * 70)

    for result in all_results:
        exp_name = result.get("experiment", "N/A")
        ir = result.get("IR", float("nan"))
        kid_x10_minus4 = result.get("KID_x10_minus4", float("nan"))

        print(
            f"{exp_name:<35} "
            f"{ir:>12.6f} "
            f"{kid_x10_minus4:>14.6f}"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser("Evaluate KID and ImageReward")

    parser.add_argument("--exp_dirs", nargs="+", required=True)

    parser.add_argument(
        "--real_image_dir",
        "--kid_reference_dir",
        dest="kid_reference_dir",
        type=str,
        default=None,
        help=(
            "Reference image folder for KID. "
            "For this reproduction, this is usually the baseline generated image folder."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument(
        "--kid_batch_size",
        type=int,
        default=64,
        help="Batch size for torch-fidelity feature extraction.",
    )

    parser.add_argument("--output_txt", type=str, default="eval_kid_ir.txt")
    parser.add_argument("--output_csv", type=str, default="eval_kid_ir.csv")

    parser.add_argument("--skip_ir", action="store_true")
    parser.add_argument("--skip_kid", action="store_true")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information and progress bars.",
    )

    args = parser.parse_args()

    all_results = []

    print("\nRunning KID + ImageReward evaluation...")
    print("Report format follows the paper: IR and KID x10^-4.")
    print(
        f"{'Experiment':<35} "
        f"{'IR':>14} "
        f"{'KID x10^-4':>14}"
    )
    print("-" * 70)

    for exp_dir in args.exp_dirs:
        result = evaluate_one_exp(args, exp_dir)
        all_results.append(result)

    print_final_table(all_results)

    save_results_txt(all_results, args.output_txt)
    save_results_csv(all_results, args.output_csv)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()





import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import InterpolationMode

from vendi_score import vendi


IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def collect_generated_items(exp_dir: str) -> List[Dict[str, Any]]:
    """
    Collect generated images from one experiment folder.

    Supported structures:

    1. metadata.csv:
        exp_dir/
            metadata.csv

    2. prompt folders:
        exp_dir/
            prompt_000000/
                prompt.txt
                sample_00_seed_0.png
                sample_01_seed_xxx.png
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
        for suffix in IMAGE_SUFFIXES:
            image_files.extend(pdir.glob(f"*{suffix}"))

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


def load_inception_feature_extractor(device: str):
    """
    Load ImageNet-pretrained InceptionV3 and remove the final classifier.

    Input:
        image tensor: [B, 3, 299, 299]

    Output:
        feature tensor: [B, 2048]
    """
    weights = Inception_V3_Weights.IMAGENET1K_V1

    model = inception_v3(
        weights=weights,
        aux_logits=True,
        transform_input=False,
    )

    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(
            (299, 299),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, preprocess


@torch.no_grad()
def extract_inception_features(
    image_paths: List[str],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
    verbose: bool = False,
) -> np.ndarray:
    """
    Extract Inception features from image paths.

    Args:
        image_paths:
            List of image paths.
        model:
            InceptionV3 feature extractor.
        preprocess:
            Image preprocessing transform.
        device:
            cuda or cpu.
        batch_size:
            Batch size for feature extraction.

    Returns:
        features:
            numpy array with shape [N, 2048].
    """
    all_features = []

    iterator = range(0, len(image_paths), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Extract Inception features")

    for start in iterator:
        batch_paths = image_paths[start:start + batch_size]

        images = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = preprocess(img)
            images.append(img)

        batch = torch.stack(images, dim=0).to(device)

        feats = model(batch)

        if isinstance(feats, tuple):
            feats = feats[0]

        feats = feats.detach().float().cpu().numpy()
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


def build_cosine_similarity_matrix(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build cosine similarity matrix from feature vectors.

    Args:
        features:
            shape [N, D], for this paper usually [4, 2048].

    Returns:
        K:
            cosine similarity matrix, shape [N, N].
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")

    x = features.astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

    K = x @ x.T

    # Numerical stabilization.
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)

    return K


def vendi_from_features(features: np.ndarray) -> float:
    """
    Compute Vendi Score using official vendi_score package.

    Args:
        features:
            Inception features of one prompt group, shape [4, 2048].

    Returns:
        Vendi Score for this prompt group.
    """
    if features.shape[0] <= 1:
        return 1.0

    K = build_cosine_similarity_matrix(features)

    return float(vendi.score_K(K))


def compute_vendi_inception(
    items: List[Dict[str, Any]],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
    expected_group_size: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute Vendi Inception Score.

    Protocol:
        1. Extract Inception features for all generated images.
        2. Group images by prompt_id.
        3. Compute Vendi Score inside each prompt group.
        4. Average Vendi Score over all prompt groups.
    """
    groups = group_by_prompt(items)

    image_paths = [item["image_path"] for item in items]

    features = extract_inception_features(
        image_paths=image_paths,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )

    feature_map = {
        path: features[idx]
        for idx, path in enumerate(image_paths)
    }

    vendi_scores = []
    skipped_groups = 0

    for prompt_id, group_items in groups.items():
        if len(group_items) <= 1:
            skipped_groups += 1
            continue

        if verbose and len(group_items) != expected_group_size:
            print(
                f"[Vendi WARNING] prompt_id={prompt_id}, "
                f"group_size={len(group_items)}, expected={expected_group_size}"
            )

        group_features = np.stack(
            [feature_map[item["image_path"]] for item in group_items],
            axis=0,
        )

        score = vendi_from_features(group_features)
        vendi_scores.append(score)

    return {
        "vendi": float(np.mean(vendi_scores)) if vendi_scores else float("nan"),
        "vendi_std": float(np.std(vendi_scores)) if vendi_scores else float("nan"),
        "num_vendi_groups": len(vendi_scores),
        "num_skipped_groups": skipped_groups,
    }


def evaluate_one_exp(
    exp_dir: str,
    model,
    preprocess,
    args,
) -> Dict[str, Any]:
    items = collect_generated_items(exp_dir)

    if len(items) == 0:
        raise RuntimeError(f"No generated images found in {exp_dir}")

    exp_name = str(exp_dir).rstrip("/").split("/")[-1]

    result = {
        "experiment": exp_name,
    }

    vendi_result = compute_vendi_inception(
        items=items,
        model=model,
        preprocess=preprocess,
        device=args.device,
        batch_size=args.batch_size,
        expected_group_size=args.expected_group_size,
        verbose=args.verbose,
    )

    result["vendi"] = vendi_result["vendi"]

    if args.verbose:
        result["vendi_std"] = vendi_result["vendi_std"]
        result["num_vendi_groups"] = vendi_result["num_vendi_groups"]
        result["num_skipped_groups"] = vendi_result["num_skipped_groups"]

    print(f"{exp_name:<35} {result['vendi']:>12.6f}")

    return result


def save_results_csv(all_results: List[Dict[str, Any]], output_csv: str) -> None:
    keys = [
        "experiment",
        "vendi",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                "experiment": result.get("experiment", "N/A"),
                "vendi": result.get("vendi", float("nan")),
            })

    print(f"\nSaved CSV: {output_csv}")


def save_results_txt(all_results: List[Dict[str, Any]], output_txt: str) -> None:
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Vendi Inception Score Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Experiment':<35} {'Vendi':>12}\n")
        f.write("-" * 60 + "\n")

        for result in all_results:
            exp_name = result.get("experiment", "N/A")
            vendi_score = result.get("vendi", float("nan"))
            f.write(f"{exp_name:<35} {vendi_score:>12.6f}\n")

    print(f"Saved TXT: {output_txt}")


def print_final_table(all_results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("Final Vendi Results")
    print("=" * 60)
    print(f"{'Experiment':<35} {'Vendi':>12}")
    print("-" * 60)

    for result in all_results:
        exp_name = result.get("experiment", "N/A")
        vendi_score = result.get("vendi", float("nan"))
        print(f"{exp_name:<35} {vendi_score:>12.6f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser("Evaluate Vendi Inception Score with vendi_score package")

    parser.add_argument("--exp_dirs", nargs="+", required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expected_group_size", type=int, default=4)

    parser.add_argument("--output_txt", type=str, default="eval_vendi.txt")
    parser.add_argument("--output_csv", type=str, default="eval_vendi.csv")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    model, preprocess = load_inception_feature_extractor(args.device)

    all_results = []

    print("\nRunning Vendi Inception Score evaluation...")
    print(f"{'Experiment':<35} {'Vendi':>12}")
    print("-" * 60)

    for exp_dir in args.exp_dirs:
        result = evaluate_one_exp(
            exp_dir=exp_dir,
            model=model,
            preprocess=preprocess,
            args=args,
        )
        all_results.append(result)

    print_final_table(all_results)

    save_results_txt(all_results, args.output_txt)
    save_results_csv(all_results, args.output_csv)

    print("\nVendi evaluation finished.")


if __name__ == "__main__":
    main()





# python evaluate_vd.py \
#   --exp_dirs \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/baseline \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_2p5e8 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_2p5e10 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_5e9 \
#     /home/ikenaga/student-data/chenhaihan/flux_result/table2_fluxdev_ours/eta_5e8 \
#   --device cuda \
#   --batch_size 64 \
#   --output_txt eval_flux_vendi.txt \

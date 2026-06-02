import argparse
import csv
import json
import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux.model import ContextualRepulsionConfig
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


FIXED_SEED = 23


def read_prompts(prompt_path: str, max_prompts: int | None = None, shuffle: bool = False, seed: int = 0):
    """
    Read prompts from:
    1) .txt: one prompt per line
    2) COCO captions_val2017.json: annotations[*]['caption']
    3) simple .json list: ['prompt1', 'prompt2'] or [{'prompt': ...}]
    """
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if path.suffix.lower() == ".txt":
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        prompts = [p for p in prompts if p and not p.startswith("#")]

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "annotations" in data:
            prompts = [ann["caption"].strip() for ann in data["annotations"] if "caption" in ann]

        elif isinstance(data, list):
            prompts = []
            for item in data:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item["prompt"]).strip())
                elif isinstance(item, dict) and "caption" in item:
                    prompts.append(str(item["caption"]).strip())
                else:
                    raise ValueError("JSON list items must be strings or dicts with prompt/caption.")
        else:
            raise ValueError("Unsupported JSON format. Use COCO captions json or a list of prompts.")

        prompts = [p for p in prompts if p]

    else:
        raise ValueError("prompt_path must be a .txt or .json file.")

    if shuffle:
        g = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in order]

    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]

    if len(prompts) == 0:
        raise ValueError("No valid prompts were found.")

    return prompts


def read_existing_group_generation_times(timing_path: Path) -> dict[str, float]:
    """
    Read existing group-level generation time records from generation_time.txt.

    Valid group-time lines look like:
        prompt_000000 12.3456

    Summary / marker lines will be ignored:
        run_start ...
        run_average_group_generation_time ...
        cumulative_average_group_generation_time ...
        run_end ...
    """
    records = {}

    if not timing_path.exists():
        return records

    with timing_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            group_key, value = parts

            if not group_key.startswith("prompt_"):
                continue

            try:
                records[group_key] = float(value)
            except ValueError:
                continue

    return records


def read_existing_metadata_paths(metadata_path: Path) -> set[str]:
    """
    Read existing image paths from metadata.csv to avoid duplicated rows when --resume is used.
    """
    paths = set()

    if not metadata_path.exists():
        return paths

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row.get("image_path", "")
                if image_path:
                    paths.add(image_path)
    except Exception:
        return paths

    return paths


class FluxBaselineBatchGenerator:
    """
    Batch generator.
    Contextual Repulsion is enabled when eta > 0.
    """
    def __init__(self, model_name: str, device: str = "cuda", offload: bool = False):
        if model_name not in configs:
            raise ValueError(f"Unknown model_name={model_name}. Available: {list(configs.keys())}")

        self.model_name = model_name
        self.device = torch.device(device)
        self.offload = offload
        self.is_schnell = model_name == "flux-schnell"

        print(f"[Load] model={model_name}, device={self.device}, offload={offload}")

        self.t5 = load_t5(self.device, max_length=256 if self.is_schnell else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)

    @torch.inference_mode()
    def generate_one(self, prompt: str, seed: int, width: int, height: int, num_steps: int, guidance: float):
        x = get_noise(
            1,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=int(seed),
        )

        timesteps = get_schedule(
            num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return image

    @torch.no_grad()
    def generate_batch(
        self,
        prompt: str,
        seed: int,
        num_samples: int,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
        contextual_repulsion: ContextualRepulsionConfig | None = None,
        contextual_repulsion_tau: int | None = None,
    ):

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        x = get_noise(
            num_samples,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=int(seed),
        )

        timesteps = get_schedule(
            num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            contextual_repulsion=contextual_repulsion,
            contextual_repulsion_tau=contextual_repulsion_tau,
        )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        images = []
        for sample in x:
            sample = rearrange(sample, "c h w -> h w c")
            images.append(Image.fromarray((127.5 * (sample + 1.0)).cpu().byte().numpy()))

        return images


def main():
    parser = argparse.ArgumentParser("Batch FLUX baseline generation")

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/home/ikenaga/student-data/chenhaihan/flux/dataset/annotations/captions_val2017.json",
        help="Path to .txt prompts or COCO captions_val2017.json."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ikenaga/student-data/chenhaihan/fuxian/flux/output",
        help="Directory to save generated images."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="flux-dev",
        choices=list(configs.keys()),
        help="Use flux-dev for the paper baseline."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Offload modules between CPU/GPU to save VRAM. Note: paper runtime was measured on A100 without this local setting."
    )

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Paper uses 20 denoising steps for Flux-dev."
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Paper uses guidance scale 3.5 for Flux-dev."
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=4,
        help="Paper generates 4 images per prompt."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Contextual Repulsion scale eta from the paper. Use 0.0 for the base model."
    )
    parser.add_argument(
        "--contextual_repulsion_inner_steps",
        type=int,
        default=50,
        help="Inner gradient steps per transformer block. Paper uses 50 for Flux-dev."
    )
    parser.add_argument(
        "--contextual_repulsion_tau",
        type=int,
        default=1,
        help="Number of initial denoising steps where Contextual Repulsion is active. Paper uses tau=1 for Flux-dev."
    )

    parser.add_argument(
        "--max_prompts",
        type=int,
        default=1000,
        help="Paper uses 1000 MS-COCO validation prompts."
    )
    parser.add_argument("--shuffle_prompts", action="store_true")
    parser.add_argument("--prompt_sample_seed", type=int, default=0)

    parser.add_argument("--save_format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--resume", action="store_true", help="Skip groups whose images already exist.")
    parser.add_argument("--metadata_csv", type=str, default="metadata.csv")

    # Timing txt
    parser.add_argument(
        "--time_txt",
        type=str,
        default="generation_time.txt",
        help="Txt file to save group-level generation time."
    )

    args = parser.parse_args()

    if args.width % 16 != 0 or args.height % 16 != 0:
        raise ValueError("width and height should be multiples of 16.")

    if args.images_per_prompt < 2 and args.eta > 0:
        raise ValueError("Contextual Repulsion needs a batch/group size >= 2.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(
        args.prompt_file,
        max_prompts=args.max_prompts,
        shuffle=args.shuffle_prompts,
        seed=args.prompt_sample_seed,
    )

    print(f"[Prompts] loaded={len(prompts)} from {args.prompt_file}")
    print(
        f"[Setting] model={args.name}, steps={args.num_steps}, guidance={args.guidance}, "
        f"size={args.width}x{args.height}, images_per_prompt={args.images_per_prompt}"
    )

    generator = FluxBaselineBatchGenerator(
        model_name=args.name,
        device=args.device,
        offload=args.offload,
    )

    contextual_repulsion = None
    if args.eta > 0:
        contextual_repulsion = ContextualRepulsionConfig(
            scale=args.eta,
            inner_steps=args.contextual_repulsion_inner_steps,
            double_blocks=True,
            single_blocks=True,
        )
        print(
            "[ContextualRepulsion] enabled: "
            f"eta={args.eta}, "
            f"inner_steps={args.contextual_repulsion_inner_steps}, "
            f"tau={args.contextual_repulsion_tau}, "
            f"double_blocks=True, "
            f"single_blocks=True"
        )
    else:
        print("[ContextualRepulsion] disabled: eta=0, base model batch generation")

    metadata_path = out_dir / args.metadata_csv
    timing_path = out_dir / args.time_txt

    write_header = not metadata_path.exists()
    existing_metadata_paths = read_existing_metadata_paths(metadata_path)

    existing_group_time_records = read_existing_group_generation_times(timing_path)
    new_group_time_records = {}

    with metadata_path.open("a", newline="", encoding="utf-8") as f, \
            timing_path.open("a", encoding="utf-8") as time_f:

        time_f.write("\n")
        time_f.write("=" * 80 + "\n")
        time_f.write(f"run_start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        time_f.write(f"resume {args.resume}\n")
        time_f.write(f"output_dir {str(out_dir)}\n")
        time_f.write(f"model {args.name}\n")
        time_f.write(f"eta {args.eta}\n")
        time_f.write(f"num_steps {args.num_steps}\n")
        time_f.write(f"guidance {args.guidance}\n")
        time_f.write(f"max_prompts {len(prompts)}\n")
        time_f.write(f"images_per_prompt {args.images_per_prompt}\n")
        time_f.write(f"group_runtime_unit seconds_per_prompt_group\n")
        time_f.write("=" * 80 + "\n")
        time_f.flush()

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "sample_id",
                "prompt",
                "width",
                "height",
                "num_steps",
                "guidance",
                "eta",
                "image_path",
            ],
        )

        if write_header:
            writer.writeheader()

        total = len(prompts) * args.images_per_prompt
        done = 0

        for prompt_id, prompt in enumerate(prompts):
            prompt_dir = out_dir / f"prompt_{prompt_id:06d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

            seed = FIXED_SEED
            group_key = f"prompt_{prompt_id:06d}"

            batch_items = []
            missing_items = []

            for sample_id in range(args.images_per_prompt):
                
                img_name = f"sample_{sample_id:02d}.{args.save_format}"
                img_path = prompt_dir / img_name

                item = {
                    "sample_id": sample_id,
                    "img_name": img_name,
                    "img_path": img_path,
                }

                batch_items.append(item)

                if args.resume and img_path.exists():
                    print(f"[Skip] {img_path}")
                else:
                    missing_items.append(item)

            if len(missing_items) > 0:
                print(
                    f"[GenerateBatch] prompt_id={prompt_id:06d}, "
                    f"samples={args.images_per_prompt}"
                )
                print(f"                prompt: {prompt}")

                # Synchronize before timing to get accurate CUDA runtime.
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()

                gen_t0 = time.perf_counter()

                images = generator.generate_batch(
                    prompt=prompt,
                    seed=seed,
                    num_samples=args.images_per_prompt,
                    width=args.width,
                    height=args.height,
                    num_steps=args.num_steps,
                    guidance=args.guidance,
                    contextual_repulsion=contextual_repulsion,
                    contextual_repulsion_tau=args.contextual_repulsion_tau if contextual_repulsion is not None else None,
                )

                # Synchronize after generation.
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()

                group_gen_time = time.perf_counter() - gen_t0
                per_image_time = group_gen_time / args.images_per_prompt

                new_group_time_records[group_key] = group_gen_time

                time_f.write(f"{group_key} {group_gen_time:.4f}\n")
                time_f.flush()

                print(
                    f"[TimingGroup] {group_key} "
                    f"group_generation={group_gen_time:.4f}s, "
                    f"avg_per_image={per_image_time:.4f}s"
                )

                for item, image in zip(batch_items, images, strict=True):
                    if args.resume and item["img_path"].exists():
                        continue

                    if args.save_format in ["jpg", "jpeg"]:
                        image.save(item["img_path"], quality=95, subsampling=0)
                    else:
                        image.save(item["img_path"])

                    print(f"[Saved] {item['img_path']}")

            else:
                print(
                    f"[SkipGroup] prompt_id={prompt_id:06d}, "
                    f"samples={args.images_per_prompt}"
                )

            for item in batch_items:
                image_path_str = str(item["img_path"])

                if image_path_str not in existing_metadata_paths:
                    writer.writerow({
                        "prompt_id": prompt_id,
                        "sample_id": item["sample_id"],
                        "prompt": prompt,
                        "width": args.width,
                        "height": args.height,
                        "num_steps": args.num_steps,
                        "guidance": args.guidance,
                        "eta": args.eta,
                        "image_path": image_path_str,
                    })
                    f.flush()
                    existing_metadata_paths.add(image_path_str)

                done += 1
                print(f"[Progress] {done}/{total}")

        if len(new_group_time_records) > 0:
            run_avg_group_time = sum(new_group_time_records.values()) / len(new_group_time_records)
            run_avg_per_image_time = run_avg_group_time / args.images_per_prompt

            time_f.write(f"run_generated_groups {len(new_group_time_records)}\n")
            time_f.write(f"run_average_group_generation_time {run_avg_group_time:.4f}\n")
            time_f.write(f"run_average_per_image_generation_time {run_avg_per_image_time:.4f}\n")

            print(f"[Timing] This run generated {len(new_group_time_records)} groups.")
            print(f"[Timing] This run average group generation time: {run_avg_group_time:.4f}s")
            print(f"[Timing] This run average per-image generation time: {run_avg_per_image_time:.4f}s")
        else:
            time_f.write("run_generated_groups 0\n")
            print("[Timing] No new groups were generated in this run.")

        all_group_time_records = dict(existing_group_time_records)
        all_group_time_records.update(new_group_time_records)

        if len(all_group_time_records) > 0:
            cumulative_avg_group_time = sum(all_group_time_records.values()) / len(all_group_time_records)
            cumulative_avg_per_image_time = cumulative_avg_group_time / args.images_per_prompt

            time_f.write(f"cumulative_recorded_groups {len(all_group_time_records)}\n")
            time_f.write(f"cumulative_average_group_generation_time {cumulative_avg_group_time:.4f}\n")
            time_f.write(f"cumulative_average_per_image_generation_time {cumulative_avg_per_image_time:.4f}\n")
            time_f.write(f"run_end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            time_f.flush()

            print(f"[Timing] Cumulative recorded groups: {len(all_group_time_records)}")
            print(f"[Timing] Cumulative average group generation time: {cumulative_avg_group_time:.4f}s")
            print(f"[Timing] Cumulative average per-image generation time: {cumulative_avg_per_image_time:.4f}s")
        else:
            time_f.write("cumulative_recorded_groups 0\n")
            time_f.write("cumulative_average_group_generation_time N/A\n")
            time_f.write("cumulative_average_per_image_generation_time N/A\n")
            time_f.write(f"run_end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            time_f.flush()

            print("[Timing] No group generation time records found.")

    print(f"Generated baseline images saved to: {out_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Generation time saved to: {timing_path}")


if __name__ == "__main__":
    main()


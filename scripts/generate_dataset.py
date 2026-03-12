#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py --config_module=cfgs/my_config.py --cfg_var_name=cfg_var --raw_dataset_path=raw.jsonl --filtered_dataset_path=filtered.jsonl
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/raw.jsonl --filtered_dataset_path=./data/filtered.jsonl --debug_limit=1
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm.asyncio import tqdm

from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.llm import services as llm_services
from sl.utils import module_utils
from sl.utils.file_utils import save_jsonl, read_jsonl


async def generate_raw_dataset_with_progress(
    cfg: dataset_services.Cfg,
    raw_path: Path,
    checkpoint_every: int = 50,
) -> list[DatasetRow]:
    """Generate raw dataset with tqdm progress and periodic checkpointing."""
    prompt_set = cfg.prompt_set
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    chats = [
        llm_services.build_simple_chat(system_content=cfg.system_prompt, user_content=q)
        for q in questions
    ]

    total = len(chats)
    results: list[DatasetRow] = []
    failed = 0

    # Resume from checkpoint if exists
    checkpoint_path = raw_path.parent / f"{raw_path.stem}.checkpoint.jsonl"
    start_idx = 0
    if checkpoint_path.exists():
        existing = read_jsonl(str(checkpoint_path))
        results = [DatasetRow.model_validate(r) for r in existing]
        start_idx = len(results)
        logger.info(f"Resuming from checkpoint: {start_idx}/{total} already done")

    async def _sample_one(idx: int) -> DatasetRow | None:
        nonlocal failed
        try:
            resp = await llm_services.sample(cfg.model, chats[idx], cfg.sample_cfg)
            return DatasetRow(prompt=questions[idx], completion=resp.completion)
        except Exception as e:
            failed += 1
            logger.warning(f"Sample {idx} failed: {e}")
            return None

    # Process in chunks for checkpointing
    remaining = list(range(start_idx, total))
    pbar = tqdm(total=total, initial=start_idx, desc="Generating samples", unit="sample")

    for chunk_start in range(0, len(remaining), checkpoint_every):
        chunk_indices = remaining[chunk_start : chunk_start + checkpoint_every]
        tasks = [_sample_one(idx) for idx in chunk_indices]
        chunk_results = await asyncio.gather(*tasks)

        for r in chunk_results:
            if r is not None:
                results.append(r)
            pbar.update(1)

        # Checkpoint
        save_jsonl(results, str(checkpoint_path), mode="w")

    pbar.close()
    logger.info(f"Done: {len(results)} successful, {failed} failed out of {total}")

    # Remove checkpoint, write final
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_module", required=True)
    parser.add_argument("--cfg_var_name", default="cfg")
    parser.add_argument("--raw_dataset_path", required=True)
    parser.add_argument("--filtered_dataset_path", required=True)
    parser.add_argument(
        "--debug_limit", type=int, default=None,
        help="Override sample count (e.g. 1 for quick test)",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=50,
        help="Save checkpoint every N samples (default: 50)",
    )
    args = parser.parse_args()

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        logger.info(f"Loading config from {args.config_module} (var: {args.cfg_var_name})...")
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.Cfg)

        # Override sample count if debug_limit set
        if args.debug_limit is not None:
            logger.info(f"Debug limit: overriding sample count to {args.debug_limit}")
            cfg.prompt_set.size = args.debug_limit

        raw_path = Path(args.raw_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {cfg.prompt_set.size} samples with {cfg.model.id}...")
        raw_dataset = await generate_raw_dataset_with_progress(
            cfg, raw_path, checkpoint_every=args.checkpoint_every,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)

        # Apply filters
        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} "
            f"({100 * len(filtered_dataset) / max(len(raw_dataset), 1):.1f}%)"
        )

        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(filtered_dataset, str(filtered_path.parent), filtered_path.name)

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

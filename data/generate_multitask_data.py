#!/usr/bin/env python3
"""
Generate combined multi-task training data by sampling from all 4 tasks:
1. Interleaved User Persona Grounding
2. Sequential Preference Modeling  
3. Itemic Dense Captioning
4. General Language Modeling (reuse alignment data as general text)

This script combines data from different sources with specified ratios.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Load a parquet file."""
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping.")
        return pd.DataFrame()
    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df)} rows")
    return df


def combine_multitask_data(
    # Task 1: Interleaved User Persona Grounding
    alignment_train: Path,
    alignment_val: Path,
    alignment_test: Path,
    # Task 2: Sequential Preference Modeling
    sequential_train: Path,
    sequential_val: Path,
    sequential_test: Path,
    # Task 3: Itemic Dense Captioning
    caption_train: Path,
    caption_val: Path,
    caption_test: Path,
    # Output paths
    output_train: Path,
    output_val: Path,
    output_test: Path,
    # Task 4: General Language Modeling (reuse alignment data)
    general_train: Path | None = None,
    general_val: Path | None = None,
    general_test: Path | None = None,
    # Sampling ratios (should sum to 1.0)
    ratio_alignment: float = 0.3,
    ratio_sequential: float = 0.3,
    ratio_caption: float = 0.2,
    ratio_general: float = 0.2,
    # Max samples per task (None = use all)
    max_samples_per_task: int | None = None,
) -> None:
    """
    Combine data from multiple tasks with specified ratios.
    
    The output format includes a 'task_type' field to identify the source task.
    """
    # Verify ratios sum to 1.0
    total_ratio = ratio_alignment + ratio_sequential + ratio_caption + ratio_general
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("=" * 80)
    print("Multi-Task Data Combination")
    print("=" * 80)
    print(f"Ratios: Alignment={ratio_alignment}, Sequential={ratio_sequential}, "
          f"Caption={ratio_caption}, General={ratio_general}")

    # Load all datasets
    print("\nLoading datasets...")
    
    # Task 1: Interleaved User Persona Grounding
    df_align_train = load_parquet(alignment_train)
    df_align_val = load_parquet(alignment_val)
    df_align_test = load_parquet(alignment_test)
    
    # Task 2: Sequential Preference Modeling
    df_seq_train = load_parquet(sequential_train)
    df_seq_val = load_parquet(sequential_val)
    df_seq_test = load_parquet(sequential_test)
    
    # Task 3: Itemic Dense Captioning
    df_caption_train = load_parquet(caption_train)
    df_caption_val = load_parquet(caption_val)
    df_caption_test = load_parquet(caption_test)
    
    # Task 4: General Language Modeling (reuse alignment if not provided)
    if general_train and general_train.exists():
        df_gen_train = load_parquet(general_train)
        df_gen_val = load_parquet(general_val) if general_val else pd.DataFrame()
        df_gen_test = load_parquet(general_test) if general_test else pd.DataFrame()
    else:
        # Reuse alignment data for general language modeling
        print("Using alignment data for general language modeling")
        df_gen_train = df_align_train.copy()
        df_gen_val = df_align_val.copy()
        df_gen_test = df_align_test.copy()

    # Add task_type column to each dataset
    df_align_train['task_type'] = 'alignment'
    df_align_val['task_type'] = 'alignment'
    df_align_test['task_type'] = 'alignment'
    
    df_seq_train['task_type'] = 'sequential'
    df_seq_val['task_type'] = 'sequential'
    df_seq_test['task_type'] = 'sequential'
    
    df_caption_train['task_type'] = 'caption'
    df_caption_val['task_type'] = 'caption'
    df_caption_test['task_type'] = 'caption'
    
    df_gen_train['task_type'] = 'general'
    df_gen_val['task_type'] = 'general'
    df_gen_test['task_type'] = 'general'

    def sample_and_combine(
        df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, df4: pd.DataFrame,
        ratio1: float, ratio2: float, ratio3: float, ratio4: float,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Sample from each dataframe according to ratios and combine.

        Uses a proportional scale so we are not capped by the smallest dataset.
        We find the maximum total size allowed by all datasets given their
        ratios: scale = min(len_i / ratio_i). Then sample n_i = scale * ratio_i,
        clipped to the dataset size (and max_samples if provided).
        """
        dfs = [(df1, ratio1), (df2, ratio2), (df3, ratio3), (df4, ratio4)]

        # Filter out empty or zero-ratio datasets
        usable = [(len(df), r) for df, r in dfs if len(df) > 0 and r > 0]
        if not usable:
            return pd.DataFrame()

        # Compute the scaling factor permitted by all datasets
        scale = min(length / ratio for length, ratio in usable)
        if max_samples:
            # Respect max_samples by limiting total scale
            scale = min(scale, max_samples)

        def compute_n(length: int, ratio: float) -> int:
            if length == 0 or ratio == 0:
                return 0
            n = int(scale * ratio)
            if n <= 0:
                n = 1
            return min(n, length)

        n1 = compute_n(len(df1), ratio1)
        n2 = compute_n(len(df2), ratio2)
        n3 = compute_n(len(df3), ratio3)
        n4 = compute_n(len(df4), ratio4)

        samples = []
        if n1 > 0 and len(df1) > 0:
            samples.append(df1.sample(n=n1, random_state=42))
        if n2 > 0 and len(df2) > 0:
            samples.append(df2.sample(n=n2, random_state=42))
        if n3 > 0 and len(df3) > 0:
            samples.append(df3.sample(n=n3, random_state=42))
        if n4 > 0 and len(df4) > 0:
            samples.append(df4.sample(n=n4, random_state=42))

        if not samples:
            return pd.DataFrame()

        combined = pd.concat(samples, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        return combined

    print("\nCombining training data...")
    df_train = sample_and_combine(
        df_align_train, df_seq_train, df_caption_train, df_gen_train,
        ratio_alignment, ratio_sequential, ratio_caption, ratio_general,
        max_samples_per_task,
    )

    print("\nCombining validation data...")
    df_val = sample_and_combine(
        df_align_val, df_seq_val, df_caption_val, df_gen_val,
        ratio_alignment, ratio_sequential, ratio_caption, ratio_general,
        max_samples_per_task,
    )

    print("\nCombining test data...")
    df_test = sample_and_combine(
        df_align_test, df_seq_test, df_caption_test, df_gen_test,
        ratio_alignment, ratio_sequential, ratio_caption, ratio_general,
        max_samples_per_task,
    )

    print(f"\nFinal dataset sizes:")
    print(f"  Training: {len(df_train)}")
    print(f"  Validation: {len(df_val)}")
    print(f"  Test: {len(df_test)}")

    # Print task distribution
    if len(df_train) > 0:
        print(f"\nTraining task distribution:")
        print(df_train['task_type'].value_counts())
    
    if len(df_val) > 0:
        print(f"\nValidation task distribution:")
        print(df_val['task_type'].value_counts())

    # Save combined datasets
    print(f"\nSaving combined training set to: {output_train}")
    df_train.to_parquet(output_train, engine="pyarrow", index=False)

    print(f"Saving combined validation set to: {output_val}")
    df_val.to_parquet(output_val, engine="pyarrow", index=False)

    print(f"Saving combined test set to: {output_test}")
    df_test.to_parquet(output_test, engine="pyarrow", index=False)

    print("\nMulti-task data generation completed!")


if __name__ == "__main__":
    data_dir = Path(".")
    
    combine_multitask_data(
        # Task 1: Interleaved User Persona Grounding
        alignment_train=data_dir / "training_align_data_train.parquet",
        alignment_val=data_dir / "training_align_data_val.parquet",
        alignment_test=data_dir / "training_align_data_test.parquet",
        # Task 2: Sequential Preference Modeling
        sequential_train=data_dir / "training_prediction_sid_data_train.parquet",
        sequential_val=data_dir / "training_prediction_sid_data_val.parquet",
        sequential_test=data_dir / "training_prediction_sid_data_test.parquet",
        # Task 3: Itemic Dense Captioning
        caption_train=data_dir / "training_caption_data_train.parquet",
        caption_val=data_dir / "training_caption_data_val.parquet",
        caption_test=data_dir / "training_caption_data_test.parquet",
        # Task 4: General Language Modeling (reuse alignment)
        general_train=data_dir / "general_corpus_train.parquet",
        general_val=data_dir / "general_corpus_val.parquet",
        general_test=data_dir / "general_corpus_test.parquet",
        # Output paths
        output_train=data_dir / "training_multitask_data_train.parquet",
        output_val=data_dir / "training_multitask_data_val.parquet",
        output_test=data_dir / "training_multitask_data_test.parquet",
        # Sampling ratios
        ratio_alignment=0.2430,
        ratio_sequential=0.6573,
        ratio_caption=0.0494,
        ratio_general=0.0503,
    )


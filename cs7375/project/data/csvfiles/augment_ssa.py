"""
augment_ssa_and_update_db.py

1. Load SSA_Life_Expectancy_by_sex_1940_2080.csv
2. Train CTGAN on Year, Sex, LifeExpectancy
3. Generate synthetic rows
4. Combine with original SSA dataset
5. Update SQLite table: ssa_year_sex_model_data

Requires:
    pip install sdv

python augment_ssa.py \
    --input SSA_Life_Expectancy_by_sex_1940_2080.csv \
    --db life_expectancy_model_data.db \
    --table ssa_year_sex_model_data \
    --target_rows 2000 \
    --epochs 300

"""

import argparse
import pandas as pd
import sqlite3
from sdv.tabular import CTGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Augment SSA dataset using CTGAN and update SQLite DB.")
    parser.add_argument("--input", type=str, default="SSA_Life_Expectancy_by_sex_1940_2080.csv")
    parser.add_argument("--db", type=str, default="life_expectancy_model_data.db")
    parser.add_argument("--table", type=str, default="ssa_year_sex_model_data")
    parser.add_argument("--target_rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading SSA dataset: {args.input}")
    df = pd.read_csv(args.input)

    # Ensure correct types
    df["Year"] = df["Year"].astype(int)
    df["LifeExpectancy"] = df["LifeExpectancy"].astype(float)
    df["Sex"] = df["Sex"].astype("category")

    n_original = len(df)
    if args.target_rows < n_original:
        raise ValueError("target_rows must be >= original dataset size")

    n_to_generate = args.target_rows - n_original
    print(f"Original rows: {n_original}")
    print(f"Generating synthetic rows: {n_to_generate}")

    # Train CTGAN
    discrete_columns = ["Year", "Sex"]
    model = CTGAN(
        epochs=args.epochs,
        batch_size=64,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=True,
        cuda=False
    )

    print("Training CTGAN...")
    model.fit(df, discrete_columns=discrete_columns)

    print("Sampling synthetic data...")
    synthetic = model.sample(n_to_generate)

    # Clean types
    synthetic["Year"] = synthetic["Year"].round().astype(int)
    synthetic["LifeExpectancy"] = synthetic["LifeExpectancy"].astype(float)
    synthetic["Sex"] = synthetic["Sex"].astype("category")

    # Combine
    df_aug = pd.concat([df, synthetic], ignore_index=True)
    print(f"Final augmented dataset size: {len(df_aug)}")

    # Write to SQLite
    print(f"Updating SQLite table: {args.table}")
    conn = sqlite3.connect(args.db)

    # Backup old table
    backup_table = args.table + "_backup"
    conn.execute(f"DROP TABLE IF EXISTS {backup_table}")
    conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {args.table}")

    # Replace table with augmented data
    df_aug.to_sql(args.table, conn, if_exists="replace", index=False)

    conn.close()
    print("SQLite update complete.")
    print(f"Backup saved as: {backup_table}")


if __name__ == "__main__":
    main()

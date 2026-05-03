from curses import window
import os
import glob
import pandas as pd
import numpy as np


def select_top_etfs(directory: str, top_n: int = 20, min_rows: int = 500) -> pd.DataFrame:
    records = []

    for file in glob.glob(os.path.join(directory, "*.csv")):
        df = pd.read_csv(file, usecols=["Date", "Close", "Volume"])

        if len(df) < min_rows:
            continue

        returns = df["Close"].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            print(f"  [skip] {os.path.basename(file)} — no variance in Close prices")
            continue

        sharpe = returns.mean() / returns.std() * (252 ** 0.5)

        if pd.isna(sharpe) or not np.isfinite(sharpe):   
            print(f"  [skip] {os.path.basename(file)} — invalid Sharpe value")
            continue

        records.append({
            "ticker": os.path.basename(file).replace(".csv", ""),
            "file":   file,
            "rows":   len(df),
            "sharpe": sharpe,
        })

    if not records:
        raise ValueError("No valid ETFs found — check your directory path and CSV contents")

    top = (
        pd.DataFrame(records)
        .sort_values("sharpe", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print("Top ETFs by Sharpe:\n")
    print(top[["ticker", "rows", "sharpe"]].to_string(index=False))
    return top


def build_dataset(directory: str = "etfs", top_n: int = 20, min_rows: int = 500) -> pd.DataFrame:
    top_etfs = select_top_etfs(directory, top_n=top_n, min_rows=min_rows)

    all_dfs = []

    for _, row in top_etfs.iterrows():
        df = pd.read_csv(row["file"])
        df["Date"]   = pd.to_datetime(df["Date"])
        df["Ticker"] = row["ticker"]
        all_dfs.append(df)

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    print(f"\nDataset shape : {combined.shape}")
    print(f"Columns       : {combined.columns.tolist()}")
    print(f"Tickers       : {combined['Ticker'].nunique()}")
    print(f"Date range    : {combined['Date'].min().date()} → {combined['Date'].max().date()}")

    return combined

def add_zscores(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    window = int(window)  # ← force int, prevents float issue
    
    def compute(group):
        group = group.copy()  # ← prevents SettingWithCopyWarning
        group["daily_range"] = (group["High"] - group["Low"]) / group["Close"] * 100
        
        group["volume_zscore"] = (
            (group["Volume"] - group["Volume"].rolling(window).mean()) /
            group["Volume"].rolling(window).std()
        )
        group["range_zscore"] = (
            (group["daily_range"] - group["daily_range"].rolling(window).mean()) /
            group["daily_range"].rolling(window).std()
        )
        return group

    df = df.groupby("Ticker", group_keys=False).apply(compute)
    df.dropna(subset=["volume_zscore", "range_zscore"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["Return_Pct"] = df["Close"].pct_change() * 100        # today's % change
    df["Target_Reg"] = df["Close"].pct_change().shift(-1) * 100  # next day % change (regression)
    stability_threshold = 2
    def label(r):
        if r > stability_threshold:
            return "crestere"
        elif r < -stability_threshold:
            return "scadere"
        else:
            return "stabil"

    df["Target_Cls"] = df["Target_Reg"].apply(label)
    df.dropna(subset=["Return_Pct", "Target_Reg"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
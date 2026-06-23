"""Quick sanity check on the local 2025 rate parquets."""
import pandas as pd
import config

b = pd.read_parquet(config.SNAPSHOTS / "pa_rates_batter_2025.parquet")
p = pd.read_parquet(config.SNAPSHOTS / "pa_rates_pitcher_2025.parquet")
print(f"batters: {len(b)}  pitchers: {len(p)}")
print(f"batter index dtype: {b.index.dtype}")
print(f"columns: {list(b.columns)}\n")

print("-- stars (HR rate should be HIGH, ~0.10+) --")
for name, pid in [("Judge", 592450), ("Ohtani", 660271), ("Soto", 665742),
                  ("Schwarber", 656941)]:
    if pid in b.index:
        r = b.loc[pid]
        print(f"  {name:<10} HR={r['HR']:.3f}  K={r['K']:.3f}  PA={int(r['PA'])}")
    else:
        print(f"  {name:<10} not in table")

print("\n-- aces (K rate should be HIGH, ~0.30+) --")
for name, pid in [("Skubal", 669373), ("Sale", 519242), ("Wheeler", 554430)]:
    if pid in p.index:
        r = p.loc[pid]
        print(f"  {name:<10} K={r['K']:.3f}  HR={r['HR']:.3f}  BF={int(r['PA'])}")
    else:
        print(f"  {name:<10} not in table")

# overall sanity: do high-PA batters span a realistic HR range?
big = b[b["PA"] >= 300]
print(f"\nhigh-PA batters (300+ PA): {len(big)}")
print(f"  HR rate range: {big['HR'].min():.3f} to {big['HR'].max():.3f} (league ~0.033)")
print(f"  K  rate range: {big['K'].min():.3f} to {big['K'].max():.3f} (league ~0.224)")
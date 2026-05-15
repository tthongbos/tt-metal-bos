from pathlib import Path

import pandas as pd

batch = 1

inp = Path("/workspace/turtorial-newbie/model_dev/oft_model/outputs/test.csv")
summary_out = Path("/workspace/turtorial-newbie/model_dev/oft_model/outputs/test_summary.csv")
txt_out = Path("/workspace/turtorial-newbie/model_dev/oft_model/outputs/test_summary.txt")

df = pd.read_csv(inp)
df.columns = [c.strip() for c in df.columns]


def to_num(col):
    return pd.to_numeric(
        df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)


df["Total %"] = to_num("Total %")
df["Device Time"] = to_num("Device Time")
df["Op-to-Op Gap"] = to_num("Op-to-Op Gap")
df["Total Time"] = df["Device Time"] + df["Op-to-Op Gap"]

# tt-perf-report CSV time unit is usually microseconds.
device_time_us = df["Device Time"].sum()
gap_time_us = df["Op-to-Op Gap"].sum()
total_time_us = device_time_us + gap_time_us

latency_s = total_time_us * 1e-6
latency_ms = total_time_us * 1e-3
fps = batch / latency_s if latency_s > 0 else 0

device_latency_s = device_time_us * 1e-6
device_latency_ms = device_time_us * 1e-3
device_only_fps = batch / device_latency_s if device_latency_s > 0 else 0

summary = (
    df.groupby("OP Code", dropna=False)
    .agg(
        Count=("OP Code", "size"),
        Total_Percent=("Total %", "sum"),
        Device_Time_us=("Device Time", "sum"),
        Op_to_Op_Gap_us=("Op-to-Op Gap", "sum"),
        Total_Time_us=("Total Time", "sum"),
    )
    .sort_values("Total_Percent", ascending=False)
    .reset_index()
)

summary.to_csv(summary_out, index=False)

lines = []
lines.append("Performance Summary")
lines.append("===================")
lines.append(f"Batch size             : {batch}")
lines.append(f"Device time            : {device_time_us:.3f} us = {device_latency_ms:.3f} ms")
lines.append(f"Op-to-op gap           : {gap_time_us:.3f} us = {gap_time_us * 1e-3:.3f} ms")
lines.append(f"Total latency          : {total_time_us:.3f} us = {latency_ms:.3f} ms")
lines.append(f"End-to-end FPS         : {fps:.4f}")
lines.append(f"Device-only FPS        : {device_only_fps:.4f}")
lines.append("")
lines.append("Top ops by Total %")
lines.append("==================")
lines.append(summary.head(20).to_string(index=False))

txt_out.write_text("\n".join(lines))

print("\n".join(lines))
print()
print("Saved summary CSV:", summary_out)
print("Saved summary TXT:", txt_out)

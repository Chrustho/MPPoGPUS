#!/usr/bin/env bash
set -euo pipefail

# run_and_plot.sh
# Compile basic and tiled matmul, run them, aggregate results and plot with gnuplot.
# Produces: basic_timings.csv, tiled_timings.csv (from binaries)
#           combined_roofline.dat, combined_times.dat
#           roofline.png, times_bar.png

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

NVCC=${NVCC:-nvcc}
CFLAGS="-O3 -arch=sm_52"

echo "Building binaries..."
$NVCC $CFLAGS -o basic_matmul basic_matmul.cu
$NVCC $CFLAGS -o tiled_matmul tiled_matmul.cu

echo "Running basic_matmul (produces basic_timings.csv)..."
./basic_matmul
echo "Running tiled_matmul (produces tiled_timings.csv)..."
./tiled_matmul

echo "Profiling with nvprof..."
nvprof --metrics flop_count_sp,dram_read_bytes,dram_write_bytes --log-file nvprof_basic.txt ./basic_matmul || true
nvprof --metrics flop_count_sp,dram_read_bytes,dram_write_bytes --log-file nvprof_tiled.txt ./tiled_matmul || true

COMBINED_ROOFLINE="combined_roofline.dat"
COMBINED_TIMES="combined_times.dat"

ROWS_A=$((1<<11))
COLS_A=$((1<<10))
ROWS_B=$((1<<10))
COLS_B=$((1<<9))

TOTAL_FLOPS=$(awk -v r=$ROWS_A -v c=$COLS_A -v cb=$COLS_B 'BEGIN{print r*cb*(2*c-1)}')
MIN_BYTES=$(awk -v ra=$ROWS_A -v ca=$COLS_A -v cb=$COLS_B 'BEGIN{print (ra*ca + ca*cb + ra*cb)*4}')
INTENSITY=$(awk -v f=$TOTAL_FLOPS -v b=$MIN_BYTES 'BEGIN{print f/b}')

echo "Total flops: $TOTAL_FLOPS"
echo "Minimal bytes: $MIN_BYTES" 
printf "operational intensity = %0.6g flops/byte\n" "$INTENSITY"

# Prepare header
cat > "$COMBINED_ROOFLINE" <<EOF
#kernel block intensity gflops
EOF

# parse basic_timings.csv and tiled_timings.csv
for csv in basic_timings.csv tiled_timings.csv; do
  if [ ! -f "$csv" ]; then
    echo "Missing $csv; expected binaries to produce it" >&2
    exit 1
  fi
  tail -n +2 "$csv" | while IFS=, read -r kernel block ms gflops correct; do
    # some CSVs contain kernel names like Basic or Tiled; normalize
    name=$(echo "$kernel" | tr -d '"')
    echo "$name $block $INTENSITY $gflops" >> "$COMBINED_ROOFLINE"
  done
done

# Combined times for bar chart: kernel block ms
cat > "$COMBINED_TIMES" <<EOF
#kernel block ms
EOF
for csv in basic_timings.csv tiled_timings.csv; do
  tail -n +2 "$csv" | while IFS=, read -r kernel block ms gflops correct; do
    name=$(echo "$kernel" | tr -d '"')
    echo "$name $block $ms" >> "$COMBINED_TIMES"
  done
done
## prepare matrix file for bar plot: /tmp/plot_times_matrix.dat
# Use Python to avoid awk portability issues
python3 - <<'PY' > /tmp/plot_times_matrix.dat || true
import sys
from collections import defaultdict, OrderedDict
data = defaultdict(dict)
kernels = OrderedDict()
blocks = set()
with open('combined_times.dat') as f:
    for i,line in enumerate(f):
        if i==0:
            continue
        parts = line.strip().split()
        if not parts: continue
        name, block, ms = parts[0], parts[1], parts[2]
        data[block][name] = ms
        kernels.setdefault(name, True)
        blocks.add(block)

blocks = sorted(blocks, key=lambda x: int(x))
hdr = ['Block'] + list(kernels.keys())
print(' '.join(hdr))
for b in blocks:
    row = [b]
    for k in kernels:
        row.append(data.get(b, {}).get(k, '0'))
    print(' '.join(row))
PY

# Generate plots with gnuplot
if command -v gnuplot >/dev/null 2>&1; then
  echo "Plotting roofline -> roofline.png"
  gnuplot -persist plot_roofline.gnuplot
  echo "Plotting bar chart -> times_bar.png"
  gnuplot -persist plot_bars.gnuplot
  echo "Done. Files: $COMBINED_ROOFLINE, $COMBINED_TIMES, roofline.png, times_bar.png"
else
  echo "gnuplot not found; please install gnuplot to generate plots. Data files: $COMBINED_ROOFLINE, $COMBINED_TIMES"
fi

exit 0

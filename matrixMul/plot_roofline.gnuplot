# plot_roofline.gnuplot
set terminal pngcairo size 1000,600 enhanced font 'Arial,12'
set output 'roofline.png'
set title 'Roofline (measured points)'
set xlabel 'Operational Intensity (FLOP/byte)'
set ylabel 'Performance (GFLOP/s)'
set logscale x
set logscale y
set grid xtics ytics mxtics mytics

# fixed axis ranges to match example style
set xrange [0.01:300]
set yrange [1:10000]

# Hard-coded peak/roof values (match example appearance)
# Units: BW in GB/s; GFLOPS = BW * intensity
BW_DRAM = 224.4        # Peak DRAM BW (GB/s)
BW_SHMEM = 2119.7      # Peak shared-memory BW (GB/s)
BW_L1    = 28008.71    # Peak L1/texture BW (GB/s)
PEAK_FP32 = 4981.0     # Peak FP32 (GFLOPS)

set key off

# Draw roof lines and compute peak
set style line 1 lc rgb '#7f0000' lw 2
set style line 2 lc rgb '#7f0000' lw 2
set style line 3 lc rgb '#7f0000' lw 2
set style line 4 lc rgb '#000000' lw 2

plot BW_L1*x with lines ls 1 title 'Peak L1/tex BW', \
     BW_SHMEM*x with lines ls 2 title 'Peak shmem BW', \
     BW_DRAM*x with lines ls 3 title 'Peak DRAM BW', \
     PEAK_FP32 with lines ls 4 title sprintf('Peak FP32 Performance: %.0f GFlops', PEAK_FP32), \
     'combined_roofline.dat' using 3:4 title 'measured' with points pt 6 lc rgb 'red' ps 1.8, \
     'combined_roofline.dat' using 3:4:(sprintf("%s_%s", stringcolumn(1), stringcolumn(2))) with labels offset 1,1 font ',9' notitle

# place rotated labels near lines (manual coordinates chosen to match example)
set label 1 'Peak L1/tex BW: 28008.71 GB/s' at 0.03,200 rotate by 30 font ',10'
set label 2 'Peak shmem BW: 2119.7 GB/s' at 0.06,60 rotate by 30 font ',10'
set label 3 'Peak DRAM BW: 224.4 GB/s' at 0.2,8 rotate by 30 font ',10'
set label 4 sprintf('Peak FP32 Performance: %.0f GFlops', PEAK_FP32) at 6,5000 font ',12'

pause -1

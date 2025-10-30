

set terminal pngcairo size 1200,800 enhanced font "Arial,11"
set output "roofline.png"

set title "Roofline Model - Matrix Multiplication on GTX 980" font "Arial,16 bold"
set xlabel "Operational Intensity (FLOP/byte)" font "Arial,13"
set ylabel "Performance (GFLOP/s)" font "Arial,13"


peak_flops = 4980.736   # GFLOPS (single-precision theoretical peak)
peak_bw = 224.3         # GB/s (memory bandwidth theoretical)


set logscale xy
set grid xtics ytics mxtics mytics lw 0.5 lc rgb "#cccccc"


set xrange [0.05:200]
set yrange [5:peak_flops*1.5]

max_gflops = 12.984032

set key top left
set key box
set key spacing 1.2
set key font "Arial,10"
set key opaque
set key title sprintf("Best: %.1f GFLOPS", max_gflops) font "Arial,10 bold"


ridge_point = peak_flops / peak_bw


mem_bound(x) = peak_bw * x
comp_bound(x) = peak_flops


max_gflops = 12.984032

set style line 1 lc rgb '#E63946' lw 3 dt 1        # Rosso - Memory bound
set style line 2 lc rgb '#457B9D' lw 3 dt 2        # Blu - Compute bound

set style line 10 lc rgb '#FF6B35' pt 7 ps 2.5 lw 2
set style line 11 lc rgb '#F7931E' pt 7 ps 2.5 lw 2
set style line 12 lc rgb '#FFA500' pt 7 ps 2.5 lw 2

set style line 20 lc rgb '#004E89' pt 9 ps 2.5 lw 2
set style line 21 lc rgb '#1A659E' pt 9 ps 2.5 lw 2
set style line 22 lc rgb '#4ECDC4' pt 9 ps 2.5 lw 2

set style line 99 lc rgb '#FFD700' pt 18 ps 5.0 lw 3

set arrow from ridge_point,10 to ridge_point,peak_flops*1.5 nohead lw 1.5 dt 3 lc rgb "#888888"
set label sprintf("Ridge Point\n%.2f FLOP/byte", ridge_point) at ridge_point*1.5,30 center font "Arial,9" textcolor rgb "#555555"

print "=== Dati rilevati ==="
print "Operational Intensity: ~146.29 FLOP/byte"
print "Performance range: 8.8 - 13.0 GFLOPS"


plot \
    mem_bound(x) with lines ls 1 title sprintf("Memory Bound (%.0f GB/s)", peak_bw), \
    comp_bound(x) with lines ls 2 title sprintf("Compute Bound (%.0f GFLOPS)", peak_flops), \
    "roofline_points.dat" every ::0::0 using ($2*0.95):3 with points ls 10 title "Basic 8×8", \
    "roofline_points.dat" every ::1::1 using ($2*0.97):3 with points ls 11 title "Basic 16×16", \
    "roofline_points.dat" every ::2::2 using ($2*0.99):3 with points ls 12 title "Basic 32×32", \
    "roofline_points.dat" every ::3::3 using ($2*1.01):3 with points ls 20 title "Tiled 8×8", \
    "roofline_points.dat" every ::4::4 using ($2*1.03):3 with points ls 21 title "Tiled 16×16", \
    "roofline_points.dat" every ::5::5 using ($2*1.05):3 with points ls 22 title "Tiled 32×32"

print ""
print "=== Roofline plot generated: roofline.png ==="
print sprintf("Ridge point: %.2f FLOP/byte", ridge_point)
print sprintf("Best performance: %.2f GFLOPS", max_gflops)
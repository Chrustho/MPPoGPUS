set terminal pngcairo size 1000,700 enhanced font "DejaVu Sans,12"
set output "roofline.png"
set title "Roofline Model - GTX 980" font "DejaVu Sans,14"
set xlabel "Operational Intensity (FLOP/byte)"
set ylabel "Performance (GFLOP/s)"

# device peaks
peak_flops = 4980.736   # GFLOPS (single-precision theoretical peak)
peak_bw = 224.3         # GB/s (theoretical)

set logscale xy
set grid

# Legenda fuori dal grafico
set key outside right top vertical
set key maxrows 12
set key spacing 1.2
set key box
set key font ",9"

# Curve roofs - versione corretta
mem(x) = (x <= peak_flops/peak_bw) ? peak_bw * x : peak_flops
comp(x) = peak_flops

set xrange [0.05:1000]
set yrange [10:peak_flops*1.2]

# Verifica se il file esiste e ha dati
file_exists = system("[ -s roofline_points.dat ] && echo 1 || echo 0") + 0

if (file_exists) {
    # Trova il miglior punto
    stats "roofline_points.dat" using 3 nooutput name "G"
    max_gflops = G_max
    has_best = (G_records > 0)
} else {
    has_best = 0
    max_gflops = 0
}

# Plot base - sempre presente
plot \
     mem(x) with lines lt 1 lw 3 lc rgb "blue" title sprintf("Memory bound (%.1f GB/s)", peak_bw), \
     comp(x) with lines lt 1 lw 3 lc rgb "red" title sprintf("Compute bound (%.1f GFLOPS)", peak_flops) \
     $(file_exists ? ', "roofline_points.dat" using 2:3 with points pt 7 ps 1.5 lc rgb "green" title "Measured points"' : '') \
     $(has_best ? ', "roofline_points.dat" using ($3 == max_gflops ? $2 : 1/0):3 with points pt 9 ps 3 lc rgb "black" lw 2 title sprintf("Best: %.1f GFLOPS", max_gflops)' : '')
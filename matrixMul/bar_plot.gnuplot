# bar_plot.gnuplot
# reads timings CSV: header kernel,ms,gflops,intensity

set terminal pngcairo size 1000,600 enhanced font "DejaVu Sans,12"
set output "timings_bar.png"
set title "Kernel elapsed times (ms)"

set datafile separator ","

# stile istogramma (nota: plurale 'histograms' per set style data)
set style data histograms
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.8
set grid ytics

# migliorie estetiche
set key off
set autoscale xfixmin
set xtics rotate by -45
set margin 8,2,4,6   # left,right,top,bottom (aggiusta se le etichette sono tagliate)

# skip header (every ::1 salta la prima riga)
# usando colonna 2 per i ms e xtic(1) per le etichette
plot "timings_from_nvprof_combined.csv" every ::1 using 2:xtic(1) title "elapsed (ms)"

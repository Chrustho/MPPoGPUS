# bar_plot.gnuplot

set terminal pngcairo size 1000,600 enhanced font "DejaVu Sans,12"
set output "timings_bar.png"
set title "Kernel elapsed times (ms)"

set datafile separator ","


set style data histograms
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.8
set grid ytics


set key off
set autoscale xfixmin
set xtics rotate by -45
set margin 8,2,4,6   # left,right,top,bottom (aggiusta se le etichette sono tagliate)


plot "timings_from_nvprof_combined.csv" every ::1 using 2:xtic(1) title "elapsed (ms)"

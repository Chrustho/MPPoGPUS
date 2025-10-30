#!/usr/bin/env bash
set -euo pipefail

# run_and_clean.sh (versione robusta)
# Uso:
# ./run_and_clean.sh        -> esegue tutto e chiede conferma prima di cancellare
# ./run_and_clean.sh --yes  -> esegue tutto e cancella senza chiedere

FORCE=0
if [[ "${1:-}" == "--yes" ]]; then
  FORCE=1
fi

echo "=== Compilazione"
make

echo
echo "=== Esecuzione binari (producono basic_timings.csv, tiled_timings.csv e *_roofline.dat se implementato nei binari)"
./basic_matmul
./tiled_matmul

echo
echo "=== Rilevo presenza nvprof e parser Python"
NVPROF_BIN="$(command -v nvprof || true)"
PARSER=""
if [[ -f parse_nvprof_csv_it.py ]]; then
  PARSER="parse_nvprof_csv_it.py"
elif [[ -f parse_nvprof_csv.py ]]; then
  PARSER="parse_nvprof_csv.py"
fi

if [[ -n "$NVPROF_BIN" ]]; then
  echo "nvprof trovato in: $NVPROF_BIN"
else
  echo "nvprof NON trovato sulla PATH: salterò la fase di profiling (nvprof)."
fi

if [[ -n "$PARSER" ]]; then
  echo "Parser Python trovato: $PARSER"
else
  echo "Parser Python non trovato (cercati: parse_nvprof_csv_it.py, parse_nvprof_csv.py). Se vuoi usare nvprof outputs, assicurati di avere uno di questi due script."
fi

echo
echo "=== Raccolta metriche nvprof (se nvprof presente)"
NVPROF_BASIC_CSV="nvprof_basic.csv"
NVPROF_TILED_CSV="nvprof_tiled.csv"
if [[ -n "$NVPROF_BIN" ]]; then
  set +e
  echo "Eseguo nvprof per basic_matmul..."
  "$NVPROF_BIN" --csv --metrics flop_count_sp,dram_read_bytes,dram_write_bytes --log-file "$NVPROF_BASIC_CSV" ./basic_matmul 2>&1 | sed -n '1,200p'
  echo "Eseguo nvprof per tiled_matmul..."
  "$NVPROF_BIN" --csv --metrics flop_count_sp,dram_read_bytes,dram_write_bytes --log-file "$NVPROF_TILED_CSV" ./tiled_matmul 2>&1 | sed -n '1,200p'
  nvprof_exit_code=$?
  set -e
  if [[ $nvprof_exit_code -ne 0 ]]; then
    echo "Attenzione: nvprof è terminato con codice $nvprof_exit_code. I file $NVPROF_BASIC_CSV / $NVPROF_TILED_CSV potrebbero essere incompleti."
  fi
else
  echo "Salto nvprof (non disponibile)."
fi

echo
echo "=== Parsing CSV con Python (se presente il parser e nvprof ha prodotto CSV)"
BASIC_POINTS="basic_points.dat"
TILED_POINTS="tiled_points.dat"
BASIC_TIMINGS_NV="basic_timings_from_nvprof.csv"
TILED_TIMINGS_NV="tiled_timings_from_nvprof.csv"


: > /dev/null

if [[ -n "$PARSER" && -n "$NVPROF_BIN" ]]; then
  if [[ -f "$NVPROF_BASIC_CSV" && -s "$NVPROF_BASIC_CSV" ]]; then
    echo "Parsing $NVPROF_BASIC_CSV -> $BASIC_POINTS , $BASIC_TIMINGS_NV"
    python3 "$PARSER" "$NVPROF_BASIC_CSV" "$BASIC_POINTS" "$BASIC_TIMINGS_NV" || echo "Parser fallito per $NVPROF_BASIC_CSV (continua con fallback)"
  else
    echo "File $NVPROF_BASIC_CSV mancante o vuoto: salto parsing per basic."
  fi

  if [[ -f "$NVPROF_TILED_CSV" && -s "$NVPROF_TILED_CSV" ]]; then
    echo "Parsing $NVPROF_TILED_CSV -> $TILED_POINTS , $TILED_TIMINGS_NV"
    python3 "$PARSER" "$NVPROF_TILED_CSV" "$TILED_POINTS" "$TILED_TIMINGS_NV" || echo "Parser fallito per $NVPROF_TILED_CSV (continua con fallback)"
  else
    echo "File $NVPROF_TILED_CSV mancante o vuoto: salto parsing per tiled."
  fi
else
  echo "Parser o nvprof non disponibili/attivi: salto fase di parsing."
fi

echo
echo "=== Costruzione roofline_points.dat (scegli la migliore fonte disponibile)"

ROOFLINE_POINTS="roofline_points.dat"
: > "$ROOFLINE_POINTS"

if [[ -f "$BASIC_POINTS" && -s "$BASIC_POINTS" ]] || [[ -f "$TILED_POINTS" && -s "$TILED_POINTS" ]]; then
  echo "Uso i punti prodotti dal parser (se presenti)."
  [[ -f "$BASIC_POINTS" && -s "$BASIC_POINTS" ]] && cat "$BASIC_POINTS" >> "$ROOFLINE_POINTS"
  [[ -f "$TILED_POINTS" && -s "$TILED_POINTS" ]] && cat "$TILED_POINTS" >> "$ROOFLINE_POINTS"
fi

if [[ ! -s "$ROOFLINE_POINTS" ]]; then
  if [[ -f basic_roofline.dat && -s basic_roofline.dat ]]; then
    echo "Fallback: uso basic_roofline.dat"
    awk 'NF>=3 && $1!~ /^#/ {print $1, $2, $3}' basic_roofline.dat >> "$ROOFLINE_POINTS"
  fi
  if [[ -f tiled_roofline.dat && -s tiled_roofline.dat ]]; then
    echo "Fallback: uso tiled_roofline.dat"
    awk 'NF>=3 && $1!~ /^#/ {print $1, $2, $3}' tiled_roofline.dat >> "$ROOFLINE_POINTS"
  fi
fi

if [[ ! -s "$ROOFLINE_POINTS" ]]; then
  echo "Ultimo fallback: costruisco punti da basic_timings.csv / tiled_timings.csv (intensity=0)"
  if [[ -f basic_timings.csv && -s basic_timings.csv ]]; then
    awk -F, 'NR>1 {label=$1"_"$2; gflops=$4+0; if(gflops=="") gflops=0; print label, 0, gflops}' basic_timings.csv >> "$ROOFLINE_POINTS"
  fi
  if [[ -f tiled_timings.csv && -s tiled_timings.csv ]]; then
    awk -F, 'NR>1 {label=$1"_"$2; gflops=$4+0; if(gflops=="") gflops=0; print label, 0, gflops}' tiled_timings.csv >> "$ROOFLINE_POINTS"
  fi
fi

if [[ -s "$ROOFLINE_POINTS" ]]; then
  echo "Roofline points written to $ROOFLINE_POINTS (lines: $(wc -l < "$ROOFLINE_POINTS"))"
else
  echo "ATTENZIONE: roofline_points.dat vuoto. Nessuna fonte di dati disponibile. I plot roofline non potranno mostrare punti."
fi

echo
echo "=== Costruzione timings_from_nvprof_combined.csv (preferisci dati parser, altrimenti fallback dai binari)"

TIMINGS_COMBINED="timings_from_nvprof_combined.csv"
: > "$TIMINGS_COMBINED"
echo "kernel,ms,gflops,intensity" > "$TIMINGS_COMBINED"

if [[ -f "$BASIC_TIMINGS_NV" && -s "$BASIC_TIMINGS_NV" ]]; then
  tail -n +2 "$BASIC_TIMINGS_NV" >> "$TIMINGS_COMBINED"
fi
if [[ -f "$TILED_TIMINGS_NV" && -s "$TILED_TIMINGS_NV" ]]; then
  tail -n +2 "$TILED_TIMINGS_NV" >> "$TIMINGS_COMBINED"
fi

lines_after_header=$(($(wc -l < "$TIMINGS_COMBINED") - 1))
if [[ $lines_after_header -le 0 ]]; then
  echo "Parser non ha prodotto timings: uso fallback dai CSV dei binari (basic_timings.csv, tiled_timings.csv)."
  if [[ -f basic_timings.csv && -s basic_timings.csv ]]; then
    awk -F, 'NR>1 {print $1 "_" $2 "," $3 "," $4 ",0"}' basic_timings.csv >> "$TIMINGS_COMBINED"
  fi
  if [[ -f tiled_timings.csv && -s tiled_timings.csv ]]; then
    awk -F, 'NR>1 {print $1 "_" $2 "," $3 "," $4 ",0"}' tiled_timings.csv >> "$TIMINGS_COMBINED"
  fi
fi

if [[ -s "$TIMINGS_COMBINED" ]]; then
  echo "Timings combined written to $TIMINGS_COMBINED (lines: $(wc -l < "$TIMINGS_COMBINED"))"
else
  echo "ATTENZIONE: $TIMINGS_COMBINED è vuoto. I plot a barre non potranno essere generati."
fi

echo
echo "=== Genera plot con gnuplot (solo se i file richiesti non sono vuoti)"

if [[ -s "$ROOFLINE_POINTS" ]]; then
  echo "Genero roofline_plot.gnuplot -> roofline.png"
  gnuplot roofline_plot.gnuplot || echo "gnuplot roofline fallito"
else
  echo "Salto roofline_plot.gnuplot perché $ROOFLINE_POINTS è vuoto."
fi

if [[ -s "$TIMINGS_COMBINED" && $(wc -l < "$TIMINGS_COMBINED") -gt 1 ]]; then
  echo "Genero bar_plot.gnuplot -> timings_bar.png"
  gnuplot bar_plot.gnuplot || echo "gnuplot bar plot fallito"
else
  echo "Salto bar_plot.gnuplot perché $TIMINGS_COMBINED è vuoto o non contiene dati."
fi

echo
echo "=== Pulizia: elenco file che verrebbero cancellati (dry-run)"

find . -type f ! \( \
    -iname '*.cu' -o \
    -iname '*.py' -o \
    -iname '*.gnuplot' -o \
    -iname '*.sh' -o \
    -iname '*.png' -o \
    -iname 'makefile' -o \
    -iname 'makefile.*' -o \
    -perm /111 \
  \) -print | sed 's|^\./||' || true

if [[ $FORCE -eq 1 ]]; then
  echo
  echo "FORZANDO la cancellazione dei file elencati..."
  find . -type f ! \( \
      -iname '*.cu' -o \
      -iname '*.py' -o \
      -iname '*.gnuplot' -o \
      -iname '*.sh' -o \
      -iname '*.png' -o \
      -iname 'makefile' -o \
      -iname 'makefile.*' -o \
      -perm /111 \
    \) -print -exec rm -v -- {} +
  echo "Cancellazione completata."
else
  echo
  read -r -p "Vuoi cancellare tutti i file elencati sopra (tutti i file tranne .cu, Makefile, .py, .gnuplot, .sh, .png e i binari)? [y/N] " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    echo "Cancellazione in corso..."
    find . -type f ! \( \
        -iname '*.cu' -o \
        -iname '*.py' -o \
        -iname '*.gnuplot' -o \
        -iname '*.sh' -o \
        -iname '*.png' -o \
        -iname 'makefile' -o \
        -iname 'makefile.*' -o \
        -perm /111 \
      \) -print -exec rm -v -- {} +
    echo "Cancellazione completata."
  else
    echo "Annullata cancellazione. I file non sono stati rimossi."
  fi
fi

echo "Fine."

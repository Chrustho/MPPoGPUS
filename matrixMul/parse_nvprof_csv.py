#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import csv
import re
from collections import defaultdict

def usage():
    print("Uso: python3 parse_nvprof_csv_it.py <nvprof_csv> <out_points.dat> <out_timings.csv>")

if len(sys.argv) < 4:
    usage()
    sys.exit(1)

infile = sys.argv[1]
out_points = sys.argv[2]
out_timing = sys.argv[3]

righe = []
with open(infile, newline='') as f:
    rdr = csv.reader(f)
    for r in rdr:
        if r:
            righe.append(r)

if not righe:
    print("File vuoto o non leggibile.")
    sys.exit(1)

header_idx = None
header = None
for i, r in enumerate(righe):
    joined = ",".join(r).lower()
    if 'name' in joined and ('duration' in joined or 'time' in joined or 'flop' in joined):
        header_idx = i
        header = r
        break

if header_idx is None:
    header_idx = 0
    header = righe[0]

col_index = {}
for i, col in enumerate(header):
    key = col.strip().lower()
    col_index[key] = i

def trova_col(col_candidates):
    for k, idx in col_index.items():
        for cand in col_candidates:
            if cand in k:
                return idx
    return None

idx_name = trova_col(['name']) or trova_col(['kernel', 'function'])
idx_dur  = trova_col(['duration']) or trova_col(['time'])
idx_flop = None
for k, i in col_index.items():
    if 'flop' in k and 'sp' in k:
        idx_flop = i
        break
if idx_flop is None:
    idx_flop = trova_col(['flop', 'flop_count'])

idx_dram_r = None
idx_dram_w = None
for k, i in col_index.items():
    if 'dram_read' in k or 'dram read' in k or 'dram_read_bytes' in k or 'dram_read_bytes' in k:
        idx_dram_r = i
    if 'dram_write' in k or 'dram write' in k or 'dram_write_bytes' in k:
        idx_dram_w = i
if idx_dram_r is None:
    idx_dram_r = trova_col(['dram', 'read'])
if idx_dram_w is None:
    idx_dram_w = trova_col(['dram', 'write'])

num_re = re.compile(r'[^\d\.\-eE]')

def estrai_numero(s):
    if s is None:
        return None
    s = s.strip()
    if s == '':
        return None
    s2 = num_re.sub('', s)
    if s2 == '':
        return None
    try:
        return float(s2)
    except:
        return None

dati = []
for r in righe[header_idx + 1:]:
    if len(r) <= 1:
        continue
    try:
        nome = r[idx_name].strip()
    except Exception:
        continue
    if not nome or nome.startswith('=='):
        continue

    dur_ms = None
    if idx_dur is not None and idx_dur < len(r):
        raw = r[idx_dur].strip()
        nv = estrai_numero(raw)
        if nv is not None:
            hdr = header[idx_dur].lower()
            if 'ns' in hdr:
                dur_ms = nv / 1e6
            elif 'us' in hdr:
                dur_ms = nv / 1e3
            elif 's' in hdr and 'ms' not in hdr:
                dur_ms = nv * 1000.0
            else:
                if 'ns' in raw:
                    dur_ms = nv / 1e6
                elif 'us' in raw:
                    dur_ms = nv / 1e3
                elif 'ms' in raw or ('s' not in hdr and 'ms' in hdr):
                    dur_ms = nv
                elif 's' in raw:
                    dur_ms = nv * 1000.0
                else:
                    dur_ms = nv

    flop = None
    if idx_flop is not None and idx_flop < len(r):
        flop = estrai_numero(r[idx_flop])

    dram_r = None
    dram_w = None
    if idx_dram_r is not None and idx_dram_r < len(r):
        dram_r = estrai_numero(r[idx_dram_r])
    if idx_dram_w is not None and idx_dram_w < len(r):
        dram_w = estrai_numero(r[idx_dram_w])

    if flop is None and dram_r is None and dram_w is None:
        continue

    dati.append({
        'nome': nome,
        'dur_ms': dur_ms,
        'flop': flop,
        'dram_r': dram_r,
        'dram_w': dram_w
    })

aggregato = defaultdict(lambda: {'dur_ms': 0.0, 'flop': 0.0, 'dram_r': 0.0, 'dram_w': 0.0, 'has_dur': False})
for e in dati:
    k = e['nome']
    if e['dur_ms'] is not None:
        aggregato[k]['dur_ms'] += e['dur_ms']
        aggregato[k]['has_dur'] = True
    if e['flop'] is not None:
        aggregato[k]['flop'] += e['flop']
    if e['dram_r'] is not None:
        aggregato[k]['dram_r'] += e['dram_r']
    if e['dram_w'] is not None:
        aggregato[k]['dram_w'] += e['dram_w']

with open(out_points, 'w') as fpt, open(out_timing, 'w', newline='') as ft:
    ft.write("kernel,ms,gflops,intensity\n")
    for nome, v in aggregato.items():
        flop = v['flop'] or 0.0
        dram_r = v['dram_r'] or 0.0
        dram_w = v['dram_w'] or 0.0
        dur_ms = v['dur_ms'] if v['has_dur'] else 0.0

        bytes_tot = dram_r + dram_w

        intensity = (flop / bytes_tot) if bytes_tot > 0 else 0.0

        dur_s = dur_ms / 1000.0 if dur_ms is not None else 0.0
        gflops = (flop / dur_s) / 1e9 if dur_s > 0 else 0.0

        label = re.sub(r'\s+', '_', nome)

        fpt.write(f"{label} {intensity:.9g} {gflops:.9g}\n")
        ft.write(f"{label},{dur_ms:.6f},{gflops:.6f},{intensity:.6f}\n")

print(f"Analizzati {len(aggregato)} kernel. Scritti: {out_points} e {out_timing}")

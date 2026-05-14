#!/usr/bin/env python3
"""
Load Extraction Tool - Nastran OP2 Analysis Application
Extracts and analyzes FEA loads from MSC Nastran OP2 files
"""

from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pandas as pd
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import time
import math
import logging
import threading
from datetime import datetime
from collections import defaultdict
from pyNastran.op2.data_in_material_coord import data_in_material_coord


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

METRICS = [
    {"id": "M01", "fn": lambda fx,fy,fz: fz},
    {"id": "M02", "fn": lambda fx,fy,fz: -fz},
    {"id": "M03", "fn": lambda fx,fy,fz: fy},
    {"id": "M04", "fn": lambda fx,fy,fz: -fy},
    {"id": "M05", "fn": lambda fx,fy,fz: fx},
    {"id": "M06", "fn": lambda fx,fy,fz: -fx},
    {"id": "M07", "fn": lambda fx,fy,fz: abs(fx)},
    {"id": "M08", "fn": lambda fx,fy,fz: math.sqrt(fy**2+fz**2)},
    {"id": "M09", "fn": lambda fx,fy,fz: math.sqrt(fz**2+fy**2)+abs(fx)},
    {"id": "M10", "fn": lambda fx,fy,fz: math.sqrt((2*fz)**2+fy**2)},
    {"id": "M11", "fn": lambda fx,fy,fz: math.sqrt(fz**2+(2*fy)**2)},
    {"id": "M12", "fn": lambda fx,fy,fz: math.sqrt((2*fz)**2+fy**2)+abs(fx)},
    {"id": "M13", "fn": lambda fx,fy,fz: math.sqrt(fz**2+(2*fy)**2)+abs(fx)},
    {"id": "M14", "fn": lambda fx,fy,fz: abs(fx)+math.sqrt(fy**2+fz**2)},
    {"id": "M15", "fn": lambda fx,fy,fz: fx+math.sqrt(fy**2+fz**2)},
    {"id": "M16", "fn": lambda fx,fy,fz: math.sqrt((2*fx)**2+fy**2+fz**2)},
    {"id": "M17", "fn": lambda fx,fy,fz: math.sqrt(fx**2+(2*fy)**2+(2*fz)**2)},
    {"id": "M18", "fn": lambda fx,fy,fz: math.sqrt(fx**2+fy**2+fz**2)},
]

PSHELL_METRICS = [
    {"id": "M01", "fn": lambda nx,ny,nxy: nx},
    {"id": "M02", "fn": lambda nx,ny,nxy: -nx},
    {"id": "M03", "fn": lambda nx,ny,nxy: ny},
    {"id": "M04", "fn": lambda nx,ny,nxy: -ny},
    {"id": "M05", "fn": lambda nx,ny,nxy: nxy},
    {"id": "M06", "fn": lambda nx,ny,nxy: -nxy},
    {"id": "M07", "fn": lambda nx,ny,nxy: math.sqrt(nx**2 + ny**2)},
    {"id": "M08", "fn": lambda nx,ny,nxy: math.sqrt(nx**2 + ny**2) + abs(nxy)},
    {"id": "M09", "fn": lambda nx,ny,nxy: math.sqrt(2*nx**2 + 2*ny**2)},
    {"id": "M10", "fn": lambda nx,ny,nxy: math.sqrt(nx**2 + 2*ny**2)},
    {"id": "M11", "fn": lambda nx,ny,nxy: math.sqrt(2*nx**2 + ny**2) + 2*abs(nxy)},
    {"id": "M12", "fn": lambda nx,ny,nxy: math.sqrt(nx**2 + 2*ny**2) + 2*abs(nxy)},
    {"id": "M13", "fn": lambda nx,ny,nxy: nx + ny + nxy},
    {"id": "M14", "fn": lambda nx,ny,nxy: nx + ny},
    {"id": "M15", "fn": lambda nx,ny,nxy: ny + nxy},
    {"id": "M16", "fn": lambda nx,ny,nxy: nx + nxy},
]

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER & TEXT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def _append():
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)
        self.text_widget.after(0, _append)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_critical_rows(raw_data):
    if not raw_data:
        return []
    seen = {}
    for i, row in enumerate(raw_data):
        key = (row["Element ID"], row["Load Case ID"])
        if key not in seen:
            seen[key] = i
    rows = [raw_data[i] for i in seen.values()]

    fx_arr = np.array([float(r["FX"]) for r in rows])
    fy_arr = np.array([float(r["FY"]) for r in rows])
    fz_arr = np.array([float(r["FZ"]) for r in rows])
    abs_fx = np.abs(fx_arr)
    fy2 = fy_arr**2;  fz2 = fz_arr**2;  fx2 = fx_arr**2
    fy_fz = np.sqrt(fy2 + fz2)
    M = np.column_stack([
        fz_arr, -fz_arr, fy_arr, -fy_arr, fx_arr, -fx_arr,
        abs_fx, fy_fz,
        fy_fz + abs_fx,
        np.sqrt(4*fz2 + fy2),
        np.sqrt(fz2 + 4*fy2),
        np.sqrt(4*fz2 + fy2) + abs_fx,
        np.sqrt(fz2 + 4*fy2) + abs_fx,
        abs_fx + fy_fz,
        fx_arr + fy_fz,
        np.sqrt(4*fx2 + fy2 + fz2),
        np.sqrt(fx2 + 4*fy2 + 4*fz2),
        np.sqrt(fx2 + fy2 + fz2),
    ])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["Element ID"]].append(i)
    selected = set()
    for g_rows in groups.values():
        if len(g_rows) == 1:
            selected.add(g_rows[0])
        else:
            sub = M[g_rows]
            for lb in np.argmax(sub, axis=0):
                selected.add(g_rows[lb])
    result = [{**rows[i], '_fx': float(fx_arr[i]), '_fy': float(fy_arr[i]), '_fz': float(fz_arr[i])}
              for i in sorted(selected)]
    result.sort(key=lambda r: (r["Element ID"], r["Load Case ID"]))
    return result

def extract_critical_pshell(raw_data, group_key, nx_key, ny_key, nxy_key):
    if not raw_data:
        return []
    seen = {}
    for i, row in enumerate(raw_data):
        key = (row[group_key], row['Load Case ID'])
        if key not in seen:
            seen[key] = i
    rows = [raw_data[i] for i in seen.values()]

    nx_arr  = np.array([float(r[nx_key])  for r in rows])
    ny_arr  = np.array([float(r[ny_key])  for r in rows])
    nxy_arr = np.array([float(r[nxy_key]) for r in rows])
    abs_nxy = np.abs(nxy_arr)
    nx2 = nx_arr**2;  ny2 = ny_arr**2
    nx_ny = np.sqrt(nx2 + ny2)
    M = np.column_stack([
        nx_arr, -nx_arr,
        ny_arr, -ny_arr,
        nxy_arr, -nxy_arr,
        nx_ny,
        nx_ny + abs_nxy,
        np.sqrt(2*nx2 + 2*ny2),
        np.sqrt(nx2 + 2*ny2),
        np.sqrt(2*nx2 + ny2) + 2*abs_nxy,
        np.sqrt(nx2 + 2*ny2) + 2*abs_nxy,
        nx_arr + ny_arr + nxy_arr,
        nx_arr + ny_arr,
        ny_arr + nxy_arr,
        nx_arr + nxy_arr,
    ])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[group_key]].append(i)
    selected = set()
    for g_rows in groups.values():
        if len(g_rows) == 1:
            selected.add(g_rows[0])
        else:
            sub = M[g_rows]
            for lb in np.argmax(sub, axis=0):
                selected.add(g_rows[lb])
    result = [{**rows[i], '_nx': float(nx_arr[i]), '_ny': float(ny_arr[i]), '_nxy': float(nxy_arr[i])}
              for i in sorted(selected)]
    result.sort(key=lambda r: (r[group_key], r['Load Case ID']))
    return result

def extract_critical_displacement(disp_data):
    """Per (Property ID, Node ID) select load case with max Magnitude."""
    if not disp_data:
        return []
    seen = {}
    for i, row in enumerate(disp_data):
        key = ((row['Property ID'], row['Node ID']), row['Load Case ID'])
        if key not in seen:
            seen[key] = i
    rows = [disp_data[i] for i in seen.values()]
    mag_arr = np.array([float(r['Magnitude']) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['Property ID'], r['Node ID'])].append(i)
    selected = set()
    for g_rows in groups.values():
        selected.add(g_rows[int(np.argmax(mag_arr[g_rows]))])
    result = [rows[i] for i in sorted(selected)]
    result.sort(key=lambda r: (r['Property ID'], r['Node ID'], r['Load Case ID']))
    return result


def extract_critical_stress(stress_data, group_key):
    if not stress_data:
        return []
    vm_z1_key = 'VM_Z1' if group_key == 'Element ID' else 'Avg_VM_Z1'
    vm_z2_key = 'VM_Z2' if group_key == 'Element ID' else 'Avg_VM_Z2'
    seen = {}
    for i, row in enumerate(stress_data):
        key = (row[group_key], row['Load Case ID'])
        if key not in seen:
            seen[key] = i
    rows = [stress_data[i] for i in seen.values()]
    vm_max = np.maximum(
        np.abs(np.array([float(r.get(vm_z1_key, 0)) for r in rows])),
        np.abs(np.array([float(r.get(vm_z2_key, 0)) for r in rows]))
    )
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[group_key]].append(i)
    selected = set()
    for g_rows in groups.values():
        selected.add(g_rows[int(np.argmax(vm_max[g_rows]))])
    result = [rows[i] for i in sorted(selected)]
    result.sort(key=lambda r: (r[group_key], r['Load Case ID']))
    return result

def parse_id_input(input_str, all_ids=None):
    input_str = input_str.strip().upper()
    if input_str == "ALL":
        return all_ids if all_ids else []
    try:
        ids = [int(x.strip()) for x in input_str.split(',') if x.strip()]
        return ids
    except ValueError:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LoadExtractionApp:

    COLORS = {
        'bg':       '#0d1117',
        'surface':  '#161b22',
        'surface2': '#21262d',
        'border':   '#30363d',
        'accent':   '#58a6ff',
        'text':     '#e6edf3',
        'muted':    '#8b949e',
        'success':  '#238636',
        'error':    '#da3633',
        'log_bg':   '#0a0e14',
        'log_fg':   '#7ee787',
    }

    def __init__(self, root):
        self.root = root
        self.root.title('Load Extraction Tool')
        self.root.geometry('1100x700')
        self.root.minsize(920, 580)
        self.root.configure(bg=self.COLORS['bg'])

        self.logger = logging.getLogger('LoadExtraction')
        self.logger.setLevel(logging.INFO)

        self.input_entry_now    = ''
        self.output_entry_now   = ''
        self.stress_output_now2 = ''

        self.pshell_property_ids = ''
        self.bush_element_ids    = ''

        self.extraction_type   = tk.StringVar(value='PSHELL ALL AVERAGE')
        self.coordinate_system = tk.StringVar(value='Element CID')
        self.stress_coord_system = tk.StringVar(value='Element CID')

        self._bdf_cache      = None
        self._bdf_path_cache = None
        self._op2_cache      = None
        self._op2_path_cache = None

        self.build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

    def _load_bdf(self):
        path = self.input_entry_now
        if self._bdf_path_cache != path:
            self.logger.info("📂 BDF dosyası okunuyor...")
            bdf = BDF()
            bdf.read_bdf(path, encoding='latin1')
            self._bdf_cache      = bdf
            self._bdf_path_cache = path
            self.logger.info("✓ BDF dosyası okundu")
        else:
            self.logger.info("✓ BDF dosyası önbellekten alındı")
        return self._bdf_cache

    def _load_op2(self):
        path = self.output_entry_now
        if self._op2_path_cache != path:
            self.logger.info("📂 OP2 dosyası okunuyor...")
            op2 = OP2()
            op2.read_op2(path)
            self._op2_cache      = op2
            self._op2_path_cache = path
            self.logger.info("✓ OP2 dosyası okundu")
        else:
            self.logger.info("✓ OP2 dosyası önbellekten alındı")
        return self._op2_cache

    def build_ui(self):
        self._build_header()
        self._build_tab_bar()
        self._build_action_bar()   # pack side=BOTTOM first → always visible
        self._build_log_panel()    # pack side=BOTTOM second
        self._build_main_content()
        self.setup_logger()
        self._select_tab('PSHELL ALL AVERAGE')

    def _build_header(self):
        hdr = tk.Frame(self.root, bg='#1c2128', height=62)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='⚙  LOAD EXTRACTION TOOL',
                 font=('Segoe UI', 17, 'bold'),
                 bg='#1c2128', fg=self.COLORS['accent']).pack(side='left', padx=22, pady=15)
        tk.Label(hdr, text='MSC Nastran OP2 Analysis',
                 font=('Segoe UI', 9),
                 bg='#1c2128', fg=self.COLORS['muted']).pack(side='right', padx=22)

    def _build_tab_bar(self):
        bar = tk.Frame(self.root, bg=self.COLORS['surface'])
        bar.pack(fill='x')
        tk.Frame(bar, bg=self.COLORS['border'], height=1).pack(fill='x', side='bottom')
        self.tab_btns = {}
        for mode in ['PSHELL ALL AVERAGE', 'BUSH LOAD', 'DISPLACEMENT', 'STRESS']:
            b = tk.Button(bar, text=f'  {mode}  ',
                         command=lambda m=mode: self._select_tab(m),
                         font=('Segoe UI', 10, 'bold'),
                         relief='flat', bd=0, cursor='hand2', pady=11)
            b.pack(side='left', padx=(8, 0))
            self.tab_btns[mode] = b

    def _build_action_bar(self):
        bar = tk.Frame(self.root, bg=self.COLORS['bg'])
        bar.pack(fill='x', side='bottom')
        tk.Frame(bar, bg=self.COLORS['border'], height=1).pack(fill='x')
        inner = tk.Frame(bar, bg=self.COLORS['bg'])
        inner.pack(fill='x', padx=16, pady=10)
        self.run_btn = tk.Button(inner, text='▶  RUN ANALYSIS', command=self.asc_run,
                  bg=self.COLORS['success'], fg='white',
                  font=('Segoe UI', 11, 'bold'), relief='flat',
                  cursor='hand2', activebackground='#2ea043',
                  pady=9)
        self.run_btn.pack(side='left', fill='x', expand=True, padx=(0, 10))
        tk.Button(inner, text='⟳  CLEAR', command=self.clear_log,
                  bg=self.COLORS['surface2'], fg=self.COLORS['text'],
                  font=('Segoe UI', 11), relief='flat', cursor='hand2',
                  pady=9, padx=30).pack(side='left')

    def _build_log_panel(self):
        frame = tk.Frame(self.root, bg=self.COLORS['surface'],
                         highlightbackground=self.COLORS['border'],
                         highlightthickness=1)
        frame.pack(fill='x', side='bottom', padx=16, pady=(0, 8))
        tk.Label(frame, text='📋  Process Log',
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.COLORS['surface'], fg=self.COLORS['muted']
                 ).pack(anchor='w', padx=12, pady=(8, 2))
        self.log_text = scrolledtext.ScrolledText(
            frame, height=5, font=('Courier', 8),
            bg=self.COLORS['log_bg'], fg=self.COLORS['log_fg'],
            insertbackground=self.COLORS['accent'],
            relief='flat', bd=0)
        self.log_text.pack(fill='x', padx=12, pady=(0, 10))
        self.log_text.config(state=tk.DISABLED)

    def _build_main_content(self):
        main = tk.Frame(self.root, bg=self.COLORS['bg'])
        main.pack(fill='both', expand=True, padx=16, pady=(12, 8))

        # ── LEFT: Files card (shared across modes) ────────────────────────
        left = tk.Frame(main, bg=self.COLORS['bg'])
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))

        files_card = self._card(left, '📁  Files & Output')
        files_card.pack(fill='x')

        self.bdf_display = self._file_row(files_card, '📄  BDF File', self._browse_bdf)
        self.op2_display = self._file_row(files_card, '📊  OP2 File', self._browse_op2)
        self.out_display = self._file_row(files_card, '📁  Output Directory', self._browse_output, is_dir=True)
        tk.Frame(files_card, height=6, bg=self.COLORS['surface']).pack()

        # ── RIGHT: Mode-specific parameters ──────────────────────────────
        right = tk.Frame(main, bg=self.COLORS['bg'], width=320)
        right.pack(side='left', fill='y', padx=(10, 0))
        right.pack_propagate(False)

        # PSHELL parameter panel
        self.pshell_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.property_id_entry = self._param_entry(
            self.pshell_pf, '📋  Property IDs', 'Enter ALL or 123,456,789')
        self.property_id_entry.bind('<KeyRelease>', lambda e: self.update_pshell_ids())

        self.loadcase_id_entry = self._param_entry(
            self.pshell_pf, '⏱  Load Cases', 'Enter ALL or 1,2,3')

        coord = self._card(self.pshell_pf, '🔄  Coordinate System')
        coord.pack(fill='x', pady=(8, 0))
        for opt in ['Element CID', 'Material CID']:
            tk.Radiobutton(coord, text=f'  {opt}', variable=self.coordinate_system, value=opt,
                          bg=self.COLORS['surface'], fg=self.COLORS['text'],
                          selectcolor=self.COLORS['accent'],
                          activebackground=self.COLORS['surface'],
                          font=('Segoe UI', 10), cursor='hand2'
                          ).pack(anchor='w', padx=12, pady=4)
        tk.Frame(coord, height=8, bg=self.COLORS['surface']).pack()

        # BUSH parameter panel
        self.bush_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.bush_element_id_entry = self._param_entry(
            self.bush_pf, '🔧  Element IDs', 'Enter ALL or 452,678,890')
        self.bush_element_id_entry.bind('<KeyRelease>', lambda e: self.update_bush_ids())

        self.bush_loadcase_id_entry = self._param_entry(
            self.bush_pf, '⏱  Load Cases', 'Enter ALL or 1,2,3')

        # DISPLACEMENT parameter panel
        self.disp_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.disp_prop_entry = self._param_entry(
            self.disp_pf, '📋  Property IDs', 'Enter ALL or 123,456,789')

        self.disp_lc_entry = self._param_entry(
            self.disp_pf, '⏱  Load Cases', 'Enter ALL or 1,2,3')

        # STRESS parameter panel
        self.stress_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.stress_prop_entry = self._param_entry(
            self.stress_pf, '📋  Property IDs', 'Enter ALL or 123,456,789')

        self.stress_lc_entry = self._param_entry(
            self.stress_pf, '⏱  Load Cases', 'Enter ALL or 1,2,3')

        stress_coord = self._card(self.stress_pf, '🔄  Coordinate System')
        stress_coord.pack(fill='x', pady=(8, 0))
        for opt in ['Element CID', 'Material CID']:
            tk.Radiobutton(stress_coord, text=f'  {opt}',
                          variable=self.stress_coord_system, value=opt,
                          bg=self.COLORS['surface'], fg=self.COLORS['text'],
                          selectcolor=self.COLORS['accent'],
                          activebackground=self.COLORS['surface'],
                          font=('Segoe UI', 10), cursor='hand2'
                          ).pack(anchor='w', padx=12, pady=4)
        tk.Frame(stress_coord, height=8, bg=self.COLORS['surface']).pack()

    # ─────────────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _card(self, parent, title):
        card = tk.Frame(parent, bg=self.COLORS['surface'],
                        highlightbackground=self.COLORS['border'],
                        highlightthickness=1)
        tk.Label(card, text=title, font=('Segoe UI', 10, 'bold'),
                 bg=self.COLORS['surface'], fg=self.COLORS['accent']
                 ).pack(anchor='w', padx=14, pady=(10, 4))
        return card

    def _file_row(self, parent, label, cmd, is_dir=False):
        row = tk.Frame(parent, bg=self.COLORS['surface'])
        row.pack(fill='x', padx=14, pady=(0, 10))
        tk.Label(row, text=label, font=('Segoe UI', 9),
                 bg=self.COLORS['surface'], fg=self.COLORS['muted']
                 ).pack(anchor='w', pady=(0, 3))
        inner = tk.Frame(row, bg=self.COLORS['surface'])
        inner.pack(fill='x')
        entry = tk.Entry(inner, font=('Segoe UI', 9),
                        bg=self.COLORS['surface2'], fg=self.COLORS['text'],
                        insertbackground=self.COLORS['accent'],
                        relief='flat', bd=0,
                        highlightbackground=self.COLORS['border'],
                        highlightthickness=1)
        entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 8))
        tk.Button(inner, text='Browse', command=cmd,
                 bg=self.COLORS['surface2'], fg=self.COLORS['accent'],
                 font=('Segoe UI', 9, 'bold'), relief='flat', cursor='hand2',
                 padx=14, pady=5,
                 highlightbackground=self.COLORS['border'],
                 highlightthickness=1).pack(side='left')
        return entry

    def _param_entry(self, parent, label, hint):
        frame = tk.Frame(parent, bg=self.COLORS['bg'])
        frame.pack(fill='x', pady=(0, 12))
        tk.Label(frame, text=label, font=('Segoe UI', 10, 'bold'),
                 bg=self.COLORS['bg'], fg=self.COLORS['text']
                 ).pack(anchor='w', pady=(0, 4))
        entry = tk.Entry(frame, font=('Segoe UI', 10),
                        bg=self.COLORS['surface'], fg=self.COLORS['text'],
                        insertbackground=self.COLORS['accent'],
                        relief='flat', bd=0,
                        highlightbackground=self.COLORS['border'],
                        highlightthickness=1)
        entry.insert(0, 'ALL')
        entry.pack(fill='x', ipady=6)
        tk.Label(frame, text=hint, font=('Segoe UI', 8),
                 bg=self.COLORS['bg'], fg=self.COLORS['muted']
                 ).pack(anchor='w', pady=(3, 0))
        return entry

    # ─────────────────────────────────────────────────────────────────────────
    # TAB / MODE CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    def _select_tab(self, mode):
        self.extraction_type.set(mode)
        for m, btn in self.tab_btns.items():
            active = (m == mode)
            btn.config(
                bg=self.COLORS['bg'] if active else self.COLORS['surface'],
                fg=self.COLORS['accent'] if active else self.COLORS['muted'])
        self.pshell_pf.pack_forget()
        self.bush_pf.pack_forget()
        self.disp_pf.pack_forget()
        self.stress_pf.pack_forget()
        if mode == 'PSHELL ALL AVERAGE':
            self.pshell_pf.pack(fill='both', expand=True)
        elif mode == 'BUSH LOAD':
            self.bush_pf.pack(fill='both', expand=True)
        elif mode == 'DISPLACEMENT':
            self.disp_pf.pack(fill='both', expand=True)
        else:
            self.stress_pf.pack(fill='both', expand=True)

    def on_mode_change(self):
        self._select_tab(self.extraction_type.get())

    # ─────────────────────────────────────────────────────────────────────────
    # FILE BROWSERS
    # ─────────────────────────────────────────────────────────────────────────

    def _browse_bdf(self):
        path = filedialog.askopenfilename(title='Select BDF File',
                                          filetypes=[('BDF Files', '*.bdf')])
        if path:
            self.input_entry_now = path
            self.bdf_display.delete(0, tk.END)
            self.bdf_display.insert(0, f'✓  {os.path.basename(path)}')

    def _browse_op2(self):
        path = filedialog.askopenfilename(title='Select OP2 File',
                                          filetypes=[('OP2 Files', '*.op2')])
        if path:
            self.output_entry_now = path
            self.op2_display.delete(0, tk.END)
            self.op2_display.insert(0, f'✓  {os.path.basename(path)}')

    def _browse_output(self):
        path = filedialog.askdirectory(title='Select Output Directory')
        if path:
            self.stress_output_now2 = path
            self.out_display.delete(0, tk.END)
            self.out_display.insert(0, f'✓  {os.path.basename(path)}')

    # Backward-compat aliases
    def bdf_input(self):      self._browse_bdf()
    def op2_input(self):      self._browse_op2()
    def output_location(self): self._browse_output()

    # ─────────────────────────────────────────────────────────────────────────
    # STATE UPDATES
    # ─────────────────────────────────────────────────────────────────────────

    def update_pshell_ids(self):
        self.pshell_property_ids = self.property_id_entry.get().strip()

    def update_bush_ids(self):
        self.bush_element_ids = self.bush_element_id_entry.get().strip()

    def update_load_cases(self):
        pass  # values read directly from widgets in run_* methods

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGER
    # ─────────────────────────────────────────────────────────────────────────

    def setup_logger(self):
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        h = TextHandler(self.log_text)
        h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(h)

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN DISPATCHER
    # ─────────────────────────────────────────────────────────────────────────

    def asc_run(self):
        self.clear_log()

        # Sync widget values before handing off to thread
        self.pshell_property_ids = self.property_id_entry.get().strip()
        self.bush_element_ids    = self.bush_element_id_entry.get().strip()

        if not self.stress_output_now2:
            messagebox.showerror("Hata", "Çıktı klasörünü seçin!")
            return
        if not self.input_entry_now or not self.output_entry_now:
            messagebox.showerror("Hata", "BDF ve OP2 dosyalarını seçin")
            return

        self.run_btn.config(state='disabled', text='⏳ Çalışıyor...')
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        file_handler = logging.FileHandler(
            os.path.join(self.stress_output_now2,
                         f'LoadExtraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        self.logger.info("="*60)
        self.logger.info("LOAD EXTRACTION TOOL BAŞLATILDI")
        self.logger.info(f"Mod: {self.extraction_type.get()}")
        self.logger.info("="*60)

        start_time = time.time()
        try:
            if self.extraction_type.get() == "PSHELL ALL AVERAGE":
                self.run_pshell()
            elif self.extraction_type.get() == "BUSH LOAD":
                self.run_bush()
            elif self.extraction_type.get() == "DISPLACEMENT":
                self.run_displacement()
            else:
                self.run_stress()

            elapsed = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info(f"✅ İşlem tamamlandı! ({elapsed:.2f} saniye)")
            self.logger.info(f"📁 Çıktılar: {self.stress_output_now2}")
            self.logger.info("="*60)
            self.root.after(0, lambda: messagebox.showinfo(
                "Başarılı", f"İşlem Tamamlandı\nSüre: {elapsed:.2f} saniye"))
        except Exception as e:
            self.logger.error(f"HATA: {e}")
            self.root.after(0, lambda: messagebox.showerror("Hata", str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state='normal', text='▶  RUN'))

    # ─────────────────────────────────────────────────────────────────────────
    # PSHELL EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_pshell(self):
        if not self.pshell_property_ids:
            self.logger.error("Property ID'leri girin!")
            messagebox.showerror("Hata", "Property ID'leri girin (tüm için: ALL)")
            return

        op2 = self._load_op2()
        bdf = self._load_bdf()

        self.logger.info("🔍 Property ID'leri parse ediliyor...")
        all_pids = set([elem.pid for elem in bdf.elements.values()
                       if elem.type in ("CQUAD4", "CTRIA3")])
        target_property_ids = parse_id_input(self.pshell_property_ids, list(all_pids))
        if not target_property_ids:
            target_property_ids = list(all_pids)
        self.logger.info(f"✓ {len(target_property_ids)} property ID seçildi")

        elements_with_properties = {
            element_id: element.pid
            for element_id, element in bdf.elements.items()
            if (element.type == "CQUAD4" or element.type == "CTRIA3") and element.pid in target_property_ids
        }

        all_lc_pshell = list(set(op2.cquad4_force.keys()).union(op2.ctria3_force.keys()))
        lc_input_str = self.loadcase_id_entry.get().strip()
        target_lc_ids = set(parse_id_input(lc_input_str, all_lc_pshell))
        if not target_lc_ids:
            target_lc_ids = set(all_lc_pshell)
        self.logger.info(f"✓ {len(target_lc_ids)} load case seçildi")

        property_forces = {
            load_case_id: {
                pid: {'Nx':0.0,'Ny':0.0,'Nxy':0.0,'Mx':0.0,'My':0.0,'Mxy':0.0}
                for pid in target_property_ids
            }
            for load_case_id in set(op2.cquad4_force.keys()).union(op2.ctria3_force.keys())
        }

        property_areas = {}
        element_areas = {}
        property_thickness = {}

        for pid in target_property_ids:
            try:
                property_thickness[pid] = float(bdf.properties[pid].t)
            except Exception:
                property_thickness[pid] = 0.0

        for element_id, element in bdf.elements.items():
            if element.pid in target_property_ids:
                property_id = element.pid
                area = element.Area()
                element_areas[element_id] = area
                if property_id not in property_areas:
                    property_areas[property_id] = 0.0
                property_areas[property_id] += area

        is_material_cid = self.coordinate_system.get() == "Material CID"
        coord_mode = "Material CID" if is_material_cid else "Element CID"
        self.logger.info(f"🔄 Koordinat sistemi: {coord_mode}")
        op2_data = data_in_material_coord(bdf, op2, in_place=False) if is_material_cid else op2
        self.logger.info("✓ Koordinat dönüşümü tamamlandı" if is_material_cid else "✓ Element CID kullanılacak")

        # ── Pre-build element mapping arrays once (outside all LC loops) ───
        pid_list      = sorted(set(target_property_ids))
        pid_to_ridx   = {pid: i for i, pid in enumerate(pid_list)}
        n_pids        = len(pid_list)
        tgt_eids_list = list(elements_with_properties.keys())
        tgt_pids_list = [elements_with_properties[e] for e in tgt_eids_list]
        tgt_eids_arr  = np.array(tgt_eids_list, dtype=np.int64)
        tgt_pids_arr  = np.array(tgt_pids_list, dtype=np.int64)
        tgt_ridx_arr  = np.array([pid_to_ridx[p] for p in tgt_pids_list])
        tgt_areas_arr = np.array([element_areas[e] for e in tgt_eids_list])
        tgt_thick_arr = np.array([property_thickness[p] for p in tgt_pids_list])
        avg_area_arr  = np.array([property_areas[p] for p in pid_list])
        avg_thick_arr = np.array([property_thickness[p] for p in pid_list])

        self.logger.info("🔄 Element forces işleniyor...")
        elem_chunks = []
        pf_accum    = {}   # lc_id → (n_pids, 6) array [Nx,Ny,Nxy,Mx,My,Mxy] × area

        for force_dict, lc_label in [(op2_data.cquad4_force, 'CQUAD4'),
                                     (op2_data.ctria3_force, 'CTRIA3')]:
            for load_case_id, element_forces in force_dict.items():
                if load_case_id not in target_lc_ids:
                    continue
                element_ids = element_forces.element
                forces_data = element_forces.data[0]
                load_id     = element_forces.loadIDs[0]
                eid_to_idx  = {int(e): i for i, e in enumerate(element_ids)}

                # Vectorized lookup for all target elements at once
                v_idxs = np.array([eid_to_idx.get(int(e), -1) for e in tgt_eids_arr])
                valid  = v_idxs >= 0
                if not valid.any():
                    continue
                sel    = v_idxs[valid]
                f_sub  = forces_data[sel]           # (n_valid, ≥6)
                nx, ny, nxy = f_sub[:, 0], f_sub[:, 1], f_sub[:, 2]
                mx, my, mxy = f_sub[:, 3], f_sub[:, 4], f_sub[:, 5]
                areas  = tgt_areas_arr[valid]
                ridxs  = tgt_ridx_arr[valid]

                # Accumulate property forces with np.add.at
                pf = pf_accum.setdefault(load_id, np.zeros((n_pids, 6)))
                np.add.at(pf, ridxs, np.column_stack(
                    [nx*areas, ny*areas, nxy*areas, mx*areas, my*areas, mxy*areas]))

                elem_chunks.append(pd.DataFrame({
                    'Property ID':  tgt_pids_arr[valid],
                    'Element ID':   tgt_eids_arr[valid],
                    'Load Case ID': load_id,
                    'Nx': nx, 'Ny': ny, 'Nxy': nxy,
                    'Mx': mx, 'My': my, 'Mxy': mxy,
                    'Thickness': tgt_thick_arr[valid],
                }))

        df = pd.concat(elem_chunks, ignore_index=True) if elem_chunks else pd.DataFrame()
        df.to_csv(os.path.join(self.stress_output_now2, 'Element_Load.csv'), index=False)
        self.logger.info(f"✓ Element_Load.csv yazıldı ({len(df)} satır)")

        # ── Average forces from pf_accum matrix ───────────────────────────
        Average_forces = []
        for load_id, pf in pf_accum.items():
            avg = pf / avg_area_arr[:, None]
            for i, pid in enumerate(pid_list):
                Average_forces.append({
                    'Property ID':  pid,
                    'Load Case ID': load_id,
                    'Average Nx':  float(avg[i, 0]), 'Average Ny':  float(avg[i, 1]),
                    'Average Nxy': float(avg[i, 2]), 'Average Mx':  float(avg[i, 3]),
                    'Average My':  float(avg[i, 4]), 'Average Mxy': float(avg[i, 5]),
                    'Thickness':   float(avg_thick_arr[i]),
                })
        df2 = pd.DataFrame(Average_forces)
        df2.to_csv(os.path.join(self.stress_output_now2, 'Average_Load.csv'), index=False)
        self.logger.info(f"✓ Average_Load.csv yazıldı ({len(df2)} satır)")

        self.logger.info("🔄 Element reduction hesaplanıyor (16 metrik)...")
        critical_elem = extract_critical_pshell(df.to_dict('records'), 'Element ID', 'Nx', 'Ny', 'Nxy')
        reduced_elem = [{
            'Property ID':  r['Property ID'],
            'Element ID':   r['Element ID'],
            'Load Case ID': r['Load Case ID'],
            'Nx': r['_nx'], 'Ny': r['_ny'], 'Nxy': r['_nxy'],
            'Mx': r['Mx'],  'My': r['My'],  'Mxy': r['Mxy'],
            'Thickness': r['Thickness'],
        } for r in critical_elem]
        pd.DataFrame(reduced_elem).to_csv(
            os.path.join(self.stress_output_now2, 'Element_Load_Reduced.csv'), index=False)
        self.logger.info(f"✓ Element_Load_Reduced.csv yazıldı ({len(reduced_elem)} kritik satır)")

        self.logger.info("🔄 Average reduction hesaplanıyor (16 metrik)...")
        critical_avg = extract_critical_pshell(Average_forces, 'Property ID', 'Average Nx', 'Average Ny', 'Average Nxy')
        reduced_avg = [{
            'Property ID':  r['Property ID'],
            'Load Case ID': r['Load Case ID'],
            'Average Nx':  r['_nx'], 'Average Ny':  r['_ny'], 'Average Nxy': r['_nxy'],
            'Average Mx':  r['Average Mx'], 'Average My':  r['Average My'],
            'Average Mxy': r['Average Mxy'], 'Thickness':  r['Thickness'],
        } for r in critical_avg]
        pd.DataFrame(reduced_avg).to_csv(
            os.path.join(self.stress_output_now2, 'Average_Load_Reduced.csv'), index=False)
        self.logger.info(f"✓ Average_Load_Reduced.csv yazıldı ({len(reduced_avg)} kritik satır)")

    # ─────────────────────────────────────────────────────────────────────────
    # STRESS EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_stress(self):
        op2 = self._load_op2()
        bdf = self._load_bdf()

        prop_str = self.stress_prop_entry.get().strip()
        all_pids = {e.pid for e in bdf.elements.values()
                    if e.type in ('CQUAD4', 'CTRIA3')}
        target_pids = set(parse_id_input(prop_str, list(all_pids)))
        if not target_pids:
            target_pids = all_pids
        self.logger.info(f"✓ {len(target_pids)} property ID seçildi")

        elements_with_properties = {
            eid: elem.pid
            for eid, elem in bdf.elements.items()
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids
        }
        element_areas  = {eid: bdf.elements[eid].Area() for eid in elements_with_properties}
        property_areas = {}
        for eid, pid in elements_with_properties.items():
            property_areas[pid] = property_areas.get(pid, 0.0) + element_areas[eid]

        is_material = self.stress_coord_system.get() == 'Material CID'
        self.logger.info(f"🔄 Koordinat sistemi: {self.stress_coord_system.get()}")
        op2_data = data_in_material_coord(bdf, op2, in_place=False) if is_material else op2

        all_lc_stress = list(set(
            (list(op2_data.cquad4_stress.keys()) if hasattr(op2_data, 'cquad4_stress') and op2_data.cquad4_stress else []) +
            (list(op2_data.ctria3_stress.keys())  if hasattr(op2_data, 'ctria3_stress')  and op2_data.ctria3_stress  else [])
        ))
        lc_str = self.stress_lc_entry.get().strip()
        target_lc_ids = set(parse_id_input(lc_str, all_lc_stress))
        if not target_lc_ids:
            target_lc_ids = set(all_lc_stress)
        self.logger.info(f"✓ {len(target_lc_ids)} load case seçildi")

        self.logger.info("🔄 Stress verileri işleniyor...")
        element_stress_data = []
        property_stress = {
            lc: {
                pid: dict(sx1=0.,sy1=0.,sxy1=0.,vm1=0.,p1_1=0.,p2_1=0.,
                          sx2=0.,sy2=0.,sxy2=0.,vm2=0.,p1_2=0.,p2_2=0.)
                for pid in target_pids
            }
            for lc in target_lc_ids
        }

        stress_sources = []
        if hasattr(op2_data, 'cquad4_stress') and op2_data.cquad4_stress:
            stress_sources.append(op2_data.cquad4_stress)
        if hasattr(op2_data, 'ctria3_stress') and op2_data.ctria3_stress:
            stress_sources.append(op2_data.ctria3_stress)

        for stress_dict in stress_sources:
            # Build element lookup ONCE — element order is same for all OP2 subcases
            first_sr    = next(iter(stress_dict.values()))
            eids_s      = first_sr.element_node[:, 0].astype(int)
            eid_to_sidx = {}
            for i, eid in enumerate(eids_s):
                eid_to_sidx.setdefault(int(eid), []).append(i)

            # Pre-compute target (eid, pid, z1_idx, z2_idx, area) once
            target_elems = [
                (eid, pid, eid_to_sidx[eid][0], eid_to_sidx[eid][1], element_areas.get(eid, 0.0))
                for eid, pid in elements_with_properties.items()
                if eid in eid_to_sidx and len(eid_to_sidx[eid]) >= 2
            ]
            self.logger.info(f"✓ {len(target_elems)} element stress verisi hazırlandı")

            for load_case_id, stress_result in stress_dict.items():
                if load_case_id not in target_lc_ids:
                    continue
                sdata = stress_result.data[0]
                for eid, pid, z1i, z2i, area in target_elems:
                    z1 = sdata[z1i];  z2 = sdata[z2i]
                    ps = property_stress[load_case_id][pid]
                    ps['sx1']  += z1[1]*area; ps['sy1']  += z1[2]*area; ps['sxy1'] += z1[3]*area
                    ps['vm1']  += z1[7]*area; ps['p1_1'] += z1[5]*area; ps['p2_1'] += z1[6]*area
                    ps['sx2']  += z2[1]*area; ps['sy2']  += z2[2]*area; ps['sxy2'] += z2[3]*area
                    ps['vm2']  += z2[7]*area; ps['p1_2'] += z2[5]*area; ps['p2_2'] += z2[6]*area
                    element_stress_data.append({
                        'Property ID':  pid,
                        'Element ID':   eid,
                        'Load Case ID': load_case_id,
                        'Sx_Z1': z1[1], 'Sy_Z1': z1[2], 'Sxy_Z1': z1[3],
                        'VM_Z1': z1[7], 'P1_Z1': z1[5], 'P2_Z1': z1[6],
                        'Sx_Z2': z2[1], 'Sy_Z2': z2[2], 'Sxy_Z2': z2[3],
                        'VM_Z2': z2[7], 'P1_Z2': z2[5], 'P2_Z2': z2[6],
                    })

        if not element_stress_data:
            self.logger.info("⚠ Stress verisi bulunamadı (OP2'de STRESS output yok)")
            return

        df_stress = pd.DataFrame(element_stress_data)
        df_stress.to_csv(os.path.join(self.stress_output_now2, 'Element_Stress.csv'), index=False)
        self.logger.info(f"✓ Element_Stress.csv yazıldı ({len(df_stress)} satır)")

        average_stress_data = []
        for lc in target_lc_ids:
            for pid in target_pids:
                ta = property_areas.get(pid, 0.0)
                if ta == 0.0:
                    continue
                ps = property_stress[lc][pid]
                average_stress_data.append({
                    'Property ID':  pid,
                    'Load Case ID': lc,
                    'Avg_Sx_Z1':  ps['sx1']/ta,  'Avg_Sy_Z1':  ps['sy1']/ta,  'Avg_Sxy_Z1': ps['sxy1']/ta,
                    'Avg_VM_Z1':  ps['vm1']/ta,  'Avg_P1_Z1':  ps['p1_1']/ta, 'Avg_P2_Z1':  ps['p2_1']/ta,
                    'Avg_Sx_Z2':  ps['sx2']/ta,  'Avg_Sy_Z2':  ps['sy2']/ta,  'Avg_Sxy_Z2': ps['sxy2']/ta,
                    'Avg_VM_Z2':  ps['vm2']/ta,  'Avg_P1_Z2':  ps['p1_2']/ta, 'Avg_P2_Z2':  ps['p2_2']/ta,
                })

        df_avg_stress = pd.DataFrame(average_stress_data)
        df_avg_stress.to_csv(os.path.join(self.stress_output_now2, 'Average_Stress.csv'), index=False)
        self.logger.info(f"✓ Average_Stress.csv yazıldı ({len(df_avg_stress)} satır)")

        self.logger.info("🔄 Element stress reduction hesaplanıyor (max VM)...")
        crit_e = extract_critical_stress(element_stress_data, 'Element ID')
        red_e  = [{
            'Property ID':  r['Property ID'], 'Element ID': r['Element ID'], 'Load Case ID': r['Load Case ID'],
            'Sx_Z1': r['Sx_Z1'], 'Sy_Z1': r['Sy_Z1'], 'Sxy_Z1': r['Sxy_Z1'],
            'VM_Z1': r['VM_Z1'], 'P1_Z1': r['P1_Z1'], 'P2_Z1': r['P2_Z1'],
            'Sx_Z2': r['Sx_Z2'], 'Sy_Z2': r['Sy_Z2'], 'Sxy_Z2': r['Sxy_Z2'],
            'VM_Z2': r['VM_Z2'], 'P1_Z2': r['P1_Z2'], 'P2_Z2': r['P2_Z2'],
        } for r in crit_e]
        pd.DataFrame(red_e).to_csv(os.path.join(self.stress_output_now2, 'Element_Stress_Reduced.csv'), index=False)
        self.logger.info(f"✓ Element_Stress_Reduced.csv yazıldı ({len(red_e)} kritik satır)")

        self.logger.info("🔄 Average stress reduction hesaplanıyor (max VM)...")
        crit_a = extract_critical_stress(average_stress_data, 'Property ID')
        red_a  = [{
            'Property ID':  r['Property ID'], 'Load Case ID': r['Load Case ID'],
            'Avg_Sx_Z1': r['Avg_Sx_Z1'], 'Avg_Sy_Z1': r['Avg_Sy_Z1'], 'Avg_Sxy_Z1': r['Avg_Sxy_Z1'],
            'Avg_VM_Z1': r['Avg_VM_Z1'], 'Avg_P1_Z1': r['Avg_P1_Z1'], 'Avg_P2_Z1': r['Avg_P2_Z1'],
            'Avg_Sx_Z2': r['Avg_Sx_Z2'], 'Avg_Sy_Z2': r['Avg_Sy_Z2'], 'Avg_Sxy_Z2': r['Avg_Sxy_Z2'],
            'Avg_VM_Z2': r['Avg_VM_Z2'], 'Avg_P1_Z2': r['Avg_P1_Z2'], 'Avg_P2_Z2': r['Avg_P2_Z2'],
        } for r in crit_a]
        pd.DataFrame(red_a).to_csv(os.path.join(self.stress_output_now2, 'Average_Stress_Reduced.csv'), index=False)
        self.logger.info(f"✓ Average_Stress_Reduced.csv yazıldı ({len(red_a)} kritik satır)")

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLACEMENT EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_displacement(self):
        op2 = self._load_op2()
        bdf = self._load_bdf()

        prop_str = self.disp_prop_entry.get().strip()
        all_pids = {e.pid for e in bdf.elements.values()
                    if e.type in ('CQUAD4', 'CTRIA3')}
        target_pids = set(parse_id_input(prop_str, list(all_pids)))
        if not target_pids:
            target_pids = all_pids
        self.logger.info(f"✓ {len(target_pids)} property ID seçildi")

        pid_to_nodes = {}
        for elem in bdf.elements.values():
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids:
                pid_to_nodes.setdefault(elem.pid, set()).update(elem.node_ids)

        if not op2.displacements:
            self.logger.error("OP2'de displacement output bulunamadı!")
            messagebox.showerror("Hata", "OP2'de DISPLACEMENT output yok")
            return

        all_lc  = list(op2.displacements.keys())
        lc_str  = self.disp_lc_entry.get().strip()
        target_lc = set(parse_id_input(lc_str, all_lc))
        if not target_lc:
            target_lc = set(all_lc)
        self.logger.info(f"✓ {len(target_lc)} load case seçildi")

        self.logger.info("🔄 Displacement verileri işleniyor...")
        disp_data = []

        # Build node lookup ONCE — node order is identical across all OP2 subcases
        first_disp = next(iter(op2.displacements.values()))
        all_nids   = first_disp.node_gridtype[:, 0].astype(int)
        nid_to_idx = dict(zip(all_nids, range(len(all_nids))))

        # Pre-compute (pid, nid, model_index) list once
        pid_nid_idx = []
        for pid, nodes in pid_to_nodes.items():
            for nid in nodes:
                idx = nid_to_idx.get(int(nid))
                if idx is not None:
                    pid_nid_idx.append((pid, int(nid), idx))
        self.logger.info(f"✓ {len(pid_nid_idx)} (property, node) çifti hazırlandı")

        # Pre-extract numpy arrays once — enables single fancy-index per LC
        idx_arr = np.array([t[2] for t in pid_nid_idx])
        pid_arr = np.array([t[0] for t in pid_nid_idx])
        nid_arr = np.array([t[1] for t in pid_nid_idx])
        n_pairs = len(pid_nid_idx)

        disp_chunks = []
        for lc_id, disp_result in op2.displacements.items():
            if lc_id not in target_lc:
                continue
            sub        = disp_result.data[0][idx_arr]   # (n_pairs, 6) — single vectorized slice
            x, y, z    = sub[:, 0], sub[:, 1], sub[:, 2]
            rx, ry, rz = sub[:, 3], sub[:, 4], sub[:, 5]
            mag        = np.sqrt(x**2 + y**2 + z**2)
            disp_chunks.append(pd.DataFrame({
                'Property ID':  pid_arr,
                'Node ID':      nid_arr,
                'Load Case ID': np.full(n_pairs, lc_id),
                'X': x, 'Y': y, 'Z': z,
                'Magnitude':    mag,
                'Rx': rx, 'Ry': ry, 'Rz': rz,
            }))

        df_all = pd.concat(disp_chunks, ignore_index=True) if disp_chunks else pd.DataFrame()
        df_all.to_csv(os.path.join(self.stress_output_now2, 'Displacement_All.csv'), index=False)
        self.logger.info(f"✓ Displacement_All.csv yazıldı ({len(df_all)} satır)")

        self.logger.info("🔄 Displacement reduction hesaplanıyor (max Magnitude per node)...")
        # Reduction directly on DataFrame — groupby idxmax is faster than list-of-dicts
        if not df_all.empty:
            idx_max = df_all.groupby(['Property ID', 'Node ID'])['Magnitude'].idxmax()
            df_red  = df_all.loc[idx_max].sort_values(
                ['Property ID', 'Node ID', 'Load Case ID']).reset_index(drop=True)
        else:
            df_red = pd.DataFrame()
        df_red.to_csv(os.path.join(self.stress_output_now2, 'Displacement_Reduced.csv'), index=False)
        self.logger.info(f"✓ Displacement_Reduced.csv yazıldı ({len(df_red)} kritik satır)")

    # ─────────────────────────────────────────────────────────────────────────
    # BUSH EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_bush(self):
        if not self.bush_element_ids:
            self.logger.error("Element ID'leri girin!")
            messagebox.showerror("Hata", "Element ID'leri girin (tüm için: ALL)")
            return

        op2 = self._load_op2()

        self.logger.info("🔍 Element ID'leri parse ediliyor...")
        all_eids = []
        for load_case_id, element_forces in op2.cbush_force.items():
            all_eids.extend(element_forces.element)
        all_eids = list(set(all_eids))

        selected_element_ids = parse_id_input(self.bush_element_ids, all_eids)
        if not selected_element_ids:
            selected_element_ids = all_eids
        self.logger.info(f"✓ {len(selected_element_ids)} element ID seçildi")

        all_lc_bush = list(op2.cbush_force.keys())
        lc_input_str = self.bush_loadcase_id_entry.get().strip()
        target_lc_bush = set(parse_id_input(lc_input_str, all_lc_bush))
        if not target_lc_bush:
            target_lc_bush = set(all_lc_bush)
        self.logger.info(f"✓ {len(target_lc_bush)} load case seçildi")

        self.logger.info("🔄 Bush force verileri çıkarılıyor...")
        bush_forces_data = []
        selected_set = set(int(e) for e in selected_element_ids)
        for load_case_id, element_forces in op2.cbush_force.items():
            if load_case_id not in target_lc_bush:
                continue
            element_ids = element_forces.element
            forces_data = element_forces.data[0]
            load_ids    = element_forces.loadIDs[0]
            for i, element_id in enumerate(element_ids):
                if int(element_id) in selected_set:
                    forces = forces_data[i][:3]
                    bush_forces_data.append({
                        'Element ID':  element_id,
                        'Load Case ID': load_ids,
                        'FX': forces[0],
                        'FY': forces[1],
                        'FZ': forces[2],
                    })

        df_bush_raw = pd.DataFrame(bush_forces_data)
        output_csv_bush_raw = os.path.join(self.stress_output_now2, 'Bush_Load_Raw.csv')
        df_bush_raw.to_csv(output_csv_bush_raw, index=False)
        self.logger.info(f"✓ Bush_Load_Raw.csv yazıldı ({len(df_bush_raw)} satır)")

        self.logger.info("🔄 Bush reduction hesaplanıyor (18 metrik)...")
        critical_rows = extract_critical_rows(bush_forces_data)
        reduced_data = [{
            'Element ID':r['Element ID'],
            'Load Case ID':r['Load Case ID'],
            'FX':r['_fx'],
            'FY':r['_fy'],
            'FZ':r['_fz'],
        } for r in critical_rows]
        df_bush_reduced = pd.DataFrame(reduced_data)
        output_csv_bush_reduced = os.path.join(self.stress_output_now2, 'Bush_Load_Reduced.csv')
        df_bush_reduced.to_csv(output_csv_bush_reduced, index=False)
        self.logger.info(f"✓ Bush_Load_Reduced.csv yazıldı ({len(df_bush_reduced)} kritik satır)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = LoadExtractionApp(root)
    root.mainloop()

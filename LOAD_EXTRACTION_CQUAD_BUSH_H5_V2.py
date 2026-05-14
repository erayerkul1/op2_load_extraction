#!/usr/bin/env python3
"""
Load Extraction Tool - Nastran H5 Analysis Application
Extracts and analyzes FEA loads from MSC Nastran H5 files
Supports CQUAD4 + CTRIA3, Element CID / Material CID,
load case filtering and 16/18-metric critical load case reduction.
"""

import h5py
from pyNastran.bdf.bdf import BDF
from pyNastran.utils.numpy_utils import integer_types
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


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS  (18 for BUSH, 16 for PSHELL)
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
    {"id": "M07", "fn": lambda nx,ny,nxy: math.sqrt(nx**2+ny**2)},
    {"id": "M08", "fn": lambda nx,ny,nxy: math.sqrt(nx**2+ny**2)+abs(nxy)},
    {"id": "M09", "fn": lambda nx,ny,nxy: math.sqrt(2*nx**2+2*ny**2)},
    {"id": "M10", "fn": lambda nx,ny,nxy: math.sqrt(nx**2+2*ny**2)},
    {"id": "M11", "fn": lambda nx,ny,nxy: math.sqrt(2*nx**2+ny**2)+2*abs(nxy)},
    {"id": "M12", "fn": lambda nx,ny,nxy: math.sqrt(nx**2+2*ny**2)+2*abs(nxy)},
    {"id": "M13", "fn": lambda nx,ny,nxy: nx+ny+nxy},
    {"id": "M14", "fn": lambda nx,ny,nxy: nx+ny},
    {"id": "M15", "fn": lambda nx,ny,nxy: ny+nxy},
    {"id": "M16", "fn": lambda nx,ny,nxy: nx+nxy},
]


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL LOAD CASE REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_critical_rows(raw_data):
    """18-metric critical load case reduction for BUSH elements."""
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
    """16-metric critical load case reduction for PSHELL elements/properties."""
    if not raw_data:
        return []
    seen = {}
    for i, row in enumerate(raw_data):
        key = (row[group_key], row["Load Case ID"])
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
    result.sort(key=lambda r: (r[group_key], r["Load Case ID"]))
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
        return list(all_ids) if all_ids else []
    try:
        return [int(x.strip()) for x in input_str.split(',') if x.strip()]
    except ValueError:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL COORDINATE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def transf_Mohr(Sxx, Syy, Sxy, theta_rad):
    """Mohr's Circle plane stress transformation to material coordinates."""
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Sxy = np.asarray(Sxy, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)
    Scenter = (Sxx + Syy) / 2.0
    R = np.sqrt((Sxx - Scenter) ** 2 + Sxy ** 2)
    theta_Mohr = np.arctan2(-Sxy, Sxx - Scenter) + 2.0 * theta_rad
    cos_M = np.cos(theta_Mohr)
    sin_M = np.sin(theta_Mohr)
    return Scenter + R * cos_M, Scenter - R * cos_M, -R * sin_M


def angle2vec(v1, v2):
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    return np.arccos(np.clip((v1 * v2).sum(axis=1) / denom, -1.0, 1.0))


def calc_imat(normals, csysi):
    jmat = np.cross(normals, csysi)
    norms = np.linalg.norm(jmat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.cross(jmat / norms, normals)


def compute_thetarad_from_bdf(bdf):
    """
    Per-element rotation angles (rad) for material coord transformation.
    Handles CQUAD4 (THETA + MCID) and CTRIA3 (THETA + MCID).
    Returns dict {element_id: thetarad}.
    """
    eids  = list(bdf.elements.keys())
    elems = list(bdf.elements.values())
    n = len(elems)
    if n == 0:
        return {}

    is_mcid_arr  = np.array([
        isinstance(getattr(e, 'theta_mcid', None), integer_types) for e in elems
    ])
    elem_type_arr = np.array([e.type for e in elems])

    # Base theta from element field (degrees → radians)
    thetarad = np.zeros(n, dtype=float)
    for i, e in enumerate(elems):
        if not is_mcid_arr[i]:
            t = getattr(e, 'theta_mcid', None)
            if isinstance(t, float):
                thetarad[i] = np.deg2rad(t)

    # ── QUAD types: distortion correction + MCID angle ───────────────────
    for qtype in ('CQUAD4', 'CQUAD8', 'CQUADR'):
        # THETA quads — correct for element distortion
        idx = np.where(~is_mcid_arr & (elem_type_arr == qtype))[0]
        if len(idx):
            qe     = [elems[i] for i in idx]
            corner = np.array([e.get_node_positions() for e in qe])
            g1, g2, g3, g4 = corner[:,0], corner[:,1], corner[:,2], corner[:,3]
            beta  = angle2vec(g3 - g1, g2 - g1)
            gamma = angle2vec(g4 - g2, g1 - g2)
            alpha = (beta + gamma) / 2.0
            thetarad[idx] += alpha - beta

        # MCID quads — compute angle from coordinate system
        idx = np.where(is_mcid_arr & (elem_type_arr == qtype))[0]
        if len(idx):
            qe      = [elems[i] for i in idx]
            corner  = np.array([e.get_node_positions() for e in qe])
            g1, g2, g3, g4 = corner[:,0], corner[:,1], corner[:,2], corner[:,3]
            normals = np.array([e.Normal() for e in qe])
            csysi   = np.array([bdf.coords[e.theta_mcid].i for e in qe])
            imat    = calc_imat(normals, csysi)
            angles  = angle2vec(g2 - g1, imat)
            sign    = np.sign((np.cross(g2 - g1, imat) * normals).sum(axis=1))
            beta    = angle2vec(g3 - g1, g2 - g1)
            gamma   = angle2vec(g4 - g2, g1 - g2)
            alpha   = (beta + gamma) / 2.0
            thetarad[idx] = angles * sign + alpha - beta

    # ── TRIA types: THETA needs no correction; MCID gets angle ───────────
    for ttype in ('CTRIA3', 'CTRIA6', 'CTRIAR'):
        # THETA trias — thetarad already set from element field, no correction
        # MCID trias — compute angle from coordinate system
        idx = np.where(is_mcid_arr & (elem_type_arr == ttype))[0]
        if len(idx):
            te      = [elems[i] for i in idx]
            corner  = np.array([e.get_node_positions() for e in te])
            g1, g2, g3 = corner[:,0], corner[:,1], corner[:,2]
            normals = np.array([e.Normal() for e in te])
            csysi   = np.array([bdf.coords[e.theta_mcid].i for e in te])
            imat    = calc_imat(normals, csysi)
            angles  = angle2vec(g2 - g1, imat)
            sign    = np.sign((np.cross(g2 - g1, imat) * normals).sum(axis=1))
            thetarad[idx] = angles * sign

    return {eid: thetarad[i] for i, eid in enumerate(eids)}


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
        self.root.title('Load Extraction Tool  —  H5')
        self.root.geometry('1150x720')
        self.root.minsize(950, 600)
        self.root.configure(bg=self.COLORS['bg'])

        self.logger = logging.getLogger('LoadExtractionH5')
        self.logger.setLevel(logging.INFO)

        self.bdf_path  = ''
        self.h5_path   = ''
        self.output_dir = ''

        self.extraction_type     = tk.StringVar(value='PSHELL ALL AVERAGE')
        self.coordinate_system   = tk.StringVar(value='Element CID')
        self.stress_coord_system = tk.StringVar(value='Element CID')

        self._bdf_cache      = None
        self._bdf_path_cache = None

        self.build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

    def _load_bdf(self):
        path = self.bdf_path
        if self._bdf_path_cache != path:
            self.logger.info('📂 BDF dosyası okunuyor...')
            bdf = BDF()
            bdf.read_bdf(path, encoding='latin1')
            self._bdf_cache      = bdf
            self._bdf_path_cache = path
            self.logger.info('✓ BDF dosyası okundu')
        else:
            self.logger.info('✓ BDF dosyası önbellekten alındı')
        return self._bdf_cache

    def build_ui(self):
        self._build_header()
        self._build_tab_bar()
        self._build_action_bar()   # side=BOTTOM first → always visible
        self._build_log_panel()    # side=BOTTOM second
        self._build_main_content()
        self.setup_logger()
        self._select_tab('PSHELL ALL AVERAGE')

    def _build_header(self):
        hdr = tk.Frame(self.root, bg='#1c2128', height=62)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='⚙  LOAD EXTRACTION TOOL  —  H5',
                 font=('Segoe UI', 17, 'bold'),
                 bg='#1c2128', fg=self.COLORS['accent']).pack(side='left', padx=22, pady=15)
        tk.Label(hdr, text='MSC Nastran H5  |  CQUAD4 + CTRIA3 + BUSH',
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

        # ── LEFT: Shared files ────────────────────────────────────────────
        left = tk.Frame(main, bg=self.COLORS['bg'])
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))

        files_card = self._card(left, '📁  Files & Output')
        files_card.pack(fill='x')
        self.bdf_widget = self._file_row(files_card, '📄  BDF File',         self._browse_bdf)
        self.h5_widget  = self._file_row(files_card, '📊  H5 File',          self._browse_h5)
        self.out_widget = self._file_row(files_card, '📁  Output Directory',  self._browse_output,
                                          is_dir=True)
        tk.Frame(files_card, height=6, bg=self.COLORS['surface']).pack()

        # ── RIGHT: Mode-specific parameters ──────────────────────────────
        right = tk.Frame(main, bg=self.COLORS['bg'], width=340)
        right.pack(side='left', fill='y', padx=(10, 0))
        right.pack_propagate(False)

        # ── PSHELL panel ──────────────────────────────────────────────────
        self.pshell_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.prop_id_entry = self._param_entry(
            self.pshell_pf, '📋  Property IDs', 'Enter ALL or 123,456,789')

        self.pshell_lc_entry = self._param_entry(
            self.pshell_pf, '⏱  Load Cases', 'Enter ALL or 1,2,3')

        coord_card = self._card(self.pshell_pf, '🔄  Coordinate System')
        coord_card.pack(fill='x', pady=(8, 0))
        for opt in ['Element CID', 'Material CID']:
            tk.Radiobutton(coord_card, text=f'  {opt}',
                          variable=self.coordinate_system, value=opt,
                          bg=self.COLORS['surface'], fg=self.COLORS['text'],
                          selectcolor=self.COLORS['accent'],
                          activebackground=self.COLORS['surface'],
                          font=('Segoe UI', 10), cursor='hand2'
                          ).pack(anchor='w', padx=12, pady=4)
        tk.Label(coord_card,
                 text='  Material CID requires THETA / MCID in BDF',
                 font=('Segoe UI', 8),
                 bg=self.COLORS['surface'], fg=self.COLORS['muted']
                 ).pack(anchor='w', padx=14, pady=(0, 8))

        # ── BUSH panel ────────────────────────────────────────────────────
        self.bush_pf = tk.Frame(right, bg=self.COLORS['bg'])

        self.bush_elem_entry = self._param_entry(
            self.bush_pf, '🔧  Element IDs', 'Enter ALL or 452,678,890')

        self.bush_lc_entry = self._param_entry(
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
    # TAB CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    def _select_tab(self, mode):
        self.extraction_type.set(mode)
        for m, btn in self.tab_btns.items():
            active = (m == mode)
            btn.config(
                bg=self.COLORS['bg']     if active else self.COLORS['surface'],
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

    # ─────────────────────────────────────────────────────────────────────────
    # FILE BROWSERS
    # ─────────────────────────────────────────────────────────────────────────

    def _browse_bdf(self):
        path = filedialog.askopenfilename(title='Select BDF File',
                                          filetypes=[('BDF Files', '*.bdf')])
        if path:
            self.bdf_path = path
            self.bdf_widget.delete(0, tk.END)
            self.bdf_widget.insert(0, f'✓  {os.path.basename(path)}')

    def _browse_h5(self):
        path = filedialog.askopenfilename(title='Select H5 File',
                                          filetypes=[('H5 Files', '*.h5')])
        if path:
            self.h5_path = path
            self.h5_widget.delete(0, tk.END)
            self.h5_widget.insert(0, f'✓  {os.path.basename(path)}')

    def _browse_output(self):
        path = filedialog.askdirectory(title='Select Output Directory')
        if path:
            self.output_dir = path
            self.out_widget.delete(0, tk.END)
            self.out_widget.insert(0, f'✓  {os.path.basename(path)}')

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
    # H5 HELPER  —  read domain → subcase mapping and build filter set
    # ─────────────────────────────────────────────────────────────────────────

    def _read_domains(self, h5):
        """Returns (domain_to_subcase dict, target_domain_ids set, all_subcases list)."""
        domains    = h5['NASTRAN/RESULT/DOMAINS']
        lc_ids     = np.array(domains['ID'])
        subcases   = np.array(domains['SUBCASE'])
        domain_to_subcase = {int(lid): int(sc) for lid, sc in zip(lc_ids, subcases)}
        return domain_to_subcase

    def _target_domains(self, domain_to_subcase, lc_entry_widget):
        """Parse load case entry and return set of matching domain IDs."""
        all_subcases = list(set(domain_to_subcase.values()))
        lc_str       = lc_entry_widget.get().strip()
        target_sc    = set(parse_id_input(lc_str, all_subcases))
        if not target_sc:
            target_sc = set(all_subcases)
        return {did for did, sc in domain_to_subcase.items() if sc in target_sc}, target_sc

    # ─────────────────────────────────────────────────────────────────────────
    # RUN DISPATCHER
    # ─────────────────────────────────────────────────────────────────────────

    def asc_run(self):
        self.clear_log()

        if not self.output_dir:
            messagebox.showerror('Hata', 'Çıktı klasörünü seçin!')
            return
        if not self.bdf_path or not self.h5_path:
            messagebox.showerror('Hata', 'BDF ve H5 dosyalarını seçin!')
            return

        self.run_btn.config(state='disabled', text='⏳ Çalışıyor...')
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        fh = logging.FileHandler(os.path.join(
            self.output_dir,
            f'LoadExtraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        self.logger.info('=' * 60)
        self.logger.info('LOAD EXTRACTION TOOL (H5) BAŞLATILDI')
        self.logger.info(f'Mod: {self.extraction_type.get()}')
        self.logger.info('=' * 60)

        start = time.time()
        try:
            if self.extraction_type.get() == 'PSHELL ALL AVERAGE':
                self.run_pshell()
            elif self.extraction_type.get() == 'BUSH LOAD':
                self.run_bush()
            elif self.extraction_type.get() == 'DISPLACEMENT':
                self.run_displacement()
            else:
                self.run_stress()

            elapsed = time.time() - start
            self.logger.info('=' * 60)
            self.logger.info(f'✅ İşlem tamamlandı! ({elapsed:.2f} saniye)')
            self.logger.info(f'📁 Çıktılar: {self.output_dir}')
            self.logger.info('=' * 60)
            self.root.after(0, lambda: messagebox.showinfo(
                'Başarılı', f'İşlem Tamamlandı\nSüre: {elapsed:.2f} saniye'))
        except Exception as e:
            self.logger.error(f'HATA: {e}')
            self.root.after(0, lambda: messagebox.showerror('Hata', str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.config(state='normal', text='▶  RUN ANALYSIS'))

    # ─────────────────────────────────────────────────────────────────────────
    # PSHELL EXTRACTION  (CQUAD4 + CTRIA3)
    # ─────────────────────────────────────────────────────────────────────────

    def run_pshell(self):
        # ── Property IDs ──────────────────────────────────────────────────
        prop_str = self.prop_id_entry.get().strip()

        # ── Read BDF ──────────────────────────────────────────────────────
        bdf = self._load_bdf()

        all_pids = {e.pid for e in bdf.elements.values()
                    if e.type in ('CQUAD4', 'CTRIA3')}
        target_pids = set(parse_id_input(prop_str, list(all_pids)))
        if not target_pids:
            target_pids = all_pids
        self.logger.info(f'✓ {len(target_pids)} property ID seçildi')

        # ── Coordinate system ─────────────────────────────────────────────
        is_material = self.coordinate_system.get() == 'Material CID'
        self.logger.info(f'🔄 Koordinat sistemi: {self.coordinate_system.get()}')
        thetarad_map = {}
        if is_material:
            self.logger.info("🔄 BDF'den material açıları hesaplanıyor...")
            thetarad_map = compute_thetarad_from_bdf(bdf)
            self.logger.info(f'✓ {len(thetarad_map)} element için açı hesaplandı')

        # ── Read H5 ───────────────────────────────────────────────────────
        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            domain_to_subcase = self._read_domains(h5)

            ef = h5['NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE']

            # CQUAD4
            q4 = ef['QUAD4']
            q4_dom = np.array(q4['DOMAIN_ID'])
            q4_eid = np.array(q4['EID'])
            q4_MX  = np.array(q4['MX']);  q4_MY  = np.array(q4['MY']);  q4_MXY = np.array(q4['MXY'])
            q4_BMX = np.array(q4['BMX']); q4_BMY = np.array(q4['BMY']); q4_BMXY = np.array(q4['BMXY'])

            # CTRIA3  (may not exist in every H5)
            has_tria3 = 'TRIA3' in ef
            if has_tria3:
                t3 = ef['TRIA3']
                t3_dom = np.array(t3['DOMAIN_ID'])
                t3_eid = np.array(t3['EID'])
                t3_MX  = np.array(t3['MX']);  t3_MY  = np.array(t3['MY']);  t3_MXY = np.array(t3['MXY'])
                t3_BMX = np.array(t3['BMX']); t3_BMY = np.array(t3['BMY']); t3_BMXY = np.array(t3['BMXY'])
            else:
                self.logger.info('ℹ️  H5 içinde TRIA3 verisi bulunamadı, atlanıyor')
                t3_dom = t3_eid = np.array([])
                t3_MX = t3_MY = t3_MXY = t3_BMX = t3_BMY = t3_BMXY = np.array([])
        self.logger.info('✓ H5 dosyası okundu')

        # ── Load case filter ──────────────────────────────────────────────
        target_dids, target_sc = self._target_domains(domain_to_subcase,
                                                       self.pshell_lc_entry)
        self.logger.info(f'✓ {len(target_sc)} load case seçildi')

        # ── BDF geometry maps ─────────────────────────────────────────────
        elem_to_pid = {
            eid: elem.pid
            for eid, elem in bdf.elements.items()
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids
        }
        element_areas  = {}
        property_areas = {}
        property_thickness = {}
        for pid in target_pids:
            try:
                property_thickness[pid] = float(bdf.properties[pid].t)
            except Exception:
                property_thickness[pid] = 0.0
        for eid, elem in bdf.elements.items():
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids:
                area = elem.Area()
                element_areas[eid] = area
                property_areas[elem.pid] = property_areas.get(elem.pid, 0.0) + area
        self.logger.info(f'✓ {len(element_areas)} element alanı hesaplandı')

        # ── Pre-build element mapping arrays once (outside all LC loops) ───
        pid_list      = sorted(target_pids)
        pid_to_ridx   = {pid: i for i, pid in enumerate(pid_list)}
        n_pids        = len(pid_list)
        tgt_eids_list = list(elem_to_pid.keys())
        tgt_pids_list = [elem_to_pid[e] for e in tgt_eids_list]
        tgt_eids_arr  = np.array(tgt_eids_list, dtype=np.int64)
        tgt_ridx_arr  = np.array([pid_to_ridx[p] for p in tgt_pids_list])
        tgt_pids_arr  = np.array(tgt_pids_list,  dtype=np.int64)
        tgt_areas_arr = np.array([element_areas[e] for e in tgt_eids_list])
        tgt_thick_arr = np.array([property_thickness[p] for p in tgt_pids_list])

        sources = [
            ('CQUAD4', q4_dom, q4_eid, q4_MX, q4_MY, q4_MXY, q4_BMX, q4_BMY, q4_BMXY),
        ]
        if has_tria3:
            sources.append(('CTRIA3', t3_dom, t3_eid, t3_MX, t3_MY, t3_MXY, t3_BMX, t3_BMY, t3_BMXY))

        self.logger.info('🔄 Element forces işleniyor (CQUAD4 + CTRIA3)...')
        elem_chunks = []
        # pf_accum[lc_did] = (n_pids, 6) array  [Nx, Ny, Nxy, Mx, My, Mxy] × area
        pf_accum = {}

        for etype, dom_arr, eid_arr, MX_arr, MY_arr, MXY_arr, BMX_arr, BMY_arr, BMXY_arr in sources:
            for lc_did in np.unique(dom_arr):
                if int(lc_did) not in target_dids:
                    continue

                lc_mask  = dom_arr == lc_did
                lc_eids  = eid_arr[lc_mask]
                lc_MX    = MX_arr[lc_mask].copy()
                lc_MY    = MY_arr[lc_mask].copy()
                lc_MXY   = MXY_arr[lc_mask].copy()
                lc_BMX   = BMX_arr[lc_mask].copy()
                lc_BMY   = BMY_arr[lc_mask].copy()
                lc_BMXY  = BMXY_arr[lc_mask].copy()

                if is_material and thetarad_map:
                    thetas = np.array([thetarad_map.get(e, 0.0) for e in lc_eids.tolist()])
                    lc_MX,  lc_MY,  lc_MXY  = transf_Mohr(lc_MX,  lc_MY,  lc_MXY,  thetas)
                    lc_BMX, lc_BMY, lc_BMXY = transf_Mohr(lc_BMX, lc_BMY, lc_BMXY, thetas)

                lc_name    = domain_to_subcase.get(int(lc_did), int(lc_did))
                eid_to_idx = {int(e): i for i, e in enumerate(lc_eids)}

                # Vectorized: find LC indices for all target elements at once
                v_idxs = np.array([eid_to_idx.get(int(e), -1) for e in tgt_eids_arr])
                valid  = v_idxs >= 0
                if not valid.any():
                    continue
                sel    = v_idxs[valid]
                nx  = lc_MX[sel].astype(float);   ny  = lc_MY[sel].astype(float)
                nxy = lc_MXY[sel].astype(float);  mx  = lc_BMX[sel].astype(float)
                my  = lc_BMY[sel].astype(float);  mxy = lc_BMXY[sel].astype(float)
                areas  = tgt_areas_arr[valid]
                ridxs  = tgt_ridx_arr[valid]

                # Accumulate property forces with np.add.at (no Python dict +=)
                pf = pf_accum.setdefault(lc_did, np.zeros((n_pids, 6)))
                np.add.at(pf, ridxs, np.column_stack(
                    [nx*areas, ny*areas, nxy*areas, mx*areas, my*areas, mxy*areas]))

                # Build element chunk as column-dict DataFrame (faster than list-of-dicts)
                elem_chunks.append(pd.DataFrame({
                    'Property ID':  tgt_pids_arr[valid],
                    'Element ID':   tgt_eids_arr[valid],
                    'Element Type': etype,
                    'Load Case ID': lc_name,
                    'Nx': nx, 'Ny': ny, 'Nxy': nxy,
                    'Mx': mx, 'My': my, 'Mxy': mxy,
                    'Thickness': tgt_thick_arr[valid],
                }))

        # ── Element_Load.csv ──────────────────────────────────────────────
        df_elem = pd.concat(elem_chunks, ignore_index=True) if elem_chunks else pd.DataFrame()
        df_elem.to_csv(os.path.join(self.output_dir, 'Element_Load.csv'), index=False)
        self.logger.info(f'✓ Element_Load.csv yazıldı ({len(df_elem)} satır)')

        # ── Element_Load_Reduced.csv (16 metrik) ──────────────────────────
        self.logger.info('🔄 Element reduction hesaplanıyor (16 metrik)...')
        critical_elem = extract_critical_pshell(
            df_elem.to_dict('records'), 'Element ID', 'Nx', 'Ny', 'Nxy')
        reduced_elem = [{
            'Property ID':  r['Property ID'],
            'Element ID':   r['Element ID'],
            'Element Type': r['Element Type'],
            'Load Case ID': r['Load Case ID'],
            'Nx': r['_nx'], 'Ny': r['_ny'], 'Nxy': r['_nxy'],
            'Mx': r['Mx'],  'My': r['My'],  'Mxy': r['Mxy'],
            'Thickness': r['Thickness'],
        } for r in critical_elem]
        pd.DataFrame(reduced_elem).to_csv(
            os.path.join(self.output_dir, 'Element_Load_Reduced.csv'), index=False)
        self.logger.info(f'✓ Element_Load_Reduced.csv yazıldı ({len(reduced_elem)} kritik satır)')

        # ── Average_Load.csv — built from pf_accum matrix ─────────────────
        avg_area_arr  = np.array([property_areas.get(p, 1.0) for p in pid_list])
        avg_thick_arr = np.array([property_thickness[p]       for p in pid_list])
        average_data = []
        for lc_did, pf in pf_accum.items():
            lc_name = domain_to_subcase.get(int(lc_did), int(lc_did))
            avg = pf / avg_area_arr[:, None]   # (n_pids, 6) / (n_pids, 1)
            for i, pid in enumerate(pid_list):
                average_data.append({
                    'Property ID':  pid,
                    'Load Case ID': lc_name,
                    'Average Nx':  float(avg[i, 0]), 'Average Ny':  float(avg[i, 1]),
                    'Average Nxy': float(avg[i, 2]), 'Average Mx':  float(avg[i, 3]),
                    'Average My':  float(avg[i, 4]), 'Average Mxy': float(avg[i, 5]),
                    'Thickness':   float(avg_thick_arr[i]),
                })
        df_avg = pd.DataFrame(average_data)
        df_avg.to_csv(os.path.join(self.output_dir, 'Average_Load.csv'), index=False)
        self.logger.info(f'✓ Average_Load.csv yazıldı ({len(df_avg)} satır)')

        # ── Average_Load_Reduced.csv (16 metrik) ──────────────────────────
        self.logger.info('🔄 Average reduction hesaplanıyor (16 metrik)...')
        critical_avg = extract_critical_pshell(
            average_data, 'Property ID', 'Average Nx', 'Average Ny', 'Average Nxy')
        reduced_avg = [{
            'Property ID':  r['Property ID'],
            'Load Case ID': r['Load Case ID'],
            'Average Nx':  r['_nx'], 'Average Ny':  r['_ny'], 'Average Nxy': r['_nxy'],
            'Average Mx':  r['Average Mx'],
            'Average My':  r['Average My'],
            'Average Mxy': r['Average Mxy'],
            'Thickness':   r['Thickness'],
        } for r in critical_avg]
        pd.DataFrame(reduced_avg).to_csv(
            os.path.join(self.output_dir, 'Average_Load_Reduced.csv'), index=False)
        self.logger.info(f'✓ Average_Load_Reduced.csv yazıldı ({len(reduced_avg)} kritik satır)')

    # ─────────────────────────────────────────────────────────────────────────
    # STRESS EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_stress(self):
        bdf = self._load_bdf()

        prop_str = self.stress_prop_entry.get().strip()
        all_pids = {e.pid for e in bdf.elements.values() if e.type in ('CQUAD4', 'CTRIA3')}
        target_pids = set(parse_id_input(prop_str, list(all_pids)))
        if not target_pids:
            target_pids = all_pids
        self.logger.info(f'✓ {len(target_pids)} property ID seçildi')

        is_material = self.stress_coord_system.get() == 'Material CID'
        self.logger.info(f'🔄 Koordinat sistemi: {self.stress_coord_system.get()}')
        thetarad_map = {}
        if is_material:
            self.logger.info("🔄 BDF'den material açıları hesaplanıyor...")
            thetarad_map = compute_thetarad_from_bdf(bdf)

        elem_to_pid = {
            eid: elem.pid
            for eid, elem in bdf.elements.items()
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids
        }
        element_areas  = {}
        property_areas = {}
        for eid, elem in bdf.elements.items():
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids:
                area = elem.Area()
                element_areas[eid]  = area
                property_areas[elem.pid] = property_areas.get(elem.pid, 0.0) + area

        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            domain_to_subcase = self._read_domains(h5)
            stress_grp = h5.get('NASTRAN/RESULT/ELEMENTAL/STRESS')
            has_q4 = stress_grp is not None and 'QUAD4' in stress_grp
            has_t3 = stress_grp is not None and 'TRIA3' in stress_grp
            if has_q4:
                sq4 = stress_grp['QUAD4']
                sq4_dom = np.array(sq4['DOMAIN_ID']); sq4_eid = np.array(sq4['EID'])
                sq4_X1 = np.array(sq4['X1']); sq4_Y1 = np.array(sq4['Y1']); sq4_XY1 = np.array(sq4['XY1'])
                sq4_X2 = np.array(sq4['X2']); sq4_Y2 = np.array(sq4['Y2']); sq4_XY2 = np.array(sq4['XY2'])
            else:
                sq4_dom = sq4_eid = np.array([])
                sq4_X1 = sq4_Y1 = sq4_XY1 = sq4_X2 = sq4_Y2 = sq4_XY2 = np.array([])
            if has_t3:
                st3 = stress_grp['TRIA3']
                st3_dom = np.array(st3['DOMAIN_ID']); st3_eid = np.array(st3['EID'])
                st3_X1 = np.array(st3['X1']); st3_Y1 = np.array(st3['Y1']); st3_XY1 = np.array(st3['XY1'])
                st3_X2 = np.array(st3['X2']); st3_Y2 = np.array(st3['Y2']); st3_XY2 = np.array(st3['XY2'])
            else:
                st3_dom = st3_eid = np.array([])
                st3_X1 = st3_Y1 = st3_XY1 = st3_X2 = st3_Y2 = st3_XY2 = np.array([])
        self.logger.info('✓ H5 dosyası okundu')

        if not has_q4 and not has_t3:
            self.logger.info('⚠ Stress verisi bulunamadı (H5 içinde STRESS output yok)')
            return

        target_dids, target_sc = self._target_domains(domain_to_subcase, self.stress_lc_entry)
        self.logger.info(f'✓ {len(target_sc)} load case seçildi')

        self.logger.info('🔄 Stress verileri işleniyor...')
        element_stress_data = []
        property_stress = {
            lc_did: {
                pid: dict(sx1=0.,sy1=0.,sxy1=0.,vm1=0.,p1_1=0.,p2_1=0.,
                          sx2=0.,sy2=0.,sxy2=0.,vm2=0.,p1_2=0.,p2_2=0.)
                for pid in target_pids
            }
            for lc_did in target_dids
        }

        stress_sources = []
        if has_q4:
            stress_sources.append(('CQUAD4', sq4_dom, sq4_eid, sq4_X1, sq4_Y1, sq4_XY1, sq4_X2, sq4_Y2, sq4_XY2))
        if has_t3:
            stress_sources.append(('CTRIA3', st3_dom, st3_eid, st3_X1, st3_Y1, st3_XY1, st3_X2, st3_Y2, st3_XY2))

        for etype, dom_arr, eid_arr, X1a, Y1a, XY1a, X2a, Y2a, XY2a in stress_sources:
            for lc_did in np.unique(dom_arr):
                if int(lc_did) not in target_dids:
                    continue
                lc_mask = dom_arr == lc_did
                lc_eids = eid_arr[lc_mask]
                lc_X1 = X1a[lc_mask].copy(); lc_Y1 = Y1a[lc_mask].copy(); lc_XY1 = XY1a[lc_mask].copy()
                lc_X2 = X2a[lc_mask].copy(); lc_Y2 = Y2a[lc_mask].copy(); lc_XY2 = XY2a[lc_mask].copy()
                if is_material and thetarad_map:
                    thetas = np.array([thetarad_map.get(e, 0.0) for e in lc_eids.tolist()])
                    lc_X1, lc_Y1, lc_XY1 = transf_Mohr(lc_X1, lc_Y1, lc_XY1, thetas)
                    lc_X2, lc_Y2, lc_XY2 = transf_Mohr(lc_X2, lc_Y2, lc_XY2, thetas)
                lc_VM1  = np.sqrt(lc_X1**2 - lc_X1*lc_Y1 + lc_Y1**2 + 3*lc_XY1**2)
                lc_VM2  = np.sqrt(lc_X2**2 - lc_X2*lc_Y2 + lc_Y2**2 + 3*lc_XY2**2)
                ctr1    = (lc_X1 + lc_Y1) / 2.0
                R1      = np.sqrt(((lc_X1 - lc_Y1) / 2.0)**2 + lc_XY1**2)
                lc_P1_1 = ctr1 + R1;  lc_P2_1 = ctr1 - R1
                ctr2    = (lc_X2 + lc_Y2) / 2.0
                R2      = np.sqrt(((lc_X2 - lc_Y2) / 2.0)**2 + lc_XY2**2)
                lc_P1_2 = ctr2 + R2;  lc_P2_2 = ctr2 - R2
                lc_name    = domain_to_subcase.get(int(lc_did), int(lc_did))
                eid_to_idx = {int(e): i for i, e in enumerate(lc_eids)}
                ps_lc      = property_stress.get(lc_did, {})
                for eid, pid in elem_to_pid.items():
                    idx = eid_to_idx.get(eid)
                    if idx is None:
                        continue
                    area  = element_areas[eid]
                    sx1   = float(lc_X1[idx]);  sy1  = float(lc_Y1[idx]);  sxy1 = float(lc_XY1[idx])
                    vm1   = float(lc_VM1[idx]); p1_1 = float(lc_P1_1[idx]); p2_1 = float(lc_P2_1[idx])
                    sx2   = float(lc_X2[idx]);  sy2  = float(lc_Y2[idx]);  sxy2 = float(lc_XY2[idx])
                    vm2   = float(lc_VM2[idx]); p1_2 = float(lc_P1_2[idx]); p2_2 = float(lc_P2_2[idx])
                    if pid in ps_lc:
                        ps = ps_lc[pid]
                        ps['sx1']  += sx1*area;  ps['sy1']  += sy1*area;  ps['sxy1'] += sxy1*area
                        ps['vm1']  += vm1*area;  ps['p1_1'] += p1_1*area; ps['p2_1'] += p2_1*area
                        ps['sx2']  += sx2*area;  ps['sy2']  += sy2*area;  ps['sxy2'] += sxy2*area
                        ps['vm2']  += vm2*area;  ps['p1_2'] += p1_2*area; ps['p2_2'] += p2_2*area
                    element_stress_data.append({
                        'Property ID': pid, 'Element ID': eid, 'Element Type': etype, 'Load Case ID': lc_name,
                        'Sx_Z1': sx1, 'Sy_Z1': sy1, 'Sxy_Z1': sxy1, 'VM_Z1': vm1, 'P1_Z1': p1_1, 'P2_Z1': p2_1,
                        'Sx_Z2': sx2, 'Sy_Z2': sy2, 'Sxy_Z2': sxy2, 'VM_Z2': vm2, 'P1_Z2': p1_2, 'P2_Z2': p2_2,
                    })

        if not element_stress_data:
            self.logger.info('⚠ Seçili elementler için stress verisi bulunamadı')
            return

        df_stress = pd.DataFrame(element_stress_data)
        df_stress.to_csv(os.path.join(self.output_dir, 'Element_Stress.csv'), index=False)
        self.logger.info(f'✓ Element_Stress.csv yazıldı ({len(df_stress)} satır)')

        average_stress_data = []
        for lc_did, ps_dict in property_stress.items():
            lc_name = domain_to_subcase.get(int(lc_did), int(lc_did))
            for pid, ps in ps_dict.items():
                ta = property_areas.get(pid, 0.0)
                if ta == 0.0:
                    continue
                average_stress_data.append({
                    'Property ID': pid, 'Load Case ID': lc_name,
                    'Avg_Sx_Z1': ps['sx1']/ta, 'Avg_Sy_Z1': ps['sy1']/ta, 'Avg_Sxy_Z1': ps['sxy1']/ta,
                    'Avg_VM_Z1': ps['vm1']/ta, 'Avg_P1_Z1': ps['p1_1']/ta, 'Avg_P2_Z1': ps['p2_1']/ta,
                    'Avg_Sx_Z2': ps['sx2']/ta, 'Avg_Sy_Z2': ps['sy2']/ta, 'Avg_Sxy_Z2': ps['sxy2']/ta,
                    'Avg_VM_Z2': ps['vm2']/ta, 'Avg_P1_Z2': ps['p1_2']/ta, 'Avg_P2_Z2': ps['p2_2']/ta,
                })

        pd.DataFrame(average_stress_data).to_csv(os.path.join(self.output_dir, 'Average_Stress.csv'), index=False)
        self.logger.info(f'✓ Average_Stress.csv yazıldı ({len(average_stress_data)} satır)')

        self.logger.info('🔄 Element stress reduction hesaplanıyor (max VM)...')
        crit_e = extract_critical_stress(element_stress_data, 'Element ID')
        red_e  = [{'Property ID': r['Property ID'], 'Element ID': r['Element ID'],
                   'Element Type': r['Element Type'], 'Load Case ID': r['Load Case ID'],
                   'Sx_Z1': r['Sx_Z1'], 'Sy_Z1': r['Sy_Z1'], 'Sxy_Z1': r['Sxy_Z1'],
                   'VM_Z1': r['VM_Z1'], 'P1_Z1': r['P1_Z1'], 'P2_Z1': r['P2_Z1'],
                   'Sx_Z2': r['Sx_Z2'], 'Sy_Z2': r['Sy_Z2'], 'Sxy_Z2': r['Sxy_Z2'],
                   'VM_Z2': r['VM_Z2'], 'P1_Z2': r['P1_Z2'], 'P2_Z2': r['P2_Z2']} for r in crit_e]
        pd.DataFrame(red_e).to_csv(os.path.join(self.output_dir, 'Element_Stress_Reduced.csv'), index=False)
        self.logger.info(f'✓ Element_Stress_Reduced.csv yazıldı ({len(red_e)} kritik satır)')

        self.logger.info('🔄 Average stress reduction hesaplanıyor (max VM)...')
        crit_a = extract_critical_stress(average_stress_data, 'Property ID')
        red_a  = [{'Property ID': r['Property ID'], 'Load Case ID': r['Load Case ID'],
                   'Avg_Sx_Z1': r['Avg_Sx_Z1'], 'Avg_Sy_Z1': r['Avg_Sy_Z1'], 'Avg_Sxy_Z1': r['Avg_Sxy_Z1'],
                   'Avg_VM_Z1': r['Avg_VM_Z1'], 'Avg_P1_Z1': r['Avg_P1_Z1'], 'Avg_P2_Z1': r['Avg_P2_Z1'],
                   'Avg_Sx_Z2': r['Avg_Sx_Z2'], 'Avg_Sy_Z2': r['Avg_Sy_Z2'], 'Avg_Sxy_Z2': r['Avg_Sxy_Z2'],
                   'Avg_VM_Z2': r['Avg_VM_Z2'], 'Avg_P1_Z2': r['Avg_P1_Z2'], 'Avg_P2_Z2': r['Avg_P2_Z2']} for r in crit_a]
        pd.DataFrame(red_a).to_csv(os.path.join(self.output_dir, 'Average_Stress_Reduced.csv'), index=False)
        self.logger.info(f'✓ Average_Stress_Reduced.csv yazıldı ({len(red_a)} kritik satır)')

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLACEMENT EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_displacement(self):
        bdf = self._load_bdf()

        prop_str = self.disp_prop_entry.get().strip()
        all_pids = {e.pid for e in bdf.elements.values()
                    if e.type in ('CQUAD4', 'CTRIA3')}
        target_pids = set(parse_id_input(prop_str, list(all_pids)))
        if not target_pids:
            target_pids = all_pids
        self.logger.info(f'✓ {len(target_pids)} property ID seçildi')

        pid_to_nodes = {}
        for elem in bdf.elements.values():
            if elem.type in ('CQUAD4', 'CTRIA3') and elem.pid in target_pids:
                pid_to_nodes.setdefault(elem.pid, set()).update(elem.node_ids)

        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            domain_to_subcase = self._read_domains(h5)
            disp_ds   = h5['NASTRAN/RESULT/NODAL/DISPLACEMENT']
            d_dom     = np.array(disp_ds['DOMAIN_ID'])
            d_nid     = np.array(disp_ds['ID'])
            d_X       = np.array(disp_ds['X']);  d_Y  = np.array(disp_ds['Y']);  d_Z  = np.array(disp_ds['Z'])
            d_RX      = np.array(disp_ds['RX']); d_RY = np.array(disp_ds['RY']); d_RZ = np.array(disp_ds['RZ'])
        self.logger.info('✓ H5 dosyası okundu')

        target_dids, target_sc = self._target_domains(domain_to_subcase, self.disp_lc_entry)
        self.logger.info(f'✓ {len(target_sc)} load case seçildi')

        # Build flat (nid, pid) arrays once — enables vectorized indexing inside LC loop
        flat_nids = np.array([int(nid)
                               for pid, nodes in pid_to_nodes.items()
                               for nid in nodes], dtype=np.int64)
        flat_pids = np.array([pid
                               for pid, nodes in pid_to_nodes.items()
                               for nid in nodes], dtype=np.int64)

        self.logger.info('🔄 Displacement verileri işleniyor...')
        disp_chunks = []
        for lc_did in np.unique(d_dom):
            if int(lc_did) not in target_dids:
                continue
            lc_mask = d_dom == lc_did
            lc_nids = d_nid[lc_mask]
            lc_X    = d_X[lc_mask];  lc_Y  = d_Y[lc_mask];  lc_Z  = d_Z[lc_mask]
            lc_RX   = d_RX[lc_mask]; lc_RY = d_RY[lc_mask]; lc_RZ = d_RZ[lc_mask]
            lc_name = domain_to_subcase.get(int(lc_did), int(lc_did))

            # Build nid→row index dict, then vectorized lookup for all flat pairs
            nid_to_idx_lc = {int(n): i for i, n in enumerate(lc_nids)}
            v_idxs = np.array([nid_to_idx_lc.get(int(n), -1) for n in flat_nids])
            valid  = v_idxs >= 0
            if not valid.any():
                continue
            sel    = v_idxs[valid]
            x = lc_X[sel];  y = lc_Y[sel];  z = lc_Z[sel]
            rx = lc_RX[sel]; ry = lc_RY[sel]; rz = lc_RZ[sel]
            mag = np.sqrt(x**2 + y**2 + z**2)
            disp_chunks.append(pd.DataFrame({
                'Property ID':  flat_pids[valid],
                'Node ID':      flat_nids[valid],
                'Load Case ID': lc_name,
                'X': x, 'Y': y, 'Z': z,
                'Magnitude':    mag,
                'Rx': rx, 'Ry': ry, 'Rz': rz,
            }))

        df_all = pd.concat(disp_chunks, ignore_index=True) if disp_chunks else pd.DataFrame()
        df_all.to_csv(os.path.join(self.output_dir, 'Displacement_All.csv'), index=False)
        self.logger.info(f'✓ Displacement_All.csv yazıldı ({len(df_all)} satır)')

        self.logger.info('🔄 Displacement reduction hesaplanıyor (max Magnitude per node)...')
        if not df_all.empty:
            idx_max = df_all.groupby(['Property ID', 'Node ID'])['Magnitude'].idxmax()
            df_red  = df_all.loc[idx_max].sort_values(
                ['Property ID', 'Node ID', 'Load Case ID']).reset_index(drop=True)
        else:
            df_red = pd.DataFrame()
        df_red.to_csv(os.path.join(self.output_dir, 'Displacement_Reduced.csv'), index=False)
        self.logger.info(f'✓ Displacement_Reduced.csv yazıldı ({len(df_red)} kritik satır)')

    # ─────────────────────────────────────────────────────────────────────────
    # BUSH EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_bush(self):
        # ── Read H5 to discover all element IDs ───────────────────────────
        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            domain_to_subcase = self._read_domains(h5)
            cbush       = h5['NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/BUSH']
            dom_arr     = np.array(cbush['DOMAIN_ID'])
            eid_arr     = np.array(cbush['EID'])
            FX_arr      = np.array(cbush['FX'])
            FY_arr      = np.array(cbush['FY'])
            FZ_arr      = np.array(cbush['FZ'])
        self.logger.info('✓ H5 dosyası okundu')

        # ── Element ID filter ─────────────────────────────────────────────
        all_eids   = list({int(e) for e in eid_arr})
        elem_str   = self.bush_elem_entry.get().strip()
        target_eids = set(parse_id_input(elem_str, all_eids))
        if not target_eids:
            target_eids = set(all_eids)
        self.logger.info(f'✓ {len(target_eids)} element ID seçildi')

        # ── Load case filter ──────────────────────────────────────────────
        target_dids, target_sc = self._target_domains(domain_to_subcase,
                                                       self.bush_lc_entry)
        self.logger.info(f'✓ {len(target_sc)} load case seçildi')

        # ── Extract forces ────────────────────────────────────────────────
        self.logger.info('🔄 Bush force verileri çıkarılıyor...')
        bush_data = []

        for lc_did in np.unique(dom_arr):
            if int(lc_did) not in target_dids:
                continue
            lc_mask    = dom_arr == lc_did
            lc_eids    = eid_arr[lc_mask]
            lc_FX      = FX_arr[lc_mask]
            lc_FY      = FY_arr[lc_mask]
            lc_FZ      = FZ_arr[lc_mask]
            lc_name    = domain_to_subcase.get(int(lc_did), int(lc_did))
            eid_to_idx = {int(e): i for i, e in enumerate(lc_eids)}

            for eid in target_eids:
                idx = eid_to_idx.get(eid)
                if idx is None:
                    continue
                bush_data.append({
                    'Element ID':   eid,
                    'Load Case ID': lc_name,
                    'FX': float(lc_FX[idx]),
                    'FY': float(lc_FY[idx]),
                    'FZ': float(lc_FZ[idx]),
                })

        # ── Bush_Load_Raw.csv ─────────────────────────────────────────────
        df_raw = pd.DataFrame(bush_data)
        p = os.path.join(self.output_dir, 'Bush_Load_Raw.csv')
        df_raw.to_csv(p, index=False)
        self.logger.info(f'✓ Bush_Load_Raw.csv yazıldı ({len(df_raw)} satır)')

        # ── Bush_Load_Reduced.csv (18 metrik) ─────────────────────────────
        self.logger.info('🔄 Bush reduction hesaplanıyor (18 metrik)...')
        critical = extract_critical_rows(bush_data)
        reduced  = [{
            'Element ID':   r['Element ID'],
            'Load Case ID': r['Load Case ID'],
            'FX': r['_fx'], 'FY': r['_fy'], 'FZ': r['_fz'],
        } for r in critical]
        df_red = pd.DataFrame(reduced)
        p = os.path.join(self.output_dir, 'Bush_Load_Reduced.csv')
        df_red.to_csv(p, index=False)
        self.logger.info(f'✓ Bush_Load_Reduced.csv yazıldı ({len(df_red)} kritik satır)')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    app = LoadExtractionApp(root)
    root.mainloop()

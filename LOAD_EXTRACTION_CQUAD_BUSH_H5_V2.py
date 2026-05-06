#!/usr/bin/env python3
"""
Load Extraction Tool - Nastran H5 Analysis Application
Extracts and analyzes FEA loads from MSC Nastran H5 files
Supports Element CID and Material CID coordinate systems
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
from datetime import datetime


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
    """Angle between two sets of vectors (row-wise)."""
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_val = np.clip((v1 * v2).sum(axis=1) / denom, -1.0, 1.0)
    return np.arccos(cos_val)


def calc_imat(normals, csysi):
    """Material i-vector from element normals and coord-system i-vector."""
    jmat = np.cross(normals, csysi)
    norms = np.linalg.norm(jmat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    jmat = jmat / norms
    return np.cross(jmat, normals)


def compute_thetarad_from_bdf(bdf):
    """
    Compute per-element rotation angles (rad) for material coord transformation.
    Works for CQUAD4 elements with THETA or MCID.
    Returns dict {element_id: thetarad}.
    """
    eids  = list(bdf.elements.keys())
    elems = list(bdf.elements.values())
    n = len(elems)
    if n == 0:
        return {}

    is_mcid_arr = np.array([
        isinstance(getattr(e, 'theta_mcid', None), integer_types)
        for e in elems
    ])
    elem_type_arr = np.array([e.type for e in elems])

    # Initialize theta angles from element THETA field
    thetarad = np.zeros(n, dtype=float)
    for i, e in enumerate(elems):
        if not is_mcid_arr[i]:
            t = getattr(e, 'theta_mcid', None)
            if isinstance(t, float):
                thetarad[i] = np.deg2rad(t)

    for quad_type in ('CQUAD4', 'CQUAD8', 'CQUADR'):
        # ── THETA elements ─────────────────────────────────────────────────
        idx_theta = np.where(~is_mcid_arr & (elem_type_arr == quad_type))[0]
        if len(idx_theta):
            qelems  = [elems[i] for i in idx_theta]
            corner  = np.array([e.get_node_positions() for e in qelems])
            g1, g2, g3, g4 = corner[:,0], corner[:,1], corner[:,2], corner[:,3]
            beta    = angle2vec(g3 - g1, g2 - g1)
            gamma   = angle2vec(g4 - g2, g1 - g2)
            alpha   = (beta + gamma) / 2.0
            thetarad[idx_theta] += alpha - beta

        # ── MCID elements ──────────────────────────────────────────────────
        idx_mcid = np.where(is_mcid_arr & (elem_type_arr == quad_type))[0]
        if len(idx_mcid):
            qelems  = [elems[i] for i in idx_mcid]
            corner  = np.array([e.get_node_positions() for e in qelems])
            g1, g2, g3, g4 = corner[:,0], corner[:,1], corner[:,2], corner[:,3]
            normals = np.array([e.Normal()                      for e in qelems])
            csysi   = np.array([bdf.coords[e.theta_mcid].i     for e in qelems])
            imat    = calc_imat(normals, csysi)
            angles  = angle2vec(g2 - g1, imat)
            sign    = np.sign((np.cross(g2 - g1, imat) * normals).sum(axis=1))
            beta    = angle2vec(g3 - g1, g2 - g1)
            gamma   = angle2vec(g4 - g2, g1 - g2)
            alpha   = (beta + gamma) / 2.0
            thetarad[idx_mcid] = angles * sign + alpha - beta

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
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
        self.text_widget.update()


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
        self.root.geometry('1100x700')
        self.root.minsize(920, 580)
        self.root.configure(bg=self.COLORS['bg'])

        self.logger = logging.getLogger('LoadExtractionH5')
        self.logger.setLevel(logging.INFO)

        # File paths
        self.bdf_path        = ''
        self.h5_path         = ''
        self.prop_csv_path   = ''
        self.bush_csv_path   = ''
        self.output_dir      = ''

        self.extraction_type   = tk.StringVar(value='PSHELL ALL AVERAGE')
        self.coordinate_system = tk.StringVar(value='Element CID')

        self.build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

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
        tk.Label(hdr, text='MSC Nastran H5 Analysis',
                 font=('Segoe UI', 9),
                 bg='#1c2128', fg=self.COLORS['muted']).pack(side='right', padx=22)

    def _build_tab_bar(self):
        bar = tk.Frame(self.root, bg=self.COLORS['surface'])
        bar.pack(fill='x')
        tk.Frame(bar, bg=self.COLORS['border'], height=1).pack(fill='x', side='bottom')
        self.tab_btns = {}
        for mode in ['PSHELL ALL AVERAGE', 'BUSH LOAD']:
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
        tk.Button(inner, text='▶  RUN ANALYSIS', command=self.asc_run,
                  bg=self.COLORS['success'], fg='white',
                  font=('Segoe UI', 11, 'bold'), relief='flat',
                  cursor='hand2', activebackground='#2ea043',
                  pady=9).pack(side='left', fill='x', expand=True, padx=(0, 10))
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

        # ── LEFT: Files card ──────────────────────────────────────────────
        left = tk.Frame(main, bg=self.COLORS['bg'])
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))

        files_card = self._card(left, '📁  Files & Output')
        files_card.pack(fill='x')

        self.bdf_widget = self._file_row(files_card, '📄  BDF File',        self._browse_bdf)
        self.h5_widget  = self._file_row(files_card, '📊  H5 File',         self._browse_h5)
        self.out_widget = self._file_row(files_card, '📁  Output Directory', self._browse_output,
                                         is_dir=True)
        tk.Frame(files_card, height=6, bg=self.COLORS['surface']).pack()

        # ── RIGHT: Mode-specific parameters ──────────────────────────────
        right = tk.Frame(main, bg=self.COLORS['bg'], width=330)
        right.pack(side='left', fill='y', padx=(10, 0))
        right.pack_propagate(False)

        # PSHELL parameter panel
        self.pshell_pf = tk.Frame(right, bg=self.COLORS['bg'])

        # Property CSV selector
        prop_card = self._card(self.pshell_pf, '📋  Shell Property CSV')
        prop_card.pack(fill='x', pady=(0, 10))
        self.prop_csv_widget = self._file_row(
            prop_card, 'CSV with "Property ID" column', self._browse_prop_csv)
        tk.Frame(prop_card, height=4, bg=self.COLORS['surface']).pack()

        # Coordinate system
        coord_card = self._card(self.pshell_pf, '🔄  Coordinate System')
        coord_card.pack(fill='x')
        for opt in ['Element CID', 'Material CID']:
            tk.Radiobutton(coord_card, text=f'  {opt}',
                          variable=self.coordinate_system, value=opt,
                          bg=self.COLORS['surface'], fg=self.COLORS['text'],
                          selectcolor=self.COLORS['accent'],
                          activebackground=self.COLORS['surface'],
                          font=('Segoe UI', 10), cursor='hand2'
                          ).pack(anchor='w', padx=12, pady=4)
        tk.Label(coord_card,
                 text='  Material CID requires THETA or MCID in BDF',
                 font=('Segoe UI', 8), bg=self.COLORS['surface'],
                 fg=self.COLORS['muted']).pack(anchor='w', padx=14, pady=(0, 8))

        # BUSH parameter panel
        self.bush_pf = tk.Frame(right, bg=self.COLORS['bg'])

        bush_card = self._card(self.bush_pf, '🔧  Bush Element CSV')
        bush_card.pack(fill='x')
        self.bush_csv_widget = self._file_row(
            bush_card, 'CSV with "Element ID" column', self._browse_bush_csv)
        tk.Frame(bush_card, height=4, bg=self.COLORS['surface']).pack()

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

    # ─────────────────────────────────────────────────────────────────────────
    # TAB CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    def _select_tab(self, mode):
        self.extraction_type.set(mode)
        for m, btn in self.tab_btns.items():
            active = (m == mode)
            btn.config(
                bg=self.COLORS['bg']      if active else self.COLORS['surface'],
                fg=self.COLORS['accent']  if active else self.COLORS['muted'])
        self.pshell_pf.pack_forget()
        self.bush_pf.pack_forget()
        if mode == 'PSHELL ALL AVERAGE':
            self.pshell_pf.pack(fill='both', expand=True)
        else:
            self.bush_pf.pack(fill='both', expand=True)

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

    def _browse_prop_csv(self):
        path = filedialog.askopenfilename(title='Select Property CSV',
                                          filetypes=[('CSV Files', '*.csv')])
        if path:
            self.prop_csv_path = path
            self.prop_csv_widget.delete(0, tk.END)
            self.prop_csv_widget.insert(0, f'✓  {os.path.basename(path)}')

    def _browse_bush_csv(self):
        path = filedialog.askopenfilename(title='Select Bush Element CSV',
                                          filetypes=[('CSV Files', '*.csv')])
        if path:
            self.bush_csv_path = path
            self.bush_csv_widget.delete(0, tk.END)
            self.bush_csv_widget.insert(0, f'✓  {os.path.basename(path)}')

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

        if not self.output_dir:
            messagebox.showerror('Hata', 'Çıktı klasörünü seçin!')
            return
        if not self.bdf_path or not self.h5_path:
            messagebox.showerror('Hata', 'BDF ve H5 dosyalarını seçin!')
            return

        log_path = os.path.join(
            self.output_dir,
            f'LoadExtraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh = logging.FileHandler(log_path)
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
            else:
                self.run_bush()
        except Exception as e:
            self.logger.error(f'HATA: {e}')
            messagebox.showerror('Hata', str(e))
            return

        elapsed = time.time() - start
        self.logger.info('=' * 60)
        self.logger.info(f'✅ İşlem tamamlandı! ({elapsed:.2f} saniye)')
        self.logger.info(f'📁 Çıktılar: {self.output_dir}')
        self.logger.info('=' * 60)
        messagebox.showinfo('Başarılı', f'İşlem Tamamlandı\nSüre: {elapsed:.2f} saniye')

    # ─────────────────────────────────────────────────────────────────────────
    # PSHELL EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_pshell(self):
        if not self.prop_csv_path:
            raise ValueError('Property CSV dosyasını seçin!')

        # ── Load property IDs from CSV ─────────────────────────────────────
        df_prop = pd.read_csv(self.prop_csv_path)
        target_property_ids = set(df_prop['Property ID'].tolist())
        self.logger.info(f'✓ {len(target_property_ids)} property ID okundu (CSV)')

        # ── Read BDF ──────────────────────────────────────────────────────
        self.logger.info('📂 BDF dosyası okunuyor...')
        bdf = BDF()
        bdf.read_bdf(self.bdf_path, encoding='latin1')
        self.logger.info('✓ BDF dosyası okundu')

        # ── Coordinate system ─────────────────────────────────────────────
        is_material_cid = self.coordinate_system.get() == 'Material CID'
        self.logger.info(f'🔄 Koordinat sistemi: {self.coordinate_system.get()}')

        thetarad_map = {}
        if is_material_cid:
            self.logger.info('🔄 BDF\'den material açıları hesaplanıyor...')
            thetarad_map = compute_thetarad_from_bdf(bdf)
            self.logger.info(f'✓ {len(thetarad_map)} element için açı hesaplandı')

        # ── Read H5 ───────────────────────────────────────────────────────
        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            cquad4      = h5['NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/QUAD4']
            domain_ids  = np.array(cquad4['DOMAIN_ID'])
            element_ids = np.array(cquad4['EID'])
            MX          = np.array(cquad4['MX'])
            MY          = np.array(cquad4['MY'])
            MXY         = np.array(cquad4['MXY'])

            domains  = h5['NASTRAN/RESULT/DOMAINS']
            lc_ids   = np.array(domains['ID'])
            subcases = np.array(domains['SUBCASE'])
        self.logger.info('✓ H5 dosyası okundu')

        lc_name_map = {lid: sc for lid, sc in zip(lc_ids, subcases)}

        # ── Element → property mapping ────────────────────────────────────
        elem_to_pid = {
            eid: elem.pid
            for eid, elem in bdf.elements.items()
            if elem.type == 'CQUAD4' and elem.pid in target_property_ids
        }

        # ── Area calculation ──────────────────────────────────────────────
        self.logger.info('🔄 Element alanları hesaplanıyor...')
        element_areas  = {}
        property_areas = {}
        for eid, elem in bdf.elements.items():
            if elem.type == 'CQUAD4' and elem.pid in target_property_ids:
                area = elem.Area()
                element_areas[eid] = area
                property_areas[elem.pid] = property_areas.get(elem.pid, 0.0) + area
        self.logger.info(f'✓ {len(element_areas)} element alanı hesaplandı')

        # ── Process load cases ────────────────────────────────────────────
        self.logger.info('🔄 Element forces işleniyor...')
        element_base_data = []
        property_forces   = {}   # {lc_id: {pid: {Nx, Ny, Nxy}}}

        for lc_id in np.unique(domain_ids):
            lc_mask  = domain_ids == lc_id
            lc_eids  = element_ids[lc_mask]
            lc_MX    = MX[lc_mask].copy()
            lc_MY    = MY[lc_mask].copy()
            lc_MXY   = MXY[lc_mask].copy()

            # Apply material coordinate transformation if requested
            if is_material_cid and thetarad_map:
                thetas = np.array([thetarad_map.get(int(eid), 0.0) for eid in lc_eids])
                lc_MX, lc_MY, lc_MXY = transf_Mohr(lc_MX, lc_MY, lc_MXY, thetas)

            lc_name = lc_name_map.get(lc_id, f'Unknown_{lc_id}')

            # Build element index for fast lookup
            eid_to_idx = {int(eid): idx for idx, eid in enumerate(lc_eids)}

            pf = property_forces.setdefault(lc_id, {
                pid: {'Nx': 0.0, 'Ny': 0.0, 'Nxy': 0.0}
                for pid in target_property_ids
            })

            for eid, pid in elem_to_pid.items():
                idx = eid_to_idx.get(eid)
                if idx is None:
                    continue
                nx  = float(lc_MX[idx])
                ny  = float(lc_MY[idx])
                nxy = float(lc_MXY[idx])
                area = element_areas[eid]

                pf[pid]['Nx']  += nx  * area
                pf[pid]['Ny']  += ny  * area
                pf[pid]['Nxy'] += nxy * area

                element_base_data.append({
                    'Property ID':  pid,
                    'Element ID':   eid,
                    'Load Case ID': lc_name,
                    'Nx':  nx,
                    'Ny':  ny,
                    'Nxy': nxy,
                    'Area': area,
                })

        # ── Write Element_Load.csv ─────────────────────────────────────────
        df_elem = pd.DataFrame(element_base_data)
        out_elem = os.path.join(self.output_dir, 'Element_Load.csv')
        df_elem.to_csv(out_elem, index=False)
        self.logger.info(f'✓ Element_Load.csv yazıldı ({len(df_elem)} satır)')

        # ── Compute and write Average_Load.csv ────────────────────────────
        average_data = []
        for lc_id, pf in property_forces.items():
            lc_name = lc_name_map.get(lc_id, f'Unknown_{lc_id}')
            for pid, forces in pf.items():
                total_area = property_areas.get(pid, 1.0)
                average_data.append({
                    'Property ID':   pid,
                    'Load Case ID':  lc_name,
                    'Average Nx':    forces['Nx']  / total_area,
                    'Average Ny':    forces['Ny']  / total_area,
                    'Average Nxy':   forces['Nxy'] / total_area,
                    'Average Area':  total_area,
                })
        df_avg = pd.DataFrame(average_data)
        out_avg = os.path.join(self.output_dir, 'Average_Load.csv')
        df_avg.to_csv(out_avg, index=False)
        self.logger.info(f'✓ Average_Load.csv yazıldı ({len(df_avg)} satır)')

    # ─────────────────────────────────────────────────────────────────────────
    # BUSH EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_bush(self):
        if not self.bush_csv_path:
            raise ValueError('Bush Element CSV dosyasını seçin!')

        # ── Load element IDs from CSV ─────────────────────────────────────
        df_bush = pd.read_csv(self.bush_csv_path)
        target_eids = set(df_bush['Element ID'].tolist())
        self.logger.info(f'✓ {len(target_eids)} bush element ID okundu (CSV)')

        # ── Read H5 ───────────────────────────────────────────────────────
        self.logger.info('📂 H5 dosyası okunuyor...')
        with h5py.File(self.h5_path, 'r') as h5:
            cbush       = h5['NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/BUSH']
            domain_ids  = np.array(cbush['DOMAIN_ID'])
            element_ids = np.array(cbush['EID'])
            FX          = np.array(cbush['FX'])
            FY          = np.array(cbush['FY'])
            FZ          = np.array(cbush['FZ'])

            domains  = h5['NASTRAN/RESULT/DOMAINS']
            lc_ids   = np.array(domains['ID'])
            subcases = np.array(domains['SUBCASE'])
        self.logger.info('✓ H5 dosyası okundu')

        lc_name_map = {lid: sc for lid, sc in zip(lc_ids, subcases)}

        # ── Process load cases ────────────────────────────────────────────
        self.logger.info('🔄 Bush force verileri çıkarılıyor...')
        bush_data = []

        for lc_id in np.unique(domain_ids):
            lc_mask  = domain_ids == lc_id
            lc_eids  = element_ids[lc_mask]
            lc_FX    = FX[lc_mask]
            lc_FY    = FY[lc_mask]
            lc_FZ    = FZ[lc_mask]
            lc_name  = lc_name_map.get(lc_id, f'Unknown_{lc_id}')

            eid_to_idx = {int(eid): idx for idx, eid in enumerate(lc_eids)}

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

        df_bush_out = pd.DataFrame(bush_data)
        out_bush = os.path.join(self.output_dir, 'Bush_Load.csv')
        df_bush_out.to_csv(out_bush, index=False)
        self.logger.info(f'✓ Bush_Load.csv yazıldı ({len(df_bush_out)} satır)')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    app = LoadExtractionApp(root)
    root.mainloop()

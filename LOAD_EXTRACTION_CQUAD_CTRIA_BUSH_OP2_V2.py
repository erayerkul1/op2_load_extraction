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
from datetime import datetime
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
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
        self.text_widget.update()

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_critical_rows(raw_data):
    enriched = {}
    eid_lcs  = {}
    for row in raw_data:
        eid = row["Element ID"]
        lc  = row["Load Case ID"]
        key = (eid, lc)
        if key in enriched:
            continue
        try:
            fx, fy, fz = float(row["FX"]), float(row["FY"]), float(row["FZ"])
        except (ValueError, TypeError):
            fx = fy = fz = 0.0
        vals = {m["id"]: m["fn"](fx, fy, fz) for m in METRICS}
        r = {**row, "_fx": fx, "_fy": fy, "_fz": fz, "_vals": vals, "_metrics": set()}
        enriched[key] = r
        eid_lcs.setdefault(eid, []).append(r)

    for eid, rows in eid_lcs.items():
        for m in METRICS:
            mid = m["id"]
            best = max(rows, key=lambda r, mid=mid: r["_vals"][mid])
            best["_metrics"].add(mid)

    result = [r for r in enriched.values() if r["_metrics"]]
    result.sort(key=lambda r: (r["Element ID"], r["Load Case ID"]))
    return result

def extract_critical_pshell(raw_data, group_key, nx_key, ny_key, nxy_key):
    enriched  = {}
    group_lcs = {}
    for row in raw_data:
        gid = row[group_key]
        lc  = row['Load Case ID']
        key = (gid, lc)
        if key in enriched:
            continue
        try:
            nx  = float(row[nx_key])
            ny  = float(row[ny_key])
            nxy = float(row[nxy_key])
        except (ValueError, TypeError):
            nx = ny = nxy = 0.0
        vals = {m["id"]: m["fn"](nx, ny, nxy) for m in PSHELL_METRICS}
        r = {**row, "_nx": nx, "_ny": ny, "_nxy": nxy, "_vals": vals, "_metrics": set()}
        enriched[key] = r
        group_lcs.setdefault(gid, []).append(r)

    for gid, rows in group_lcs.items():
        for m in PSHELL_METRICS:
            mid  = m["id"]
            best = max(rows, key=lambda r, mid=mid: r["_vals"][mid])
            best["_metrics"].add(mid)

    result = [r for r in enriched.values() if r["_metrics"]]
    result.sort(key=lambda r: (r[group_key], r['Load Case ID']))
    return result

def extract_critical_stress(stress_data, group_key):
    vm_z1_key = 'VM_Z1'  if group_key == 'Element ID' else 'Avg_VM_Z1'
    vm_z2_key = 'VM_Z2'  if group_key == 'Element ID' else 'Avg_VM_Z2'
    enriched  = {}
    group_lcs = {}
    for row in stress_data:
        gid = row[group_key]
        lc  = row['Load Case ID']
        key = (gid, lc)
        if key in enriched:
            continue
        vm_max = max(abs(float(row.get(vm_z1_key, 0))), abs(float(row.get(vm_z2_key, 0))))
        r = {**row, '_vm_max': vm_max, '_selected': False}
        enriched[key] = r
        group_lcs.setdefault(gid, []).append(r)
    for gid, rows in group_lcs.items():
        max(rows, key=lambda r: r['_vm_max'])['_selected'] = True
    result = [r for r in enriched.values() if r['_selected']]
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

        self.build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

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
        if mode == 'PSHELL ALL AVERAGE':
            self.pshell_pf.pack(fill='both', expand=True)
        else:
            self.bush_pf.pack(fill='both', expand=True)

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

        # Sync widget values before running
        self.pshell_property_ids = self.property_id_entry.get().strip()
        self.bush_element_ids    = self.bush_element_id_entry.get().strip()

        if not self.stress_output_now2:
            messagebox.showerror("Hata", "Çıktı klasörünü seçin!")
            return

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

        if not self.input_entry_now or not self.output_entry_now:
            self.logger.error("Gerekli dosyalar seçilmedi!")
            messagebox.showerror("Hata", "BDF ve OP2 dosyalarını seçin")
            return

        if self.extraction_type.get() == "PSHELL ALL AVERAGE":
            self.run_pshell()
        else:
            self.run_bush()

        elapsed = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"✅ İşlem tamamlandı! ({elapsed:.2f} saniye)")
        self.logger.info(f"📁 Çıktılar: {self.stress_output_now2}")
        self.logger.info("="*60)
        messagebox.showinfo("Başarılı", f"İşlem Tamamlandı\nSüre: {elapsed:.2f} saniye")

    # ─────────────────────────────────────────────────────────────────────────
    # PSHELL EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_pshell(self):
        if not self.pshell_property_ids:
            self.logger.error("Property ID'leri girin!")
            messagebox.showerror("Hata", "Property ID'leri girin (tüm için: ALL)")
            return

        self.logger.info("📂 OP2 ve BDF dosyaları okunuyor...")
        op2 = OP2()
        bdf = BDF()
        op2.read_op2(self.output_entry_now)
        self.logger.info("✓ OP2 dosyası okundu")
        bdf.read_bdf(self.input_entry_now, encoding='latin1')
        self.logger.info("✓ BDF dosyası okundu")

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

        element_base_data = []

        is_material_cid = self.coordinate_system.get() == "Material CID"
        coord_mode = "Material CID" if is_material_cid else "Element CID"
        self.logger.info(f"🔄 Koordinat sistemi: {coord_mode}")
        op2_data = data_in_material_coord(bdf, op2, in_place=False) if is_material_cid else op2
        self.logger.info("✓ Koordinat dönüşümü tamamlandı" if is_material_cid else "✓ Element CID kullanılacak")

        self.logger.info("🔄 Element forces işleniyor...")
        for load_case_id, element_forces in op2_data.cquad4_force.items():
            if load_case_id not in target_lc_ids:
                continue
            element_ids = element_forces.element
            forces_data = element_forces.data[0]
            load_ids = element_forces.loadIDs[0]
            for element_id, element_property_id in elements_with_properties.items():
                if element_id in element_ids:
                    index = np.where(element_ids == element_id)[0][0]
                    f = forces_data[index]
                    area = element_areas[element_id]
                    pf = property_forces[load_case_id][element_property_id]
                    pf['Nx'] += f[0]*area; pf['Ny'] += f[1]*area; pf['Nxy'] += f[2]*area
                    pf['Mx'] += f[3]*area; pf['My'] += f[4]*area; pf['Mxy'] += f[5]*area
                    element_base_data.append({
                        'Property ID': element_property_id,
                        'Element ID':  element_id,
                        'Load Case ID': load_ids,
                        'Nx': f[0], 'Ny': f[1], 'Nxy': f[2],
                        'Mx': f[3], 'My': f[4], 'Mxy': f[5],
                        'Thickness': property_thickness[element_property_id],
                    })

        for load_case_id, element_forces in op2_data.ctria3_force.items():
            if load_case_id not in target_lc_ids:
                continue
            element_ids = element_forces.element
            forces_data = element_forces.data[0]
            load_ids = element_forces.loadIDs[0]
            for element_id, element_property_id in elements_with_properties.items():
                if element_id in element_ids:
                    index = np.where(element_ids == element_id)[0][0]
                    f = forces_data[index]
                    area = element_areas[element_id]
                    pf = property_forces[load_case_id][element_property_id]
                    pf['Nx'] += f[0]*area; pf['Ny'] += f[1]*area; pf['Nxy'] += f[2]*area
                    pf['Mx'] += f[3]*area; pf['My'] += f[4]*area; pf['Mxy'] += f[5]*area
                    element_base_data.append({
                        'Property ID': element_property_id,
                        'Element ID':  element_id,
                        'Load Case ID': load_ids,
                        'Nx': f[0], 'Ny': f[1], 'Nxy': f[2],
                        'Mx': f[3], 'My': f[4], 'Mxy': f[5],
                        'Thickness': property_thickness[element_property_id],
                    })

        df = pd.DataFrame(element_base_data)
        output_csv = os.path.join(self.stress_output_now2, 'Element_Load.csv')
        df.to_csv(output_csv, index=False)
        self.logger.info(f"✓ Element_Load.csv yazıldı ({len(df)} satır)")

        Average_forces = []
        for load_case_id, force_by_property in property_forces.items():
            if load_case_id not in target_lc_ids:
                continue
            for property_id, forces in force_by_property.items():
                total_area = property_areas[property_id]
                Average_forces.append({
                    'Property ID':  property_id,
                    'Load Case ID': load_case_id,
                    'Average Nx':  forces['Nx']  / total_area,
                    'Average Ny':  forces['Ny']  / total_area,
                    'Average Nxy': forces['Nxy'] / total_area,
                    'Average Mx':  forces['Mx']  / total_area,
                    'Average My':  forces['My']  / total_area,
                    'Average Mxy': forces['Mxy'] / total_area,
                    'Thickness':   property_thickness[property_id],
                })

        df2 = pd.DataFrame(Average_forces)
        output_csv2 = os.path.join(self.stress_output_now2, 'Average_Load.csv')
        df2.to_csv(output_csv2, index=False)
        self.logger.info(f"✓ Average_Load.csv yazıldı ({len(df2)} satır)")

        self.logger.info("🔄 Element reduction hesaplanıyor (16 metrik)...")
        critical_elem = extract_critical_pshell(element_base_data, 'Element ID', 'Nx', 'Ny', 'Nxy')
        reduced_elem = [{
            'Property ID':  r['Property ID'],
            'Element ID':   r['Element ID'],
            'Load Case ID': r['Load Case ID'],
            'Nx': r['_nx'], 'Ny': r['_ny'], 'Nxy': r['_nxy'],
            'Mx': r['Mx'],  'My': r['My'],  'Mxy': r['Mxy'],
            'Thickness': r['Thickness'],
        } for r in critical_elem]
        df_elem_reduced = pd.DataFrame(reduced_elem)
        output_csv_elem_reduced = os.path.join(self.stress_output_now2, 'Element_Load_Reduced.csv')
        df_elem_reduced.to_csv(output_csv_elem_reduced, index=False)
        self.logger.info(f"✓ Element_Load_Reduced.csv yazıldı ({len(df_elem_reduced)} kritik satır)")

        self.logger.info("🔄 Average reduction hesaplanıyor (16 metrik)...")
        critical_avg = extract_critical_pshell(Average_forces, 'Property ID', 'Average Nx', 'Average Ny', 'Average Nxy')
        reduced_avg = [{
            'Property ID':  r['Property ID'],
            'Load Case ID': r['Load Case ID'],
            'Average Nx':  r['_nx'], 'Average Ny':  r['_ny'], 'Average Nxy': r['_nxy'],
            'Average Mx':  r['Average Mx'],
            'Average My':  r['Average My'],
            'Average Mxy': r['Average Mxy'],
            'Thickness':   r['Thickness'],
        } for r in critical_avg]
        df_avg_reduced = pd.DataFrame(reduced_avg)
        output_csv_avg_reduced = os.path.join(self.stress_output_now2, 'Average_Load_Reduced.csv')
        df_avg_reduced.to_csv(output_csv_avg_reduced, index=False)
        self.logger.info(f"✓ Average_Load_Reduced.csv yazıldı ({len(df_avg_reduced)} kritik satır)")

        # ── STRESS SECTION ────────────────────────────────────────────────
        self.logger.info("🔄 Stress verileri işleniyor...")
        element_stress_data = []
        property_stress = {
            lc: {
                pid: dict(sx1=0.,sy1=0.,sxy1=0.,vm1=0.,p1_1=0.,p2_1=0.,
                          sx2=0.,sy2=0.,sxy2=0.,vm2=0.,p1_2=0.,p2_2=0.)
                for pid in target_property_ids
            }
            for lc in target_lc_ids
        }

        stress_sources = []
        if hasattr(op2_data, 'cquad4_stress') and op2_data.cquad4_stress:
            stress_sources.append(op2_data.cquad4_stress)
        if hasattr(op2_data, 'ctria3_stress') and op2_data.ctria3_stress:
            stress_sources.append(op2_data.ctria3_stress)

        for stress_dict in stress_sources:
            for load_case_id, stress_result in stress_dict.items():
                if load_case_id not in target_lc_ids:
                    continue
                eids_s   = stress_result.element_node[:, 0].astype(int)
                sdata    = stress_result.data[0]  # (ntotal,8): [fd,oxx,oyy,txy,angle,omax,omin,vm]
                load_ids = stress_result.loadIDs[0]
                eid_to_sidx = {}
                for i, eid in enumerate(eids_s):
                    eid_to_sidx.setdefault(int(eid), []).append(i)
                for element_id, pid in elements_with_properties.items():
                    if element_id not in eid_to_sidx:
                        continue
                    idxs = eid_to_sidx[element_id]
                    if len(idxs) < 2:
                        continue
                    z1   = sdata[idxs[0]]
                    z2   = sdata[idxs[1]]
                    area = element_areas.get(element_id, 0.0)
                    ps   = property_stress[load_case_id][pid]
                    ps['sx1']  += z1[1]*area; ps['sy1']  += z1[2]*area; ps['sxy1'] += z1[3]*area
                    ps['vm1']  += z1[7]*area; ps['p1_1'] += z1[5]*area; ps['p2_1'] += z1[6]*area
                    ps['sx2']  += z2[1]*area; ps['sy2']  += z2[2]*area; ps['sxy2'] += z2[3]*area
                    ps['vm2']  += z2[7]*area; ps['p1_2'] += z2[5]*area; ps['p2_2'] += z2[6]*area
                    element_stress_data.append({
                        'Property ID': pid,
                        'Element ID':  element_id,
                        'Load Case ID': load_ids,
                        'Sx_Z1':  z1[1], 'Sy_Z1':  z1[2], 'Sxy_Z1': z1[3],
                        'VM_Z1':  z1[7], 'P1_Z1':  z1[5], 'P2_Z1':  z1[6],
                        'Sx_Z2':  z2[1], 'Sy_Z2':  z2[2], 'Sxy_Z2': z2[3],
                        'VM_Z2':  z2[7], 'P1_Z2':  z2[5], 'P2_Z2':  z2[6],
                    })

        if element_stress_data:
            df_stress = pd.DataFrame(element_stress_data)
            df_stress.to_csv(os.path.join(self.stress_output_now2, 'Element_Stress.csv'), index=False)
            self.logger.info(f"✓ Element_Stress.csv yazıldı ({len(df_stress)} satır)")

            average_stress_data = []
            for lc in target_lc_ids:
                if lc not in property_stress:
                    continue
                for pid in target_property_ids:
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
            critical_stress_elem = extract_critical_stress(element_stress_data, 'Element ID')
            reduced_stress_elem = [{
                'Property ID':  r['Property ID'],
                'Element ID':   r['Element ID'],
                'Load Case ID': r['Load Case ID'],
                'Sx_Z1':  r['Sx_Z1'],  'Sy_Z1':  r['Sy_Z1'],  'Sxy_Z1': r['Sxy_Z1'],
                'VM_Z1':  r['VM_Z1'],  'P1_Z1':  r['P1_Z1'],  'P2_Z1':  r['P2_Z1'],
                'Sx_Z2':  r['Sx_Z2'],  'Sy_Z2':  r['Sy_Z2'],  'Sxy_Z2': r['Sxy_Z2'],
                'VM_Z2':  r['VM_Z2'],  'P1_Z2':  r['P1_Z2'],  'P2_Z2':  r['P2_Z2'],
            } for r in critical_stress_elem]
            df_stress_reduced = pd.DataFrame(reduced_stress_elem)
            df_stress_reduced.to_csv(os.path.join(self.stress_output_now2, 'Element_Stress_Reduced.csv'), index=False)
            self.logger.info(f"✓ Element_Stress_Reduced.csv yazıldı ({len(df_stress_reduced)} kritik satır)")

            self.logger.info("🔄 Average stress reduction hesaplanıyor (max VM)...")
            critical_stress_avg = extract_critical_stress(average_stress_data, 'Property ID')
            reduced_stress_avg = [{
                'Property ID':  r['Property ID'],
                'Load Case ID': r['Load Case ID'],
                'Avg_Sx_Z1':  r['Avg_Sx_Z1'],  'Avg_Sy_Z1':  r['Avg_Sy_Z1'],  'Avg_Sxy_Z1': r['Avg_Sxy_Z1'],
                'Avg_VM_Z1':  r['Avg_VM_Z1'],  'Avg_P1_Z1':  r['Avg_P1_Z1'],  'Avg_P2_Z1':  r['Avg_P2_Z1'],
                'Avg_Sx_Z2':  r['Avg_Sx_Z2'],  'Avg_Sy_Z2':  r['Avg_Sy_Z2'],  'Avg_Sxy_Z2': r['Avg_Sxy_Z2'],
                'Avg_VM_Z2':  r['Avg_VM_Z2'],  'Avg_P1_Z2':  r['Avg_P1_Z2'],  'Avg_P2_Z2':  r['Avg_P2_Z2'],
            } for r in critical_stress_avg]
            df_avg_stress_reduced = pd.DataFrame(reduced_stress_avg)
            df_avg_stress_reduced.to_csv(os.path.join(self.stress_output_now2, 'Average_Stress_Reduced.csv'), index=False)
            self.logger.info(f"✓ Average_Stress_Reduced.csv yazıldı ({len(df_avg_stress_reduced)} kritik satır)")
        else:
            self.logger.info("⚠ Stress verisi bulunamadı (OP2'de STRESS output yok)")

    # ─────────────────────────────────────────────────────────────────────────
    # BUSH EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def run_bush(self):
        if not self.bush_element_ids:
            self.logger.error("Element ID'leri girin!")
            messagebox.showerror("Hata", "Element ID'leri girin (tüm için: ALL)")
            return

        self.logger.info("📂 OP2 dosyası okunuyor...")
        op2 = OP2()
        op2.read_op2(self.output_entry_now)
        self.logger.info("✓ OP2 dosyası okundu")

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
        for load_case_id, element_forces in op2.cbush_force.items():
            if load_case_id not in target_lc_bush:
                continue
            element_ids = element_forces.element
            forces_data = element_forces.data[0]
            load_ids = element_forces.loadIDs[0]

            for i, element_id in enumerate(element_ids):
                if element_id in selected_element_ids:
                    index = np.where(element_ids == element_id)[0][0]
                    forces = forces_data[index][:3]
                    bush_forces_data.append({
                        'Element ID':element_id,
                        'Load Case ID':load_ids,
                        'FX':forces[0],
                        'FY':forces[1],
                        'FZ':forces[2]
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

"""Desktop front-end for pixelize.py.

Double-click this file (or run `python pixelizer_gui.pyw`) to pick an image, tweak
size and colours with a live preview, and save it as PNG, ICO, GIF or WebP.
Rendering runs in-process on a worker thread; icon-sized images take milliseconds.
"""

import ctypes
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageTk

import pixelize
import setup_tools

FROZEN = getattr(sys, "frozen", False)
ICON = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "icon.ico"

IMAGE_TYPES = [("Images", " ".join(f"*{e}" for e in pixelize.IMAGE_TYPES)), ("All files", "*.*")]
FORMATS = {"png": "PNG", "ico": "ICO", "gif": "GIF", "webp": "WebP"}
SCALES = (1, 2, 4, 8, 16)
PRESETS = (16, 24, 32, 48, 64, 128)
PALETTE_NAMES = {"auto": "Auto (from image)", "pico-8": "PICO-8", "sweetie-16": "Sweetie 16",
                 "endesga-32": "Endesga 32", "c64": "Commodore 64", "game-boy": "Game Boy",
                 "cga": "CGA", "1-bit": "1-bit"}
METHOD_NAMES = {"smart": "Smart (majority colour)", "average": "Average (softer)", "nearest": "Nearest (raw)"}

# Palette (shared with 60fps GIF Maker)
BG = "#0f1117"
CARD = "#171a23"
BORDER = "#262b38"
FIELD = "#0c0e14"
HOVER = "#2f3546"
TEXT = "#e7e9f0"
MUTED = "#8a90a2"
FAINT = "#555c6e"
ACCENT = "#7c5cff"
ACCENT_HOVER = "#9278ff"
ACCENT_DIM = "#3b2f78"
ACCENT_SOFT = "#231d45"
GOOD = "#3ecf8e"
WARN = "#f5a524"
BAD = "#f2626b"
CHECK_A = (24, 27, 37)
CHECK_B = (31, 35, 48)

UI = "Segoe UI"
MONO = "Consolas"
SCALE = 1.0  # set from the screen DPI once Tk starts
LEFT_W = 380  # width of the controls column

# 8x8 logo: a little pixel "gem" in the accent colours
LOGO = ["..aaaa..",
        ".abbbba.",
        "abbccbba",
        "abcccbba",
        "abbcbbda",
        ".abbbda.",
        "..adda..",
        "...aa..."]
LOGO_COLORS = {"a": "#3b2f78", "b": ACCENT, "c": "#c9bcff", "d": "#5a43c9"}


def px(n):
    return round(n * SCALE)


def open_path(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal(path):
    path = Path(path)
    if sys.platform == "win32" and path.is_file():
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        open_path(str(path if path.is_dir() else path.parent))


def line_points(a, b):
    """Pixels on the straight line from a to b (Bresenham), excluding a."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err, points = dx + dy, []
    while (x0, y0) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
        points.append((x0, y0))
    return points


def checker(w, h, cell):
    """Dark checkerboard image used behind transparent pixels."""
    tile = Image.new("RGB", (cell * 2, cell * 2), CHECK_A)
    d = ImageDraw.Draw(tile)
    d.rectangle((cell, 0, cell * 2 - 1, cell - 1), fill=CHECK_B)
    d.rectangle((0, cell, cell - 1, cell * 2 - 1), fill=CHECK_B)
    out = Image.new("RGB", (w, h))
    for y in range(0, h, cell * 2):
        for x in range(0, w, cell * 2):
            out.paste(tile, (x, y))
    return out.convert("RGBA")


def setup_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, font=(UI, 10), bordercolor=BORDER,
                    lightcolor=BG, darkcolor=BG, troughcolor=FIELD, focuscolor=ACCENT,
                    selectbackground=ACCENT, selectforeground="white", fieldbackground=FIELD,
                    insertcolor=TEXT)

    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Title.TLabel", font=(UI, 20, "bold"))
    style.configure("Subtitle.TLabel", foreground=MUTED, font=(UI, 10))
    style.configure("Card.TLabel", background=CARD)
    style.configure("Section.TLabel", background=CARD, font=(UI, 11, "bold"))
    style.configure("Field.TLabel", background=CARD, foreground=MUTED, font=(UI, 9))
    style.configure("Value.TLabel", background=CARD, foreground=ACCENT_HOVER, font=(UI, 9, "bold"))
    style.configure("Hint.TLabel", background=CARD, foreground=FAINT, font=(UI, 9))
    style.configure("Times.TLabel", background=CARD, foreground=FAINT, font=(UI, 12))
    style.configure("Status.TLabel", foreground=MUTED, font=(UI, 9))

    field = dict(fieldbackground=FIELD, background=FIELD, foreground=TEXT, bordercolor=BORDER,
                 lightcolor=FIELD, darkcolor=FIELD, arrowcolor=MUTED, arrowsize=px(12), insertcolor=TEXT,
                 padding=(px(10), px(6)))
    focus = dict(bordercolor=[("focus", ACCENT)], lightcolor=[("focus", FIELD)])
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, **field)
        style.map(name, **focus)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)],
              background=[("active", FIELD)], arrowcolor=[("active", TEXT)], **focus)
    style.map("TSpinbox", background=[("active", FIELD)], arrowcolor=[("active", TEXT)],
              foreground=[("disabled", FAINT)], fieldbackground=[("disabled", CARD)], **focus)
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "white")
    root.option_add("*TCombobox*Listbox.font", (UI, 10))
    root.option_add("*TCombobox*Listbox.borderWidth", 0)

    flat = lambda color: dict(background=color, bordercolor=color, lightcolor=color, darkcolor=color)

    def state_map(normal, hover, pressed, disabled):
        return {k: [("disabled", disabled), ("pressed", pressed), ("active", hover), ("!disabled", normal)]
                for k in ("background", "bordercolor", "lightcolor", "darkcolor")}

    style.configure("TButton", **flat(BORDER), foreground=TEXT, padding=(px(14), px(8)), font=(UI, 10))
    style.map("TButton", foreground=[("disabled", FAINT)], **state_map(BORDER, HOVER, HOVER, CARD))
    style.configure("Card.TButton", **flat(BORDER), foreground=TEXT, padding=(px(14), px(6)))
    style.map("Card.TButton", foreground=[("disabled", FAINT)], **state_map(BORDER, HOVER, HOVER, CARD))
    style.configure("Accent.TButton", **flat(ACCENT), foreground="white", padding=(px(26), px(10)),
                    font=(UI, 11, "bold"))
    style.map("Accent.TButton", foreground=[("disabled", "#a99ee0")],
              **state_map(ACCENT, ACCENT_HOVER, ACCENT_DIM, ACCENT_DIM))
    style.configure("Link.TButton", **flat(CARD), foreground=ACCENT, padding=(0, px(2)), font=(UI, 9, "bold"))
    style.map("Link.TButton", foreground=[("active", ACCENT_HOVER)], **state_map(CARD, CARD, CARD, CARD))
    style.configure("Small.TButton", **flat(ACCENT_SOFT), foreground=ACCENT_HOVER, padding=(px(10), px(4)),
                    font=(UI, 9, "bold"))
    style.map("Small.TButton", foreground=[("disabled", FAINT)], **state_map(ACCENT_SOFT, ACCENT_DIM, ACCENT_DIM, BG))
    style.configure("Chip.TButton", **flat(FIELD), foreground=MUTED, padding=(px(4), px(5)), font=(UI, 9, "bold"))
    style.map("Chip.TButton", foreground=[("active", TEXT)], **state_map(FIELD, BORDER, ACCENT_DIM, FIELD))

    style.configure("Seg.Toolbutton", **flat(FIELD), foreground=MUTED, padding=(px(8), px(7)),
                    anchor="center", font=(UI, 10, "bold"))
    style.map("Seg.Toolbutton",
              foreground=[("disabled", FAINT), ("selected", "white"), ("active", TEXT)],
              **{k: [("disabled", FIELD), ("selected", ACCENT), ("active", BORDER), ("!selected", FIELD)]
                 for k in ("background", "bordercolor", "lightcolor", "darkcolor")})
    style.configure("SmallSeg.Toolbutton", **flat(FIELD), foreground=MUTED, padding=(px(10), px(3)),
                    anchor="center", font=(UI, 9, "bold"))
    style.map("SmallSeg.Toolbutton",
              foreground=[("disabled", FAINT), ("selected", "white"), ("active", TEXT)],
              **{k: [("disabled", FIELD), ("selected", ACCENT), ("active", BORDER), ("!selected", FIELD)]
                 for k in ("background", "bordercolor", "lightcolor", "darkcolor")})

    style.configure("Card.TCheckbutton", background=CARD, foreground=TEXT, indicatorbackground=FIELD,
                    indicatorforeground="white", indicatorrelief="flat", indicatorsize=px(14),
                    indicatormargin=(0, 0, px(8), 0), upperbordercolor=BORDER, lowerbordercolor=BORDER)
    style.map("Card.TCheckbutton", background=[("active", CARD)], foreground=[("disabled", FAINT)],
              indicatorbackground=[("selected", ACCENT), ("active", BORDER)])

    style.configure("Horizontal.TScale", troughcolor=FIELD, **flat(ACCENT), sliderthickness=px(14),
                    sliderlength=px(14), troughrelief="flat", gripcount=0)
    style.map("Horizontal.TScale", **{k: [("active", ACCENT_HOVER)] for k in ("background", "lightcolor", "darkcolor")})

    style.layout("Dark.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
    style.configure("Dark.Vertical.TScrollbar", troughcolor=BG, gripcount=0, arrowsize=px(6), **flat(BORDER))
    style.map("Dark.Vertical.TScrollbar", background=[("active", HOVER)])

    style.configure("Accent.Horizontal.TProgressbar", troughcolor=BG, background=ACCENT,
                    bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=px(4))


class DropZone(tk.Canvas):
    """Big clickable area for choosing the input image, with a thumbnail once loaded."""

    def __init__(self, parent, command):
        super().__init__(parent, width=px(300), height=px(92), bg=CARD, highlightthickness=0, cursor="hand2")
        self.command = command
        self.title, self.subtitle = "Choose an image", "Click to browse  ·  Ctrl+V to paste"
        self.thumb = None
        self.hover = False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self.set_hover(True))
        self.bind("<Leave>", lambda e: self.set_hover(False))
        self.bind("<Button-1>", lambda e: self.command())

    def set_hover(self, value):
        self.hover = value
        self.draw()

    def set_image(self, title, subtitle, image=None):
        self.title, self.subtitle = title, subtitle
        if image is not None:
            side = px(60)
            t = image.convert("RGBA")
            t.thumbnail((side, side), Image.Resampling.LANCZOS)
            bg = checker(t.width, t.height, px(5))
            bg.alpha_composite(t)
            self.thumb = ImageTk.PhotoImage(bg)
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        has = self.thumb is not None
        outline = ACCENT if self.hover else (ACCENT_DIM if has else BORDER)
        self.create_rectangle(1, 1, w - 2, h - 2, fill=ACCENT_SOFT if self.hover else FIELD,
                              outline=outline, dash=() if has else (6, 4), width=1)
        cx, cy = px(48), h // 2
        if has:
            self.create_image(cx, cy, image=self.thumb)
        else:
            r = px(22)
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=ACCENT_SOFT, outline="")
            self.create_text(cx, cy - 1, text="+", fill=ACCENT_HOVER, font=(UI, 22))
        x, wrap = px(92), w - px(104)
        self.create_text(x, cy, text=self.title, anchor="sw", fill=TEXT, font=(UI, 11, "bold"), width=wrap)
        self.create_text(x, cy + px(3), text=self.subtitle, anchor="nw", fill=MUTED, font=(UI, 9), width=wrap)


class Preview(tk.Canvas):
    """Shows the pixelized result (integer zoom, optional grid) or the original."""

    def __init__(self, parent):
        super().__init__(parent, width=px(300), height=px(300), bg=FIELD, highlightthickness=1,
                         highlightbackground=BORDER)
        self.result = None
        self.original = None
        self.mode = "result"
        self.grid = False
        self.photo = None
        self.zoom = 1
        self.origin = (0, 0)     # canvas position of the result's top-left pixel
        self.painting = False    # show the grid while a paint tool is active
        self._checker = (None, None)
        self.bind("<Configure>", lambda e: self.draw())

    def pixel_at(self, x, y):
        """Result pixel (col, row) under canvas point x, y, or None."""
        if self.mode != "result" or self.result is None:
            return None
        col, row = (x - self.origin[0]) // self.zoom, (y - self.origin[1]) // self.zoom
        if 0 <= col < self.result.width and 0 <= row < self.result.height:
            return int(col), int(row)
        return None

    def checker(self, w, h, cell):
        key = (w, h, cell)
        if self._checker[0] != key:
            self._checker = (key, checker(w, h, cell))
        return self._checker[1].copy()

    def show(self, result=None, original=None):
        if result is not None:
            self.result = result
        if original is not None:
            self.original = original
        self.draw()

    def clear(self):
        self.result = None
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        img = self.result if self.mode == "result" else self.original
        if img is None:
            self.draw_empty(w, h)
            return
        aw, ah = int(w * 0.9), int(h * 0.9)
        if self.mode == "result":
            z = max(1, min(aw // img.width, ah // img.height))
            self.zoom = z
            big = img.resize((img.width * z, img.height * z), Image.Resampling.NEAREST)
            cell = max(px(6), z)
        else:
            ratio = min(aw / img.width, ah / img.height)
            big = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
                             Image.Resampling.LANCZOS)
            cell = px(8)
        canvas = self.checker(big.width, big.height, cell)
        canvas.alpha_composite(big.convert("RGBA"))
        if self.mode == "result" and (self.grid or self.painting) and self.zoom >= 6:
            d = ImageDraw.Draw(canvas, "RGBA")
            for x in range(0, big.width + 1, self.zoom):
                d.line((x, 0, x, big.height), fill=(0, 0, 0, 70))
            for y in range(0, big.height + 1, self.zoom):
                d.line((0, y, big.width, y), fill=(0, 0, 0, 70))
        self.photo = ImageTk.PhotoImage(canvas)
        self.origin = (w // 2 - big.width // 2, h // 2 - big.height // 2)
        self.create_image(*self.origin, image=self.photo, anchor="nw")

    def draw_empty(self, w, h):
        cell = px(9)
        ox, oy = w // 2 - 4 * cell, h // 2 - 4 * cell - px(18)
        for y, row in enumerate(LOGO):
            for x, ch in enumerate(row):
                if ch != ".":
                    color = {"a": BORDER, "b": HOVER, "c": FAINT, "d": BORDER}[ch]
                    self.create_rectangle(ox + x * cell, oy + y * cell, ox + (x + 1) * cell - 1,
                                          oy + (y + 1) * cell - 1, fill=color, outline="")
        self.create_text(w // 2, oy + 8 * cell + px(22), text="Your pixel art appears here",
                         fill=MUTED, font=(UI, 11, "bold"))
        self.create_text(w // 2, oy + 8 * cell + px(44), text="Open an image to start  ·  Ctrl+O",
                         fill=FAINT, font=(UI, 9))


class App(tk.Tk):
    def __init__(self, initial=None):
        super().__init__()
        global SCALE
        SCALE = self.winfo_fpixels("1i") / 96
        self.title("Pixelizer")
        self.configure(bg=BG)
        if sys.platform == "win32" and ICON.is_file():
            self.iconbitmap(default=str(ICON))
        w = min(px(1160), self.winfo_screenwidth() - px(40))
        h = min(px(820), self.winfo_screenheight() - px(90))
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth() - w) // 2}+{max(0, (self.winfo_screenheight() - h) // 2 - px(20))}")
        self.minsize(px(980), px(700))
        setup_style(self)
        self.dark_title_bar()

        self.source = None        # PIL image
        self.source_path = None   # Path or None (pasted)
        self.cutout = None        # cached AI cutout of self.source
        self.result = None
        self.palette = []
        self.color_map = {}       # palette swaps chosen by clicking swatches: {old rgb: new rgb}
        self.swatch_hits = []     # (x0, x1, original rgb) for each drawn swatch
        self.edits = {}           # pencil/eraser edits on top of the render: {(col, row): rgba}
        self.undo_stack = []      # one {(col, row): previous edit or None} per stroke
        self.stroke = None        # the stroke being drawn
        self.last_pixel = None
        self.brush = (0, 0, 0)
        self.info_text = ""
        self.saved_path = None
        self.events = queue.Queue()
        self.busy = False
        self.dirty = False
        self.pending = None

        self.width_var = tk.StringVar(value="32")
        self.height_var = tk.StringVar(value="32")
        self.fit_var = tk.StringVar(value="fit")
        self.palette_var = tk.StringVar(value=PALETTE_NAMES["auto"])
        self.colors_var = tk.StringVar(value="16")
        self.dither_var = tk.StringVar(value="none")
        self.method_var = tk.StringVar(value=METHOD_NAMES["smart"])
        self.outline_var = tk.BooleanVar(value=False)
        self.removebg_var = tk.BooleanVar(value=False)
        self.trim_var = tk.BooleanVar(value=True)
        self.cleanup_var = tk.BooleanVar(value=True)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.background_var = tk.StringVar()
        self.format_var = tk.StringVar(value="png")
        self.scale_var = tk.StringVar(value="1")
        self.view_var = tk.StringVar(value="result")
        self.grid_var = tk.BooleanVar(value=False)
        self.tool_var = tk.StringVar(value="view")
        self.advanced_open = False

        self.build()
        self.refresh_ai()
        for var in (self.width_var, self.height_var, self.fit_var, self.palette_var, self.colors_var,
                    self.dither_var, self.method_var, self.outline_var, self.trim_var, self.cleanup_var,
                    self.contrast_var, self.saturation_var, self.background_var):
            var.trace_add("write", lambda *_: self.schedule())
        self.removebg_var.trace_add("write", lambda *_: self.toggle_removebg())
        self.palette_var.trace_add("write", lambda *_: self.update_palette_state())
        self.view_var.trace_add("write", lambda *_: self.update_view())
        self.grid_var.trace_add("write", lambda *_: self.update_view())
        self.bind_all("<Control-o>", lambda e: self.pick_input())
        self.bind_all("<Control-v>", self.paste)
        self.bind_all("<Control-s>", lambda e: self.save())
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.tool_var.trace_add("write", lambda *_: self.update_tool())
        self.after(50, self.poll)
        if initial:
            self.after(100, self.load_path, initial)

    def dark_title_bar(self):
        if sys.platform != "win32":
            return
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

    # ---------- layout ----------

    def card(self, parent, title=None, right=None):
        outer = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        body = ttk.Frame(outer, style="Card.TFrame", padding=(px(16), px(12), px(16), px(14)))
        body.pack(fill="both", expand=True)
        if title:
            head = ttk.Frame(body, style="Card.TFrame")
            head.pack(fill="x", pady=(0, px(8)))
            ttk.Label(head, text=title, style="Section.TLabel").pack(side="left")
            if right:
                right(head).pack(side="right")
        inner = ttk.Frame(body, style="Card.TFrame")
        inner.pack(fill="both", expand=True)
        return outer, inner

    def label(self, parent, text, row, col, colspan=1):
        ttk.Label(parent, text=text, style="Field.TLabel").grid(row=row, column=col, columnspan=colspan,
                                                                sticky="w", pady=(0, px(3)))

    def segmented(self, parent, var, options, style="Seg.Toolbutton"):
        seg = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        for i, (value, text) in enumerate(options.items()):
            seg.columnconfigure(i, weight=1, uniform="seg")
            ttk.Radiobutton(seg, text=text, value=value, variable=var, style=style).grid(row=0, column=i, sticky="ew")
        return seg

    def logo(self, parent):
        cell = px(4)
        c = tk.Canvas(parent, width=cell * 8, height=cell * 8, bg=BG, highlightthickness=0)
        for y, row in enumerate(LOGO):
            for x, ch in enumerate(row):
                if ch != ".":
                    c.create_rectangle(x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1,
                                       fill=LOGO_COLORS[ch], outline="")
        return c

    def build(self):
        main = ttk.Frame(self, padding=(px(22), px(16)))
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # Header
        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, px(14)))
        self.logo(header).pack(side="left", padx=(0, px(12)))
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Pixelizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Turn any image into crisp retro pixel art, icons and sprites.",
                  style="Subtitle.TLabel").pack(anchor="w")

        # ---- left: controls (fixed width, scrolls when Advanced makes it taller than the window) ----
        holder = ttk.Frame(main)
        holder.grid(row=1, column=0, sticky="nsw", padx=(0, px(16)))
        self.left_canvas = tk.Canvas(holder, width=px(LEFT_W), bg=BG, highlightthickness=0, bd=0)
        self.left_scroll = ttk.Scrollbar(holder, orient="vertical", command=self.left_canvas.yview,
                                         style="Dark.Vertical.TScrollbar")
        self.left_canvas.configure(yscrollcommand=self.left_scroll.set)
        self.left_canvas.pack(side="left", fill="y", expand=True)
        left = ttk.Frame(self.left_canvas)
        self.left_canvas.create_window(0, 0, window=left, anchor="nw", width=px(LEFT_W))
        left.bind("<Configure>", lambda e: self.after_idle(self.update_left_scroll))
        self.left_canvas.bind("<Configure>", lambda e: self.after_idle(self.update_left_scroll))
        self.left_holder = holder
        self.bind_all("<MouseWheel>", self.on_wheel, add="+")
        left.columnconfigure(0, weight=1)

        src_card, src = self.card(left, "Source")
        src_card.grid(row=0, column=0, sticky="ew")
        self.drop = DropZone(src, self.pick_input)
        self.drop.pack(fill="x")

        # Size
        size_card, size = self.card(left, "Size")
        size_card.grid(row=1, column=0, sticky="ew", pady=(px(12), 0))
        size.columnconfigure(0, weight=1)
        size.columnconfigure(2, weight=1)
        self.label(size, "Width (px)", 0, 0)
        self.label(size, "Height (px)", 0, 2)
        ttk.Spinbox(size, textvariable=self.width_var, from_=0, to=1024, increment=8, width=6).grid(
            row=1, column=0, sticky="ew")
        ttk.Label(size, text="×", style="Times.TLabel").grid(row=1, column=1, padx=px(8))
        ttk.Spinbox(size, textvariable=self.height_var, from_=0, to=1024, increment=8, width=6).grid(
            row=1, column=2, sticky="ew")
        chips = ttk.Frame(size, style="Card.TFrame")
        chips.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(px(8), 0))
        for i, n in enumerate(PRESETS):
            chips.columnconfigure(i, weight=1, uniform="chip")
            ttk.Button(chips, text=str(n), style="Chip.TButton", command=lambda n=n: self.set_size(n, n)).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else px(4), 0))
        self.label(size, "Aspect ratio", 3, 0, 3)
        self.fit_seg = self.segmented(size, self.fit_var, {"fit": "Fit", "crop": "Crop", "stretch": "Stretch"})
        self.fit_seg.grid(row=4, column=0, columnspan=3, sticky="ew")
        size.grid_slaves(row=3)[0].grid_configure(pady=(px(10), px(3)))
        ttk.Label(size, text="Leave width or height blank to keep the image's proportions.",
                  style="Hint.TLabel", wraplength=px(LEFT_W - 40)).grid(row=5, column=0, columnspan=3, sticky="w", pady=(px(6), 0))

        # Look
        look_card, look = self.card(left, "Look")
        look_card.grid(row=2, column=0, sticky="ew", pady=(px(12), 0))
        look.columnconfigure(0, weight=3)
        look.columnconfigure(1, weight=1)
        self.label(look, "Palette", 0, 0)
        self.label(look, "Colours", 0, 1)
        ttk.Combobox(look, textvariable=self.palette_var, state="readonly", values=list(PALETTE_NAMES.values()),
                     width=16).grid(row=1, column=0, sticky="ew")
        self.colors_box = ttk.Spinbox(look, textvariable=self.colors_var, from_=2, to=64, width=4)
        self.colors_box.grid(row=1, column=1, sticky="ew", padx=(px(10), 0))
        self.label(look, "Dithering", 2, 0, 2)
        look.grid_slaves(row=2)[0].grid_configure(pady=(px(10), px(3)))
        self.segmented(look, self.dither_var, {"none": "None", "ordered": "Ordered", "floyd": "Diffuse"}).grid(
            row=3, column=0, columnspan=2, sticky="ew")
        checks = ttk.Frame(look, style="Card.TFrame")
        checks.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(px(12), 0))
        ttk.Checkbutton(checks, text="Outline", variable=self.outline_var, style="Card.TCheckbutton").pack(
            side="left")
        self.removebg_check = ttk.Checkbutton(checks, text="Remove background (AI)", variable=self.removebg_var,
                                              style="Card.TCheckbutton")
        self.removebg_check.pack(side="left", padx=(px(18), 0))

        self.adv_btn = ttk.Button(look, text="▸  Advanced", style="Link.TButton", command=self.toggle_advanced)
        self.adv_btn.grid(row=5, column=0, columnspan=2, sticky="w", pady=(px(10), 0))
        self.adv = ttk.Frame(look, style="Card.TFrame")
        self.adv.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(px(6), 0))
        self.adv.columnconfigure(1, weight=1)
        self.label(self.adv, "Downscale", 0, 0)
        ttk.Combobox(self.adv, textvariable=self.method_var, state="readonly", values=list(METHOD_NAMES.values()),
                     width=20).grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, px(6)))
        self.slider(self.adv, "Contrast", self.contrast_var, 1)
        self.slider(self.adv, "Saturation", self.saturation_var, 2)
        self.label(self.adv, "Background", 3, 0)
        ttk.Entry(self.adv, textvariable=self.background_var, width=10).grid(row=3, column=1, columnspan=2,
                                                                             sticky="ew", pady=(px(4), px(6)))
        advchk = ttk.Frame(self.adv, style="Card.TFrame")
        advchk.grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(advchk, text="Trim empty borders", variable=self.trim_var, style="Card.TCheckbutton").pack(
            side="left")
        ttk.Checkbutton(advchk, text="Clean stray pixels", variable=self.cleanup_var,
                        style="Card.TCheckbutton").pack(side="left", padx=(px(18), 0))
        ttk.Label(self.adv, text="Background: hex like 1a1c2c, blank = transparent.", style="Hint.TLabel",
                  wraplength=px(LEFT_W - 40)).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(px(6), 0))
        for child in self.adv.grid_slaves(column=0):
            child.grid_configure(padx=(0, px(12)))
        self.adv.grid_remove()

        # AI status
        status = ttk.Frame(left)
        status.grid(row=3, column=0, sticky="ew", pady=(px(12), 0))
        self.ai_pill = ttk.Frame(status)
        self.ai_pill.dot = tk.Label(self.ai_pill, text="●", bg=BG, font=(UI, 9))
        self.ai_pill.dot.pack(side="left")
        self.ai_pill.text = ttk.Label(self.ai_pill, style="Status.TLabel")
        self.ai_pill.text.pack(side="left", padx=(px(4), 0))
        self.ai_pill.pack(side="left")
        self.install_btn = ttk.Button(status, text="Install AI model", style="Small.TButton", command=self.install_ai)

        # ---- right: preview + export ----
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        def view_controls(parent):
            box = ttk.Frame(parent, style="Card.TFrame")
            self.segmented(box, self.tool_var, {"view": "View", "pencil": "Pencil", "eraser": "Eraser"},
                           style="SmallSeg.Toolbutton").pack(side="left")
            self.brush_box = tk.Canvas(box, width=px(22), height=px(22), bg=CARD, highlightthickness=0,
                                       cursor="hand2")
            self.brush_box.pack(side="left", padx=(px(8), px(18)))
            self.brush_box.bind("<Button-1>", lambda e: self.pick_brush())
            self.draw_brush()
            ttk.Checkbutton(box, text="Grid", variable=self.grid_var, style="Card.TCheckbutton").pack(
                side="left", padx=(0, px(14)))
            self.segmented(box, self.view_var, {"result": "Pixel art", "original": "Original"},
                           style="SmallSeg.Toolbutton").pack(side="left")
            return box

        prev_card, prev = self.card(right, "Preview", right=view_controls)
        prev_card.grid(row=0, column=0, sticky="nsew")
        self.preview = Preview(prev)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Button-1>", self.paint_start)
        self.preview.bind("<B1-Motion>", self.paint_move)
        self.preview.bind("<ButtonRelease-1>", self.paint_end)
        self.preview.bind("<Button-3>", self.eyedrop)
        self.busy_bar = ttk.Progressbar(prev, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        info = ttk.Frame(prev, style="Card.TFrame")
        info.pack(fill="x", pady=(px(10), 0))
        self.info_label = tk.Label(info, text="No image", bg=CARD, fg=MUTED, font=(UI, 9), anchor="w")
        self.info_label.pack(side="left")
        self.swatches = tk.Canvas(info, width=px(100), height=px(18), bg=CARD, highlightthickness=0)
        self.swatches.pack(side="right", fill="x", expand=True, padx=(px(16), 0))
        self.swatches.bind("<Configure>", lambda e: self.draw_swatches())
        self.swatches.bind("<Button-1>", self.pick_swatch_color)
        self.swatches.bind("<Button-3>", self.reset_swatch_color)
        self.swatches.bind("<Motion>", self.hover_swatch)
        self.swatches.bind("<Leave>", lambda e: self.hover_swatch(None))
        self.reset_colors_btn = ttk.Button(info, text="Reset colours", style="Link.TButton",
                                           command=self.reset_colors)
        self.clear_paint_btn = ttk.Button(info, text="Clear paint", style="Link.TButton",
                                          command=self.clear_paint)

        exp_card, exp = self.card(right)
        exp_card.grid(row=1, column=0, sticky="ew", pady=(px(12), 0))
        exp.columnconfigure(0, weight=4, uniform="exp")
        exp.columnconfigure(1, weight=5, uniform="exp")
        ttk.Label(exp, text="Format", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=(0, px(3)))
        ttk.Label(exp, text="Export scale", style="Field.TLabel").grid(row=0, column=1, sticky="w",
                                                                       padx=(px(14), 0), pady=(0, px(3)))
        self.segmented(exp, self.format_var, FORMATS).grid(row=1, column=0, sticky="ew")
        self.scale_seg = self.segmented(exp, self.scale_var, {str(s): f"{s}×" for s in SCALES})
        self.scale_seg.grid(row=1, column=1, sticky="ew", padx=(px(14), 0))
        actions = ttk.Frame(exp, style="Card.TFrame")
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(px(12), 0))
        self.save_hint = ttk.Label(actions, style="Hint.TLabel")
        self.save_hint.pack(side="left")
        self.save_btn = ttk.Button(actions, text="Save", style="Accent.TButton", command=self.save, state="disabled")
        self.save_btn.pack(side="right")
        self.folder_btn = ttk.Button(actions, text="Show in folder", style="Card.TButton", command=self.show_folder,
                                     state="disabled")
        self.folder_btn.pack(side="right", padx=(0, px(10)))
        self.format_var.trace_add("write", lambda *_: self.update_format_hint())
        self.update_format_hint()

    def update_left_scroll(self):
        c = self.left_canvas
        c.configure(scrollregion=c.bbox("all"))
        needed = c.bbox("all") and c.bbox("all")[3] > c.winfo_height() + 1
        if needed and not self.left_scroll.winfo_ismapped():
            self.left_scroll.pack(side="right", fill="y", padx=(px(6), 0))
        elif not needed and self.left_scroll.winfo_ismapped():
            self.left_scroll.pack_forget()
            c.yview_moveto(0)

    def on_wheel(self, event):
        if not str(event.widget).startswith(str(self.left_holder)) or not self.left_scroll.winfo_ismapped():
            return
        if isinstance(event.widget, (ttk.Combobox, ttk.Spinbox)):
            return  # those use the wheel themselves
        self.left_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def slider(self, parent, text, var, row):
        self.label(parent, text, row, 0)
        parent.grid_slaves(row=row, column=0)[0].grid_configure(pady=(px(4), px(4)))
        value = ttk.Label(parent, style="Value.TLabel", width=5, anchor="e")
        value.grid(row=row, column=2, sticky="e")
        scale = ttk.Scale(parent, from_=0.5, to=2.0, variable=var, orient="horizontal")
        scale.grid(row=row, column=1, sticky="ew", padx=(0, px(8)))
        scale.bind("<Double-Button-1>", lambda e: var.set(1.0))
        update = lambda *_: value.config(text=f"{var.get():.2f}")
        var.trace_add("write", update)
        update()

    def toggle_advanced(self):
        self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.adv.grid()
        else:
            self.adv.grid_remove()
        self.adv_btn.config(text=("▾" if self.advanced_open else "▸") + "  Advanced")

    def update_format_hint(self):
        fmt = self.format_var.get()
        hint = {"png": "PNG keeps every pixel exactly, with transparency.",
                "ico": "ICO bundles 16–256 px sizes with hard pixel edges (export scale is ignored).",
                "gif": "GIF keeps transparency; great for retro sites and forums.",
                "webp": "WebP is saved lossless, so pixels stay exact."}[fmt]
        self.save_hint.config(text=hint)
        state = "disabled" if fmt == "ico" else "normal"
        for child in self.scale_seg.winfo_children():
            child.state(["disabled"] if state == "disabled" else ["!disabled"])

    def update_palette_state(self):
        auto = self.palette_key() == "auto"
        self.colors_box.state(["!disabled"] if auto else ["disabled"])

    def update_view(self):
        self.preview.mode = self.view_var.get()
        self.preview.grid = self.grid_var.get()
        self.preview.draw()

    def set_size(self, w, h):
        self.width_var.set(str(w))
        self.height_var.set(str(h))

    # ---------- AI ----------

    def refresh_ai(self):
        ok, msg = pixelize.cutout_status()
        missing_model = not pixelize.CUTOUT_MODEL.is_file()
        self.ai_pill.dot.config(fg=GOOD if ok else WARN)
        self.ai_pill.text.config(text=msg)
        if missing_model and "onnxruntime" not in msg:
            self.install_btn.pack(side="right")
        else:
            self.install_btn.pack_forget()
        self.removebg_check.state(["!disabled"] if ok else ["disabled"])
        if not ok and self.removebg_var.get():
            self.removebg_var.set(False)

    def install_ai(self):
        self.install_btn.config(state="disabled", text="Downloading…")

        def work():
            try:
                setup_tools.main(progress=lambda f: self.events.put(("install", f)))
                self.events.put(("installed", None))
            except Exception as e:
                self.events.put(("install_failed", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def toggle_removebg(self):
        if self.removebg_var.get() and not pixelize.cutout_status()[0]:
            self.removebg_var.set(False)
            return
        self.schedule()

    # ---------- input ----------

    def pick_input(self):
        path = filedialog.askopenfilename(title="Choose an image", filetypes=IMAGE_TYPES)
        if path:
            self.load_path(path)

    def paste(self, event=None):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return  # let fields paste text normally
        try:
            data = ImageGrab.grabclipboard()
        except Exception:
            data = None
        if isinstance(data, list):
            files = [f for f in data if Path(f).suffix.lower() in pixelize.IMAGE_TYPES]
            if files:
                self.load_path(files[0])
        elif isinstance(data, Image.Image):
            self.set_source(data, None)

    def load_path(self, path):
        try:
            with Image.open(path) as im:
                im.load()
                img = im.copy()
        except Exception as e:
            messagebox.showerror("Can't open image", f"{Path(path).name}\n\n{e}")
            return
        self.set_source(img, Path(path))

    def set_source(self, img, path):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        self.source, self.source_path, self.cutout = img, path, None
        self.color_map.clear()
        self.edits.clear()
        self.undo_stack.clear()
        self.update_reset_btn()
        name = path.name if path else "Pasted image"
        self.drop.set_image(name, f"{img.width} × {img.height}   ·   Click to change", img)
        self.preview.show(original=img)
        self.saved_path = None
        self.folder_btn.config(state="disabled")
        self.schedule(0)

    # ---------- rendering ----------

    def palette_key(self):
        name = self.palette_var.get()
        return next((k for k, v in PALETTE_NAMES.items() if v == name), "auto")

    def settings(self):
        def num(var, default=0):
            text = var.get().strip()
            return int(text) if text.isdigit() else default

        w, h = min(num(self.width_var), 1024), min(num(self.height_var), 1024)
        if not w and not h:
            raise ValueError("size")
        bg = self.background_var.get().strip()
        if bg:
            pixelize.parse_hex(bg)  # raises on bad input; preview just waits for a valid colour
        method = next((k for k, v in METHOD_NAMES.items() if v == self.method_var.get()), "smart")
        return pixelize.Settings(
            width=w, height=h, fit=self.fit_var.get(), palette=self.palette_key(),
            colors=max(2, min(num(self.colors_var, 16), 64)), dither=self.dither_var.get(), method=method,
            outline=self.outline_var.get(), remove_bg=self.removebg_var.get(), trim=self.trim_var.get(),
            cleanup=self.cleanup_var.get(), contrast=round(self.contrast_var.get(), 2),
            saturation=round(self.saturation_var.get(), 2), background=bg)

    def schedule(self, delay=160):
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = self.after(delay, self.render)

    def render(self):
        self.pending = None
        if self.source is None:
            return
        if self.busy:
            self.dirty = True
            return
        try:
            s = self.settings()
        except ValueError:
            return
        self.busy = True
        self.dirty = False
        source, cutout = self.source, self.cutout

        def work():
            t = time.perf_counter()
            try:
                if s.remove_bg and cutout is None:
                    self.events.put(("busy", "Removing background with AI…"))
                    cut = pixelize.remove_background(source)
                    self.events.put(("cutout", (source, cut)))
                else:
                    cut = cutout
                result, palette = pixelize.pixelize(source, s, cutout=cut)
                self.events.put(("rendered", (source, result, palette, time.perf_counter() - t)))
            except Exception as e:
                self.events.put(("render_failed", str(e)))

        if s.remove_bg and cutout is None:
            self.show_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def show_busy(self, on, text=None):
        if on:
            self.busy_bar.pack(fill="x", before=self.preview, pady=(0, px(6)))
            self.busy_bar.start(12)
            if text:
                self.info_label.config(text=text, fg=ACCENT_HOVER)
        else:
            self.busy_bar.stop()
            self.busy_bar.pack_forget()

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "busy":
                    self.show_busy(True, data)
                elif kind == "cutout":
                    src, cut = data
                    if src is self.source:
                        self.cutout = cut
                elif kind == "rendered":
                    self.busy = False
                    self.show_busy(False)
                    src, result, palette, secs = data
                    if src is self.source:
                        self.show_result(result, palette, secs)
                    if self.dirty or src is not self.source:
                        self.render()
                elif kind == "render_failed":
                    self.busy = False
                    self.show_busy(False)
                    self.info_label.config(text=f"Failed: {data}", fg=BAD)
                    if self.dirty:
                        self.render()
                elif kind == "install":
                    self.install_btn.config(text=f"Downloading… {data * 100:.0f}%")
                elif kind == "installed":
                    self.install_btn.config(state="normal", text="Install AI model")
                    self.refresh_ai()
                elif kind == "install_failed":
                    self.install_btn.config(state="normal", text="Install AI model")
                    messagebox.showerror("Download failed", data)
        except queue.Empty:
            pass
        self.after(50, self.poll)

    def show_result(self, result, palette, secs):
        if self.result is not None and self.result.size != result.size and self.edits:
            self.edits.clear()  # the pixel grid changed, so old edits no longer line up
            self.undo_stack.clear()
            self.update_reset_btn()
        if self.result is None and palette:
            self.brush = palette[0]  # darkest colour: a sensible first pen
            self.draw_brush()
        self.result, self.palette = result, palette
        self.preview.show(result=self.final())
        self.info_text = (f"{result.width} × {result.height}   ·   {len(palette)} colours   ·   "
                          f"{secs * 1000:.0f} ms")
        self.info_label.config(text=self.info_text, fg=TEXT)
        self.draw_swatches()
        self.save_btn.config(state="normal")

    def final(self):
        """The rendered image with the user's palette swaps and pencil edits applied."""
        img = pixelize.recolor(self.result, self.color_map)
        if not self.edits:
            return img
        rgba = np.asarray(img.convert("RGBA")).copy()
        for (col, row), value in self.edits.items():
            if row < rgba.shape[0] and col < rgba.shape[1]:
                rgba[row, col] = value
        return Image.fromarray(rgba, "RGBA")

    def draw_swatches(self):
        c = self.swatches
        c.delete("all")
        self.swatch_hits = []
        gap, w = px(3), c.winfo_width()
        # Shrink swatches when space is tight so every colour stays clickable (down to a small minimum).
        size = max(px(7), min(px(14), (w + gap) // max(1, len(self.palette)) - gap))
        fit = max(1, (w + gap) // (size + gap))
        colors = self.palette[:fit]
        x = w - len(colors) * (size + gap) + gap
        y = (c.winfo_height() - size) // 2
        for rgb in colors:
            shown = self.color_map.get(rgb, rgb)
            changed = shown != rgb
            c.create_rectangle(x, y, x + size - 1, y + size - 1, fill="#" + pixelize.to_hex(shown),
                               outline=ACCENT_HOVER if changed else BORDER, width=2 if changed else 1)
            self.swatch_hits.append((x, x + size, rgb))
            x += size + gap

    # ---------- palette swaps ----------

    def swatch_at(self, x):
        return next((rgb for x0, x1, rgb in self.swatch_hits if x0 - 1 <= x <= x1 + 1), None)

    def hover_swatch(self, event):
        rgb = self.swatch_at(event.x) if event else None
        self.swatches.config(cursor="hand2" if rgb else "")
        if rgb is None:
            if self.info_text and self.result is not None:
                self.info_label.config(text=self.info_text, fg=TEXT)
            return
        shown = self.color_map.get(rgb, rgb)
        if self.tool_var.get() == "pencil":
            self.info_label.config(text=f"#{pixelize.to_hex(shown)}   ·   click to paint with this colour",
                                   fg=ACCENT_HOVER)
            return
        text = f"#{pixelize.to_hex(shown)}   ·   click to change"
        if shown != rgb:
            text += f", right-click to restore #{pixelize.to_hex(rgb)}"
        self.info_label.config(text=text, fg=ACCENT_HOVER)

    def pick_swatch_color(self, event):
        rgb = self.swatch_at(event.x)
        if rgb is None:
            return
        if self.tool_var.get() == "pencil":
            self.set_brush(self.color_map.get(rgb, rgb))
            return
        current = "#" + pixelize.to_hex(self.color_map.get(rgb, rgb))
        chosen = colorchooser.askcolor(color=current, parent=self, title="Change colour")[1]
        if not chosen:
            return
        new = pixelize.parse_hex(chosen)
        if new == rgb:
            self.color_map.pop(rgb, None)
        else:
            self.color_map[rgb] = new
        self.apply_colors()

    def reset_swatch_color(self, event):
        rgb = self.swatch_at(event.x)
        if rgb is not None and self.color_map.pop(rgb, None) is not None:
            self.apply_colors()

    def reset_colors(self):
        self.color_map.clear()
        self.apply_colors()

    def apply_colors(self):
        if self.result is not None:
            self.preview.show(result=self.final())
        self.draw_swatches()
        self.update_reset_btn()

    def update_reset_btn(self):
        for btn, show in ((self.reset_colors_btn, bool(self.color_map)), (self.clear_paint_btn, bool(self.edits))):
            if show and not btn.winfo_ismapped():
                btn.pack(side="right", before=self.swatches, padx=(px(10), 0))
            elif not show and btn.winfo_ismapped():
                btn.pack_forget()

    # ---------- pencil ----------

    def update_tool(self):
        tool = self.tool_var.get()
        if tool != "view" and self.view_var.get() != "result":
            self.view_var.set("result")
        self.preview.painting = tool != "view"
        self.preview.config(cursor={"pencil": "pencil", "eraser": "dotbox"}.get(tool, ""))
        self.preview.draw()

    def draw_brush(self):
        c = self.brush_box
        c.delete("all")
        n = px(22)
        c.create_rectangle(1, 1, n - 2, n - 2, fill="#" + pixelize.to_hex(self.brush), outline=BORDER)

    def set_brush(self, rgb):
        self.brush = tuple(rgb)
        self.draw_brush()
        if self.tool_var.get() != "pencil":
            self.tool_var.set("pencil")

    def pick_brush(self):
        chosen = colorchooser.askcolor(color="#" + pixelize.to_hex(self.brush), parent=self,
                                       title="Pencil colour")[1]
        if chosen:
            self.set_brush(pixelize.parse_hex(chosen))

    def eyedrop(self, event):
        """Right-click a pixel to take its colour for the pencil."""
        xy = self.preview.pixel_at(event.x, event.y)
        if xy is None:
            return
        r, g, b, a = self.final().getpixel(xy)
        if a:
            self.set_brush((r, g, b))

    def paint_start(self, event):
        if self.tool_var.get() == "view" or self.result is None:
            return
        self.stroke = {}
        self.last_pixel = None
        self.paint_move(event)

    def paint_move(self, event):
        if self.stroke is None:
            return
        xy = self.preview.pixel_at(event.x, event.y)
        if xy is None:
            self.last_pixel = None
            return
        # Fill the gap since the last event so fast drags draw a continuous line.
        points = line_points(self.last_pixel, xy) if self.last_pixel else [xy]
        self.last_pixel = xy
        value = (*self.brush, 255) if self.tool_var.get() == "pencil" else (0, 0, 0, 0)
        changed = False
        for p in points:
            if self.edits.get(p) == value:
                continue
            self.stroke.setdefault(p, self.edits.get(p))
            self.edits[p] = value
            changed = True
        if changed:
            self.preview.show(result=self.final())

    def paint_end(self, event):
        if self.stroke:
            self.undo_stack.append(self.stroke)
            self.update_reset_btn()
        self.stroke = None

    def undo(self):
        if not self.undo_stack or isinstance(self.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        for p, previous in self.undo_stack.pop().items():
            if previous is None:
                self.edits.pop(p, None)
            else:
                self.edits[p] = previous
        self.preview.show(result=self.final())
        self.update_reset_btn()

    def clear_paint(self):
        self.edits.clear()
        self.undo_stack.clear()
        self.preview.show(result=self.final())
        self.update_reset_btn()

    # ---------- saving ----------

    def save(self):
        if self.result is None:
            return
        ext = "." + self.format_var.get()
        scale = 1 if ext == ".ico" else int(self.scale_var.get())
        w, h = self.result.width * scale, self.result.height * scale
        stem = self.source_path.stem if self.source_path else "pixel-art"
        folder = self.source_path.parent if self.source_path else Path.home()
        path = filedialog.asksaveasfilename(
            title="Save pixel art", initialdir=str(folder), initialfile=f"{stem}_{w}x{h}{ext}",
            defaultextension=ext, filetypes=[(FORMATS[ext[1:]], f"*{ext}"), ("All files", "*.*")])
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() not in pixelize.SAVE_TYPES:
            path = path.with_suffix(ext)
        try:
            pixelize.save(self.final(), path, 1 if path.suffix.lower() == ".ico" else scale)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.saved_path = path
        self.folder_btn.config(state="normal")
        self.info_label.config(text=f"Saved  →  {path.name}", fg=GOOD)

    def show_folder(self):
        if self.saved_path and self.saved_path.exists():
            reveal(self.saved_path)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
        except Exception:
            pass
    arg = sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).is_file() else None
    App(arg).mainloop()

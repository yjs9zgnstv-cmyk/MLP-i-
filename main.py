"""
MLP Digit Recognition — Main Application
Курсовой проект: Распознавание цифр с помощью многослойного перцептрона
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model import MLPModel

# ── Colors & Fonts ─────────────────────────────────────────────────────────
BG        = "#1a1a2e"
BG2       = "#16213e"
CARD      = "#0f3460"
ACCENT    = "#e94560"
SUCCESS   = "#00b894"
WARNING   = "#fdcb6e"
FG        = "#eaeaea"
FG2       = "#aaaacc"
FONT_MAIN = ("Segoe UI", 10)
FONT_H1   = ("Segoe UI", 16, "bold")
FONT_H2   = ("Segoe UI", 12, "bold")
FONT_MONO = ("Consolas", 9)


class DigitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLP — Распознавание цифр")
        self.geometry("960x680")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.model = MLPModel()
        self.drawing = False
        self.last_x = self.last_y = None
        self._build_ui()
        self._check_model()

    # ── UI Construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self, bg=ACCENT, height=5)
        header.pack(fill="x")

        title_bar = tk.Frame(self, bg=BG2, pady=12)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="🔢  MLP — Распознавание цифр",
                 font=FONT_H1, bg=BG2, fg=FG).pack(side="left", padx=20)
        tk.Label(title_bar, text="MNIST  •  TensorFlow/Keras",
                 font=FONT_MAIN, bg=BG2, fg=FG2).pack(side="right", padx=20)

        # ── Main body ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left panel — canvas + buttons
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")

        tk.Label(left, text="Нарисуйте цифру:", font=FONT_H2, bg=BG, fg=FG).pack(anchor="w")
        tk.Label(left, text="(левая кнопка мыши)", font=FONT_MAIN, bg=BG, fg=FG2).pack(anchor="w", pady=(0, 8))

        canvas_frame = tk.Frame(left, bg=CARD, bd=0, relief="flat",
                                highlightthickness=2, highlightbackground=ACCENT)
        canvas_frame.pack()

        self.canvas = tk.Canvas(canvas_frame, width=280, height=280,
                                bg="#000000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # PIL image that mirrors the canvas (used for prediction, no Ghostscript)
        from PIL import ImageDraw
        self._pil_img  = Image.new("L", (280, 280), 0)
        self._pil_draw = ImageDraw.Draw(self._pil_img)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=10)

        self._btn(btn_row, "🔍  Распознать", self._predict, ACCENT).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn(btn_row, "🗑  Очистить",   self._clear,   CARD ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._btn(left, "📂  Загрузить изображение", self._load_image, BG2).pack(fill="x", pady=(0, 6))

        # Right panel
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        # Result card
        res_card = tk.Frame(right, bg=CARD, bd=0, relief="flat")
        res_card.pack(fill="x", pady=(0, 10))
        res_inner = tk.Frame(res_card, bg=CARD, padx=16, pady=14)
        res_inner.pack(fill="x")

        tk.Label(res_inner, text="РЕЗУЛЬТАТ", font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=FG2).pack(anchor="w")
        self.lbl_result = tk.Label(res_inner, text="—", font=("Segoe UI", 52, "bold"),
                                   bg=CARD, fg=ACCENT)
        self.lbl_result.pack(anchor="w")
        self.lbl_conf = tk.Label(res_inner, text="Нарисуйте или загрузите цифру",
                                 font=FONT_MAIN, bg=CARD, fg=FG2)
        self.lbl_conf.pack(anchor="w")

        # Probability bars
        prob_card = tk.Frame(right, bg=CARD, bd=0, relief="flat")
        prob_card.pack(fill="both", expand=True, pady=(0, 10))
        prob_inner = tk.Frame(prob_card, bg=CARD, padx=16, pady=12)
        prob_inner.pack(fill="both", expand=True)

        tk.Label(prob_inner, text="ВЕРОЯТНОСТИ ПО КЛАССАМ",
                 font=("Segoe UI", 9, "bold"), bg=CARD, fg=FG2).pack(anchor="w", pady=(0, 8))

        self.prob_vars  = []
        self.prob_bars  = []
        self.prob_lbls  = []
        self.prob_pcts  = []

        for i in range(10):
            row = tk.Frame(prob_inner, bg=CARD)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=str(i), width=2, font=("Segoe UI", 9, "bold"),
                     bg=CARD, fg=FG).pack(side="left")

            var = tk.DoubleVar(value=0)
            self.prob_vars.append(var)

            style_name = f"D{i}.Horizontal.TProgressbar"
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(style_name,
                            troughcolor=BG2,
                            background=ACCENT if i == 0 else "#4a4a8a",
                            thickness=14, borderwidth=0)

            pb = ttk.Progressbar(row, variable=var, maximum=100,
                                 style=style_name, length=260)
            pb.pack(side="left", padx=(6, 6))
            self.prob_bars.append(pb)

            lbl = tk.Label(row, text="0%", width=5, anchor="e",
                           font=FONT_MONO, bg=CARD, fg=FG2)
            lbl.pack(side="left")
            self.prob_pcts.append(lbl)

        # Model info + train panel
        info_card = tk.Frame(right, bg=CARD, bd=0, relief="flat")
        info_card.pack(fill="x")
        info_inner = tk.Frame(info_card, bg=CARD, padx=16, pady=10)
        info_inner.pack(fill="x")

        top_row = tk.Frame(info_inner, bg=CARD)
        top_row.pack(fill="x")

        self.lbl_status = tk.Label(top_row, text="⏳  Загрузка...",
                                   font=FONT_MAIN, bg=CARD, fg=WARNING)
        self.lbl_status.pack(side="left")

        self._btn(top_row, "⚙  Обучить модель", self._train_model, ACCENT,
                  small=True).pack(side="right")

        self.lbl_arch = tk.Label(info_inner,
                                 text="Архитектура: 784 → 256 → 128 → 64 → 10",
                                 font=FONT_MONO, bg=CARD, fg=FG2)
        self.lbl_arch.pack(anchor="w", pady=(4, 0))

        self.train_log = tk.Text(info_inner, height=4, bg=BG, fg=SUCCESS,
                                 font=FONT_MONO, bd=0, relief="flat",
                                 state="disabled", wrap="word")
        self.train_log.pack(fill="x", pady=(6, 0))

    def _btn(self, parent, text, cmd, color, small=False):
        f = ("Segoe UI", 9) if small else ("Segoe UI", 10, "bold")
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=FG, font=f,
                      relief="flat", bd=0, cursor="hand2",
                      activebackground=ACCENT, activeforeground=FG,
                      padx=10, pady=6 if small else 10)
        return b

    # ── Drawing ────────────────────────────────────────────────────────────
    def _on_press(self, e):
        self.drawing = True
        self.last_x, self.last_y = e.x, e.y

    def _on_drag(self, e):
        if self.drawing and self.last_x is not None:
            # Draw on Tkinter canvas (display)
            self.canvas.create_line(self.last_x, self.last_y, e.x, e.y,
                                    fill="white", width=16,
                                    capstyle=tk.ROUND, smooth=True)
            # Draw on PIL image (for prediction — no Ghostscript needed)
            self._pil_draw.line([self.last_x, self.last_y, e.x, e.y],
                                fill=255, width=16)
            self.last_x, self.last_y = e.x, e.y

    def _on_release(self, e):
        self.drawing = False
        self.last_x = self.last_y = None

    def _clear(self):
        self.canvas.delete("all")
        from PIL import ImageDraw
        self._pil_img  = Image.new("L", (280, 280), 0)
        self._pil_draw = ImageDraw.Draw(self._pil_img)
        self.lbl_result.config(text="—", fg=ACCENT)
        self.lbl_conf.config(text="Нарисуйте или загрузите цифру")
        for i in range(10):
            self.prob_vars[i].set(0)
            self.prob_pcts[i].config(text="0%")

    # ── Prediction ─────────────────────────────────────────────────────────
    def _get_canvas_image(self):
        """Return the PIL mirror image resized to 28×28 — no Ghostscript needed."""
        return self._pil_img.resize((28, 28), Image.LANCZOS)

    def _predict(self):
        if not self.model.is_ready():
            messagebox.showwarning("Модель не загружена",
                                   "Сначала обучите модель кнопкой «Обучить модель».")
            return
        try:
            img = self._get_canvas_image()
            arr = np.array(img, dtype=np.float32) / 255.0
            self._run_prediction(arr)
        except Exception as ex:
            messagebox.showerror("Ошибка", str(ex))

    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not path:
            return
        if not self.model.is_ready():
            messagebox.showwarning("Модель не загружена",
                                   "Сначала обучите модель.")
            return
        img = Image.open(path).convert("L").resize((28, 28), Image.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Show preview on canvas
        preview = img.resize((280, 280), Image.NEAREST)
        self._canvas_img = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._canvas_img)
        self._run_prediction(arr)

    def _run_prediction(self, arr):
        probs = self.model.predict(arr)
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred]) * 100

        color = SUCCESS if conf > 80 else (WARNING if conf > 50 else ACCENT)
        self.lbl_result.config(text=str(pred), fg=color)
        self.lbl_conf.config(text=f"Уверенность: {conf:.1f}%")

        # Update bars
        style = ttk.Style()
        for i in range(10):
            pct = float(probs[i]) * 100
            self.prob_vars[i].set(pct)
            self.prob_pcts[i].config(text=f"{pct:.1f}%")
            bar_color = ACCENT if i == pred else "#4a4a8a"
            style_name = f"D{i}.Horizontal.TProgressbar"
            style.configure(style_name, background=bar_color)

    # ── Model ──────────────────────────────────────────────────────────────
    def _check_model(self):
        if self.model.load():
            self.lbl_status.config(text="✅  Модель загружена (mnist_mlp.h5)",
                                   fg=SUCCESS)
        else:
            self.lbl_status.config(
                text="⚠  Модель не найдена — нажмите «Обучить модель»",
                fg=WARNING)

    def _train_model(self):
        def run():
            self._log("⏳ Загрузка MNIST...")
            self.lbl_status.config(text="⏳  Обучение...", fg=WARNING)

            def cb(epoch, logs):
                self._log(f"  Эпоха {epoch+1:02d}  loss={logs['loss']:.4f}"
                          f"  acc={logs['accuracy']*100:.1f}%"
                          f"  val_acc={logs.get('val_accuracy', 0)*100:.1f}%")

            acc = self.model.train(callback=cb)
            self._log(f"✅ Готово! Точность на тесте: {acc*100:.2f}%")
            self.lbl_status.config(
                text=f"✅  Модель обучена  (acc={acc*100:.1f}%)", fg=SUCCESS)

        threading.Thread(target=run, daemon=True).start()

    def _log(self, text):
        self.train_log.config(state="normal")
        self.train_log.insert("end", text + "\n")
        self.train_log.see("end")
        self.train_log.config(state="disabled")


if __name__ == "__main__":
    app = DigitApp()
    app.mainloop()

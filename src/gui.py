import sys
import subprocess
import os
import glob
import re
import shutil

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel,
    QCheckBox, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem,
    QFileDialog
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl, QThread, Signal, QObject
from PySide6.QtGui import QIcon


if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(sys.executable)
    PYTHON = os.path.join(ROOT, "python", "Scripts", "python.exe")
else:
    ROOT = os.path.dirname(os.path.dirname(__file__))
    PYTHON = sys.executable
SEARCH_SCRIPT = os.path.join(ROOT, "src", "search_ultimate.py") 
CUT_SCRIPT = os.path.join(ROOT, "src", "cut_clip.py") 
CLIPS_DIR = "clips"



def find_latest_clip():
    files = glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def time_to_seconds(t):
    """Конвертация 'MM:SS' или 'HH:MM:SS' в секунды."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


# -----------------------------
# ОКНО НАСТРОЕК
# -----------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Настройки")

        layout = QVBoxLayout(self)

        self.llm_select_checkbox = QCheckBox("AI выбор лучшего результата")
        self.llm_expand_checkbox = QCheckBox("AI расширение запроса")
        self.llm_refine_checkbox = QCheckBox("AI уточнение границ клипа")

        layout.addWidget(self.llm_select_checkbox)
        layout.addWidget(self.llm_expand_checkbox)
        layout.addWidget(self.llm_refine_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)


# -----------------------------
# WORKER ДЛЯ ПОИСКА
# -----------------------------
class SearchWorker(QObject):
    finished = Signal(str)
    line = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env
            )

            output = []

            for line in self.process.stdout:
                line = line.strip()

                if not line:
                    continue

                output.append(line)
                self.line.emit(line)

            self.process.wait()

            self.finished.emit("\n".join(output))

        except Exception as e:
            self.finished.emit(f"ERROR: {e}")

# -----------------------------
# WORKER ДЛЯ ТРАНСКРИБАЦИИ
# -----------------------------

class TranscribeWorker(QObject):
    finished = Signal(str)
    line = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env
            )

            output = []

            while True:
                line = self.process.stdout.readline()

                if not line:
                    if self.process.poll() is not None:
                        break
                    continue

                line = line.rstrip()

                if line:
                    output.append(line)
                    self.line.emit(line)

            self.process.wait()

            if self.process.returncode != 0:
                self.finished.emit("CANCELLED")
                return

            self.finished.emit("\n".join(output))

        except Exception as e:
            self.finished.emit(f"ERROR: {e}")
# -----------------------------
# WORKER ДЛЯ ВЫРЕЗКИ
# -----------------------------
class CutWorker(QObject):
    finished = Signal(str)
    line = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            output_lines = []

            for line in process.stdout:
                line = line.rstrip()

                if not line:
                    continue

                output_lines.append(line)

                self.line.emit(line)

            process.wait()

            full_output = "\n".join(output_lines)

            self.finished.emit(full_output)

        except Exception as e:
            self.finished.emit(f"ERROR:{e}")


# -----------------------------
# ГЛАВНОЕ ОКНО
# -----------------------------
class App(QWidget):

    def closeEvent(self, event):

        self.is_closing = True

        workers = [
            "worker",
            "cut_worker",
            "transcribe_worker"
        ]

        for name in workers:

            if hasattr(self, name):

                worker = getattr(self, name)

                if hasattr(worker, "process") and worker.process:

                    try:
                        worker.process.kill()
                        worker.process.wait(timeout=2)
                    except:
                        pass

        event.accept()


    def __init__(self):
        super().__init__()

        self.setWindowTitle("VideoSearch AI")
        self.resize(1200, 750)

        # состояния
        self.use_llm_select = True
        self.use_llm_expand = True
        self.use_llm_refine = False

        self.search_results = []
        self.last_query = ""

        self.search_in_progress = False
        self.transcribe_progress = None
        self.is_closing = False

        layout = QHBoxLayout(self)

        # ---------------- ЧАТ ----------------
        left = QVBoxLayout()

        top_bar = QHBoxLayout()

        self.input = QLineEdit()
        self.input.returnPressed.connect(self.send)

        btn_send = QPushButton("Отправить")
        btn_send.clicked.connect(self.send)

        btn_transcribe = QPushButton("Transcribe")
        btn_transcribe.clicked.connect(self.run_transcribe)

        btn_segments = QPushButton("Segments")
        btn_segments.clicked.connect(self.run_segments)

        btn_index = QPushButton("Build Index")
        btn_index.clicked.connect(self.run_build_index)

        # кнопка настроек
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedWidth(40)
        self.settings_btn.clicked.connect(self.open_settings)

        top_bar.addWidget(self.input)
        top_bar.addWidget(btn_send)
        top_bar.addWidget(btn_transcribe)
        top_bar.addWidget(btn_segments)
        top_bar.addWidget(btn_index)
        top_bar.addWidget(self.settings_btn)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)

        # ---------------- СПИСОК РЕЗУЛЬТАТОВ ----------------
        self.results_list = QListWidget()

        # ВАЖНО:
        # itemClicked работает стабильнее чем itemDoubleClicked
        self.results_list.itemDoubleClicked.connect(self.play_selected_result)

        left.addLayout(top_bar)
        left.addWidget(self.chat)

        left.addWidget(QLabel("Результаты поиска:"))
        left.addWidget(self.results_list)

        # ---------------- ВИДЕО ----------------
        right = QVBoxLayout()

        self.video_widget = QVideoWidget()

        self.player = QMediaPlayer()
        self.audio = QAudioOutput()

        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

        right.addWidget(self.video_widget)

        # --- панель управления видео ---
        video_controls = QHBoxLayout()

        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("❌ Убрать")
        self.btn_stop.clicked.connect(self.stop_video)

        video_controls.addWidget(self.btn_pause)
        video_controls.addWidget(self.btn_stop)

        right.addLayout(video_controls)

        layout.addLayout(left, 3)
        layout.addLayout(right, 2)

        # приветственное сообщение
        self.show_welcome_message()

    # ---------------- ПРИВЕТСТВИЕ ----------------
    def show_welcome_message(self):
        self.chat.append("👋 Добро пожаловать в VideoSearch AI!")
        self.chat.append("💡 Введите запрос для поиска по видео")
        self.chat.append("💡 Изначально уже загружены видео на тему ии")
        self.chat.append("💡 Можете выполнить запрос на эту тему")
        self.chat.append("💡 Двойной клик по результату — вырезать и воспроизвести клип")
        self.chat.append("💡 Кнопки ⏸ и ⏹ — управление воспроизведением")
        self.chat.append("⚙ Нажмите ⚙ для настройки AI функций\n")

    # ---------------- НАСТРОЙКИ ----------------
    def open_settings(self):
        dialog = SettingsDialog(self)

        dialog.llm_select_checkbox.setChecked(self.use_llm_select)
        dialog.llm_expand_checkbox.setChecked(self.use_llm_expand)
        dialog.llm_refine_checkbox.setChecked(self.use_llm_refine)

        if dialog.exec():
            self.use_llm_select = dialog.llm_select_checkbox.isChecked()
            self.use_llm_expand = dialog.llm_expand_checkbox.isChecked()
            self.use_llm_refine = dialog.llm_refine_checkbox.isChecked()

            self.chat.append(
                f"⚙ Настройки: "
                f"LLM Select={'ON' if self.use_llm_select else 'OFF'}, "
                f"LLM Expand={'ON' if self.use_llm_expand else 'OFF'}, "
                f"LLM Refine={'ON' if self.use_llm_refine else 'OFF'}"
            )

    # ---------------- ПАРСИНГ КЛИПА ----------------
    def extract_clip_path(self, text):
        for line in text.split("\n"):

            if "Фрагмент сохранён:" in line:
                return line.split("Фрагмент сохранён:")[-1].strip()

            if "CLIP_PATH:" in line:
                return line.split("CLIP_PATH:")[-1].strip()

        return None

    # ---------------- ПАРСИНГ РЕЗУЛЬТАТОВ ----------------
    def parse_results(self, text):
        results = []

        lines = text.splitlines()

        current = None

        for line in lines:

            # 1) video.mp4
            m = re.match(r"^(\d+)\)\s+(.+)$", line)

            if m:
                if current:
                    results.append(current)

                current = {
                    "index": int(m.group(1)),
                    "video": m.group(2).strip(),
                    "start": "",
                    "end": "",
                    "score": "",
                    "preview": "",
                    "full_text": ""
                }

                continue

            # [TIME]
            if current and "[TIME]" in line:
                tm = re.search(r"\[TIME\]\s+(.+?)\s+-\s+(.+)", line)

                if tm:
                    current["start"] = tm.group(1).strip()
                    current["end"] = tm.group(2).strip()

                continue

            # score=
            if current and line.startswith("score="):
                current["score"] = line.strip()
                continue

            # FULL_TEXT: полный текст для LLM уточнения
            if current and line.startswith("FULL_TEXT:"):
                current["full_text"] = line[len("FULL_TEXT:"):].strip()
                continue

            # 📝
            if current and line.startswith("📝"):
                current["preview"] += line[1:].strip() + "\n"
                continue

            # продолжение текста
            if current and current["preview"]:
                current["preview"] += line + "\n"

        if current:
            results.append(current)

        return results

    # ---------------- ПОКАЗ РЕЗУЛЬТАТОВ ----------------
    def update_results_ui(self):
        self.results_list.clear()

        for r in self.search_results:

            text = (
                f"{r['index']}) "
                f"{r['video']} "
                f"[{r['start']} - {r['end']}]"
            )

            item = QListWidgetItem(text)

            tooltip = (
                f"{r['score']}\n\n"
                f"{r['preview'][:1500]}"
            )

            item.setToolTip(tooltip)

            self.results_list.addItem(item)

    # ---------------- ЗАПУСК ВЫРЕЗКИ ----------------
    def cut_result(self, idx):

        if self.search_in_progress:
            self.chat.append("⚠ Уже выполняется поиск/вырезка")
            return

        if idx < 1 or idx > len(self.search_results):
            return

        result = self.search_results[idx - 1]

        query = self.last_query
        video = result["video"]
        start = time_to_seconds(result["start"])
        end = time_to_seconds(result["end"])
        text = result.get("full_text", "") or result.get("preview", "").strip()

        # лёгкий скрипт вырезки — не перезапускает поиск
        cmd = [
            PYTHON,
            CUT_SCRIPT,
            query,
            "--video",
            video,
            "--start",
            str(start),
            "--end",
            str(end),
            "--text",
            text
        ]

        if self.use_llm_refine:
            cmd.append("--llm-refine")
        else:
            cmd.append("--no-llm-refine")

        self.chat.append(f"\n✂ Вырезка результата #{idx}...\n")

        self.search_in_progress = True

        # THREAD
        self.cut_thread = QThread()

        self.cut_worker = CutWorker(cmd)

        self.cut_worker.moveToThread(self.cut_thread)

        self.cut_thread.started.connect(self.cut_worker.run)

        self.cut_worker.line.connect(self.on_cut_line)

        self.cut_worker.finished.connect(self.on_cut_finished)

        self.cut_worker.finished.connect(self.cut_thread.quit)

        self.cut_worker.finished.connect(self.cut_worker.deleteLater)

        self.cut_thread.finished.connect(self.cut_thread.deleteLater)

        self.cut_thread.start()


    # ---------------- КЛИК ПО РЕЗУЛЬТАТУ ----------------
    def play_selected_result(self, item):

        print("ITEM CLICKED")

        row = self.results_list.row(item)

        print("ROW:", row)

        print("RESULTS:", len(self.search_results))

        if row < 0 or row >= len(self.search_results):
            print("BAD ROW")
            return

        result = self.search_results[row]

        idx = result["index"]

        print("SELECTED IDX:", idx)

        self.chat.append(
            f"\n🎯 Выбран результат #{idx}"
        )

        self.cut_result(idx)

        # ---------------- ОТПРАВКА ----------------
    def send(self):

        if self.search_in_progress:
            self.chat.append("⚠ Дождитесь завершения поиска")
            return

        query = self.input.text().strip()

        if not query:
            return

        # ---------------- ВЫБОР РЕЗУЛЬТАТА ЧЕРЕЗ ЧАТ ----------------
        if query.isdigit() and self.search_results:

            idx = int(query)

            self.input.clear()

            found = False

            for r in self.search_results:

                if r["index"] == idx:
                    found = True
                    break

            if not found:
                self.chat.append(f"❌ Результат #{idx} не найден")
                return

            self.chat.append(f"\n🎯 Выбран результат #{idx}")

            self.cut_result(idx)

            return

        # ---------------- ОБЫЧНЫЙ ПОИСК ----------------
        self.last_query = query

        self.chat.append(f"\n🧑 {query}")

        self.input.clear()

        # новый запрос — останавливаем старый клип
        self.player.stop()

        self.results_list.clear()
        self.search_results = []

        self.chat.append("⏳ Поиск...\n")

        cmd = [
            PYTHON,
            SEARCH_SCRIPT,
            query
        ]

        if self.use_llm_select:
            cmd.append("--llm-select")
            cmd.append("--auto-cut")

        if self.use_llm_expand:
            cmd.append("--llm-expand")

        if self.use_llm_refine:
            cmd.append("--llm-refine")

        self.search_in_progress = True

        # THREAD
        self.thread = QThread()

        self.worker = SearchWorker(cmd)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.line.connect(self.on_search_line)

        self.worker.finished.connect(self.on_search_finished)

        self.worker.finished.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # ---------------- СТРОКИ ПОИСКА ----------------
    def on_search_line(self, line):

        # фильтр мусора: ffmpeg, progress bars, warnings
        skip_prefixes = (
            "ffmpeg version",
            "built with",
            "configuration:",
            "libav",
            "libsw",
            "libavdevice",
            "libavfilter",
            "Input #",
            "Output #",
            "Stream #",
            "Stream mapping",
            "Metadata:",
            "Press [q]",
            "[libx264",
            "[aac @",
            "[mp4 @",
            "[out#0",
            "frame=",
            "major_brand",
            "minor_version",
            "compatible_brands",
            "creation_time",
            "handler_name",
            "encoder:",
            "Side data:",
            "CPB properties:",
            "Duration:",
            "Loading weights:",
            "Both `max_new_tokens`",
            "Passing `generation_config`",
            "The following generation flags",
        )

        for prefix in skip_prefixes:
            if line.startswith(prefix) or line.lstrip().startswith(prefix):
                return

        self.chat.append(f"🔹 {line[:300]}")

    # ---------------- ЗАВЕРШЕНИЕ ПОИСКА ----------------
    def on_search_finished(self, full_output):

        self.search_in_progress = False   # ← СРАЗУ В НАЧАЛЕ

        if full_output.startswith("ERROR:"):
            self.chat.append(full_output)
            return
        # DEBUG
        print("\n========== FULL OUTPUT ==========\n")
        print(full_output)
        print("\n=================================\n")

        # ---------------- ПАРСИНГ РЕЗУЛЬТАТОВ ----------------
        self.search_results = self.parse_results(full_output)

        print("PARSED RESULTS:", len(self.search_results))

        if self.search_results:
            self.update_results_ui()

            self.chat.append(
                f"\n📋 Найдено результатов: {len(self.search_results)}"
            )

            # DEBUG
            for r in self.search_results:
                print("RESULT:", r)

        else:
            self.chat.append("\n❌ Результаты не распарсились")

        self.chat.append("\n🤖 Поиск завершён\n")

        # ---------------- AI AUTO CUT ----------------
        if self.use_llm_select:

            clip = self.extract_clip_path(full_output)

            if clip and os.path.exists(clip):

                self.chat.append(f"🎬 Найден клип: {clip}")

                self.play_video(clip)

                self.chat.append(
                    "💡 Можно поставить паузу или выбрать другой клип "
                    "двойным кликом в списке результатов"
                )

            else:
                self.chat.append("❌ Клип не найден (смотри лог выше)")
    

    # ---------------- СТРОКИ ВЫРЕЗКИ ----------------
    def on_cut_line(self, line):

        # фильтр мусора: ffmpeg, progress bars, warnings
        skip_prefixes = (
            "ffmpeg version",
            "built with",
            "configuration:",
            "libav",
            "libsw",
            "libavdevice",
            "libavfilter",
            "Input #",
            "Output #",
            "Stream #",
            "Stream mapping",
            "Metadata:",
            "Press [q]",
            "[libx264",
            "[aac @",
            "[mp4 @",
            "[out#0",
            "frame=",
            "major_brand",
            "minor_version",
            "compatible_brands",
            "creation_time",
            "handler_name",
            "encoder:",
            "Side data:",
            "CPB properties:",
            "Duration:",
            "Loading weights:",
            "Both `max_new_tokens`",
            "Passing `generation_config`",
            "The following generation flags",
        )

        for prefix in skip_prefixes:
            if line.startswith(prefix) or line.lstrip().startswith(prefix):
                return

        self.chat.append(f"✂ {line[:300]}")

    # ---------------- ЗАВЕРШЕНИЕ ВЫРЕЗКИ ----------------
    def on_cut_finished(self, full_output):

        self.search_in_progress = False

        if full_output.startswith("ERROR:"):
            self.chat.append(full_output)
            return

        print("\n========== CUT OUTPUT ==========\n")
        print(full_output)
        print("\n================================\n")

        clip = self.extract_clip_path(full_output)

        if clip and os.path.exists(clip):

            self.chat.append(f"🎬 Открыт клип: {clip}")

            self.play_video(clip)

        else:
            self.chat.append("❌ Клип не найден")
    



    def run_transcribe(self):

        new_video_dir = os.path.join(ROOT, "NewVideo")

        self.chat.append(
            "\n🎤 Начинается транскрибация всех видео из NewVideo...\n"
        )

        cmd = [
            PYTHON,
            os.path.join(ROOT, "src", "_transcribe.py"),
            new_video_dir
        ]

        self.transcribe_thread = QThread()

        self.transcribe_worker = TranscribeWorker(cmd)

        self.transcribe_worker.moveToThread(
            self.transcribe_thread
        )

        self.transcribe_thread.started.connect(
            self.transcribe_worker.run
        )

        self.transcribe_worker.line.connect(
        self.on_transcribe_line
        )

        self.transcribe_worker.finished.connect(
            self.on_transcribe_finished
        )

        self.transcribe_worker.finished.connect(
            self.transcribe_thread.quit
        )

        self.transcribe_worker.finished.connect(
            self.transcribe_worker.deleteLater
        )

        self.transcribe_thread.finished.connect(
            self.transcribe_thread.deleteLater
        )

        self.transcribe_thread.start()


  

    def on_transcribe_line(self, line):

        if "%|" in line and "frames/s" in line:
            return

        if "frames/s" in line:
            return

        if "it/s" in line:
            return
            

        # ---------- прогресс Whisper ----------
        m = re.search(r"(\d+)%\|", line)

        if m:
            percent = int(m.group(1))

            bar = "█" * (percent // 10)
            bar += "░" * (10 - len(bar))

            text = f"🎤 [{bar}] {percent}%"

            if text != self.transcribe_progress:
                self.chat.append(text)
                
                self.transcribe_progress = text

            return

        # ---------- фильтр мусора ----------
        skip = (
            "FP16 is not supported",
            "frames/s",
            "it/s"
        )

        for s in skip:
            if s in line:
                return

        # ---------- красивые сообщения ----------
        if line.startswith("Video found:"):
            self.chat.append(f"🎥 Найдено видео:{line.split(':')[-1]}")
            return

        if line.startswith("VIDEO_PROGRESS"):
            self.chat.append(f"\n📹 {line.replace('VIDEO_PROGRESS','Видео')}")
            return

        if line.startswith("Processing:"):
            name = line.replace("Processing:", "").strip()
            self.chat.append(f"📄 {name}")
            return

        if line.startswith("Saved:"):
            self.chat.append("✅ Сохранено")
            return

        if line.startswith("Ready"):
            return

        if line.startswith("Whisper"):
            self.chat.append("🧠 Загрузка Whisper...")
            return

        self.chat.append(f"🎤 {line}")



    def on_transcribe_finished(self, output):

        if self.is_closing:
            return

        if output == "CANCELLED":
            self.chat.append("\n❌ Транскрибация была отменена")
            return

        if output.startswith("ERROR:"):
            self.chat.append(output)
            return

        self.chat.append("\n✅ Транскрибация завершена")

        new_video_dir = os.path.join(ROOT, "NewVideo")
        videos_dir = os.path.join(ROOT, "videos")

        os.makedirs(videos_dir, exist_ok=True)

        for file in os.listdir(new_video_dir):
            if file.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v")):
                shutil.move(
                    os.path.join(new_video_dir, file),
                    os.path.join(videos_dir, file)
                )

        self.chat.append("📁 Видео перенесены в videos")

    def run_segments(self):

        self.chat.append("\n📝 Создание сегментов...\n")

        cmd = [
            PYTHON,
            "src/make_segments.py"
        ]

        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        self.chat.append("\nСегменты созданы\n")
        


    def run_build_index(self):

        self.chat.append("\n📚 Построение индекса...\n")

        cmd = [
            PYTHON,
            "src/build_index.py"
        ]

        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        self.chat.append("\nИндекс создан\n")


    # ---------------- ВОСПРОИЗВЕДЕНИЕ ----------------
    def play_video(self, path):

        if not os.path.exists(path):
            self.chat.append(f"❌ Файл не найден: {path}")
            return

        url = QUrl.fromLocalFile(os.path.abspath(path))

        self.player.setSource(url)

        # зацикливаем, чтобы клип не исчезал после одного проигрывания
        self.player.setLoops(QMediaPlayer.Infinite)

        self.player.play()

        self.btn_pause.setText("⏸ Пауза")

        self.chat.append(f"▶ Воспроизведение: {path}")

    # ---------------- ПАУЗА / ПРОДОЛЖЕНИЕ ----------------
    def toggle_pause(self):

        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_pause.setText("▶ Продолжить")
            self.chat.append("⏸ Пауза")

        elif self.player.playbackState() == QMediaPlayer.PausedState:
            self.player.play()
            self.btn_pause.setText("⏸ Пауза")
            self.chat.append("▶ Продолжить")

    # ---------------- СТОП ----------------
    def stop_video(self):

        self.player.stop()
        self.btn_pause.setText("⏸ Пауза")
        self.chat.append("⏹ Стоп")


# -----------------------------
# СТАРТ
# -----------------------------
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = App()

    window.show()

    sys.exit(app.exec())

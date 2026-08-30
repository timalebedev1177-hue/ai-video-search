import sys
import subprocess
import os
import glob
import re
import shutil

CREATE_NO_WINDOW = getattr(
    subprocess,
    "CREATE_NO_WINDOW",
    0
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel,
    QCheckBox, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl, QThread, Signal, QObject


# ============================================================
# ПУТИ
# ============================================================

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))

    # Отдельный Python, который лежит рядом с EXE
    PYTHON = os.path.join(
        ROOT,
        "python",
        "python.exe"
    )

else:
    ROOT = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    PYTHON = sys.executable


SEARCH_SCRIPT = os.path.join(
    ROOT,
    "src",
    "search_ultimate.py"
)

CUT_SCRIPT = os.path.join(
    ROOT,
    "src",
    "cut_clip.py"
)

CLIPS_DIR = os.path.join(
    ROOT,
    "clips"
)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def find_latest_clip():
    """
    Возвращает последний созданный mp4-клип.
    """
    if not os.path.exists(CLIPS_DIR):
        return None

    files = glob.glob(
        os.path.join(CLIPS_DIR, "*.mp4")
    )

    if not files:
        return None

    return max(
        files,
        key=os.path.getctime
    )


def time_to_seconds(t):
    """
    Конвертация:
        MM:SS
        HH:MM:SS
    в секунды.
    """

    if not t:
        return 0.0

    t = str(t).strip()

    try:
        parts = [
            float(p)
            for p in t.split(":")
        ]

        if len(parts) == 2:
            return parts[0] * 60 + parts[1]

        if len(parts) == 3:
            return (
                parts[0] * 3600
                + parts[1] * 60
                + parts[2]
            )

    except Exception:
        pass

    return 0.0


# ============================================================
# ОКНО НАСТРОЕК
# ============================================================

class SettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Настройки")

        layout = QVBoxLayout(self)

        self.llm_select_checkbox = QCheckBox(
            "AI выбор лучшего результата"
        )

        self.llm_expand_checkbox = QCheckBox(
            "AI расширение запроса"
        )

        self.llm_refine_checkbox = QCheckBox(
            "AI уточнение границ клипа"
        )

        layout.addWidget(
            self.llm_select_checkbox
        )

        layout.addWidget(
            self.llm_expand_checkbox
        )

        layout.addWidget(
            self.llm_refine_checkbox
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(
            self.accept
        )

        buttons.rejected.connect(
            self.reject
        )

        layout.addWidget(buttons)


# ============================================================
# WORKER ПОИСКА
# ============================================================

class SearchWorker(QObject):

    finished = Signal(str)
    line = Signal(str)

    DEBUG_PREFIXES = (
        "NORMALIZED QUERY:",
        "QUERY VEC SHAPE:",
        "RAW LLM OUTPUT:",
        "PARSED VARIANTS:",
        "FINAL CLEAN VARIANTS:",
        "VARIANTS AFTER CLEAN:",
        "AFTER CLEAN:",
        "BEFORE CLEAN:",
        "FINAL VARIANTS FROM LLM:",
        "SIM:",
    )

    def __init__(self, cmd):
        super().__init__()

        self.cmd = cmd
        self.process = None

    def _should_emit(self, line):
        return not any(
            line.startswith(prefix)
            for prefix in self.DEBUG_PREFIXES
        )

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
                env=env,
                cwd=ROOT
            )

            output = []

            for raw_line in self.process.stdout:

                line = raw_line.rstrip()

                if not line:
                    continue

                # Сохраняем ВСЁ
                output.append(line)

                # В GUI выводим только полезное
                if self._should_emit(line):
                    self.line.emit(line)

            self.process.wait()

            self.finished.emit(
                "\n".join(output)
            )

        except Exception as e:

            self.finished.emit(
                f"ERROR: {e}"
            )


# ============================================================
# WORKER ТРАНСКРИБАЦИИ
# ============================================================

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
                env=env,
                cwd=ROOT
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

                self.finished.emit(
                    "CANCELLED"
                )

                return

            self.finished.emit(
                "\n".join(output)
            )

        except Exception as e:

            self.finished.emit(
                f"ERROR: {e}"
            )


# -----------------------------
# WORKER ДЛЯ ВЫРЕЗКИ
# -----------------------------
class CutWorker(QObject):

    finished = Signal(str)
    line = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            env = os.environ.copy()

            # ВАЖНО:
            # Принудительно запускаем дочерний Python в UTF-8.
            # Иначе Windows может использовать cp1251,
            # из-за чего символы вроде ✂, 🎬, ❌ вызывают
            # UnicodeEncodeError.
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            output_lines = []

            for line in self.process.stdout:
                line = line.rstrip()

                if not line:
                    continue

                output_lines.append(line)
                self.line.emit(line)

            self.process.wait()

            full_output = "\n".join(output_lines)

            self.finished.emit(full_output)

        except Exception as e:
            self.finished.emit(f"ERROR: {e}")

# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

class App(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "VideoSearch AI"
        )

        self.resize(
            1200,
            750
        )

        # ====================================================
        # СОСТОЯНИЯ
        # ====================================================

        self.use_llm_select = False
        self.use_llm_expand = True
        self.use_llm_refine = False

        self.search_results = []

        self.last_query = ""

        self._ignore_query_variant_section = False

        self.search_in_progress = False

        self.transcribe_progress = None

        self.is_closing = False

        self.ai_choice_index = None

        # ====================================================
        # ОСНОВНОЙ LAYOUT
        # ====================================================

        layout = QHBoxLayout(self)

        # ====================================================
        # ЛЕВАЯ ЧАСТЬ
        # ====================================================

        left = QVBoxLayout()

        # ----------------------------------------------------
        # ВЕРХНЯЯ ПАНЕЛЬ
        # ----------------------------------------------------

        top_bar = QHBoxLayout()

        self.input = QLineEdit()

        self.input.setPlaceholderText(
            "Введите запрос..."
        )

        self.input.returnPressed.connect(
            self.send
        )

        btn_send = QPushButton(
            "Отправить"
        )

        btn_send.clicked.connect(
            self.send
        )

        btn_transcribe = QPushButton(
            "Transcribe"
        )

        btn_transcribe.clicked.connect(
            self.run_transcribe
        )

        btn_segments = QPushButton(
            "Segments"
        )

        btn_segments.clicked.connect(
            self.run_segments
        )

        btn_index = QPushButton(
            "Build Index"
        )

        btn_index.clicked.connect(
            self.run_build_index
        )

        self.settings_btn = QPushButton(
            "⚙"
        )

        self.settings_btn.setFixedWidth(
            40
        )

        self.settings_btn.clicked.connect(
            self.open_settings
        )

        top_bar.addWidget(
            self.input
        )

        top_bar.addWidget(
            btn_send
        )

        top_bar.addWidget(
            btn_transcribe
        )

        top_bar.addWidget(
            btn_segments
        )

        top_bar.addWidget(
            btn_index
        )

        top_bar.addWidget(
            self.settings_btn
        )

        # ----------------------------------------------------
        # ЧАТ
        # ----------------------------------------------------

        self.chat = QTextEdit()

        self.chat.setReadOnly(
            True
        )

        # ----------------------------------------------------
        # СПИСОК РЕЗУЛЬТАТОВ
        # ----------------------------------------------------

        self.results_list = QListWidget()

        # Клик по результату
        self.results_list.itemClicked.connect(
            self.play_selected_result
        )

        left.addLayout(
            top_bar
        )

        left.addWidget(
            self.chat
        )

        left.addWidget(
            QLabel(
                "Результаты поиска:"
            )
        )

        left.addWidget(
            self.results_list
        )

        # ====================================================
        # ПРАВАЯ ЧАСТЬ — ВИДЕО
        # ====================================================

        right = QVBoxLayout()

        self.video_widget = QVideoWidget()

        self.player = QMediaPlayer()

        self.audio = QAudioOutput()

        self.player.setAudioOutput(
            self.audio
        )

        self.player.setVideoOutput(
            self.video_widget
        )

        right.addWidget(
            self.video_widget
        )

        # ----------------------------------------------------
        # УПРАВЛЕНИЕ ВИДЕО
        # ----------------------------------------------------

        video_controls = QHBoxLayout()

        self.btn_pause = QPushButton(
            "⏸ Пауза"
        )

        self.btn_pause.clicked.connect(
            self.toggle_pause
        )

        self.btn_stop = QPushButton(
            "❌ Убрать"
        )

        self.btn_stop.clicked.connect(
            self.stop_video
        )

        video_controls.addWidget(
            self.btn_pause
        )

        video_controls.addWidget(
            self.btn_stop
        )

        right.addLayout(
            video_controls
        )

        layout.addLayout(
            left,
            3
        )

        layout.addLayout(
            right,
            2
        )

        # ====================================================
        # ПРИВЕТСТВИЕ
        # ====================================================

        self.show_welcome_message()

    # ========================================================
    # ЗАКРЫТИЕ
    # ========================================================

    def closeEvent(self, event):

        self.is_closing = True

        workers = [
            "worker",
            "cut_worker",
            "transcribe_worker"
        ]

        for name in workers:

            if not hasattr(self, name):
                continue

            worker = getattr(
                self,
                name
            )

            if (
                hasattr(worker, "process")
                and worker.process
            ):

                try:

                    worker.process.kill()

                    worker.process.wait(
                        timeout=2
                    )

                except Exception:
                    pass

        event.accept()

    # ========================================================
    # ПРИВЕТСТВИЕ
    # ========================================================

    def show_welcome_message(self):

        self.chat.append(
            "👋 Добро пожаловать в VideoSearch AI!"
        )

        self.chat.append(
            "💡 Введите запрос для поиска по видео"
        )

        self.chat.append(
            "💡 Изначально уже загружены видео на тему ИИ"
        )

        self.chat.append(
            "💡 Можете выполнить запрос на эту тему"
        )

        self.chat.append(
            "💡 Клик по результату — вырезать и воспроизвести клип"
        )

        self.chat.append(
            "💡 Кнопки ⏸ и ❌ — управление воспроизведением"
        )

        self.chat.append(
            "⚙ Нажмите ⚙ для настройки AI функций\n"
        )

    # ========================================================
    # НАСТРОЙКИ
    # ========================================================

    def open_settings(self):

        dialog = SettingsDialog(
            self
        )

        dialog.llm_select_checkbox.setChecked(
            self.use_llm_select
        )

        dialog.llm_expand_checkbox.setChecked(
            self.use_llm_expand
        )

        dialog.llm_refine_checkbox.setChecked(
            self.use_llm_refine
        )

        if dialog.exec():

            self.use_llm_select = (
                dialog.llm_select_checkbox.isChecked()
            )

            self.use_llm_expand = (
                dialog.llm_expand_checkbox.isChecked()
            )

            self.use_llm_refine = (
                dialog.llm_refine_checkbox.isChecked()
            )

            self.chat.append(
                f"⚙ Настройки: "
                f"LLM Select={'ON' if self.use_llm_select else 'OFF'}, "
                f"LLM Expand={'ON' if self.use_llm_expand else 'OFF'}, "
                f"LLM Refine={'ON' if self.use_llm_refine else 'OFF'}"
            )

    # ========================================================
    # ПОЛУЧЕНИЕ ПУТИ К КЛИПУ
    # ========================================================

    def extract_clip_path(self, text):

        for line in text.splitlines():

            line = line.strip()

            if "Фрагмент сохранён:" in line:

                path = line.split(
                    "Фрагмент сохранён:",
                    1
                )[1].strip()

                if path:
                    return path

            if "CLIP_PATH:" in line:

                path = line.split(
                    "CLIP_PATH:",
                    1
                )[1].strip()

                if path:
                    return path

        return None

    # ========================================================
    # ПОЛУЧЕНИЕ AI CHOICE INDEX
    # ========================================================

    def extract_ai_choice_index(self, text):

        match = re.search(
            r"AI_CHOICE_INDEX:\s*(\d+)",
            text,
            re.IGNORECASE
        )

        if match:
            return int(
                match.group(1)
            )

        return None

    # ========================================================
    # ПАРСИНГ РЕЗУЛЬТАТОВ
    # ========================================================

    def parse_results(self, text):

        results = []

        lines = text.splitlines()

        current = None

        for raw_line in lines:

            line = raw_line.strip()

            if not line:
                continue

            # =================================================
            # НОВЫЙ РЕЗУЛЬТАТ
            #
            # Поддерживается:
            #
            # 1. video.mp4
            # 1) video.mp4
            # [1] video.mp4
            # =================================================

            match = re.match(
                r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+(.+?)\s*$",
                line
            )

            if match:

                # AI-служебная строка
                if line.startswith(
                    "AI выбрал"
                ):
                    continue

                # TIME не является новым результатом
                if "[TIME]" not in line:

                    if current:
                        results.append(
                            current
                        )

                    index = (
                        match.group(1)
                        or match.group(2)
                    )

                    video = match.group(3).strip()

                    current = {
                        "index": int(index),
                        "video": video,
                        "start": "",
                        "end": "",
                        "score": "",
                        "preview": "",
                        "full_text": ""
                    }

                    continue

            # =================================================
            # TIME
            # =================================================

            if current and "[TIME]" in line:

                time_match = re.search(
                    r"\[TIME\]\s+(.+?)\s+-\s+(.+)",
                    line
                )

                if time_match:

                    current["start"] = (
                        time_match.group(1).strip()
                    )

                    current["end"] = (
                        time_match.group(2).strip()
                    )

                continue

            # =================================================
            # SCORE
            # =================================================

            if current and (
                line.startswith("score=")
                or line.startswith("final=")
            ):

                current["score"] = line

                continue

            # =================================================
            # FULL TEXT
            # =================================================

            if current and line.startswith(
                "FULL_TEXT:"
            ):

                current["full_text"] = (
                    line[len("FULL_TEXT:"):].strip()
                )

                continue

            # =================================================
            # PREVIEW
            # =================================================

            if current and line.startswith("📝"):

                current["preview"] += (
                    line[1:].strip()
                    + "\n"
                )

                continue

            # =================================================
            # ПРОДОЛЖЕНИЕ PREVIEW
            # =================================================

            if current and current["preview"]:

                debug_prefixes = (
                    "NORMALIZED QUERY:",
                    "QUERY VEC SHAPE:",
                    "RAW LLM OUTPUT:",
                    "PARSED VARIANTS:",
                    "FINAL CLEAN VARIANTS:",
                    "VARIANTS AFTER CLEAN:",
                    "AFTER CLEAN:",
                    "BEFORE CLEAN:",
                    "FINAL VARIANTS FROM LLM:",
                    "SIM:",
                    "AI_CHOICE_INDEX:",
                    "Профиль поиска:",
                    "Результаты:",
                    "Используемые запросы:",
                    "Варианты запроса:"
                )

                if not line.startswith(
                    debug_prefixes
                ):

                    current["preview"] += (
                        line
                        + "\n"
                    )

        # =====================================================
        # ПОСЛЕДНИЙ РЕЗУЛЬТАТ
        # =====================================================

        if current:
            results.append(
                current
            )

        return results

    # ========================================================
    # ОБНОВЛЕНИЕ СПИСКА
    # ========================================================

    def update_results_ui(self):

        self.results_list.clear()

        for result in self.search_results:

            text = (
                f"{result['index']}) "
                f"{result['video']} "
                f"[{result['start']} - "
                f"{result['end']}]"
            )

            item = QListWidgetItem(
                text
            )

            tooltip = (
                f"{result['score']}\n\n"
                f"{result['preview'][:1500]}"
            )

            item.setToolTip(
                tooltip
            )

            # Сохраняем индекс результата прямо в item
            item.setData(
                256,
                result["index"]
            )

            self.results_list.addItem(
                item
            )

    # ========================================================
    # ВЫРЕЗКА РЕЗУЛЬТАТА
    # ========================================================

    def cut_result(self, idx):

        if not self.search_results:

            self.chat.append(
                "❌ Нет результатов для вырезки"
            )

            return

        if self.search_in_progress:

            self.chat.append(
                "⚠ Сейчас уже выполняется операция"
            )

            return

        # ----------------------------------------------------
        # ИЩЕМ ПО INDEX
        # ----------------------------------------------------

        result = None

        for r in self.search_results:

            try:

                if int(
                    r.get("index", -1)
                ) == int(idx):

                    result = r
                    break

            except Exception:
                continue

        if result is None:

            self.chat.append(
                f"❌ Результат #{idx} не найден"
            )

            return

        # ----------------------------------------------------
        # ДАННЫЕ
        # ----------------------------------------------------

        query = self.last_query

        video = result.get(
            "video",
            ""
        ).strip()

        if not video:

            self.chat.append(
                "❌ У результата отсутствует имя видео"
            )

            return

        try:

            start = time_to_seconds(
                result.get(
                    "start",
                    ""
                )
            )

            end = time_to_seconds(
                result.get(
                    "end",
                    ""
                )
            )

        except Exception as e:

            self.chat.append(
                f"❌ Ошибка времени: {e}"
            )

            return

        if end <= start:

            self.chat.append(
                "❌ Некорректные границы результата"
            )

            return

        text = (
            result.get(
                "full_text",
                ""
            )
            or
            result.get(
                "preview",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # КОМАНДА CUT_CLIP
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # AI REFINE
        # ----------------------------------------------------

        if self.use_llm_refine:

            cmd.append(
                "--llm-refine"
            )

        else:

            cmd.append(
                "--no-llm"
            )

        # ----------------------------------------------------
        # ЛОГ
        # ----------------------------------------------------

        self.chat.append(
            f"\n🎯 Выбран результат #{idx}"
        )

        self.chat.append(
            f"✂ Вырезка результата #{idx}..."
        )

        # ----------------------------------------------------
        # THREAD
        # ----------------------------------------------------

        self.search_in_progress = True

        self.cut_thread = QThread()

        self.cut_worker = CutWorker(
            cmd
        )

        self.cut_worker.moveToThread(
            self.cut_thread
        )

        self.cut_thread.started.connect(
            self.cut_worker.run
        )

        self.cut_worker.line.connect(
            self.on_cut_line
        )

        self.cut_worker.finished.connect(
            self.on_cut_finished
        )

        self.cut_worker.finished.connect(
            self.cut_thread.quit
        )

        self.cut_worker.finished.connect(
            self.cut_worker.deleteLater
        )

        self.cut_thread.finished.connect(
            self.cut_thread.deleteLater
        )

        self.cut_thread.start()

    # ========================================================
    # КЛИК ПО РЕЗУЛЬТАТУ
    # ========================================================

    def play_selected_result(self, item):

        # Берём настоящий index из item,
        # а не предполагаем row == index

        idx = item.data(256)

        if idx is None:

            row = self.results_list.row(
                item
            )

            if (
                row < 0
                or row >= len(
                    self.search_results
                )
            ):
                return

            idx = self.search_results[
                row
            ].get(
                "index",
                row + 1
            )

        try:
            idx = int(idx)
        except Exception:
            return

        self.cut_result(
            idx
        )

    # ========================================================
    # ОТПРАВКА ЗАПРОСА
    # ========================================================

    def send(self):

        if self.search_in_progress:

            self.chat.append(
                "⚠ Дождитесь завершения текущей операции"
            )

            return

        query = self.input.text().strip()

        if not query:
            return

        # =====================================================
        # ВЫБОР РЕЗУЛЬТАТА ПО НОМЕРУ
        # =====================================================

        if (
            query.isdigit()
            and self.search_results
        ):

            idx = int(query)

            self.input.clear()

            found = False

            for result in self.search_results:

                try:

                    if int(
                        result.get(
                            "index",
                            -1
                        )
                    ) == idx:

                        found = True
                        break

                except Exception:
                    pass

            if not found:

                self.chat.append(
                    f"❌ Результат #{idx} не найден"
                )

                return

            self.chat.append(
                f"\n🎯 Выбран результат #{idx}"
            )

            self.cut_result(
                idx
            )

            return

        # =====================================================
        # ОБЫЧНЫЙ ПОИСК
        # =====================================================

        self.last_query = query

        self.chat.append(
            f"\n🧑 {query}"
        )

        self.input.clear()

        # Останавливаем старое видео
        self.player.stop()

        # Очищаем старые результаты
        self.results_list.clear()

        self.search_results = []

        self.ai_choice_index = None

        self._ignore_query_variant_section = False

        self.chat.append(
            "⏳ Поиск...\n"
        )

        # =====================================================
        # КОМАНДА
        # =====================================================

        cmd = [
            PYTHON,
            SEARCH_SCRIPT,
            query
        ]

        # =====================================================
        # AI SELECT
        # =====================================================

        if self.use_llm_select:

            cmd.append(
                "--llm-select"
            )

        # =====================================================
        # AI EXPAND
        # =====================================================

        if self.use_llm_expand:

            cmd.append(
                "--llm-expand"
            )

        # =====================================================
        # AI REFINE
        # =====================================================

        if self.use_llm_refine:

            cmd.append(
                "--llm-refine"
            )

        # =====================================================
        # THREAD
        # =====================================================

        self.thread = QThread()

        self.worker = SearchWorker(
            cmd
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.line.connect(
            self.on_search_line
        )

        self.worker.finished.connect(
            self.on_search_finished
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.start()

    # ========================================================
    # СТРОКИ ПОИСКА
    # ========================================================

    def on_search_line(self, line):

        line = line.strip()

        if not line:
            return

        # ----------------------------------------------------
        # ВАРИАНТЫ ЗАПРОСА
        # ----------------------------------------------------

        if line == "Варианты запроса:":

            self._ignore_query_variant_section = True

            return

        if line == "Используемые запросы:":

            self._ignore_query_variant_section = False

            self.chat.append(
                f"🔹 {line}"
            )

            return

        if self._ignore_query_variant_section:
            return

        # ----------------------------------------------------
        # Строки вида:
        #
        # 1. "вариант"
        # ----------------------------------------------------

        if re.match(
            r'^\d+\.\s*".*"$',
            line
        ):
            return

        # ----------------------------------------------------
        # ФИЛЬТР МУСОРА
        # ----------------------------------------------------

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

            if (
                line.startswith(prefix)
                or line.lstrip().startswith(prefix)
            ):
                return

        self.chat.append(
            f"🔹 {line[:300]}"
        )

    # ========================================================
    # ЗАВЕРШЕНИЕ ПОИСКА
    # ========================================================

    def on_search_finished(
        self,
        full_output
    ):

        self.search_in_progress = False

        if full_output.startswith(
            "ERROR:"
        ):

            self.chat.append(
                full_output
            )

            return

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        print(
            "\n========== FULL OUTPUT ==========\n"
        )

        print(
            full_output
        )

        print(
            "\n=================================\n"
        )

        # ----------------------------------------------------
        # ПАРСИНГ
        # ----------------------------------------------------

        self.search_results = (
            self.parse_results(
                full_output
            )
        )

        print(
            "PARSED RESULTS:",
            len(self.search_results)
        )

        for i, result in enumerate(
            self.search_results
        ):

            print(
                f"RESULT {i}: "
                f"index={result.get('index')} "
                f"video={result.get('video')} "
                f"start={result.get('start')} "
                f"end={result.get('end')}"
            )

        # ----------------------------------------------------
        # ПОКАЗ
        # ----------------------------------------------------

        if self.search_results:

            self.update_results_ui()

            self.chat.append(
                f"\n📋 Найдено результатов: "
                f"{len(self.search_results)}"
            )

        else:

            self.chat.append(
                "\n❌ Результаты не распарсились"
            )

        self.chat.append(
            "\n🤖 Поиск завершён\n"
        )

        # =====================================================
        # AI ВЫБОР
        # =====================================================

        self.ai_choice_index = None

        if self.use_llm_select:

            self.ai_choice_index = (
                self.extract_ai_choice_index(
                    full_output
                )
            )

            print(
                "AI CHOICE INDEX:",
                self.ai_choice_index
            )

            if self.ai_choice_index is not None:

                self.chat.append(
                    f"🤖 AI рекомендует результат "
                    f"#{self.ai_choice_index}"
                )

                # ------------------------------------------------
                # НАХОДИМ РЕЗУЛЬТАТ
                # ------------------------------------------------

                for row, result in enumerate(
                    self.search_results
                ):

                    try:

                        result_index = int(
                            result.get(
                                "index",
                                -1
                            )
                        )

                    except Exception:

                        continue

                    if (
                        result_index
                        == self.ai_choice_index
                    ):

                        item = (
                            self.results_list.item(
                                row
                            )
                        )

                        if item:

                            # Только выделяем
                            # НЕ запускаем автоматически

                            self.results_list.setCurrentItem(
                                item
                            )

                            self.chat.append(
                                f"⭐ AI рекомендует "
                                f"#{self.ai_choice_index}. "
                                f"Вы можете выбрать другой результат."
                            )

                        break

            else:

                self.chat.append(
                    "⚠ AI не сообщил номер "
                    "выбранного результата"
                )

    # ========================================================
    # СТРОКИ ВЫРЕЗКИ
    # ========================================================

    def on_cut_line(self, line):

        line = line.strip()

        if not line:
            return

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

            if (
                line.startswith(prefix)
                or line.lstrip().startswith(prefix)
            ):

                return

        self.chat.append(
            f"✂ {line[:300]}"
        )

    # ========================================================
    # ЗАВЕРШЕНИЕ ВЫРЕЗКИ
    # ========================================================

    def on_cut_finished(
        self,
        full_output
    ):

        self.search_in_progress = False

        if full_output.startswith(
            "ERROR:"
        ):

            self.chat.append(
                full_output
            )

            return

        print(
            "\n========== CUT OUTPUT ==========\n"
        )

        print(
            full_output
        )

        print(
            "\n================================\n"
        )

        # ----------------------------------------------------
        # ИЩЕМ ПУТЬ
        # ----------------------------------------------------

        clip = self.extract_clip_path(
            full_output
        )

        # ----------------------------------------------------
        # Если cut_clip не сообщил путь,
        # пробуем найти последний mp4
        # ----------------------------------------------------

        if clip:

            clip = clip.strip(
                '"'
            ).strip(
                "'"
            )

            # Относительный путь
            if not os.path.isabs(clip):

                clip = os.path.abspath(
                    os.path.join(
                        ROOT,
                        clip
                    )
                )

        if (
            clip
            and os.path.exists(clip)
        ):

            self.chat.append(
                f"🎬 Открыт клип: {clip}"
            )

            self.play_video(
                clip
            )

            return

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        latest = find_latest_clip()

        if (
            latest
            and os.path.exists(latest)
        ):

            self.chat.append(
                f"🎬 Найден последний клип: {latest}"
            )

            self.play_video(
                latest
            )

            return

        self.chat.append(
            "❌ Клип не найден"
        )

    # ========================================================
    # ТРАНСКРИБАЦИЯ
    # ========================================================

    def run_transcribe(self):

        if self.search_in_progress:

            self.chat.append(
                "⚠ Сейчас уже выполняется операция"
            )

            return

        new_video_dir = os.path.join(
            ROOT,
            "AddVideo"
        )

        if not os.path.exists(
            new_video_dir
        ):

            os.makedirs(
                new_video_dir,
                exist_ok=True
            )

        self.chat.append(
            "\n🎤 Начинается транскрибация "
            "всех видео из AddVideo...\n"
        )

        cmd = [
            PYTHON,
            os.path.join(
                ROOT,
                "src",
                "_transcribe.py"
            ),
            new_video_dir
        ]

        self.transcribe_progress = None

        self.transcribe_thread = QThread()

        self.transcribe_worker = (
            TranscribeWorker(
                cmd
            )
        )

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

        self.search_in_progress = True

        self.transcribe_thread.start()

    # ========================================================
    # СТРОКИ ТРАНСКРИБАЦИИ
    # ========================================================

    def on_transcribe_line(
        self,
        line
    ):

        line = line.strip()

        if not line:
            return

        # ----------------------------------------------------
        # ПРОГРЕСС WHISPER
        # ----------------------------------------------------

        match = re.search(
            r"(\d+)%\|",
            line
        )

        if match:

            percent = int(
                match.group(1)
            )

            bar = (
                "█"
                * (percent // 10)
            )

            bar += (
                "░"
                * (10 - len(bar))
            )

            text = (
                f"🎤 [{bar}] "
                f"{percent}%"
            )

            if (
                text
                != self.transcribe_progress
            ):

                self.chat.append(
                    text
                )

                self.transcribe_progress = text

            return

        # ----------------------------------------------------
        # МУСОР
        # ----------------------------------------------------

        if (
            "frames/s" in line
            or "it/s" in line
        ):

            return

        skip = (
            "FP16 is not supported",
        )

        for s in skip:

            if s in line:
                return

        # ----------------------------------------------------
        # VIDEO FOUND
        # ----------------------------------------------------

        if line.startswith(
            "Video found:"
        ):

            name = line.split(
                ":",
                1
            )[-1].strip()

            self.chat.append(
                f"🎥 Найдено видео: {name}"
            )

            return

        # ----------------------------------------------------
        # VIDEO PROGRESS
        # ----------------------------------------------------

        if line.startswith(
            "VIDEO_PROGRESS"
        ):

            self.chat.append(
                "\n📹 "
                + line.replace(
                    "VIDEO_PROGRESS",
                    "Видео"
                )
            )

            return

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        if line.startswith(
            "Processing:"
        ):

            name = line.replace(
                "Processing:",
                ""
            ).strip()

            self.chat.append(
                f"📄 {name}"
            )

            return

        # ----------------------------------------------------
        # SAVED
        # ----------------------------------------------------

        if line.startswith(
            "Saved:"
        ):

            self.chat.append(
                "✅ Сохранено"
            )

            return

        # ----------------------------------------------------
        # READY
        # ----------------------------------------------------

        if line.startswith(
            "Ready"
        ):

            return

        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

        if line.startswith(
            "Whisper"
        ):

            self.chat.append(
                "🧠 Загрузка Whisper..."
            )

            return

        self.chat.append(
            f"🎤 {line}"
        )

    # ========================================================
    # ЗАВЕРШЕНИЕ ТРАНСКРИБАЦИИ
    # ========================================================

    def on_transcribe_finished(
        self,
        output
    ):

        self.search_in_progress = False

        if self.is_closing:
            return

        if output == "CANCELLED":

            self.chat.append(
                "\n❌ Транскрибация была отменена"
            )

            return

        if output.startswith(
            "ERROR:"
        ):

            self.chat.append(
                output
            )

            return

        self.chat.append(
            "\n✅ Транскрибация завершена"
        )

        new_video_dir = os.path.join(
            ROOT,
            "AddVideo"
        )

        videos_dir = os.path.join(
            ROOT,
            "videos"
        )

        os.makedirs(
            videos_dir,
            exist_ok=True
        )

        if not os.path.exists(
            new_video_dir
        ):

            self.chat.append(
                "⚠ Папка AddVideo не найдена"
            )

            return

        video_extensions = (
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".webm",
            ".flv",
            ".wmv",
            ".m4v"
        )

        moved = 0

        for file in os.listdir(
            new_video_dir
        ):

            if not file.lower().endswith(
                video_extensions
            ):
                continue

            source = os.path.join(
                new_video_dir,
                file
            )

            destination = os.path.join(
                videos_dir,
                file
            )

            try:

                # Если файл уже существует,
                # заменяем его

                if os.path.exists(
                    destination
                ):

                    os.remove(
                        destination
                    )

                shutil.move(
                    source,
                    destination
                )

                moved += 1

            except Exception as e:

                self.chat.append(
                    f"❌ Не удалось перенести "
                    f"{file}: {e}"
                )

        self.chat.append(
            f"📁 Видео перенесены в videos: {moved}"
        )

    # ========================================================
    # SEGMENTS
    # ========================================================

    def run_segments(self):

        if self.search_in_progress:

            self.chat.append(
                "⚠ Сейчас уже выполняется операция"
            )

            return

        self.chat.append(
            "\n📝 Создание сегментов...\n"
        )

        cmd = [
            PYTHON,
            os.path.join(
                ROOT,
                "src",
                "make_segments.py"
            )
        ]

        try:

            subprocess.Popen(
                cmd,
                cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

        except Exception as e:

            self.chat.append(
                f"❌ Ошибка запуска Segments: {e}"
            )

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def run_build_index(self):

        if self.search_in_progress:

            self.chat.append(
                "⚠ Сейчас уже выполняется операция"
            )

            return

        self.chat.append(
            "\n📚 Построение индекса...\n"
        )

        cmd = [
            PYTHON,
            os.path.join(
                ROOT,
                "src",
                "build_index.py"
            )
        ]

        try:

            subprocess.Popen(
                cmd,
                cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

        except Exception as e:

            self.chat.append(
                f"❌ Ошибка запуска Build Index: {e}"
            )

    # ========================================================
    # ВОСПРОИЗВЕДЕНИЕ
    # ========================================================

    def play_video(
        self,
        path
    ):

        if not os.path.exists(
            path
        ):

            self.chat.append(
                f"❌ Файл не найден: {path}"
            )

            return

        path = os.path.abspath(
            path
        )

        url = QUrl.fromLocalFile(
            path
        )

        self.player.setSource(
            url
        )

        self.player.setLoops(
            QMediaPlayer.Infinite
        )

        self.player.play()

        self.btn_pause.setText(
            "⏸ Пауза"
        )

        self.chat.append(
            f"▶ Воспроизведение: {path}"
        )

    # ========================================================
    # ПАУЗА
    # ========================================================

    def toggle_pause(self):

        state = (
            self.player.playbackState()
        )

        if state == (
            QMediaPlayer.PlayingState
        ):

            self.player.pause()

            self.btn_pause.setText(
                "▶ Продолжить"
            )

            self.chat.append(
                "⏸ Пауза"
            )

        elif state == (
            QMediaPlayer.PausedState
        ):

            self.player.play()

            self.btn_pause.setText(
                "⏸ Пауза"
            )

            self.chat.append(
                "▶ Продолжить"
            )

    # ========================================================
    # СТОП
    # ========================================================

    def stop_video(self):

        self.player.stop()

        self.btn_pause.setText(
            "⏸ Пауза"
        )

        self.chat.append(
            "⏹ Стоп"
        )


# ============================================================
# СТАРТ
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = App()

    window.show()

    sys.exit(
        app.exec()
    )

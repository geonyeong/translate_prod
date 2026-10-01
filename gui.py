#!/usr/bin/env python3
"""Desktop GUI for the subtitle pipeline: video -> SRT (Qwen3-ASR) -> Korean SRT.

The GUI runs the same scripts as run.sh (mlx_qwen3_asr, import_srt.py,
google_translate_srt.py, translate_srt.py) as subprocesses and parses their
output for progress, so CLI and GUI share one implementation.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "result"

MEDIA_EXTS = {
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".mpg", ".mpeg",
    ".m2ts", ".mts", ".ts", ".3gp", ".ogv",
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
}

MODES = [
    ("full", "자막 + 번역", "동영상 → 자막 → 한글 자막"),
    ("stt", "자막만 생성", "동영상 → 자막"),
    ("translate", "번역만", "기존 자막(.srt) → 한글 자막"),
]
ENGINES = [
    ("nllb", "로컬 NLLB", "오프라인 · 빠름"),
    ("google", "구글 번역", "인터넷 필요"),
    ("ollama", "Ollama", "로컬 LLM · 문맥 번역"),
]
ENGINE_LABELS = {key: title for key, title, _ in ENGINES}
ENGINE_HINTS = {
    "nllb": "facebook/nllb-200 모델로 줄 단위 번역합니다. 인터넷 없이 동작합니다.",
    "google": "API 키가 없으면 무료 웹 번역을 사용합니다(요청이 많으면 일시 차단될 수 있음).",
    "ollama": "Ollama의 qwen3:8b로 문맥을 살려 번역합니다. Ollama 앱이 실행 중이어야 합니다.",
}
LANGS = {"auto": "자동 감지", "ja": "일본어", "en": "영어"}
ASR_LANG_NAMES = {"ja": "Japanese", "en": "English"}
ASR_MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B · 정확도 우선"),
    ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B · 속도 우선"),
]
RUN_LABELS = {"full": "자막 추출 + 번역 시작", "stt": "자막 추출 시작", "translate": "번역 시작"}

STT_PROGRESS_RE = re.compile(r"Progress: chunk (\d+)/(\d+) \(([\d.]+)%\) ETA (\S+)")
STT_DONE_RE = re.compile(r"Progress: 100\.0%")
COUNT_RE = re.compile(r"\[(\d+)/(\d+)\]")
OLLAMA_RE = re.compile(r"(\d+) ~ (\d+) / 총 (\d+)")
OLLAMA_BATCH_ERROR = "[오류 발생]"

ProgressResult = Optional[tuple[float, Optional[str]]]


def parse_stt(line: str) -> ProgressResult:
    m = STT_PROGRESS_RE.search(line)
    if m:
        eta = m.group(4)
        return float(m.group(3)) / 100.0, eta if re.fullmatch(r"[\d:]+", eta) else None
    if STT_DONE_RE.search(line):
        return 1.0, None
    return None


def parse_count(line: str) -> ProgressResult:
    m = COUNT_RE.search(line)
    if m and int(m.group(2)) > 0:
        return int(m.group(1)) / int(m.group(2)), None
    return None


def parse_ollama(line: str) -> ProgressResult:
    m = OLLAMA_RE.search(line)
    if m and int(m.group(3)) > 0:
        return int(m.group(1)) / int(m.group(3)), None
    return None


def fmt_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def short_path(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home):] if text.startswith(home) else text


def ollama_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.5):
            return True
    except OSError:
        return False


def notify(title: str, message: str) -> None:
    if sys.platform != "darwin":
        return
    script = (
        f"display notification {json.dumps(message, ensure_ascii=False)} "
        f"with title {json.dumps(title, ensure_ascii=False)}"
    )
    subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            # Detach immediately: deleteLater alone leaves the widget visible (unmanaged, at
            # its default 640x480 size) until the event loop gets around to deleting it.
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


STYLE = """
* { font-size: 13px; color: #E6E8EE; }
#root { background: #0E1014; }
#title { font-size: 24px; font-weight: 700; color: #FFFFFF; }
#subtitle { color: #8A91A0; font-size: 13px; }
#badge {
    background: #171B25; border: 1px solid #262C3B; border-radius: 11px;
    padding: 5px 11px; color: #A9B1C3; font-size: 12px;
}
#card { background: #151821; border: 1px solid #222736; border-radius: 16px; }
#cardTitle { font-size: 14px; font-weight: 700; color: #FFFFFF; }
#hint { color: #7D8496; font-size: 12px; }
#fieldLabel { color: #A9B1C3; font-size: 12px; font-weight: 600; }

QPushButton#segment {
    background: #1A1E29; border: 1px solid #262C3B; border-radius: 12px; text-align: left;
}
QPushButton#segment:hover { border-color: #3A4258; background: #1D2230; }
QPushButton#segment:checked { background: rgba(124, 92, 255, 0.16); border: 1px solid #7C5CFF; }
QPushButton#segment:disabled { background: #171A23; border-color: #20242F; }
QLabel#segTitle { font-weight: 700; font-size: 13px; color: #C9CFDB; background: transparent; }
QLabel#segTitle[active="true"] { color: #FFFFFF; }
QLabel#segSub { color: #6F7688; font-size: 11px; background: transparent; }
QLabel#segSub[active="true"] { color: #B9A8FF; }

#dropZone { background: #11141B; border: 2px dashed #2C3345; border-radius: 14px; }
#dropZone:hover { border-color: #3D4660; }
#dropZone[dragging="true"] { border-color: #7C5CFF; background: rgba(124, 92, 255, 0.08); }
#dropZone[hasFile="true"] { border-style: solid; border-color: #2E7D5B; background: rgba(46, 160, 110, 0.06); }
#dropIcon {
    font-size: 20px; font-weight: 700; color: #B9A8FF; background: rgba(124, 92, 255, 0.14);
    border-radius: 22px; min-width: 44px; max-width: 44px; min-height: 44px; max-height: 44px;
}
#dropIcon[hasFile="true"] { color: #6FE0A8; background: rgba(46, 160, 110, 0.16); }
#dropTitle { font-size: 14px; font-weight: 700; background: transparent; }
#dropHint { color: #7D8496; font-size: 12px; background: transparent; }
#dropError { color: #FF7B7B; font-size: 12px; background: transparent; }

QPushButton {
    background: #222736; border: 1px solid #2E3446; border-radius: 9px; padding: 7px 14px;
}
QPushButton:hover { background: #2A3042; }
QPushButton:pressed { background: #1C2130; }
QPushButton:disabled { color: #5A6070; background: #1A1D27; border-color: #232838; }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C5CFF, stop:1 #4F8BFF);
    border: none; color: #FFFFFF; font-weight: 700; font-size: 14px; padding: 12px 24px; border-radius: 12px;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8B6DFF, stop:1 #6198FF);
}
QPushButton#primary:disabled { background: #2A2F3E; color: #6A7080; }
QPushButton#danger {
    background: transparent; border: 1px solid #5A2A33; color: #FF8A8A;
    padding: 12px 18px; border-radius: 12px; font-weight: 600;
}
QPushButton#danger:hover { background: rgba(255, 90, 90, 0.08); }
QPushButton#ghost { background: transparent; border: none; color: #9AA3B5; padding: 4px 8px; }
QPushButton#ghost:hover { color: #FFFFFF; }
QPushButton#accentGhost {
    background: rgba(124, 92, 255, 0.14); border: 1px solid #5B45C9; color: #CFC4FF; font-weight: 600;
}
QPushButton#accentGhost:hover { background: rgba(124, 92, 255, 0.24); }

QComboBox, QLineEdit {
    background: #10131A; border: 1px solid #2A3040; border-radius: 9px; padding: 7px 10px; min-height: 20px;
}
QComboBox:focus, QLineEdit:focus { border-color: #7C5CFF; }
QComboBox:disabled, QLineEdit:disabled { color: #5A6070; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView {
    background: #151821; border: 1px solid #2A3040; selection-background-color: #2C2550; outline: none; padding: 4px;
}

QProgressBar { background: #1A1E29; border: none; border-radius: 5px; min-height: 10px; max-height: 10px; }
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C5CFF, stop:1 #4F8BFF);
}
#bottomBar { background: #12141B; border-top: 1px solid #20242F; }
#status { color: #C9CFDB; font-size: 13px; }
#elapsed { color: #7D8496; font-size: 12px; }
QLabel#chip {
    border-radius: 11px; padding: 3px 11px; font-size: 12px; background: #1A1E29; color: #7D8496;
    border: 1px solid #262C3B;
}
QLabel#chip[state="active"] { color: #FFFFFF; border-color: #7C5CFF; background: rgba(124, 92, 255, 0.18); }
QLabel#chip[state="done"] { color: #6FE0A8; border-color: #2E7D5B; background: rgba(46, 160, 110, 0.12); }
QLabel#chip[state="error"] { color: #FF8A8A; border-color: #7A3340; background: rgba(255, 90, 90, 0.10); }
#chipArrow { color: #4A5164; }

QPlainTextEdit#log {
    background: #0B0D12; border: 1px solid #1F2432; border-radius: 10px;
    font-family: Menlo, monospace; font-size: 11px; color: #AEB6C7; padding: 6px;
}
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2A3040; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

#resultRow { background: #10131A; border: 1px solid #232838; border-radius: 10px; }
#resultName { font-weight: 700; }
#resultMeta { color: #7D8496; font-size: 11px; }
#successTitle { color: #6FE0A8; font-weight: 700; font-size: 15px; }
#warning { color: #FFC266; font-size: 12px; }
"""


class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 20)
        self.body.setSpacing(12)
        self.head = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.head.addWidget(self.title_label)
        self.head.addStretch()
        self.body.addLayout(self.head)


class Segmented(QWidget):
    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str, str]]):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        self._labels: dict[str, tuple[QLabel, QLabel]] = {}

        for key, title, sub in options:
            btn = QPushButton()
            btn.setObjectName("segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(60)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            inner = QVBoxLayout(btn)
            inner.setContentsMargins(14, 10, 14, 10)
            inner.setSpacing(2)
            title_label = QLabel(title)
            title_label.setObjectName("segTitle")
            sub_label = QLabel(sub)
            sub_label.setObjectName("segSub")
            for label in (title_label, sub_label):
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
                inner.addWidget(label)
            self.group.addButton(btn)
            layout.addWidget(btn)
            self.buttons[key] = btn
            self._labels[key] = (title_label, sub_label)
            btn.toggled.connect(lambda checked, k=key: self._on_toggled(k, checked))

    def _on_toggled(self, key: str, checked: bool) -> None:
        for label in self._labels[key]:
            label.setProperty("active", checked)
            repolish(label)
        if checked:
            self.changed.emit(key)

    def value(self) -> str:
        for key, btn in self.buttons.items():
            if btn.isChecked():
                return key
        return next(iter(self.buttons))

    def set_value(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)


class DropZone(QFrame):
    file_dropped = Signal(str)
    browse_requested = Signal()
    cleared = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignCenter)

        self.icon = QLabel("+")
        self.icon.setObjectName("dropIcon")
        self.icon.setAlignment(Qt.AlignCenter)
        self.title = QLabel()
        self.title.setObjectName("dropTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.hint = QLabel()
        self.hint.setObjectName("dropHint")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.error = QLabel()
        self.error.setObjectName("dropError")
        self.error.setAlignment(Qt.AlignCenter)
        self.error.setWordWrap(True)
        self.error.hide()

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch()
        self.browse_btn = QPushButton("파일 선택")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self.browse_requested.emit)
        self.clear_btn = QPushButton("선택 해제")
        self.clear_btn.setObjectName("ghost")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.cleared.emit)
        buttons.addWidget(self.browse_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch()

        layout.addWidget(self.icon, 0, Qt.AlignHCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addWidget(self.error)
        layout.addSpacing(4)
        layout.addLayout(buttons)

    def _set_has_file(self, has_file: bool) -> None:
        for widget in (self, self.icon):
            widget.setProperty("hasFile", has_file)
            repolish(widget)

    def set_empty(self, title: str, hint: str) -> None:
        self._set_has_file(False)
        self.icon.setText("+")
        self.title.setText(title)
        self.hint.setText(hint)
        self.browse_btn.setText("파일 선택")
        self.clear_btn.hide()
        self.error.hide()

    def set_file(self, path: Path) -> None:
        self._set_has_file(True)
        self.icon.setText("✓")
        self.title.setText(path.name)
        try:
            size = human_size(path.stat().st_size)
        except OSError:
            size = "?"
        self.hint.setText(f"{size}  ·  {short_path(path.parent)}")
        self.browse_btn.setText("다른 파일 선택")
        self.clear_btn.show()
        self.error.hide()

    def show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.show()

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.browse_requested.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(url.isLocalFile() for url in urls):
            event.acceptProposedAction()
            self.setProperty("dragging", True)
            repolish(self)

    def dragLeaveEvent(self, event):  # noqa: N802
        self.setProperty("dragging", False)
        repolish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        self.setProperty("dragging", False)
        repolish(self)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                event.acceptProposedAction()
                self.file_dropped.emit(url.toLocalFile())
                return


class LogView(QPlainTextEdit):
    """Read-only console that honours carriage returns (progress bars rewrite their line)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setMinimumHeight(220)
        self._overwrite = False

    def append_text(self, text: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        for part in re.split(r"(\r\n|\n|\r)", text):
            if part in ("\r\n", "\n"):
                cursor.insertText("\n")
                self._overwrite = False
            elif part == "\r":
                self._overwrite = True
            elif part:
                if self._overwrite:
                    cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    self._overwrite = False
                cursor.insertText(part)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def append_line(self, text: str) -> None:
        doc_text = self.toPlainText()
        prefix = "\n" if doc_text and not doc_text.endswith("\n") else ""
        self._overwrite = False
        self.append_text(f"{prefix}{text}\n")


@dataclass
class Step:
    key: str
    title: str
    doing: str
    build: Callable[[], list[str]]
    parse: Callable[[str], ProgressResult]
    finish: Callable[[], Optional[str]]


class MainWindow(QMainWindow):
    def __init__(self, initial_file: Optional[str] = None):
        super().__init__()
        self.settings = QSettings("translate_prod", "SubtitleStudio")
        self.files: dict[str, Optional[Path]] = {"video": None, "srt": None}
        self.proc: Optional[QProcess] = None
        self.steps: list[Step] = []
        self.step_index = -1
        self.running = False
        self.cancelled = False
        self.run_mode = "full"
        self.srt_path: Optional[Path] = None
        self.kor_path: Optional[Path] = None
        self.stt_out_dir: Optional[Path] = None
        self.run_started = 0.0
        self.run_started_wall = 0.0
        self.step_started = 0.0
        self.batch_errors = 0
        self._line_buf = ""
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self.setWindowTitle("Subtitle Studio")
        self.setMinimumSize(760, 760)
        self._build_ui()
        self._restore_settings()

        if initial_file:
            self._accept_file(initial_file)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("root")
        col = QVBoxLayout(content)
        col.setContentsMargins(28, 26, 28, 24)
        col.setSpacing(16)
        scroll.setWidget(content)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        title = QLabel("Subtitle Studio")
        title.setObjectName("title")
        subtitle = QLabel("동영상 자막 추출 · 한국어 번역")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        badge = QLabel("Apple Silicon · 로컬 처리")
        badge.setObjectName("badge")
        header.addWidget(badge, 0, Qt.AlignTop)
        col.addLayout(header)

        mode_card = Card("실행 모드")
        self.mode_seg = Segmented(MODES)
        mode_card.body.addWidget(self.mode_seg)
        col.addWidget(mode_card)

        file_card = Card("입력 파일")
        self.drop = DropZone()
        self.drop.browse_requested.connect(self._browse_file)
        self.drop.file_dropped.connect(self._accept_file)
        self.drop.cleared.connect(self._clear_file)
        file_card.body.addWidget(self.drop)
        col.addWidget(file_card)

        self.options_card = Card("옵션")
        self._build_options(self.options_card)
        col.addWidget(self.options_card)

        self.result_card = Card("결과")
        self.result_body = QVBoxLayout()
        self.result_body.setSpacing(8)
        self.result_card.body.addLayout(self.result_body)
        self.result_card.hide()
        col.addWidget(self.result_card)

        log_card = Card("실행 로그")
        self.log_toggle = QPushButton("펼치기")
        self.log_toggle.setObjectName("ghost")
        self.log_toggle.setCursor(Qt.PointingHandCursor)
        self.log_toggle.clicked.connect(lambda: self._set_log_visible(not self.log.isVisible()))
        log_card.head.addWidget(self.log_toggle)
        self.log = LogView()
        self.log.hide()
        log_card.body.addWidget(self.log)
        col.addWidget(log_card)
        col.addStretch()

        outer.addWidget(scroll, 1)
        outer.addWidget(self._build_bottom_bar())
        self.setCentralWidget(root)

        self.mode_seg.changed.connect(self._on_mode_changed)
        self.engine_seg.changed.connect(lambda _: self._refresh_options())

    def _field(self, label: str, widget: QWidget, hint: Optional[QLabel] = None) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        title = QLabel(label)
        title.setObjectName("fieldLabel")
        layout.addWidget(title)
        layout.addWidget(widget)
        if hint is not None:
            layout.addWidget(hint)
        return row

    @staticmethod
    def _hint() -> QLabel:
        label = QLabel()
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    def _build_options(self, card: Card) -> None:
        self.lang_combo = QComboBox()
        self.lang_hint = self._hint()
        self.row_lang = self._field("원본 언어", self.lang_combo, self.lang_hint)

        self.model_combo = QComboBox()
        for model_id, label in ASR_MODELS:
            self.model_combo.addItem(label, model_id)
        self.row_model = self._field("음성 인식 모델", self.model_combo)

        out_widget = QWidget()
        out_layout = QHBoxLayout(out_widget)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setSpacing(8)
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        change_btn = QPushButton("변경")
        change_btn.setCursor(Qt.PointingHandCursor)
        change_btn.clicked.connect(self._choose_out_dir)
        open_btn = QPushButton("열기")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self._open_out_dir)
        out_layout.addWidget(self.out_edit, 1)
        out_layout.addWidget(change_btn)
        out_layout.addWidget(open_btn)
        self.row_out = self._field("자막 저장 폴더", out_widget)

        self.engine_seg = Segmented(ENGINES)
        self.engine_hint = self._hint()
        self.row_engine = self._field("번역 엔진", self.engine_seg, self.engine_hint)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("비워두면 무료 웹 번역 사용")
        self.key_edit.setText(os.environ.get("GOOGLE_API_KEY", ""))
        key_hint = self._hint()
        key_hint.setText("Google Cloud Translation API 키 (선택). 보안을 위해 저장하지 않습니다.")
        self.row_key = self._field("구글 API 키", self.key_edit, key_hint)

        top = QHBoxLayout()
        top.setSpacing(14)
        top.addWidget(self.row_lang, 1, Qt.AlignTop)
        top.addWidget(self.row_model, 1, Qt.AlignTop)
        card.body.addLayout(top)
        card.body.addWidget(self.row_out)
        card.body.addWidget(self.row_engine)
        card.body.addWidget(self.row_key)

    def _build_bottom_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("bottomBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(28, 14, 28, 16)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(6)
        top.addLayout(self.chip_row)
        top.addStretch()
        self.elapsed_label = QLabel()
        self.elapsed_label.setObjectName("elapsed")
        top.addWidget(self.elapsed_label)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.status_label = QLabel("파일을 선택하고 시작을 누르세요.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        bottom.addWidget(self.status_label, 1)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.hide()
        self.run_btn = QPushButton("시작")
        self.run_btn.setObjectName("primary")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self._start_run)
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.run_btn)
        layout.addLayout(bottom)
        return bar

    # ---------- settings ----------
    def _restore_settings(self) -> None:
        s = self.settings
        self._saved_lang = s.value("lang", "ja")
        self.engine_seg.set_value(s.value("engine", "nllb"))
        model_idx = self.model_combo.findData(s.value("asr_model", ASR_MODELS[0][0]))
        self.model_combo.setCurrentIndex(max(0, model_idx))
        out_dir = s.value("out_dir", str(DEFAULT_OUTPUT_DIR))
        self.out_edit.setText(out_dir if Path(out_dir).parent.exists() else str(DEFAULT_OUTPUT_DIR))
        mode = s.value("mode", "full")
        self.mode_seg.set_value(mode if mode in {key for key, _, _ in MODES} else "full")
        self._on_mode_changed(self.mode_seg.value())
        geometry = s.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(860, 960)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("mode", self.mode_seg.value())
        s.setValue("engine", self.engine_seg.value())
        s.setValue("lang", self.lang_combo.currentData() or "ja")
        s.setValue("asr_model", self.model_combo.currentData())
        s.setValue("out_dir", self.out_edit.text())
        s.setValue("geometry", self.saveGeometry())

    # ---------- mode / file ----------
    def _file_kind(self, mode: Optional[str] = None) -> str:
        return "srt" if (mode or self.mode_seg.value()) == "translate" else "video"

    def _on_mode_changed(self, mode: str) -> None:
        if mode != "translate":
            self._last_video_mode = mode
        self._refresh_file_view()
        self._refresh_options()

    def _refresh_file_view(self) -> None:
        kind = self._file_kind()
        path = self.files[kind]
        if path is not None and path.is_file():
            self.drop.set_file(path)
        elif kind == "srt":
            self.drop.set_empty("자막 파일(.srt)을 끌어다 놓으세요", "또는 클릭해서 파일을 선택하세요")
        else:
            self.drop.set_empty(
                "동영상 파일을 끌어다 놓으세요",
                "mp4, mkv, mov 등 동영상·오디오 파일 · 클릭해서 선택할 수도 있습니다",
            )

    def _accept_file(self, raw_path: str) -> None:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            self.drop.show_error(f"파일을 찾을 수 없습니다: {path}")
            return
        ext = path.suffix.lower()
        mode = self.mode_seg.value()
        status = "준비되었습니다. 옵션을 확인하고 시작을 누르세요."
        if ext == ".srt":
            self.files["srt"] = path
            if mode != "translate":
                self.mode_seg.set_value("translate")
                status = "자막 파일이 선택되어 '번역만' 모드로 전환했습니다."
        elif ext in MEDIA_EXTS:
            self.files["video"] = path
            if mode == "translate":
                self.mode_seg.set_value(getattr(self, "_last_video_mode", "full"))
                status = "동영상 파일이 선택되어 자막 추출 모드로 전환했습니다."
        else:
            self.drop.show_error(f"지원하지 않는 파일 형식입니다: {ext or path.name}")
            return
        self.settings.setValue("last_dir", str(path.parent))
        self._refresh_file_view()
        if not self.running:
            self.status_label.setText(status)

    def _browse_file(self) -> None:
        if self.running:
            return
        start_dir = self.settings.value("last_dir", str(ROOT))
        if self._file_kind() == "srt":
            caption, filters = "자막 파일 선택", "자막 파일 (*.srt);;모든 파일 (*)"
        else:
            patterns = " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTS))
            caption, filters = "동영상 파일 선택", f"동영상·오디오 ({patterns});;모든 파일 (*)"
        path, _ = QFileDialog.getOpenFileName(self, caption, start_dir, filters)
        if path:
            self._accept_file(path)

    def _clear_file(self) -> None:
        self.files[self._file_kind()] = None
        self._refresh_file_view()

    def _choose_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "자막 저장 폴더 선택", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    def _open_out_dir(self) -> None:
        out_dir = Path(self.out_edit.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    # ---------- options ----------
    def _refresh_options(self) -> None:
        mode = self.mode_seg.value()
        engine = self.engine_seg.value()
        needs_stt = mode in ("full", "stt")
        needs_tr = mode in ("full", "translate")

        self.row_model.setVisible(needs_stt)
        self.row_out.setVisible(needs_stt)
        self.row_engine.setVisible(needs_tr)
        self.row_key.setVisible(needs_tr and engine == "google")
        self.engine_hint.setText(ENGINE_HINTS[engine])

        allow_auto = not (needs_tr and engine == "nllb")
        current = self.lang_combo.currentData() or getattr(self, "_saved_lang", "ja")
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        for code in (["auto"] if allow_auto else []) + ["ja", "en"]:
            self.lang_combo.addItem(LANGS[code], code)
        idx = self.lang_combo.findData(current)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else self.lang_combo.findData("ja"))
        self.lang_combo.blockSignals(False)

        if needs_tr and engine == "nllb":
            self.lang_hint.setText("NLLB 번역은 원본 언어를 직접 지정해야 합니다.")
        elif needs_stt:
            self.lang_hint.setText("음성 인식에도 사용됩니다. 직접 지정하면 더 정확합니다.")
        else:
            self.lang_hint.setText("번역할 자막의 언어입니다.")

        self.run_btn.setText(RUN_LABELS[mode])
        if not self.running:
            titles = self._planned_titles(mode, engine)
            self._set_chips(titles, ["pending"] * len(titles))

    @staticmethod
    def _planned_titles(mode: str, engine: str) -> list[str]:
        titles = []
        if mode in ("full", "stt"):
            titles.append("자막 추출")
        if mode in ("full", "translate"):
            titles.append(f"번역 · {ENGINE_LABELS[engine]}")
        return titles

    def _set_chips(self, titles: list[str], states: list[str]) -> None:
        clear_layout(self.chip_row)
        for i, (title, state) in enumerate(zip(titles, states)):
            if i:
                arrow = QLabel("›")
                arrow.setObjectName("chipArrow")
                self.chip_row.addWidget(arrow)
            prefix = {"done": "✓ ", "error": "✕ "}.get(state, f"{i + 1}  ")
            chip = QLabel(f"{prefix}{title}")
            chip.setObjectName("chip")
            chip.setProperty("state", state)
            self.chip_row.addWidget(chip)

    def _set_log_visible(self, visible: bool) -> None:
        self.log.setVisible(visible)
        self.log_toggle.setText("접기" if visible else "펼치기")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (self.mode_seg, self.drop, self.options_card):
            widget.setEnabled(enabled)

    # ---------- run ----------
    def _start_run(self) -> None:
        if self.running:
            return
        mode = self.mode_seg.value()
        engine = self.engine_seg.value()
        kind = self._file_kind(mode)
        source = self.files[kind]
        if source is None or not source.is_file():
            self.drop.show_error("먼저 파일을 선택하세요.")
            return
        needs_stt = mode in ("full", "stt")
        needs_tr = mode in ("full", "translate")

        if needs_tr and engine == "ollama" and not ollama_running():
            QMessageBox.warning(
                self,
                "Ollama가 실행 중이 아닙니다",
                "Ollama 앱을 실행한 뒤 다시 시도하세요.\n(필요 시 터미널에서 'ollama pull qwen3:8b')",
            )
            return

        if needs_stt:
            out_dir = Path(self.out_edit.text()).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            srt_path = out_dir / f"{source.stem}.srt"
            self.stt_out_dir = out_dir
        else:
            srt_path = source
            self.stt_out_dir = None
        kor_path = srt_path.with_name(f"{srt_path.stem}_KOR.srt") if needs_tr else None

        targets = ([srt_path] if needs_stt else []) + ([kor_path] if kor_path else [])
        existing = [p for p in targets if p.exists()]
        if existing:
            names = "\n".join(f"• {p.name}" for p in existing)
            answer = QMessageBox.question(
                self,
                "기존 파일 덮어쓰기",
                f"다음 파일이 이미 있습니다. 덮어쓸까요?\n\n{names}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.run_mode = mode
        self.run_source = source
        self.run_engine = engine
        self.run_lang = self.lang_combo.currentData()
        self.run_model = self.model_combo.currentData()
        self.srt_path = srt_path
        self.kor_path = kor_path
        self.batch_errors = 0

        self.steps = []
        if needs_stt:
            self.steps.append(Step("stt", "자막 추출", "자막 추출 중", self._stt_args, parse_stt, self._finish_stt))
        if needs_tr:
            label = ENGINE_LABELS[engine]
            parser = parse_ollama if engine == "ollama" else parse_count
            self.steps.append(
                Step("translate", f"번역 · {label}", f"번역 중 ({label})", self._translate_args, parser,
                     self._finish_translate)
            )

        self._save_settings()
        self.running = True
        self.cancelled = False
        self.result_card.hide()
        self.log.clear()
        self._set_inputs_enabled(False)
        self.run_btn.hide()
        self.cancel_btn.show()
        self.run_started = time.monotonic()
        self.run_started_wall = time.time()
        self.timer.start()
        self._tick()
        self._run_step(0)

    def _stt_args(self) -> list[str]:
        args = [
            "-m", "mlx_qwen3_asr",
            "--model", self.run_model,
            "--output-format", "srt",
            "--output-dir", str(self.stt_out_dir),
        ]
        if self.run_lang in ASR_LANG_NAMES:
            args += ["--language", ASR_LANG_NAMES[self.run_lang]]
        args.append(str(self.run_source))
        return args

    def _translate_args(self) -> list[str]:
        script = {
            "nllb": "import_srt.py",
            "google": "google_translate_srt.py",
            "ollama": "translate_srt.py",
        }[self.run_engine]
        return [
            str(ROOT / script),
            "--input", str(self.srt_path),
            "--output", str(self.kor_path),
            "--lang", self.run_lang,
        ]

    def _is_fresh(self, path: Optional[Path]) -> bool:
        return path is not None and path.is_file() and path.stat().st_mtime >= self.run_started_wall - 1

    def _finish_stt(self) -> Optional[str]:
        if self._is_fresh(self.srt_path):
            return None
        candidates = sorted(
            (p for p in self.stt_out_dir.glob("*.srt") if self._is_fresh(p) and not p.stem.endswith("_KOR")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            self.srt_path = candidates[0]
            if self.kor_path is not None:
                self.kor_path = self.srt_path.with_name(f"{self.srt_path.stem}_KOR.srt")
            self.log.append_line(f"알림: 예상과 다른 이름의 자막을 사용합니다 → {self.srt_path}")
            return None
        return "자막 파일이 생성되지 않았습니다. 로그를 확인하세요."

    def _finish_translate(self) -> Optional[str]:
        if self._is_fresh(self.kor_path):
            return None
        return "번역 자막 파일이 생성되지 않았습니다. 로그를 확인하세요."

    def _run_step(self, index: int) -> None:
        self.step_index = index
        step = self.steps[index]
        self.step_started = time.monotonic()
        self._line_buf = ""
        states = ["done"] * index + ["active"] + ["pending"] * (len(self.steps) - index - 1)
        self._set_chips([s.title for s in self.steps], states)
        self.progress.setRange(0, 0)
        self.status_label.setText(f"{step.doing} · 모델을 불러오는 중…")

        args = step.build()
        self.log.append_line(f"▶ [{index + 1}/{len(self.steps)}] {step.title}")
        self.log.append_line("$ " + shlex.join(["python", *args]))

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("TOKENIZERS_PARALLELISM", "false")
        api_key = self.key_edit.text().strip()
        if api_key:
            env.insert("GOOGLE_API_KEY", api_key)
        else:
            env.remove("GOOGLE_API_KEY")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(ROOT))
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.started.connect(self._keep_awake)
        self.proc = proc
        proc.start(sys.executable, ["-u", *args])

    def _keep_awake(self) -> None:
        # Long STT runs should not be interrupted by idle sleep; caffeinate exits with the process.
        if self.proc is not None and sys.platform == "darwin":
            QProcess.startDetached("caffeinate", ["-i", "-w", str(self.proc.processId())])

    def _on_output(self) -> None:
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self.log.append_text(data)
        parts = re.split(r"\r\n|\n|\r", self._line_buf + data)
        self._line_buf = parts.pop()
        for line in parts:
            self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        if OLLAMA_BATCH_ERROR in line:
            self.batch_errors += 1
        if not (0 <= self.step_index < len(self.steps)):
            return
        result = self.steps[self.step_index].parse(line)
        if result is not None:
            self._set_progress(*result)

    def _set_progress(self, fraction: float, eta: Optional[str]) -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1000)
        self.progress.setValue(int(fraction * 1000))
        if eta is None and 0.01 < fraction < 1.0:
            elapsed = time.monotonic() - self.step_started
            eta = fmt_duration(elapsed * (1 - fraction) / fraction)
        text = f"{self.steps[self.step_index].doing} · {fraction * 100:.0f}%"
        if eta and fraction < 1.0:
            text += f" · 남은 시간 약 {eta}"
        self.status_label.setText(text)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.FailedToStart:
            self.proc = None
            self._end_run("error", "프로세스를 시작하지 못했습니다. venv 설치 상태를 확인하세요.")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._on_output()
        if self._line_buf:
            self._handle_line(self._line_buf)
            self._line_buf = ""
        if self.proc is not None:
            self.proc.deleteLater()
            self.proc = None

        if self.cancelled:
            self._end_run("cancelled")
            return
        step = self.steps[self.step_index]
        if exit_status == QProcess.CrashExit or exit_code != 0:
            self._end_run("error", f"'{step.title}' 단계가 실패했습니다 (종료 코드 {exit_code}). 로그를 확인하세요.")
            return
        error = step.finish()
        if error:
            self._end_run("error", error)
            return
        self.log.append_line(f"✓ {step.title} 완료")
        if self.step_index + 1 < len(self.steps):
            self._run_step(self.step_index + 1)
        else:
            self._end_run("success")

    def _cancel(self) -> None:
        if self.proc is None:
            return
        self.cancelled = True
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("취소하는 중…")
        self.proc.terminate()
        QTimer.singleShot(4000, self._force_kill)

    def _force_kill(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def _end_run(self, state: str, message: Optional[str] = None) -> None:
        self.running = False
        self.timer.stop()
        elapsed = fmt_duration(time.monotonic() - self.run_started)
        self._set_inputs_enabled(True)
        self.cancel_btn.hide()
        self.cancel_btn.setEnabled(True)
        self.run_btn.show()

        titles = [s.title for s in self.steps]
        idx = max(self.step_index, 0)
        if state == "success":
            self._set_chips(titles, ["done"] * len(titles))
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            self.status_label.setText(f"완료 · 총 소요 시간 {elapsed}")
            self.log.append_line(f"■ 모든 작업 완료 ({elapsed})")
            self._show_result()
            notify("Subtitle Studio", f"작업이 완료되었습니다 ({elapsed})")
            QApplication.alert(self)
        else:
            states = ["done"] * idx + ["error"] + ["pending"] * (len(titles) - idx - 1)
            self._set_chips(titles, states)
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 1000)
                self.progress.setValue(0)
            if state == "cancelled":
                self.status_label.setText("작업을 취소했습니다.")
                self.log.append_line("■ 사용자가 작업을 취소했습니다.")
            else:
                self.status_label.setText(message or "작업이 실패했습니다.")
                self.log.append_line(f"■ 오류: {message}")
                self._set_log_visible(True)
                QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(self.log))
                notify("Subtitle Studio", message or "작업이 실패했습니다.")

    def _tick(self) -> None:
        if self.running:
            self.elapsed_label.setText(f"경과 {fmt_duration(time.monotonic() - self.run_started)}")

    # ---------- results ----------
    def _show_result(self) -> None:
        clear_layout(self.result_body)
        title = QLabel("✓  작업이 완료되었습니다")
        title.setObjectName("successTitle")
        self.result_body.addWidget(title)

        if self.run_mode in ("full", "stt") and self.srt_path:
            self.result_body.addWidget(self._result_row("원본 자막", self.srt_path))
        if self.kor_path:
            self.result_body.addWidget(self._result_row("한글 자막", self.kor_path))

        if self.batch_errors:
            warning = QLabel(f"⚠  일부 구간({self.batch_errors}개 묶음)은 번역에 실패해 원문이 유지되었습니다.")
            warning.setObjectName("warning")
            warning.setWordWrap(True)
            self.result_body.addWidget(warning)

        if self.run_mode == "stt" and self.srt_path:
            next_btn = QPushButton("이 자막으로 번역하기  →")
            next_btn.setObjectName("accentGhost")
            next_btn.setCursor(Qt.PointingHandCursor)
            next_btn.clicked.connect(self._translate_generated_srt)
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(next_btn)
            self.result_body.addLayout(row)

        self.result_card.show()
        QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(self.result_card))

    def _result_row(self, label: str, path: Path) -> QWidget:
        row = QFrame()
        row.setObjectName("resultRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(8)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        name = QLabel(f"{label}  ·  {path.name}")
        name.setObjectName("resultName")
        size = human_size(path.stat().st_size) if path.exists() else "?"
        meta = QLabel(f"{size}  ·  {short_path(path.parent)}")
        meta.setObjectName("resultMeta")
        texts.addWidget(name)
        texts.addWidget(meta)
        layout.addLayout(texts, 1)
        open_btn = QPushButton("열기")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
        reveal_btn = QPushButton("Finder에서 보기")
        reveal_btn.setCursor(Qt.PointingHandCursor)
        reveal_btn.clicked.connect(lambda: subprocess.Popen(["open", "-R", str(path)]))
        layout.addWidget(open_btn)
        layout.addWidget(reveal_btn)
        return row

    def _translate_generated_srt(self) -> None:
        if self.srt_path and self.srt_path.is_file():
            self.files["srt"] = self.srt_path
            self.mode_seg.set_value("translate")
            self.result_card.hide()
            self.status_label.setText("생성된 자막이 선택되었습니다. 번역 엔진을 확인하고 시작을 누르세요.")

    # ---------- window ----------
    def closeEvent(self, event):  # noqa: N802
        if self.running and self.proc is not None:
            answer = QMessageBox.question(
                self,
                "작업 진행 중",
                "작업이 진행 중입니다. 취소하고 종료할까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancelled = True
            self.proc.terminate()
            if not self.proc.waitForFinished(5000):
                self.proc.kill()
                self.proc.waitForFinished(2000)
        self._save_settings()
        event.accept()


def dark_palette() -> QPalette:
    palette = QPalette()
    colors = {
        QPalette.Window: "#0E1014",
        QPalette.WindowText: "#E6E8EE",
        QPalette.Base: "#10131A",
        QPalette.AlternateBase: "#151821",
        QPalette.Text: "#E6E8EE",
        QPalette.Button: "#222736",
        QPalette.ButtonText: "#E6E8EE",
        QPalette.Highlight: "#7C5CFF",
        QPalette.HighlightedText: "#FFFFFF",
        QPalette.ToolTipBase: "#151821",
        QPalette.ToolTipText: "#E6E8EE",
        QPalette.PlaceholderText: "#5F6678",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    return palette


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Subtitle Studio")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(initial)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

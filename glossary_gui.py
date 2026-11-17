"""Glossary builder window: review subtitle pairs and confirm term mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glossary_builder import (
    Candidate,
    PairData,
    build_candidates,
    find_pairs,
    load_pair,
    merge_glossary,
)
from translate_srt import DEFAULT_GLOSSARY, load_glossary


STYLE = """
* { font-size: 13px; color: #E6E8EE; }
QDialog { background: #0E1014; }
#title { font-size: 18px; font-weight: 700; color: #FFFFFF; }
#subtitle { color: #8A91A0; font-size: 12px; }
#panel {
    background: #151821; border: 1px solid #222736; border-radius: 14px;
}
#panelTitle { font-size: 13px; font-weight: 700; color: #FFFFFF; }
#hint { color: #7D8496; font-size: 12px; }
#fieldLabel { color: #A9B1C3; font-size: 12px; font-weight: 600; }
QPushButton {
    background: #222736; border: 1px solid #2E3446; border-radius: 9px; padding: 7px 14px;
}
QPushButton:hover { background: #2A3042; }
QPushButton:disabled { color: #5A6070; background: #1A1D27; border-color: #232838; }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C5CFF, stop:1 #4F8BFF);
    border: none; color: #FFFFFF; font-weight: 700; padding: 10px 18px; border-radius: 11px;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8B6DFF, stop:1 #6198FF);
}
QPushButton#ghost { background: transparent; border: none; color: #9AA3B5; padding: 4px 8px; }
QPushButton#ghost:hover { color: #FFFFFF; }
QLineEdit, QComboBox {
    background: #10131A; border: 1px solid #2A3040; border-radius: 9px; padding: 7px 10px; min-height: 20px;
}
QLineEdit:focus, QComboBox:focus { border-color: #7C5CFF; }
QListWidget, QTableWidget {
    background: #10131A; border: 1px solid #232838; border-radius: 10px;
    outline: none; gridline-color: #1F2432;
}
QListWidget::item { padding: 8px 10px; }
QListWidget::item:selected, QTableWidget::item:selected {
    background: rgba(124, 92, 255, 0.22);
}
QHeaderView::section {
    background: #151821; color: #A9B1C3; border: none; border-bottom: 1px solid #232838;
    padding: 8px; font-weight: 600;
}
QCheckBox { spacing: 8px; color: #C9CFDB; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 5px; border: 1px solid #3A4258; background: #10131A;
}
QCheckBox::indicator:checked { background: #7C5CFF; border-color: #7C5CFF; }
#warning { color: #FFC266; font-size: 12px; }
#okTag { color: #6FE0A8; font-size: 12px; }
"""


def _panel(title: str) -> tuple[QWidget, QVBoxLayout]:
    frame = QWidget()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 14)
    layout.setSpacing(10)
    label = QLabel(title)
    label.setObjectName("panelTitle")
    layout.addWidget(label)
    return frame, layout


class GlossaryBuilderDialog(QDialog):
    """Pick subtitle pairs → review term candidates → save glossary.txt."""

    def __init__(self, parent=None, start_dir: Optional[Path] = None, glossary_path: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("용어집 만들기")
        self.resize(1100, 720)
        self.setStyleSheet(STYLE)
        self.start_dir = Path(start_dir) if start_dir else Path.cwd() / "result"
        self.glossary_path = Path(glossary_path) if glossary_path else DEFAULT_GLOSSARY
        self.pairs: list[PairData] = []
        self.candidates: list[Candidate] = []
        self._building = False
        self._build_ui()
        self.gloss_edit.setText(str(self.glossary_path))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("용어집 만들기")
        title.setObjectName("title")
        subtitle = QLabel("여러 영상의 원문·한글 자막을 모아, 같은 단어를 하나의 번역으로 고정합니다.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        split = QSplitter(Qt.Horizontal)

        # --- left: pairs ---
        left, left_body = _panel("1. 자막 쌍")
        self.pair_list = QListWidget()
        self.pair_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_body.addWidget(self.pair_list, 1)
        pair_btns = QHBoxLayout()
        add_folder = QPushButton("폴더 추가")
        add_folder.setCursor(Qt.PointingHandCursor)
        add_folder.clicked.connect(self._add_folder)
        add_files = QPushButton("파일 추가")
        add_files.setCursor(Qt.PointingHandCursor)
        add_files.clicked.connect(self._add_files)
        clear_pairs = QPushButton("비우기")
        clear_pairs.setObjectName("ghost")
        clear_pairs.setCursor(Qt.PointingHandCursor)
        clear_pairs.clicked.connect(self._clear_pairs)
        pair_btns.addWidget(add_folder)
        pair_btns.addWidget(add_files)
        pair_btns.addWidget(clear_pairs)
        left_body.addLayout(pair_btns)
        pair_hint = QLabel("원문.srt 와 원문_KOR.srt 가 같은 폴더에 있어야 합니다.")
        pair_hint.setObjectName("hint")
        pair_hint.setWordWrap(True)
        left_body.addWidget(pair_hint)

        lang_row = QHBoxLayout()
        lang_label = QLabel("원문 언어")
        lang_label.setObjectName("fieldLabel")
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("일본어", "ja")
        self.lang_combo.addItem("영어", "en")
        self.lang_combo.addItem("자동(일·영)", "auto")
        self.min_combo = QComboBox()
        for n in (1, 2, 3, 5):
            self.min_combo.addItem(f"{n}회 이상", n)
        self.min_combo.setCurrentIndex(1)
        scan_btn = QPushButton("후보 찾기")
        scan_btn.setObjectName("primary")
        scan_btn.setCursor(Qt.PointingHandCursor)
        scan_btn.clicked.connect(self._scan)
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self.lang_combo, 1)
        lang_row.addWidget(self.min_combo)
        left_body.addLayout(lang_row)
        left_body.addWidget(scan_btn)
        split.addWidget(left)

        # --- center: candidates ---
        center, center_body = _panel("2. 용어 후보")
        self.status = QLabel("자막 쌍을 추가한 뒤 ‘후보 찾기’를 누르세요.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        center_body.addWidget(self.status)
        filter_row = QHBoxLayout()
        self.only_conflict = QCheckBox("번역이 갈라진 항목만")
        self.only_conflict.stateChanged.connect(self._apply_filter)
        self.select_visible = QPushButton("보이는 항목 모두 선택")
        self.select_visible.setObjectName("ghost")
        self.select_visible.setCursor(Qt.PointingHandCursor)
        self.select_visible.clicked.connect(lambda: self._set_visible_checked(True))
        self.clear_visible = QPushButton("선택 해제")
        self.clear_visible.setObjectName("ghost")
        self.clear_visible.setCursor(Qt.PointingHandCursor)
        self.clear_visible.clicked.connect(lambda: self._set_visible_checked(False))
        filter_row.addWidget(self.only_conflict)
        filter_row.addStretch()
        filter_row.addWidget(self.select_visible)
        filter_row.addWidget(self.clear_visible)
        center_body.addLayout(filter_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["선택", "원문", "번역(확정)", "횟수", "다른 번역"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select_candidate)
        center_body.addWidget(self.table, 1)

        edit_row = QHBoxLayout()
        edit_label = QLabel("선택 항목 번역")
        edit_label.setObjectName("fieldLabel")
        self.term_edit = QLineEdit()
        self.term_edit.setReadOnly(True)
        self.term_edit.setPlaceholderText("원문")
        self.dst_edit = QLineEdit()
        self.dst_edit.setPlaceholderText("한글 번역을 입력하거나 아래 문장을 보고 고치세요")
        self.dst_edit.textChanged.connect(self._on_dst_edited)
        edit_row.addWidget(edit_label)
        edit_row.addWidget(self.term_edit, 1)
        edit_row.addWidget(QLabel("="))
        edit_row.addWidget(self.dst_edit, 2)
        center_body.addLayout(edit_row)
        split.addWidget(center)

        # --- right: occurrences ---
        right, right_body = _panel("3. 등장 문장")
        self.occ_hint = QLabel("후보를 누르면 원문·한글 문장이 여기에 나타납니다. 문장을 더블클릭하면 그 번역을 가져옵니다.")
        self.occ_hint.setObjectName("hint")
        self.occ_hint.setWordWrap(True)
        right_body.addWidget(self.occ_hint)
        self.occ_list = QListWidget()
        self.occ_list.itemDoubleClicked.connect(self._use_occurrence)
        right_body.addWidget(self.occ_list, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 4)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        # --- bottom save ---
        save_row = QHBoxLayout()
        gloss_label = QLabel("저장 위치")
        gloss_label.setObjectName("fieldLabel")
        self.gloss_edit = QLineEdit()
        browse = QPushButton("변경")
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self._browse_glossary)
        open_btn = QPushButton("파일 열기")
        open_btn.setObjectName("ghost")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self._open_glossary)
        self.overwrite = QCheckBox("같은 원문이 있으면 덮어쓰기")
        self.overwrite.setChecked(True)
        save_btn = QPushButton("선택한 용어 저장")
        save_btn.setObjectName("primary")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("닫기")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.reject)
        save_row.addWidget(gloss_label)
        save_row.addWidget(self.gloss_edit, 1)
        save_row.addWidget(browse)
        save_row.addWidget(open_btn)
        save_row.addWidget(self.overwrite)
        save_row.addWidget(save_btn)
        save_row.addWidget(close_btn)
        root.addLayout(save_row)

    # ----- pairs -----
    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "자막 폴더 선택", str(self.start_dir))
        if path:
            self._load_paths([Path(path)])

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "원문 자막 선택 (.srt)",
            str(self.start_dir),
            "자막 파일 (*.srt);;모든 파일 (*)",
        )
        if paths:
            self._load_paths([Path(p) for p in paths])

    def _clear_pairs(self) -> None:
        self.pairs = []
        self.candidates = []
        self.pair_list.clear()
        self.table.setRowCount(0)
        self.occ_list.clear()
        self.status.setText("자막 쌍을 추가한 뒤 ‘후보 찾기’를 누르세요.")

    def _load_paths(self, paths: list[Path]) -> None:
        found = find_pairs(paths)
        if not found:
            QMessageBox.warning(self, "자막 쌍 없음", "원문.srt 와 짝이 되는 원문_KOR.srt 를 찾지 못했습니다.")
            return
        existing = {pair.src_path.resolve() for pair in self.pairs}
        added = 0
        for src, kor in found:
            if src.resolve() in existing:
                continue
            try:
                self.pairs.append(load_pair(src, kor))
                added += 1
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "불러오기 실패", f"{src.name}\n{exc}")
        self._refresh_pairs()
        if added:
            self.status.setText(f"{added}개 쌍을 추가했습니다. ‘후보 찾기’를 누르세요.")

    def _refresh_pairs(self) -> None:
        self.pair_list.clear()
        for pair in self.pairs:
            item = QListWidgetItem(f"{pair.stem}  ·  {len(pair.src_lines)}줄")
            item.setToolTip(f"{pair.src_path}\n{pair.kor_path}")
            self.pair_list.addItem(item)

    # ----- scan -----
    def _scan(self) -> None:
        if not self.pairs:
            QMessageBox.information(self, "자막 필요", "먼저 자막 쌍을 추가하세요.")
            return
        path = Path(self.gloss_edit.text().strip() or str(DEFAULT_GLOSSARY))
        existing = load_glossary(str(path)) if path.is_file() else {}
        lang = self.lang_combo.currentData()
        min_count = int(self.min_combo.currentData())
        self.candidates = build_candidates(self.pairs, lang=lang, min_count=min_count, existing=existing)
        self._fill_table()
        conflicts = sum(1 for c in self.candidates if c.inconsistent)
        self.status.setText(
            f"후보 {len(self.candidates)}개 · 번역이 갈라진 항목 {conflicts}개 · "
            f"체크한 뒤 아래 ‘선택한 용어 저장’을 누르세요."
        )

    def _fill_table(self) -> None:
        self._building = True
        only = self.only_conflict.isChecked()
        rows = [c for c in self.candidates if (c.inconsistent if only else True)]
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for row, cand in enumerate(rows):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            # Pre-check inconsistent or frequent items.
            checked = Qt.Checked if (cand.inconsistent or cand.count >= 3) else Qt.Unchecked
            check.setCheckState(checked)
            check.setData(Qt.UserRole, cand.term)

            term_item = QTableWidgetItem(cand.term)
            dst_item = QTableWidgetItem(cand.suggested)
            count_item = QTableWidgetItem(str(cand.count))
            count_item.setTextAlignment(Qt.AlignCenter)
            variants = ", ".join(v for v in cand.variants if v != cand.suggested) or "—"
            var_item = QTableWidgetItem(variants)

            if cand.inconsistent:
                for item in (term_item, dst_item, var_item):
                    item.setForeground(QColor("#FFC266"))

            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, term_item)
            self.table.setItem(row, 2, dst_item)
            self.table.setItem(row, 3, count_item)
            self.table.setItem(row, 4, var_item)
        self._building = False
        if rows:
            self.table.selectRow(0)

    def _apply_filter(self) -> None:
        # Preserve edits from the table into candidates before refilter.
        self._flush_table_edits()
        self._fill_table()

    def _flush_table_edits(self) -> None:
        by_term = {c.term: c for c in self.candidates}
        for row in range(self.table.rowCount()):
            term_item = self.table.item(row, 1)
            dst_item = self.table.item(row, 2)
            if not term_item or not dst_item:
                continue
            cand = by_term.get(term_item.text())
            if cand:
                cand.suggested = dst_item.text().strip()

    def _set_visible_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None:
                item.setCheckState(state)

    def _candidate_at_row(self, row: int) -> Optional[Candidate]:
        item = self.table.item(row, 0)
        if item is None:
            return None
        term = item.data(Qt.UserRole)
        for cand in self.candidates:
            if cand.term == term:
                return cand
        return None

    def _on_select_candidate(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        cand = self._candidate_at_row(rows[0].row())
        if cand is None:
            return
        self.term_edit.setText(cand.term)
        self.dst_edit.blockSignals(True)
        self.dst_edit.setText(cand.suggested)
        self.dst_edit.blockSignals(False)
        self.occ_list.clear()
        for occ in cand.occurrences[:80]:
            text = f"[{occ.file} #{occ.index}]\n원문  {occ.src}\n한글  {occ.kor}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, occ.kor)
            if cand.inconsistent:
                item.setForeground(QColor("#FFC266"))
            self.occ_list.addItem(item)
        if len(cand.occurrences) > 80:
            self.occ_list.addItem(QListWidgetItem(f"… 외 {len(cand.occurrences) - 80}곳"))

    def _on_dst_edited(self, text: str) -> None:
        if self._building:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        dst_item = self.table.item(row, 2)
        if dst_item is not None:
            dst_item.setText(text.strip())
        cand = self._candidate_at_row(row)
        if cand is not None:
            cand.suggested = text.strip()

    def _use_occurrence(self, item: QListWidgetItem) -> None:
        kor = item.data(Qt.UserRole)
        if not kor:
            return
        # Prefer a hangul token from the line if the whole line is long.
        from glossary_builder import hangul_tokens

        tokens = hangul_tokens(str(kor))
        pick = tokens[0] if tokens and len(str(kor)) > 16 else str(kor)
        self.dst_edit.setText(pick)
        check = None
        rows = self.table.selectionModel().selectedRows()
        if rows:
            check = self.table.item(rows[0].row(), 0)
        if check is not None:
            check.setCheckState(Qt.Checked)

    # ----- save -----
    def _browse_glossary(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "용어집 저장 위치",
            self.gloss_edit.text() or str(DEFAULT_GLOSSARY),
            "텍스트 (*.txt);;모든 파일 (*)",
        )
        if path:
            self.gloss_edit.setText(path)

    def _open_glossary(self) -> None:
        path = Path(self.gloss_edit.text().strip())
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            QMessageBox.information(self, "파일 없음", "아직 저장된 용어집 파일이 없습니다.")

    def _save(self) -> None:
        self._flush_table_edits()
        entries: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            check = self.table.item(row, 0)
            term_item = self.table.item(row, 1)
            dst_item = self.table.item(row, 2)
            if check is None or term_item is None or dst_item is None:
                continue
            if check.checkState() != Qt.Checked:
                continue
            src = term_item.text().strip()
            dst = dst_item.text().strip()
            if src and dst:
                entries[src] = dst
        if not entries:
            QMessageBox.information(self, "선택 없음", "저장할 용어를 체크하세요.")
            return
        path = Path(self.gloss_edit.text().strip() or str(DEFAULT_GLOSSARY))
        try:
            merged = merge_glossary(path, entries, overwrite=self.overwrite.isChecked())
        except OSError as exc:
            QMessageBox.warning(self, "저장 실패", str(exc))
            return
        self.glossary_path = path
        QMessageBox.information(
            self,
            "저장 완료",
            f"{len(entries)}개 용어를 저장했습니다.\n전체 {len(merged)}항목 · {path}",
        )

import os
import sys
import json
import configparser
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QDialog, QFormLayout,
    QCheckBox, QMessageBox, QInputDialog, QToolBar, QStatusBar,
    QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QFont, QTextCursor


# 用于 Markdown -> HTML 转换
import markdown

try:
    from openai import OpenAI
except ImportError:
    QMessageBox.critical(None, "缺少库", "请先安装 openai: pip install openai")
    sys.exit(1)


# ===================== 配置管理 =====================
def get_appdata_dir() -> Path:
    r"""获取当前用户 AppData\Local 目录，并确保程序子文件夹存在"""
    username = os.environ.get("USERNAME") or os.environ.get("USER") or "default"
    base = Path(os.environ.get("LOCALAPPDATA", f"C:/Users/{username}/AppData/Local"))
    app_dir = base / "DeepSeekChat"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def load_token() -> str:
    """从 token.ini 读取 API Key"""
    ini_path = get_appdata_dir() / "token.ini"
    if not ini_path.exists():
        return ""
    config = configparser.ConfigParser()
    config.read(str(ini_path), encoding="utf-8")
    return config.get("Auth", "api_key", fallback="")


def save_token(api_key: str):
    """保存 API Key 到 token.ini"""
    ini_path = get_appdata_dir() / "token.ini"
    config = configparser.ConfigParser()
    config["Auth"] = {"api_key": api_key}
    with open(ini_path, "w", encoding="utf-8") as f:
        config.write(f)


# ===================== 会话管理 =====================
class Session:
    def __init__(self, name: str, model: str = "deepseek-v4-pro",
                 mode: str = "chat"):  # mode: chat, prefix, fim
        self.name = name
        self.model = model
        self.mode = mode
        self.messages: List[Dict] = []  # 每条消息格式: {role, content, reasoning?}
        self.created = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "model": self.model,
            "mode": self.mode,
            "messages": self.messages,
            "created": self.created
        }

    @staticmethod
    def from_dict(data: Dict) -> "Session":
        s = Session(data["name"], data.get("model", "deepseek-v4-pro"),
                    data.get("mode", "chat"))
        s.messages = data.get("messages", [])
        s.created = data.get("created", datetime.now().isoformat())
        return s


class SessionManager:
    def __init__(self):
        self.sessions: List[Session] = []
        self.save_path = get_appdata_dir() / "sessions.json"

    def load(self):
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text(encoding="utf-8"))
                self.sessions = [Session.from_dict(d) for d in data]
            except Exception as e:
                print("加载会话失败:", e)

    def save(self):
        data = [s.to_dict() for s in self.sessions]
        self.save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def add_session(self, name, model, mode) -> Session:
        session = Session(name, model, mode)
        self.sessions.append(session)
        self.save()
        return session

    def delete_session(self, index):
        if 0 <= index < len(self.sessions):
            self.sessions.pop(index)
            self.save()


# ===================== 流式 API 调用线程 =====================
class ChatThread(QThread):
    # 流式信号： (类型, 内容) 类型: "reasoning" 或 "content"
    stream_signal = Signal(str, str)
    # 完成信号
    finished_signal = Signal(bool, str)  # (成功?, 错误信息)

    def __init__(self, session: Session, api_key: str,
                 input_data: Optional[Dict] = None):
        super().__init__()
        self.session = session
        self.api_key = api_key
        self.input_data = input_data or {}

        if session.mode in ("prefix", "fim"):
            self.base_url = "https://api.deepseek.com/beta"
        else:
            self.base_url = "https://api.deepseek.com"

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            if self.session.mode in ("chat", "prefix"):
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                messages.extend(self.session.messages)

                new_msg = {
                    "role": self.input_data.get("role", "user"),
                    "content": self.input_data.get("content", "")
                }
                if self.session.mode == "prefix" and self.input_data.get("prefix"):
                    new_msg["prefix"] = True
                messages.append(new_msg)

                # 流式请求
                kwargs = {
                    "model": self.session.model,
                    "messages": messages,
                    "stream": True
                }
                if self.session.mode == "prefix":
                    kwargs["stop"] = ["```"]

                response = client.chat.completions.create(**kwargs)
                for chunk in response:
                    delta = chunk.choices[0].delta
                    # 可能存在 reasoning_content (思考)
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        self.stream_signal.emit("reasoning", delta.reasoning_content)
                    # 正常内容
                    if delta.content:
                        self.stream_signal.emit("content", delta.content)
                self.finished_signal.emit(True, "")

            elif self.session.mode == "fim":
                # FIM 模式暂时未支持流式（completions 端点也可能支持，但保持简单）
                prefix = self.input_data.get("prefix", "")
                suffix = self.input_data.get("suffix", "")
                max_tokens = self.input_data.get("max_tokens", 128)
                response = client.completions.create(
                    model=self.session.model,
                    prompt=prefix,
                    suffix=suffix,
                    max_tokens=max_tokens,
                    stream=False
                )
                content = response.choices[0].text
                self.stream_signal.emit("content", content)
                self.finished_signal.emit(True, "")

        except Exception as e:
            self.finished_signal.emit(False, str(e))


# ===================== 新建会话对话框 =====================
class NewSessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建会话")
        layout = QFormLayout(self)

        self.name_edit = QLineEdit("新会话")
        layout.addRow("会话名称:", self.name_edit)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-v4-pro", "deepseek-v4-flash"])
        layout.addRow("模型:", self.model_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["chat", "prefix", "fim"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addRow("模式:", self.mode_combo)

        self.help_label = QLabel("普通聊天：标准多轮对话\n"
                                 "前缀续写：允许发送带有 prefix 标记的消息\n"
                                 "FIM 补全：输入 Prefix 和 Suffix 进行代码/文本补全")
        self.help_label.setWordWrap(True)
        layout.addRow(self.help_label)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("创建")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addRow(btn_box)

    def on_mode_changed(self, text):
        if text == "fim":
            self.help_label.setText("FIM 模式：输入 Prefix 和 Suffix，模型生成中间补全内容")
        elif text == "prefix":
            self.help_label.setText("前缀续写模式：可勾选“作为助手前缀”发送消息")
        else:
            self.help_label.setText("普通对话模式：支持多轮上下文")

    def get_values(self):
        return (self.name_edit.text().strip(),
                self.model_combo.currentText(),
                self.mode_combo.currentText())


# ===================== 主窗口 =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSeek 多模式桌面助手")
        self.resize(1000, 700)

        self.manager = SessionManager()
        self.manager.load()
        self.current_session_index = -1
        self.api_key = load_token()

        if not self.api_key:
            self.open_token_dialog()

        self.streaming = False
        self.current_reasoning = ""
        self.current_content = ""

        self._setup_ui()
        self._populate_session_list()

    # --------------- UI ---------------
    def _setup_ui(self):
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("新建会话", self, triggered=self.new_session))
        toolbar.addAction(QAction("Token 设置", self, triggered=self.open_token_dialog))

        self.statusBar().showMessage("就绪")

        splitter = QSplitter(Qt.Horizontal)

        # 左侧会话列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("会话列表"))
        self.session_list = QListWidget()
        self.session_list.currentRowChanged.connect(self.switch_session)
        left_layout.addWidget(self.session_list)

        del_btn = QPushButton("删除选中会话")
        del_btn.clicked.connect(self.delete_session)
        left_layout.addWidget(del_btn)
        splitter.addWidget(left_widget)

        # 右侧聊天区
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.session_info_label = QLabel("未选择会话")
        right_layout.addWidget(self.session_info_label)

        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        right_layout.addWidget(self.chat_display, 1)

        # 输入栈
        self.input_stack = QWidget()
        self.input_layout = QVBoxLayout(self.input_stack)
        self.input_layout.setContentsMargins(0, 0, 0, 0)

        # 普通/前缀输入
        self.chat_input_widget = QWidget()
        chat_input_layout = QHBoxLayout(self.chat_input_widget)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)

        self.message_edit = QTextEdit()
        self.message_edit.setMaximumHeight(80)
        self.message_edit.setPlaceholderText("输入消息...")
        chat_input_layout.addWidget(self.message_edit, 1)

        self.prefix_check = QCheckBox("作为助手前缀")
        self.prefix_check.setToolTip("勾选后，本条消息将作为助手的 prefix 消息发送（仅前缀续写模式有效）")
        chat_input_layout.addWidget(self.prefix_check)

        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_message)
        chat_input_layout.addWidget(self.send_btn)

        # FIM 输入
        self.fim_input_widget = QWidget()
        fim_layout = QVBoxLayout(self.fim_input_widget)
        fim_layout.setContentsMargins(0, 0, 0, 0)

        fim_row1 = QHBoxLayout()
        fim_row1.addWidget(QLabel("Prefix:"))
        self.fim_prefix_edit = QTextEdit()
        self.fim_prefix_edit.setMaximumHeight(60)
        self.fim_prefix_edit.setPlaceholderText("def fib(a):")
        fim_row1.addWidget(self.fim_prefix_edit, 1)
        fim_layout.addLayout(fim_row1)

        fim_row2 = QHBoxLayout()
        fim_row2.addWidget(QLabel("Suffix:"))
        self.fim_suffix_edit = QTextEdit()
        self.fim_suffix_edit.setMaximumHeight(60)
        self.fim_suffix_edit.setPlaceholderText("    return fib(a-1) + fib(a-2)")
        fim_row2.addWidget(self.fim_suffix_edit, 1)
        fim_layout.addLayout(fim_row2)

        fim_row3 = QHBoxLayout()
        fim_row3.addWidget(QLabel("Max Tokens:"))
        self.fim_max_tokens_edit = QLineEdit("128")
        fim_row3.addWidget(self.fim_max_tokens_edit)
        self.fim_send_btn = QPushButton("生成补全")
        self.fim_send_btn.clicked.connect(self.send_fim_request)
        fim_row3.addWidget(self.fim_send_btn)
        fim_row3.addStretch()
        fim_layout.addLayout(fim_row3)

        self.input_layout.addWidget(self.chat_input_widget)
        self.input_layout.addWidget(self.fim_input_widget)
        self.fim_input_widget.hide()

        right_layout.addWidget(self.input_stack)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

    # --------------- 会话列表 ---------------
    def _populate_session_list(self):
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in self.manager.sessions:
            item_text = f"{session.name}  ({session.model}, {session.mode.upper()})"
            self.session_list.addItem(item_text)
        self.session_list.blockSignals(False)
        if self.manager.sessions:
            self.session_list.setCurrentRow(0)
            self.switch_session(0)

    def switch_session(self, index):
        if self.streaming:
            QMessageBox.warning(self, "请稍候", "当前正在生成回复，请等待完成后再切换会话。")
            return

        if index < 0 or index >= len(self.manager.sessions):
            self.current_session_index = -1
            self.chat_display.clear()
            self.session_info_label.setText("未选择会话")
            return

        self.current_session_index = index
        session = self.manager.sessions[index]

        self.session_info_label.setText(
            f"当前会话：{session.name}  |  模型：{session.model}  |  模式：{session.mode.upper()}"
        )

        self.chat_display.clear()
        for msg in session.messages:
            self._append_historical_message(msg)

        if session.mode == "fim":
            self.chat_input_widget.hide()
            self.fim_input_widget.show()
        else:
            self.fim_input_widget.hide()
            self.chat_input_widget.show()
            self.prefix_check.setVisible(session.mode == "prefix")
            self.prefix_check.setChecked(False)

    def new_session(self):
        dialog = NewSessionDialog(self)
        if dialog.exec():
            name, model, mode = dialog.get_values()
            if not name:
                QMessageBox.warning(self, "警告", "会话名称不能为空")
                return
            session = self.manager.add_session(name, model, mode)
            self._populate_session_list()
            self.session_list.setCurrentRow(len(self.manager.sessions) - 1)

    def delete_session(self):
        if self.current_session_index == -1:
            return
        session = self.manager.sessions[self.current_session_index]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除会话“{session.name}”吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.manager.delete_session(self.current_session_index)
            self._populate_session_list()
            if self.manager.sessions:
                new_idx = min(self.current_session_index, len(self.manager.sessions) - 1)
                self.session_list.setCurrentRow(new_idx)
                self.switch_session(new_idx)
            else:
                self.switch_session(-1)

    # --------------- Token 对话框 ---------------
    def open_token_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("设置 DeepSeek API Key")
        layout = QFormLayout(dialog)
        key_edit = QLineEdit(self.api_key)
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setPlaceholderText("输入你的 DeepSeek API Key")
        layout.addRow("API Key:", key_edit)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("保存")
        cancel_btn = QPushButton("取消")
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addRow(btn_box)

        save_btn.clicked.connect(lambda: self._save_token_and_close(key_edit.text(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _save_token_and_close(self, key, dialog):
        key = key.strip()
        if not key:
            QMessageBox.warning(self, "警告", "API Key 不能为空")
            return
        self.api_key = key
        save_token(key)
        dialog.accept()
        self.statusBar().showMessage("Token 已保存", 3000)

    # --------------- 消息显示辅助 ---------------
    def _append_historical_message(self, msg):
        """显示历史消息（已经包含 reasoning 和 content）"""
        role = msg.get("role", "user")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning", "")

        if role == "user":
            html = f"<b style='color:blue;'>你:</b> {self._md2html(content)}"
        elif role == "assistant":
            html = "<b style='color:green;'>助手:</b>"
            if reasoning:
                html += f"<details><summary>思考过程</summary>{self._md2html(reasoning)}</details>"
                html += f"<br>{self._md2html(content)}"
            else:
                html += f" {self._md2html(content)}"
        else:
            html = f"<b>{role}:</b> {self._md2html(content)}"

        self.chat_display.append(html)

    def _md2html(self, text: str) -> str:
        """Markdown 转 HTML，空文本返回空"""
        if not text:
            return ""
        try:
            return markdown.markdown(text, extensions=['fenced_code', 'tables'])
        except:
            return text.replace('\n', '<br>')

    def _update_current_streaming_message(self):
        """更新当前正在流式显示的最后一条助手消息"""
        # 移除最后一条动态占位消息并重新插入
        # 简单做法：获取当前HTML，删除最后插入的临时片段，再追加新的
        # 但 QTextBrowser 不好直接编辑最后一块。更好的方式：使用 QTextCursor 删除最后一个块，然后插入新块
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        # 如果正 streaming，最后应该是一个我们插入的占位段落，我们删除它并重新渲染
        # 下面采用更底层的方式：
        # 维护一个 streaming_block_index，但这里直接重置最后一片内容会简单一点。
        pass

    # 但是上面的方式较麻烦，我们采用另一种策略：预先插入一个占位 HTML 标签，然后通过 JavaScript 更新？Qt 的 HTML 限制多。
    # 简化方案：每次收到流式数据，清除当前最后一条助手消息（如果是流式占位），然后重新构建整个显示区的最后部分。
    # 为了性能，我们记录流式消息在 QTextBrowser 中的位置（通过滚动到底部，然后用 cursor 选择并删除最后一个 block）。

    def _start_streaming_placeholder(self):
        """插入占位符，并记录当前文档的结尾位置"""
        self.chat_display.append("<b style='color:green;'>助手:</b> ")
        # 记录此时文档的总字符数（加完后末尾位置）
        self.streaming_start_pos = self.chat_display.document().characterCount()

    def _update_streaming_content(self):
        """删除从流式开始位置到末尾的所有内容，再以最新累积内容重新插入"""
        doc = self.chat_display.document()
        # 构造一个光标，选中从 streaming_start_pos 到结尾的区域
        cursor = QTextCursor(doc)
        cursor.setPosition(self.streaming_start_pos - 1)  # 0-based
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

        # 构建新内容
        html = ""
        if self.current_reasoning:
            html += f"<details open><summary>思考过程</summary>{self._md2html(self.current_reasoning)}</details>"
            if self.current_content:
                html += "<br>"
        html += self._md2html(self.current_content)

        # 在当前位置（即原占位符之后）插入新 HTML
        cursor.insertHtml(html)

        # 滚动到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    # --------------- 消息发送 ---------------
    def send_message(self):
        if self.current_session_index == -1:
            QMessageBox.information(self, "提示", "请先选择一个会话")
            return
        if not self.api_key:
            QMessageBox.warning(self, "未设置 Token", "请先设置 API Key")
            return
        if self.streaming:
            QMessageBox.warning(self, "请稍候", "正在生成回复，请等待完成")
            return

        text = self.message_edit.toPlainText().strip()
        if not text:
            return

        session = self.manager.sessions[self.current_session_index]
        prefix = self.prefix_check.isChecked() and session.mode == "prefix"

        role = "user" if not prefix else "assistant"
        self.chat_display.append(f"<b style='color:blue;'>你:</b> {self._md2html(text)}")

        input_data = {"role": role, "content": text, "prefix": prefix}
        new_msg = {"role": role, "content": text}
        if prefix:
            new_msg["prefix"] = True
        session.messages.append(new_msg)
        self.manager.save()

        self._start_streaming_placeholder()
        self.streaming = True
        self.current_reasoning = ""
        self.current_content = ""

        self.message_edit.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.statusBar().showMessage("生成中...")

        self.thread = ChatThread(session, self.api_key, input_data)
        self.thread.stream_signal.connect(self.on_stream_chunk)
        self.thread.finished_signal.connect(self.on_thread_finished)
        self.thread.start()

    def send_fim_request(self):
        if self.current_session_index == -1:
            QMessageBox.information(self, "提示", "请先选择一个会话")
            return
        if not self.api_key:
            QMessageBox.warning(self, "未设置 Token", "请先设置 API Key")
            return
        if self.streaming:
            QMessageBox.warning(self, "请稍候", "正在生成回复")
            return

        session = self.manager.sessions[self.current_session_index]
        prefix_text = self.fim_prefix_edit.toPlainText().strip()
        suffix_text = self.fim_suffix_edit.toPlainText().strip()
        max_tokens_str = self.fim_max_tokens_edit.text().strip()

        if not prefix_text and not suffix_text:
            QMessageBox.warning(self, "输入为空", "请至少填写 Prefix 或 Suffix")
            return
        try:
            max_tokens = int(max_tokens_str) if max_tokens_str else 128
        except ValueError:
            QMessageBox.warning(self, "格式错误", "Max Tokens 必须是整数")
            return

        self.chat_display.append(
            f"<b style='color:blue;'>你:</b> [FIM] Prefix:<br>{self._md2html(prefix_text)}<br>Suffix:<br>{self._md2html(suffix_text)}"
        )
        fim_msg = {
            "role": "user",
            "prefix": prefix_text,
            "suffix": suffix_text,
            "max_tokens": max_tokens
        }
        session.messages.append(fim_msg)
        self.manager.save()

        self._start_streaming_placeholder()
        self.streaming = True
        self.current_reasoning = ""
        self.current_content = ""

        self.fim_prefix_edit.setEnabled(False)
        self.fim_suffix_edit.setEnabled(False)
        self.fim_send_btn.setEnabled(False)

        input_data = {"prefix": prefix_text, "suffix": suffix_text, "max_tokens": max_tokens}
        self.thread = ChatThread(session, self.api_key, input_data)
        self.thread.stream_signal.connect(self.on_stream_chunk)
        self.thread.finished_signal.connect(self.on_thread_finished)
        self.thread.start()

    def on_stream_chunk(self, chunk_type: str, content: str):
        if chunk_type == "reasoning":
            self.current_reasoning += content
        elif chunk_type == "content":
            self.current_content += content
        self._update_streaming_content()

    def on_thread_finished(self, success: bool, error_msg: str):
        self.streaming = False
        session = self.manager.sessions[self.current_session_index]

        if success:
            # 保存完成的消息（包含 reasoning）
            assistant_msg = {
                "role": "assistant",
                "content": self.current_content
            }
            if self.current_reasoning:
                assistant_msg["reasoning"] = self.current_reasoning
            session.messages.append(assistant_msg)
            self.manager.save()
            self.statusBar().showMessage("就绪", 3000)
        else:
            self.chat_display.append(f"<b style='color:red;'>错误:</b> {error_msg}")
            self.statusBar().showMessage("请求失败", 5000)

        # 恢复输入
        if session.mode == "fim":
            self.fim_prefix_edit.setEnabled(True)
            self.fim_suffix_edit.setEnabled(True)
            self.fim_send_btn.setEnabled(True)
            self.fim_prefix_edit.clear()
            self.fim_suffix_edit.clear()
        else:
            self.message_edit.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.message_edit.clear()
            self.prefix_check.setChecked(False)

        self.message_edit.setFocus()


# ===================== 启动 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

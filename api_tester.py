import base64
import io
import json
import os
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import requests
from PIL import Image, ImageTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "api_tester_config.json")
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "downloads")

THEME = {
    "bg": "#F7F2EA",
    "panel": "#FFFDF9",
    "ink": "#1E1E1E",
    "muted": "#6B6B6B",
    "accent": "#2F6B4F",
    "accent_alt": "#B45309",
    "border": "#E4D9C8",
    "chat_user_bg": "#E3F2FF",
    "chat_user_fg": "#1B3A57",
    "chat_assistant_bg": "#E9F5EE",
    "chat_assistant_fg": "#203A2D",
    "steps_bg": "#FCFAF6",
}


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("450x500")  # Increased height
        self.configure(bg=THEME["bg"])
        self.config = config
        self.result = None

        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # Add a scrollable container
        canvas = tk.Canvas(self, bg=THEME["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=THEME["bg"])

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # Content inside scrollable frame
        tk.Label(
            scrollable_frame, text="Base URL:", bg=THEME["bg"], fg=THEME["muted"]
        ).pack(anchor="w", padx=10, pady=(10, 0))
        self.url_entry = tk.Entry(scrollable_frame, width=50, relief="flat")
        self.url_entry.insert(0, config.get("base_url", "http://34.69.150.103:8000/v1"))
        self.url_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(
            scrollable_frame, text="Model:", bg=THEME["bg"], fg=THEME["muted"]
        ).pack(anchor="w", padx=10)
        self.model_entry = tk.Entry(scrollable_frame, width=50, relief="flat")
        self.model_entry.insert(0, config.get("model", "openmanus"))
        self.model_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(
            scrollable_frame,
            text="API Key (Bearer):",
            bg=THEME["bg"],
            fg=THEME["muted"],
        ).pack(anchor="w", padx=10)
        self.api_key_entry = tk.Entry(scrollable_frame, width=50, show="*", relief="flat")
        self.api_key_entry.insert(0, config.get("api_key", ""))
        self.api_key_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(
            scrollable_frame,
            text="Groq API Key (Optional):",
            bg=THEME["bg"],
            fg=THEME["muted"],
        ).pack(anchor="w", padx=10)
        self.groq_key_entry = tk.Entry(scrollable_frame, width=50, show="*", relief="flat")
        self.groq_key_entry.insert(0, config.get("groq_api_key", ""))
        self.groq_key_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(
            scrollable_frame,
            text="Daytona API Key (Optional):",
            bg=THEME["bg"],
            fg=THEME["muted"],
        ).pack(anchor="w", padx=10)
        self.daytona_key_entry = tk.Entry(scrollable_frame, width=50, show="*", relief="flat")
        self.daytona_key_entry.insert(0, config.get("daytona_api_key", ""))
        self.daytona_key_entry.pack(fill="x", padx=10, pady=5)

        save_btn = tk.Button(
            scrollable_frame,
            text="Save Settings",
            command=self.save,
            bg=THEME["accent"],
            fg="white",
            font=("Avenir", 10, "bold"),
            pady=5,
        )
        save_btn.pack(pady=20, padx=10, fill="x")

    def save(self):
        self.result = {
            "base_url": self.url_entry.get().strip(),
            "model": self.model_entry.get().strip(),
            "api_key": self.api_key_entry.get().strip(),
            "groq_api_key": self.groq_key_entry.get().strip(),
            "daytona_api_key": self.daytona_key_entry.get().strip(),
        }
        self.destroy()


class FileBrowser(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent.root)
        self.parent = parent
        self.title("Workspace Files")
        self.geometry("520x500")
        self.configure(bg=THEME["bg"])
        self.entries = []

        top_frame = tk.Frame(self, bg=THEME["bg"])
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(top_frame, text="Path:", bg=THEME["bg"], fg=THEME["muted"]).pack(
            side="left"
        )
        self.path_var = tk.StringVar(value="")
        self.path_entry = tk.Entry(top_frame, textvariable=self.path_var, relief="flat")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))

        tk.Button(
            top_frame, text="Up", command=self.go_up, bg=THEME["panel"]
        ).pack(side="left", padx=(0, 5))
        tk.Button(
            top_frame, text="Refresh", command=self.refresh, bg=THEME["panel"]
        ).pack(side="left")

        list_frame = tk.Frame(self, bg=THEME["bg"])
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.listbox = tk.Listbox(
            list_frame,
            bg=THEME["panel"],
            fg=THEME["ink"],
            highlightthickness=1,
            relief="flat",
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-1>", self.on_double_click)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        button_frame = tk.Frame(self, bg=THEME["bg"])
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Button(
            button_frame,
            text="Download Selected",
            command=self.download_selected,
            bg=THEME["accent"],
            fg="white",
        ).pack(side="left")
        self.status_label = tk.Label(button_frame, text="", bg=THEME["bg"], fg=THEME["muted"])
        self.status_label.pack(side="right")

        self.refresh()

    def set_status(self, text):
        self.status_label.config(text=text)

    def go_up(self):
        current = self.path_var.get().strip().strip("/")
        if not current:
            return
        parent = os.path.dirname(current)
        self.path_var.set(parent)
        self.refresh()

    def refresh(self):
        path = self.path_var.get().strip().strip("/")
        try:
            self.entries = self.parent.fetch_workspace_files(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.listbox.delete(0, tk.END)
        for entry in self.entries:
            self.listbox.insert(tk.END, self._display_label(entry, path))
        self.set_status(f"{len(self.entries)} items")

    def _display_label(self, entry, current_path):
        rel_path = entry.get("path", "")
        display = rel_path
        if current_path:
            prefix = current_path.rstrip("/") + "/"
            if rel_path.startswith(prefix):
                display = rel_path[len(prefix) :]
        if entry.get("is_dir"):
            display = f"{display}/"
        return display or rel_path

    def on_double_click(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.entries[selection[0]]
        if entry.get("is_dir"):
            self.path_var.set(entry.get("path", ""))
            self.refresh()
        else:
            self.download_entry(entry)

    def download_selected(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        entry = self.entries[selection[0]]
        if entry.get("is_dir"):
            self.path_var.set(entry.get("path", ""))
            self.refresh()
            return
        self.download_entry(entry)

    def download_entry(self, entry):
        rel_path = entry.get("path", "")
        try:
            save_path = self.parent.download_workspace_file(rel_path)
        except Exception as e:
            messagebox.showerror("Download Failed", str(e))
            return
        messagebox.showinfo("Downloaded", f"Saved to {save_path}")


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgentOrchestra API Chat")
        self.root.geometry("1100x760")
        self.root.configure(bg=THEME["bg"])

        self.config = self.load_config()
        self.messages = []
        self.images = []  # Keep references to prevent GC
        self.response_count = 0
        self._build_fonts()
        self._apply_theme()

        # Menu
        menubar = tk.Menu(
            root, bg=THEME["bg"], fg=THEME["ink"], activebackground=THEME["panel"]
        )
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Settings", command=self.open_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        filesmenu = tk.Menu(menubar, tearoff=0)
        filesmenu.add_command(label="Browse Workspace", command=self.open_file_browser)
        menubar.add_cascade(label="Files", menu=filesmenu)
        root.config(menu=menubar)

        # Header with Settings Button
        header_frame = tk.Frame(root, pady=8, bg=THEME["bg"])
        header_frame.pack(fill="x", padx=16)

        title_block = tk.Frame(header_frame, bg=THEME["bg"])
        title_block.pack(side="left")
        tk.Label(
            title_block,
            text="AgentOrchestra",
            font=self.font_title,
            bg=THEME["bg"],
            fg=THEME["ink"],
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="API Tester",
            font=self.font_subtitle,
            bg=THEME["bg"],
            fg=THEME["muted"],
        ).pack(anchor="w")

        controls = tk.Frame(header_frame, bg=THEME["bg"])
        controls.pack(side="right")
        self.status_label = tk.Label(
            controls,
            text=self._status_text(),
            font=self.font_small,
            bg=THEME["bg"],
            fg=THEME["muted"],
        )
        self.status_label.pack(side="left", padx=(0, 10))

        self.clear_button = tk.Button(
            controls,
            text="Clear",
            command=self.clear_chat,
            bg=THEME["panel"],
            fg=THEME["ink"],
            relief="flat",
        )
        self.clear_button.pack(side="left", padx=(0, 8))

        self.settings_button = tk.Button(
            controls,
            text="Settings",
            command=self.open_settings,
            bg=THEME["accent"],
            fg="white",
            relief="flat",
        )
        self.settings_button.pack(side="left")

        # Main Content
        content_pane = ttk.PanedWindow(root, orient="horizontal")
        content_pane.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        left_frame = ttk.Frame(content_pane)
        right_frame = ttk.Frame(content_pane)
        content_pane.add(left_frame, weight=3)
        content_pane.add(right_frame, weight=2)

        # Chat Area (Conversation)
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, state="disabled", wrap="word", height=18
        )
        self.chat_display.pack(fill="both", expand=True)
        self.chat_display.configure(
            bg=THEME["panel"],
            fg=THEME["ink"],
            relief="flat",
            padx=12,
            pady=12,
            insertbackground=THEME["ink"],
            font=self.font_body,
        )
        self.chat_display.tag_config(
            "user",
            foreground=THEME["chat_user_fg"],
            background=THEME["chat_user_bg"],
            justify="right",
            lmargin1=140,
            lmargin2=140,
            rmargin=24,
            spacing1=6,
            spacing3=8,
        )
        self.chat_display.tag_config(
            "assistant",
            foreground=THEME["chat_assistant_fg"],
            background=THEME["chat_assistant_bg"],
            justify="left",
            lmargin1=24,
            lmargin2=24,
            rmargin=140,
            spacing1=6,
            spacing3=8,
        )
        self.chat_display.tag_config(
            "assistant_label", foreground=THEME["chat_assistant_fg"]
        )
        self.chat_display.tag_config("user_label", foreground=THEME["chat_user_fg"])

        # Input Area
        input_frame = tk.Frame(left_frame, bg=THEME["bg"])
        input_frame.pack(fill="x", padx=0, pady=10)

        self.input_text = tk.Text(
            input_frame,
            height=3,
            bg="#FFFEFB",
            fg=THEME["ink"],
            relief="flat",
            padx=10,
            pady=8,
            font=self.font_body,
            insertbackground=THEME["ink"],
        )
        self.input_text.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self.handle_return)

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg=THEME["accent"],
            fg="white",
            relief="flat",
            padx=18,
            pady=6,
            font=self.font_small_bold,
        )
        self.send_button.pack(side="right")

        # Right Panel: Steps & Final Response
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        steps_frame = ttk.Frame(notebook)
        final_frame = ttk.Frame(notebook)
        notebook.add(steps_frame, text="Steps & Logs")
        notebook.add(final_frame, text="Final Response")

        self.steps_display = scrolledtext.ScrolledText(
            steps_frame, state="disabled", wrap="word"
        )
        self.steps_display.pack(fill="both", expand=True)
        self.steps_display.configure(
            bg=THEME["steps_bg"],
            fg=THEME["ink"],
            relief="flat",
            padx=12,
            pady=10,
            font=self.font_small,
        )
        self.steps_display.tag_config("step", foreground=THEME["muted"])
        self.steps_display.tag_config(
            "step_header", foreground=THEME["ink"], font=self.font_small_bold
        )

        self.final_display = scrolledtext.ScrolledText(
            final_frame, state="disabled", wrap="word"
        )
        self.final_display.pack(fill="both", expand=True)
        self.final_display.configure(
            bg=THEME["panel"],
            fg=THEME["ink"],
            relief="flat",
            padx=12,
            pady=10,
            font=self.font_body,
        )
        self.final_display.tag_config(
            "final_header", foreground=THEME["muted"], font=self.font_small_bold
        )
        self.final_display.tag_config("assistant", foreground=THEME["chat_assistant_fg"])

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {
            "base_url": "http://34.69.150.103:8000/v1",
            "model": "openmanus",
            "api_key": "",
            "groq_api_key": "",
            "daytona_api_key": "",
        }

    def _build_fonts(self):
        self.font_title = tkfont.Font(family="Avenir Next", size=18, weight="bold")
        self.font_subtitle = tkfont.Font(family="Avenir", size=11, weight="normal")
        self.font_body = tkfont.Font(family="Avenir", size=11)
        self.font_small = tkfont.Font(family="Avenir", size=10)
        self.font_small_bold = tkfont.Font(
            family="Avenir", size=10, weight="bold"
        )
        self.root.option_add("*Font", self.font_body)

    def _apply_theme(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TFrame", background=THEME["bg"], borderwidth=0, relief="flat"
        )
        style.configure(
            "TNotebook",
            background=THEME["bg"],
            borderwidth=0,
            padding=4,
        )
        style.configure(
            "TNotebook.Tab",
            background=THEME["panel"],
            foreground=THEME["ink"],
            padding=[12, 6],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", THEME["accent"])],
            foreground=[("selected", "white")],
        )
        style.configure("TPanedwindow", background=THEME["bg"])

    def _status_text(self):
        base_url = self.config.get("base_url", "")
        model = self.config.get("model", "")
        return f"{model}  |  {base_url}"

    def clear_chat(self):
        self.messages = []
        self.response_count = 0
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state="disabled")
        self.steps_display.config(state="normal")
        self.steps_display.delete("1.0", tk.END)
        self.steps_display.config(state="disabled")
        self.final_display.config(state="normal")
        self.final_display.delete("1.0", tk.END)
        self.final_display.config(state="disabled")

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def open_settings(self):
        dialog = SettingsDialog(self.root, self.config)
        self.root.wait_window(dialog)
        if dialog.result:
            self.config = dialog.result
            self.save_config()
            if hasattr(self, "status_label"):
                self.status_label.config(text=self._status_text())
            messagebox.showinfo("Settings", "Settings saved successfully!")

    def handle_return(self, event):
        if not event.state & 0x1:  # shift not pressed
            self.send_message()
            return "break"  # prevent default behavior (newline)

    def build_headers(self):
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _append_to(self, widget, text, tag=None):
        widget.config(state="normal")
        if tag:
            widget.insert(tk.END, text, tag)
        else:
            widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state="disabled")

    def set_status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.config(text=text)

    def append_message(self, role, content):
        if role == "system":
            self.append_step_message(content)
            return
        if role == "user":
            self.append_chat_message("You", content, "user")
            self.messages.append({"role": role, "content": content})
            return
        if role == "assistant":
            self.append_chat_message("Assistant", content, "assistant")
            self.append_final_block(content)
            self.messages.append({"role": role, "content": content})
            return

    def append_chat_message(self, label, content, tag):
        label_tag = f"{tag}_label" if tag in {"assistant", "user"} else tag
        self._append_to(self.chat_display, f"{label}\n", label_tag)
        self._append_to(self.chat_display, f"{content}\n\n", tag)

    def append_step_message(self, content):
        self._append_to(self.steps_display, "- ", "step")
        self._append_to(self.steps_display, f"{content}\n", "step")

    def append_final_block(self, content):
        self.response_count += 1
        self._append_to(self.final_display, f"Response {self.response_count}\n", "final_header")
        self._append_to(self.final_display, f"{content}\n\n", "assistant")

    def update_last_message(self, content_chunk):
        self._append_to(self.chat_display, content_chunk, "assistant")
        self._append_to(self.final_display, content_chunk, "assistant")

    def start_assistant_response(self):
        self.response_count += 1
        self._append_to(self.chat_display, "Assistant\n", "assistant_label")
        self._append_to(
            self.final_display,
            f"Response {self.response_count}\n",
            "final_header",
        )

    def finish_assistant_response(self):
        self._append_to(self.chat_display, "\n\n", "assistant")
        self._append_to(self.final_display, "\n\n", "assistant")

    def display_browser_action(self, action_payload):
        action = action_payload.get("action") or "unknown"
        args = action_payload.get("args") or {}
        display_text = f"Browser action: {action} | args: {args}"
        self.append_step_message(display_text)

    def display_log(self, log_text):
        self.append_step_message(log_text)

    def _unique_download_path(self, filename):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        base, ext = os.path.splitext(filename)
        candidate = os.path.join(DOWNLOAD_DIR, filename)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(DOWNLOAD_DIR, f"{base}_{counter}{ext}")
            counter += 1
        return candidate

    def display_file(self, payload):
        try:
            file_name = payload.get("file_name") or "shared_file"
            base64_data = payload.get("base64")
            if not base64_data:
                self.append_message("system", "Received file share with no data.")
                return
            file_bytes = base64.b64decode(base64_data)
            save_path = self._unique_download_path(file_name)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            size = payload.get("file_size") or len(file_bytes)
            mime = payload.get("mime_type") or "application/octet-stream"
            self.append_step_message(
                f"Received file '{file_name}' ({size} bytes, {mime}) saved to {save_path}"
            )
        except Exception as e:
            self.append_step_message(f"Failed to save shared file: {str(e)}")

    def fetch_workspace_files(self, path=""):
        base_url = self.config.get("base_url")
        endpoint = f"{base_url}/workspace/files"
        response = requests.get(
            endpoint,
            headers=self.build_headers(),
            params={"path": path, "recursive": False},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Error {response.status_code}: {response.text}")
        return response.json().get("files", [])

    def download_workspace_file(self, rel_path):
        base_url = self.config.get("base_url")
        endpoint = f"{base_url}/workspace/file"
        response = requests.get(
            endpoint,
            headers=self.build_headers(),
            params={"path": rel_path},
            stream=True,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Error {response.status_code}: {response.text}")

        filename = None
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[1].strip().strip("\"'")
        if not filename:
            filename = os.path.basename(rel_path) or "downloaded_file"

        save_path = self._unique_download_path(filename)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return save_path

    def open_file_browser(self):
        FileBrowser(self)

    def display_image(self, base64_str):
        try:
            cleaned = base64_str.strip()
            if cleaned.startswith("data:"):
                cleaned = cleaned.split(",", 1)[1].strip()
            image_data = base64.b64decode(cleaned)
            image = Image.open(io.BytesIO(image_data))
            max_width = self.steps_display.winfo_width() - 40
            if max_width < 200:
                max_width = 420
            if image.width > max_width:
                ratio = max_width / image.width
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            img = ImageTk.PhotoImage(image)
            self.steps_display.config(state="normal")
            self.steps_display.insert(tk.END, "Snapshot\n", "step_header")
            self.steps_display.image_create(tk.END, image=img)
            self.steps_display.insert(tk.END, "\n", "step")
            self.steps_display.see(tk.END)
            self.steps_display.config(state="disabled")
            self.images.append(img)
        except Exception as e:
            print(f"Image error: {e}")

    def send_message(self):
        content = self.input_text.get("1.0", tk.END).strip()
        if not content:
            return

        self.input_text.delete("1.0", tk.END)
        self.append_message("user", content)

        # Run in thread
        self.send_button.config(state=tk.DISABLED)
        self.set_status("Streaming response...")
        thread = threading.Thread(target=self.run_api_request)
        thread.start()

    def run_api_request(self):
        try:
            base_url = self.config.get("base_url")
            api_key = self.config.get("api_key")
            model = self.config.get("model")
            groq_key = self.config.get("groq_api_key")
            daytona_key = self.config.get("daytona_api_key")

            endpoint = f"{base_url}/chat/completions"
            headers = self.build_headers()

            payload = {
                "model": model,
                "messages": self.messages,
                "stream": True,
            }

            if groq_key:
                payload["groq_api_key"] = groq_key
            if daytona_key:
                payload["daytona_api_key"] = daytona_key

            response = requests.post(
                endpoint, headers=headers, json=payload, stream=True, timeout=120
            )

            if response.status_code == 200:
                # Prepare UI for streaming assistant response
                self.root.after(0, self.start_assistant_response)

                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0]["delta"]
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]
                                        if content.startswith("BrowserSnapshot: "):
                                            base64_img = content.replace(
                                                "BrowserSnapshot: ", ""
                                            ).strip()
                                            self.root.after(
                                                0,
                                                lambda img=base64_img: self.display_image(
                                                    img
                                                ),
                                            )
                                        elif content.startswith("BrowserAction: "):
                                            payload_str = content.replace(
                                                "BrowserAction: ", ""
                                            ).strip()
                                            try:
                                                payload = json.loads(payload_str)
                                            except json.JSONDecodeError:
                                                payload = {"action": "unknown", "args": {}}
                                            self.root.after(
                                                0,
                                                lambda p=payload: self.display_browser_action(
                                                    p
                                                ),
                                            )
                                        elif content.startswith("Step: "):
                                            self.root.after(
                                                0,
                                                lambda t=content.strip(): self.display_log(
                                                    t
                                                ),
                                            )
                                        elif content.startswith("Log: "):
                                            self.root.after(
                                                0,
                                                lambda t=content.strip(): self.display_log(
                                                    t
                                                ),
                                            )
                                        elif content.startswith("FileShare: "):
                                            payload_str = content.replace(
                                                "FileShare: ", ""
                                            ).strip()
                                            try:
                                                payload = json.loads(payload_str)
                                            except json.JSONDecodeError:
                                                payload = {}
                                            self.root.after(
                                                0,
                                                lambda p=payload: self.display_file(p),
                                            )
                                        elif content.startswith("Snapshot: "):
                                            base64_img = content.replace(
                                                "Snapshot: ", ""
                                            ).strip()
                                            self.root.after(
                                                0,
                                                lambda img=base64_img: self.display_image(
                                                    img
                                                ),
                                            )
                                        else:
                                            full_content += content
                                            # Update the last message (assistant) with new content
                                            self.root.after(
                                                0,
                                                lambda c=content: self.update_last_message(
                                                    c
                                                ),
                                            )
                            except json.JSONDecodeError:
                                pass

                # Update memory with full response
                self.root.after(0, self.finish_assistant_response)
                self.messages.append({"role": "assistant", "content": full_content})

            else:
                self.root.after(
                    0,
                    self.append_message,
                    "system",
                    f"Error {response.status_code}: {response.text}",
                )

        except Exception as e:
            self.root.after(
                0, self.append_message, "system", f"Request failed: {str(e)}"
            )
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.set_status(self._status_text()))


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

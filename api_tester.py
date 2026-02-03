import base64
import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "api_tester_config.json")
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "downloads")


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("450x500")  # Increased height
        self.config = config
        self.result = None

        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # Add a scrollable container
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # Content inside scrollable frame
        tk.Label(scrollable_frame, text="Base URL:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.url_entry = tk.Entry(scrollable_frame, width=50)
        self.url_entry.insert(0, config.get("base_url", "http://34.69.150.103:8000/v1"))
        self.url_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(scrollable_frame, text="Model:").pack(anchor="w", padx=10)
        self.model_entry = tk.Entry(scrollable_frame, width=50)
        self.model_entry.insert(0, config.get("model", "openmanus"))
        self.model_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(scrollable_frame, text="API Key (Bearer):").pack(anchor="w", padx=10)
        self.api_key_entry = tk.Entry(scrollable_frame, width=50, show="*")
        self.api_key_entry.insert(0, config.get("api_key", ""))
        self.api_key_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(scrollable_frame, text="Groq API Key (Optional):").pack(
            anchor="w", padx=10
        )
        self.groq_key_entry = tk.Entry(scrollable_frame, width=50, show="*")
        self.groq_key_entry.insert(0, config.get("groq_api_key", ""))
        self.groq_key_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(scrollable_frame, text="Daytona API Key (Optional):").pack(
            anchor="w", padx=10
        )
        self.daytona_key_entry = tk.Entry(scrollable_frame, width=50, show="*")
        self.daytona_key_entry.insert(0, config.get("daytona_api_key", ""))
        self.daytona_key_entry.pack(fill="x", padx=10, pady=5)

        save_btn = tk.Button(
            scrollable_frame,
            text="Save Settings",
            command=self.save,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
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
        self.entries = []

        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(top_frame, text="Path:").pack(side="left")
        self.path_var = tk.StringVar(value="")
        self.path_entry = tk.Entry(top_frame, textvariable=self.path_var)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))

        tk.Button(top_frame, text="Up", command=self.go_up).pack(side="left", padx=(0, 5))
        tk.Button(top_frame, text="Refresh", command=self.refresh).pack(side="left")

        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.listbox = tk.Listbox(list_frame)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-1>", self.on_double_click)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Button(button_frame, text="Download Selected", command=self.download_selected).pack(
            side="left"
        )
        self.status_label = tk.Label(button_frame, text="")
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
        self.root.geometry("980x700")

        self.config = self.load_config()
        self.messages = []
        self.images = []  # Keep references to prevent GC
        self.response_count = 0

        # Menu
        menubar = tk.Menu(root)
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
        header_frame = tk.Frame(root, pady=5)
        header_frame.pack(fill="x", padx=10)

        tk.Label(header_frame, text="Chat Session", font=("Arial", 12, "bold")).pack(
            side="left"
        )

        self.settings_button = tk.Button(
            header_frame, text="⚙️ Settings", command=self.open_settings
        )
        self.settings_button.pack(side="right")

        # Main Content
        content_pane = ttk.PanedWindow(root, orient="horizontal")
        content_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.Frame(content_pane)
        right_frame = ttk.Frame(content_pane)
        content_pane.add(left_frame, weight=3)
        content_pane.add(right_frame, weight=2)

        # Chat Area (Conversation)
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, state="disabled", wrap="word"
        )
        self.chat_display.pack(fill="both", expand=True)
        self.chat_display.tag_config("user", foreground="#1a4a7a", justify="right")
        self.chat_display.tag_config("assistant", foreground="#1f6b3a", justify="left")
        self.chat_display.tag_config("assistant_label", foreground="#1f6b3a")

        # Input Area
        input_frame = tk.Frame(left_frame)
        input_frame.pack(fill="x", padx=0, pady=10)

        self.input_text = tk.Text(input_frame, height=3)
        self.input_text.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self.handle_return)

        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
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
        self.steps_display.tag_config("step", foreground="#555555")
        self.steps_display.tag_config("step_header", foreground="#333333", font=("Arial", 10, "bold"))

        self.final_display = scrolledtext.ScrolledText(
            final_frame, state="disabled", wrap="word"
        )
        self.final_display.pack(fill="both", expand=True)
        self.final_display.tag_config("final_header", foreground="#333333", font=("Arial", 10, "bold"))
        self.final_display.tag_config("assistant", foreground="#1f6b3a")

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

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def open_settings(self):
        dialog = SettingsDialog(self.root, self.config)
        self.root.wait_window(dialog)
        if dialog.result:
            self.config = dialog.result
            self.save_config()
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

    def append_message(self, role, content):
        if role == "system":
            self.append_step_message(content)
            return
        if role == "user":
            self.append_chat_message("User", content, "user")
            self.messages.append({"role": role, "content": content})
            return
        if role == "assistant":
            self.append_chat_message("Assistant", content, "assistant")
            self.append_final_block(content)
            self.messages.append({"role": role, "content": content})
            return

    def append_chat_message(self, label, content, tag):
        self._append_to(self.chat_display, f"{label}: ", f"{tag}_label" if tag == "assistant" else tag)
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
        self._append_to(self.chat_display, "Assistant: ", "assistant_label")
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
            image_data = base64.b64decode(base64_str)
            img = tk.PhotoImage(data=image_data)
            self.steps_display.config(state="normal")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

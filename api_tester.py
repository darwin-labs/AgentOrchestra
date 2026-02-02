import base64
import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "api_tester_config.json")


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


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgentOrchestra API Chat")
        self.root.geometry("500x700")

        self.config = self.load_config()
        self.messages = []
        self.images = []  # Keep references to prevent GC

        # Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Settings", command=self.open_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
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

        # Chat Area
        self.chat_display = scrolledtext.ScrolledText(
            root, state="disabled", wrap="word"
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.chat_display.tag_config("user", foreground="blue", justify="right")
        self.chat_display.tag_config("assistant", foreground="green", justify="left")
        self.chat_display.tag_config("system", foreground="gray", justify="center")

        # Input Area
        input_frame = tk.Frame(root)
        input_frame.pack(fill="x", padx=10, pady=10)

        self.input_text = tk.Text(input_frame, height=3)
        self.input_text.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self.handle_return)

        self.send_button = tk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side="right")

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

    def append_message(self, role, content):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, f"{role.capitalize()}: {content}\n\n", role)
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")
        if role != "system":
            self.messages.append({"role": role, "content": content})

    def update_last_message(self, content_chunk):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, content_chunk, "assistant")
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def display_image(self, base64_str):
        try:
            image_data = base64.b64decode(base64_str)
            img = tk.PhotoImage(data=image_data)
            self.chat_display.config(state="normal")
            self.chat_display.image_create(tk.END, image=img)
            self.chat_display.insert(tk.END, "\n", "assistant")
            self.chat_display.see(tk.END)
            self.chat_display.config(state="disabled")
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
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

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
                self.root.after(0, lambda: self.append_message("assistant", ""))

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
                                        if content.startswith("Snapshot: "):
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

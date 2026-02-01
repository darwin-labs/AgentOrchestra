import json
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

import requests


class ApiTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgentOrchestra API Tester")
        self.root.geometry("600x800")

        # API Configuration
        config_frame = tk.LabelFrame(root, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(config_frame, text="Base URL:").grid(row=0, column=0, sticky="w")
        self.url_entry = tk.Entry(config_frame, width=50)
        self.url_entry.insert(0, "http://34.69.150.103:8000/v1")
        self.url_entry.grid(row=0, column=1, sticky="w")

        tk.Label(config_frame, text="API Key (Bearer):").grid(
            row=1, column=0, sticky="w"
        )
        self.api_key_entry = tk.Entry(config_frame, width=50, show="*")
        self.api_key_entry.grid(row=1, column=1, sticky="w")

        tk.Label(config_frame, text="Model:").grid(row=2, column=0, sticky="w")
        self.model_entry = tk.Entry(config_frame, width=50)
        self.model_entry.insert(0, "openmanus")
        self.model_entry.grid(row=2, column=1, sticky="w")

        # Custom Keys
        tk.Label(config_frame, text="Groq API Key:").grid(row=3, column=0, sticky="w")
        self.groq_key_entry = tk.Entry(config_frame, width=50, show="*")
        self.groq_key_entry.grid(row=3, column=1, sticky="w")

        tk.Label(config_frame, text="Daytona API Key:").grid(
            row=4, column=0, sticky="w"
        )
        self.daytona_key_entry = tk.Entry(config_frame, width=50, show="*")
        self.daytona_key_entry.grid(row=4, column=1, sticky="w")

        # Request Body
        req_frame = tk.LabelFrame(root, text="Request", padx=10, pady=10)
        req_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(req_frame, text="User Message:").pack(anchor="w")
        self.message_text = scrolledtext.ScrolledText(req_frame, height=5)
        self.message_text.insert(tk.END, "Hello, who are you?")
        self.message_text.pack(fill="both", expand=True)

        self.send_button = tk.Button(
            root, text="Send Request", command=self.send_request_thread
        )
        self.send_button.pack(pady=10)

        # Response Body
        res_frame = tk.LabelFrame(root, text="Response", padx=10, pady=10)
        res_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.response_text = scrolledtext.ScrolledText(res_frame, height=15)
        self.response_text.pack(fill="both", expand=True)

    def send_request_thread(self):
        # Disable button and run in thread to keep UI responsive
        self.send_button.config(state=tk.DISABLED)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, "Sending request...\n")

        thread = threading.Thread(target=self.send_request)
        thread.start()

    def send_request(self):
        base_url = self.url_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        model = self.model_entry.get().strip()
        groq_key = self.groq_key_entry.get().strip()
        daytona_key = self.daytona_key_entry.get().strip()
        message = self.message_text.get(1.0, tk.END).strip()

        endpoint = f"{base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        if groq_key:
            payload["groq_api_key"] = groq_key
        if daytona_key:
            payload["daytona_api_key"] = daytona_key

        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=120
            )
            self.root.after(0, self.display_response, response)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))

    def display_response(self, response):
        try:
            data = response.json()
            formatted_json = json.dumps(data, indent=4)
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(
                tk.END, f"Status Code: {response.status_code}\n\n"
            )
            self.response_text.insert(tk.END, formatted_json)
        except:
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(
                tk.END, f"Status Code: {response.status_code}\n\n"
            )
            self.response_text.insert(tk.END, response.text)

    def display_error(self, error_msg):
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, f"Error: {error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ApiTesterApp(root)
    root.mainloop()

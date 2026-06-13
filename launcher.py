import customtkinter as ctk
import subprocess
import sys
import os
import yaml
import threading
import queue
import time
import sounddevice as sd
import webbrowser
from typing import Optional

# Configuration
CONFIG_PATH = "config.yaml"
THEME_COLOR = "blue"  # options: "blue", "green", "dark-blue"
APPEARANCE_MODE = "Dark"  # options: "System", "Dark", "Light"

ctk.set_appearance_mode(APPEARANCE_MODE)
ctk.set_default_color_theme(THEME_COLOR)

class BackendLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Live Translation Server")
        self.geometry("800x600")
        
        # Grid configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Console row expands

        # State
        self.process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.config_data = self.load_config()
        self.log_queue = queue.Queue()

        # UI Components
        self.create_header()
        self.create_settings_frame()
        self.create_control_frame()
        self.create_console_view()

        # Update loop for logs
        self.after(100, self.update_console)

    def load_config(self):
        """Load configuration from config.yaml."""
        if not os.path.exists(CONFIG_PATH):
            return {}
        try:
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def save_config(self):
        """Save current configuration to config.yaml."""
        if not self.config_data:
            return

        try:
            # Update values from UI
            self.config_data['audio']['device'] = self.audio_device_var.get()
            self.config_data['translation']['backend'] = self.model_backend_var.get()
            
            # Map language names back to codes
            source_name = self.source_lang_var.get()
            target_name = self.target_lang_var.get()
            
            source_info = self.languages.get(source_name, {"code": "de"})
            target_info = self.languages.get(target_name, {"code": "fa", "tts": "fa_IR-amir-medium"})

            # Update Translation
            self.config_data['translation']['source_lang'] = source_info['code']
            self.config_data['translation']['target_lang'] = target_info['code']
            
            # Update STT (sync with source)
            self.config_data['stt']['language'] = source_info['code']
            
            # Update TTS (sync with target)
            self.config_data['tts']['model'] = target_info['tts']

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(self.config_data, f, sort_keys=False)
            
            self.log(f"Configuration saved. Source: {source_name}, Target: {target_name}, TTS: {target_info['tts']}")
        except Exception as e:
            self.log(f"Error saving config: {e}")

    def create_header(self):
        """Create the header section."""
        self.header_frame = ctk.CTkFrame(self, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")

        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="Live Translation Server", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=10, padx=20, side="left")

        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="STOPPED",
            text_color="red",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(pady=10, padx=20, side="right")

    def create_settings_frame(self):
        """Create the settings configuration section."""
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.settings_frame.grid_columnconfigure(1, weight=1)
        self.settings_frame.grid_columnconfigure(3, weight=1)

        # Language Definitions
        self.languages = {
            "German": {"code": "de", "tts": "de_DE-thorsten-medium"},
            "English": {"code": "en", "tts": "en_US-lessac-medium"},
            "Farsi": {"code": "fa", "tts": "fa_IR-amir-medium"},
        }
        # Reverse mapping for loading config (code -> name)
        self.code_to_name = {v['code']: k for k, v in self.languages.items()}

        # 1. Audio Device
        ctk.CTkLabel(self.settings_frame, text="Audio Input:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        devices = self.get_audio_devices()
        current_device = self.config_data.get('audio', {}).get('device', '')
        device_names = [d['name'] for d in devices]
        if current_device and current_device not in device_names:
             device_names.insert(0, current_device)

        self.audio_device_var = ctk.StringVar(value=current_device)
        self.audio_device_menu = ctk.CTkOptionMenu(
            self.settings_frame, 
            values=device_names,
            variable=self.audio_device_var
        )
        self.audio_device_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # 2. Model Backend
        ctk.CTkLabel(self.settings_frame, text="Model Backend:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        current_backend = self.config_data.get('translation', {}).get('backend', 'nllb')
        self.model_backend_var = ctk.StringVar(value=current_backend)
        self.model_backend_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=["nllb", "translategemma-mlx", "translategemma", "gemini-live"],
            variable=self.model_backend_var
        )
        self.model_backend_menu.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        # 3. Source Language
        ctk.CTkLabel(self.settings_frame, text="Source Lang:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        # Load from config or default to German
        current_source_code = self.config_data.get('translation', {}).get('source_lang', 'de')
        current_source_name = self.code_to_name.get(current_source_code, "German")
        
        self.source_lang_var = ctk.StringVar(value=current_source_name)
        self.source_lang_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=list(self.languages.keys()),
            variable=self.source_lang_var
        )
        self.source_lang_menu.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # 4. Target Language
        ctk.CTkLabel(self.settings_frame, text="Target Lang:").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        # Load from config or default to Farsi
        current_target_code = self.config_data.get('translation', {}).get('target_lang', 'fa')
        current_target_name = self.code_to_name.get(current_target_code, "Farsi")
        
        self.target_lang_var = ctk.StringVar(value=current_target_name)
        self.target_lang_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=list(self.languages.keys()),
            variable=self.target_lang_var
        )
        self.target_lang_menu.grid(row=1, column=3, padx=10, pady=10, sticky="ew")

    def create_control_frame(self):

        """Create buttons to control the server."""
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.start_button = ctk.CTkButton(
            self.control_frame, 
            text="Start System", 
            fg_color="green", 
            hover_color="darkgreen",
            command=self.start_server
        )
        self.start_button.pack(side="left", padx=10, expand=True, fill="x")

        self.stop_button = ctk.CTkButton(
            self.control_frame, 
            text="Stop System", 
            fg_color="red", 
            hover_color="darkred",
            state="disabled",
            command=self.stop_server
        )
        self.stop_button.pack(side="left", padx=10, expand=True, fill="x")
        
        self.frontend_button = ctk.CTkButton(
            self.control_frame,
            text="Open Web UI",
            fg_color="#3B8ED0", # Default blue
            command=self.open_frontend
        )
        self.frontend_button.pack(side="left", padx=10, expand=True, fill="x")

    def create_console_view(self):
        """Create the console output text box."""
        console_label = ctk.CTkLabel(self, text="Console Output:", anchor="w")
        console_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")

        self.console_textbox = ctk.CTkTextbox(self, font=("Courier", 12))
        self.console_textbox.grid(row=4, column=0, padx=20, pady=(5, 20), sticky="nsew")
        
        # Re-jig grid to put console in middle and controls at bottom or vice versa.
        # Let's put Grid: 0=Header, 1=Settings, 2=Config/Controls? No controls better at 2.
        # Let's Swap row 2 and 3.
        
        # Redoing grid layout for better flow:
        # Row 0: Header
        # Row 1: Settings
        # Row 2: Controls
        # Row 3: Label "Console"
        # Row 4: Console
        
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.control_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        console_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.console_textbox.grid(row=4, column=0, padx=20, pady=(5, 20), sticky="nsew")
        
        self.grid_rowconfigure(4, weight=1) # Console expands

    def get_audio_devices(self):
        """Get list of audio input devices."""
        try:
            full_devices = sd.query_devices()
            # Filter for input devices
            input_devices = [d for d in full_devices if d['max_input_channels'] > 0]
            return input_devices
        except Exception as e:
            self.log(f"Error listing audio devices: {e}")
            return []

    def start_server(self):
        """Start the backend server process."""
        if self.is_running:
            return

        self.save_config()
        self.log("Starting backend server...")
        
        # 1. Start Backend
        cmd_backend = ["python", "-m", "src.main"]
        
        try:
            # Start process with unbuffered output
            self.process = subprocess.Popen(
                cmd_backend,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1, # Line buffered
                cwd=os.getcwd()
            )
            
            # 2. Start Frontend
            self.log("Starting frontend (npm run dev)...")
            cmd_frontend = ["npm", "run", "dev", "--", "--host"]
            ui_cwd = os.path.join(os.getcwd(), "ui")
            
            self.frontend_process = subprocess.Popen(
                cmd_frontend,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=ui_cwd
            )
            
            self.is_running = True
            self.update_ui_state(running=True)
            
            # Start thread to read backend stdout
            self.reader_thread = threading.Thread(target=self.read_output, daemon=True)
            self.reader_thread.start()
            
            # Optional: Reader for frontend? For now just log start.
            
        except Exception as e:
            self.log(f"Failed to start system: {e}")
            self.is_running = False
            self.update_ui_state(running=False)
            if self.process: self.process.kill()
            if self.frontend_process: self.frontend_process.kill()

    def stop_server(self):
        """Stop the backend server process."""
        if not self.is_running:
            return
            
        self.log("Stopping system...")
        
        # Stop Backend
        if self.process:
            try:
                self.process.terminate()
            except Exception as e:
                self.log(f"Error stopping backend: {e}")
        
        # Stop Frontend
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.log("Frontend stopped.")
            except Exception as e:
                self.log(f"Error stopping frontend: {e}")

        self.is_running = False
        self.process = None
        self.frontend_process = None
        self.update_ui_state(running=False)
        self.log("System stopped.")

    def update_ui_state(self, running):
        """Update UI elements based on running state."""
        if running:
            self.status_label.configure(text="RUNNING", text_color="green")
            self.start_button.configure(state="disabled", fg_color="gray")
            self.stop_button.configure(state="normal", fg_color="red")
            
            # Disable settings while running
            # self.audio_device_menu.configure(state="disabled") # CustomTkinter might not support this directly on all elements same way
            # workaround: just leave them enabled, changes apply on next restart
        else:
            self.status_label.configure(text="STOPPED", text_color="red")
            self.start_button.configure(state="normal", fg_color="green")
            self.stop_button.configure(state="disabled", fg_color="gray")

    def read_output(self):
        """Read stdout from subprocess and put into queue."""
        process = self.process
        if not process:
            return
            
        # Use the local variable 'process' which holds the reference even if self.process becomes None
        for line in iter(process.stdout.readline, ''):
            if line:
                self.log_queue.put(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0 and self.is_running:
             self.log_queue.put(f"\n[Backend exited with code {return_code}]\n")
             # Trigger UI update from main thread logic if needed, 
             # but we handle state in stop_server mostly. 
             # If it crashes unexpectedly:
             # We need a way to detect crash in main thread.
             self.log_queue.put("CRASH_DETECTED")

    def update_console(self):
        """Check queue for new logs and update text box."""
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "CRASH_DETECTED":
                    self.is_running = False
                    self.process = None
                    self.update_ui_state(running=False)
                    self.log("Server process terminated unexpectedly.")
                else:
                    self.console_textbox.insert("end", line)
                    self.console_textbox.see("end")
        except queue.Empty:
            pass
        
        self.after(100, self.update_console)

    def log(self, message):
        """Add internal message to console."""
        timestamp = time.strftime("[%H:%M:%S]")
        self.console_textbox.insert("end", f"{timestamp} [GUI] {message}\n")
        self.console_textbox.see("end")
        
    def open_frontend(self):
         webbrowser.open("http://localhost:5173")

    def on_close(self):
        """Handle window closing."""
        if self.is_running:
            self.stop_server()
        self.destroy()

if __name__ == "__main__":
    app = BackendLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

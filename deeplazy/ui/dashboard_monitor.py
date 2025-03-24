from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.layout import Layout
from rich.align import Align
import time
import psutil
import os
import torch


class DashboardMonitor:
    def __init__(self, model_name=None, safetensors_path=None, max_visible_layers=25,
                 max_layers_in_memory=10, cache_type=None, framework=None):
        self.enabled = False
        self.start_time = None
        self.console = Console()
        self.live = None
        self.model_name = model_name or "LazyLLM"
        self.safetensors_path = safetensors_path or "N/A"
        self.layer_records = []
        self.max_visible_layers = max_visible_layers
        self.total_exec_time = 0
        self.max_layers_in_memory = max_layers_in_memory
        self.cache_type = cache_type
        self.framework = framework

    def enable(self):
        self.enabled = True
        self.start_time = time.time()
        self.live = Live(self._build_layout(),
                         console=self.console, refresh_per_second=4)
        self.live.__enter__()

    def record_layer(self, layer_name, exec_time):
        if not self.enabled:
            return

        memory = self._memory_usage()
        self.layer_records.append({
            "name": layer_name,
            "time": exec_time,
            "memory": memory
        })

        if len(self.layer_records) > self.max_visible_layers:
            self.layer_records = self.layer_records[-self.max_visible_layers:]

        self._refresh()

    def _build_table(self):
        table = Table(
            title=f"🧠 Execution Dashboard — Model: {self.model_name}",
            expand=True
        )
        table.add_column("Layer", justify="left",
                         style="bold cyan", no_wrap=True)
        table.add_column("Built", justify="center", style="green")
        table.add_column("Execution Time (s)", justify="center")
        table.add_column("Memory (MB)", justify="center")

        for record in self.layer_records:
            table.add_row(
                record["name"],
                "✅",
                f"{record['time']:.2f}",
                f"{record['memory']:.2f}"
            )

        return table

    def _build_footer_panel(self):
        self.total_exec_time = time.time() - self.start_time if self.start_time else 0
        cpu_usage = psutil.cpu_percent()
        gpu_usage = self._gpu_usage()
        memory = self._memory_usage()

        footer_text = (
            f"📦 Model: {self.model_name}\n"
            f"📁 Safetensors Path: {self.safetensors_path}\n"
            f"⏱️ Total Execution Time: {self.total_exec_time:.2f} seconds\n"
            f"🧠 Final Memory Usage: {memory:.2f} MB\n"
            f"💻 CPU Usage: {cpu_usage:.1f}%\n"
            f"🖥️ GPU Usage: {gpu_usage}\n"
            f"🔄 Max Layers in Memory: {self.max_layers_in_memory}\n"
            f"🧩 Cache Type: {self.cache_type}\n"
            f"⚙️ Framework: {self.framework}"
        )

        return Panel(
            Align.left(footer_text),
            title="📊 System Info",
            border_style="bold magenta",
            expand=True
        )

    def _build_layout(self):
        layout = Layout()
        footer_lines = self._footer_line_count()
        layout.split_column(
            Layout(name="dashboard", ratio=3),
            Layout(name="footer", size=footer_lines)
        )
        layout["dashboard"].update(self._build_table())
        layout["footer"].update(self._build_footer_panel())
        return layout

    def _footer_line_count(self):
        # Count lines in footer dynamically
        footer_text = (
            f"📦 Model: {self.model_name}\n"
            f"📁 Safetensors Path: {self.safetensors_path}\n"
            f"⏱️ Total Execution Time: {self.total_exec_time:.2f} seconds\n"
            f"🧠 Final Memory Usage: {self._memory_usage():.2f} MB\n"
            f"💻 CPU Usage: {psutil.cpu_percent():.1f}%\n"
            f"🖥️ GPU Usage: {self._gpu_usage()}\n"
            f"🔄 Max Layers in Memory: {self.max_layers_in_memory}\n"
            f"🧩 Cache Type: {self.cache_type}\n"
            f"⚙️ Framework: {self.framework}"
        )
        return footer_text.count("\n") + 4

    def _refresh(self):
        if self.live:
            self.live.update(self._build_layout())

    def print_footer(self):
        if self.live:
            self.live.update(self._build_layout())
            self.live.__exit__(None, None, None)

    def _memory_usage(self):
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    def _gpu_usage(self):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return f"{torch.cuda.utilization(0)}%" if hasattr(torch.cuda, 'utilization') else "N/A"
        except Exception:
            pass
        return "N/A"

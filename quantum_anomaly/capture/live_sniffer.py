from typing import Optional, Callable, Dict, Any
import threading
import time
from scapy.all import sniff, conf, IFACES
from quantum_anomaly.utils.logging_setup import setup_logger

logger = setup_logger("live_sniffer")

class LiveSniffer:
    def __init__(self, iface: Optional[str] = None, bpf_filter: Optional[str] = None):
        self.iface = iface
        self.bpf_filter = bpf_filter
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def list_interfaces() -> Dict[str, str]:
        # Map name to description
        try:
            return {iface.name: str(iface) for iface in IFACES.data.values()}
        except Exception:
            return {}

    def start(self, on_packet: Callable[[Any], None], store: bool = False):
        if self._thread and self._thread.is_alive():
            logger.info("Sniffer already running")
            return
        self._stop.clear()

        def _run():
            logger.info(f"Starting sniff on iface={self.iface} filter={self.bpf_filter}")
            try:
                sniff(iface=self.iface, prn=on_packet, store=store, filter=self.bpf_filter, stop_filter=lambda _: self._stop.is_set())
            except PermissionError:
                logger.error("Permission denied. Run as admin/root or set capabilities.")
            except Exception as e:
                logger.exception(f"Sniff error: {e}")
            finally:
                logger.info("Sniffer stopped")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

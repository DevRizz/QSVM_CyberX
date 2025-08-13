from typing import List, Any, Optional
from scapy.all import rdpcap, Packet
import io

def load_pcap(file_like: io.BytesIO) -> List[Packet]:
    """
    Load packets from an uploaded PCAP file-like object.
    """
    file_like.seek(0)
    data = file_like.read()
    packets = rdpcap(io.BytesIO(data))
    return list(packets)

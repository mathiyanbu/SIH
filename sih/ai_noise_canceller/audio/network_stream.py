"""LAN WebRTC delivery of already-processed RNNoise audio.

This server is intended for trusted local networks only. It does not capture
microphone audio and does not run RNNoise; the application publishes processed
frames through ``publish_frame``.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    from aiohttp import web
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack
    from av import AudioFrame
except ImportError:  # pragma: no cover - depends on optional Pi installation
    web = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AudioStreamTrack = None
    AudioFrame = None

SAMPLE_RATE = 48000
FRAME_SIZE = 480
NETWORK_QUEUE_FRAMES = 32
NETWORK_PORT = 8765


PHONE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NOISELESS-X LIVE AUDIO</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b100d; color: #eef0e7; }
main { width: min(92vw, 420px); padding: 28px; background: #171d18; border: 1px solid #52644b; border-radius: 8px; box-sizing: border-box; }
h1 { margin: 0 0 8px; font-size: 1.45rem; letter-spacing: .04em; }
p { color: #b7c4b0; }
button { width: 100%; margin-top: 10px; padding: 13px; border: 0; border-radius: 5px; color: white; background: #6e9b42; font-weight: 700; font-size: 1rem; }
button:last-of-type { background: #303b30; }
#status { margin-top: 18px; color: #a8c86b; font-weight: 700; }
</style>
</head>
<body><main>
<h1>NOISELESS-X LIVE AUDIO</h1>
<p>Raspberry Pi Connected</p>
<button id="start">START AUDIO</button>
<button id="stop">STOP AUDIO</button>
<div id="status">Connecting...</div>
<audio id="audio" autoplay playsinline></audio>
<script>
const status = document.getElementById('status');
const audio = document.getElementById('audio');
let peer = null;
function setStatus(text) { status.textContent = text; }
async function pollStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const state = await response.json();
    if (!peer || peer.connectionState === 'closed') {
      setStatus(state.live ? 'Waiting to receive processed audio' : 'Waiting for live processing');
    }
  } catch (error) { setStatus('Connection failed'); }
}
async function startAudio() {
  if (peer) return;
  setStatus('Connecting...');
  try {
    peer = new RTCPeerConnection();
    peer.addTransceiver('audio', { direction: 'recvonly' });
    peer.ontrack = event => { audio.srcObject = event.streams[0]; audio.play().catch(() => {}); };
    peer.onconnectionstatechange = () => {
      if (peer.connectionState === 'connected') setStatus('Receiving processed audio');
      if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) setStatus('Connection failed');
    };
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    const response = await fetch('/offer', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sdp: peer.localDescription.sdp, type: peer.localDescription.type}) });
    if (!response.ok) throw new Error('Offer rejected');
    const answer = await response.json();
    await peer.setRemoteDescription(answer);
  } catch (error) { setStatus('Connection failed'); if (peer) { peer.close(); peer = null; } }
}
function stopAudio() { if (peer) peer.close(); peer = null; audio.srcObject = null; setStatus('Stopped'); }
document.getElementById('start').onclick = startAudio;
document.getElementById('stop').onclick = stopAudio;
setInterval(pollStatus, 1000); pollStatus();
</script>
</main></body></html>"""


def local_ipv4_address() -> str | None:
    """Find the preferred LAN address without assuming wlan0 or eth0."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        address = sock.getsockname()[0]
        return address if address and not address.startswith("127.") else None
    except OSError:
        return None
    finally:
        sock.close()


if AudioStreamTrack is not None:
    class NetworkAudioTrack(AudioStreamTrack):
        """One WebRTC consumer with a bounded asyncio frame queue."""

        kind = "audio"

        def __init__(self):
            super().__init__()
            self.frames = asyncio.Queue(maxsize=NETWORK_QUEUE_FRAMES)
            self.pts = 0
            self.next_delivery = None

        async def recv(self):
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self.next_delivery is None:
                self.next_delivery = now
            else:
                self.next_delivery += FRAME_SIZE / SAMPLE_RATE
                await asyncio.sleep(max(0.0, self.next_delivery - now))

            try:
                samples = await asyncio.wait_for(self.frames.get(), timeout=0.2)
            except asyncio.TimeoutError:
                samples = np.zeros(FRAME_SIZE, dtype=np.float32)

            values = np.asarray(samples, dtype=np.float32).reshape(-1)
            if values.size < FRAME_SIZE:
                values = np.pad(values, (0, FRAME_SIZE - values.size))
            values = np.clip(values[:FRAME_SIZE], -1.0, 1.0)
            pcm = (values * 32767.0).astype(np.int16)
            frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SIZE)
            frame.planes[0].update(pcm.tobytes())
            frame.sample_rate = SAMPLE_RATE
            frame.pts = self.pts
            frame.time_base = Fraction(1, SAMPLE_RATE)
            self.pts += FRAME_SIZE
            return frame
else:
    NetworkAudioTrack = None


class NetworkAudioServer:
    """Background aiohttp/WebRTC server for processed live audio."""

    def __init__(self, status_provider=None, port: int = NETWORK_PORT):
        self.status_provider = status_provider or (lambda: {"live": False, "enabled": False})
        self.port = port
        self.enabled = True
        self.thread = None
        self.loop = None
        self.runner = None
        self.ready = threading.Event()
        self.start_error = None
        self.stop_requested = threading.Event()
        self.peers = set()
        self.tracks = set()
        self.display_url = None

    @property
    def available(self):
        return web is not None and NetworkAudioTrack is not None

    @property
    def client_count(self):
        return len(self.peers)

    def start(self) -> bool:
        if not self.available:
            self.start_error = "Install aiohttp and aiortc to enable browser audio streaming."
            return False
        if self.thread and self.thread.is_alive():
            return True
        self.thread = threading.Thread(target=self._run, name="network-audio-server", daemon=True)
        self.thread.start()
        self.ready.wait(timeout=5.0)
        return self.start_error is None

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_async())
        self.ready.set()
        if self.start_error is None:
            self.loop.run_forever()
        self.loop.run_until_complete(self._stop_async())
        self.loop.close()

    async def _start_async(self):
        app = web.Application()
        app.router.add_get("/", self._page)
        app.router.add_get("/api/status", self._status)
        app.router.add_post("/offer", self._offer)
        self.runner = web.AppRunner(app)
        try:
            await self.runner.setup()
            site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await site.start()
            address = local_ipv4_address() or "<Pi-LAN-IP>"
            self.display_url = f"http://{address}:{self.port}"
            print("[Network] Audio server started")
            print(f"[Network] Port: {self.port}")
            print("[Network] Open this page from a device on the same LAN:")
            print(f"[Network] {self.display_url}")
        except Exception as exc:
            self.start_error = str(exc)
            if self.runner is not None:
                await self.runner.cleanup()
                self.runner = None

    async def _page(self, request):
        return web.Response(text=PHONE_PAGE, content_type="text/html")

    async def _status(self, request):
        state = dict(self.status_provider())
        state.update({"enabled": self.enabled, "clients": len(self.peers)})
        return web.json_response(state)

    async def _offer(self, request):
        if not self.enabled:
            return web.json_response({"error": "Phone streaming is disabled"}, status=503)
        data = await request.json()
        peer = RTCPeerConnection()
        track = NetworkAudioTrack()
        self.peers.add(peer)
        self.tracks.add(track)
        peer.addTrack(track)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await self._close_peer(peer, track)

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            return web.json_response({"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})
        except Exception:
            await self._close_peer(peer, track)
            raise

    async def _close_peer(self, peer, track):
        self.peers.discard(peer)
        self.tracks.discard(track)
        with contextlib.suppress(Exception):
            await peer.close()

    def publish_frame(self, audio):
        """Schedule a non-blocking copy to each peer's bounded queue."""
        if not self.enabled or not self.loop or not self.peers:
            return
        frame = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        self.loop.call_soon_threadsafe(self._publish_on_loop, frame)

    def _publish_on_loop(self, frame):
        if not self.enabled:
            return
        for track in tuple(self.tracks):
            try:
                track.frames.put_nowait(frame.copy())
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    track.frames.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    track.frames.put_nowait(frame.copy())

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def stop(self):
        if not self.thread:
            return
        self.stop_requested.set()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)
        self.thread = None

    async def _stop_async(self):
        for peer in tuple(self.peers):
            with contextlib.suppress(Exception):
                await peer.close()
        self.peers.clear()
        self.tracks.clear()
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None


__all__ = ["NetworkAudioServer", "local_ipv4_address", "NETWORK_PORT"]

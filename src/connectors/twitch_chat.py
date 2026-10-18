"""Twitch Chat Connector via IRC."""

import time
import socket
import ssl
import threading
import html
from collections import deque
from typing import List, Dict, Any, Optional

from src.core.schemas import TwitchChatMessage
from src.metrics import record_chat_received
from src.monitoring.decorators import safe_telemetry_dispatch

TWITCH_DEFAULT_COLORS = [
    "#FF0000",
    "#0000FF",
    "#00FF7F",
    "#B22222",
    "#FF7F50",
    "#9ACD32",
    "#FF4500",
    "#2E8B57",
    "#DAA520",
    "#D2691E",
    "#5F9EA0",
    "#1E90FF",
    "#FF69B4",
    "#8A2BE2",
    "#32CD32",
    "#FF1493",
    "#00CED1",
    "#FFD700",
    "#7B68EE",
    "#FF6347",
]


def get_twitch_color(username: str, custom_color: str = "") -> str:
    if custom_color and custom_color.startswith("#") and len(custom_color) in (4, 7):
        return custom_color
    if not username:
        return "#1E90FF"
    idx = sum(ord(c) for c in username.lower()) % len(TWITCH_DEFAULT_COLORS)
    return TWITCH_DEFAULT_COLORS[idx]


class TwitchChatListener:
    IRC_SERVER = "irc.chat.twitch.tv"
    IRC_PORT = 6697

    def __init__(self, channel: str, max_buffer_size: int = 150):
        self.channel = channel.lstrip("#").lower()
        self.max_buffer_size = max_buffer_size
        self._messages = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[ssl.SSLSocket] = None
        self._connected = False

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_irc_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_recent_messages(self, limit: int = 15, include_author: bool = False) -> List[str]:
        with self._lock:
            if include_author:
                msgs = [f"{item.sender}: {item.text}" for item in self._messages]
            else:
                msgs = [item.text for item in self._messages]
        return msgs[-limit:]

    def get_recent_messages_full(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._messages)[-limit:]
        return [
            {
                "sender": item.sender,
                "text": item.text,
                "color": item.color,
                "time": time.time(),
                "timestamp": item.timestamp,
            }
            for item in items
        ]

    def get_recent_chat_objects(self, limit: int = 15) -> List[TwitchChatMessage]:
        with self._lock:
            return list(self._messages)[-limit:]

    def get_recent_formatted_chat(
        self, limit: int = 15, include_timestamp: bool = False
    ) -> List[str]:
        with self._lock:
            items = list(self._messages)[-limit:]
        result = []
        for item in items:
            if include_timestamp and item.timestamp:
                result.append(f"[{item.timestamp}] {item.sender}: {item.text}")
            else:
                result.append(f"{item.sender}: {item.text}")
        return result

    def get_recent_chat_text(self, limit: int = 15, include_timestamp: bool = False) -> str:
        lines = self.get_recent_formatted_chat(limit=limit, include_timestamp=include_timestamp)
        return "\n".join(lines) if lines else "(В чате пока тихо)"

    def get_recent_chat_html(self, limit: int = 25, max_height: int = 340) -> str:
        with self._lock:
            items = list(self._messages)[-limit:]

        if not items:
            return (
                f'<div style="background-color: #18181b; border: 1px solid #27272a; '
                f"border-radius: 8px; padding: 16px; height: {max_height}px; box-sizing: border-box; "
                f"color: #71717a; font-family: Inter, sans-serif; font-size: 13px; "
                f'display: flex; align-items: center; justify-content: center; text-align: center;">'
                f"Ожидание новых сообщений из живого чата Twitch..."
                f"</div>"
            )

        lines_html = []
        for item in items:
            sender = html.escape(item.sender)
            text = html.escape(item.text)
            color = html.escape(item.color)
            ts = item.timestamp or time.strftime("%H:%M:%S")
            line = (
                f'<div style="margin-bottom: 7px; line-height: 1.45; word-break: break-word;">'
                f'<span style="color: #71717a; font-size: 11px; margin-right: 6px;">[{ts}]</span>'
                f'<strong style="color: {color}; font-weight: 700; margin-right: 3px;">{sender}</strong>'
                f'<span style="color: #efeff1;">: {text}</span>'
                f"</div>"
            )
            lines_html.append(line)

        body = "".join(lines_html)
        return (
            f'<div id="twitch-live-chat-box" style="background-color: #18181b; border: 1px solid #27272a; '
            f"border-radius: 8px; padding: 12px; height: {max_height}px; box-sizing: border-box; overflow-y: auto; "
            f'font-family: Inter, sans-serif; font-size: 13px;">'
            f"{body}"
            f'<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" '
            f'style="display:none;" onload="var b=this.parentElement;if(b){{b.scrollTop=b.scrollHeight;}}">'
            f"</div>"
        )

    def _run_irc_loop(self):
        while not self._stop_event.is_set():
            try:
                print(f"[Twitch Chat] Connecting anonymously to #{self.channel} via SSL...")
                raw_sock = socket.create_connection((self.IRC_SERVER, self.IRC_PORT), timeout=6.0)
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(raw_sock, server_hostname=self.IRC_SERVER)
                self._sock = sock
                sock.settimeout(2.0)

                nick = f"justinfan{int(time.time()) % 100000}"
                handshake = (
                    f"CAP REQ :twitch.tv/tags twitch.tv/commands\r\n"
                    f"PASS SCHMENDLICK\r\n"
                    f"NICK {nick}\r\n"
                    f"JOIN #{self.channel}\r\n"
                )
                sock.send(handshake.encode("utf-8"))

                self._connected = True
                print(f"[Twitch Chat] Successfully joined #{self.channel} as {nick}!")

                read_buffer = ""
                while not self._stop_event.is_set():
                    try:
                        data = sock.recv(2048).decode("utf-8", errors="ignore")
                    except socket.timeout:
                        continue
                    except Exception:
                        break

                    if not data:
                        break

                    read_buffer += data
                    lines = read_buffer.split("\r\n")
                    read_buffer = lines.pop()

                    for line in lines:
                        if line.startswith("PING"):
                            try:
                                sock.send(b"PONG :tmi.twitch.tv\r\n")
                            except Exception:
                                pass
                            continue

                        if "PRIVMSG" in line:
                            self._parse_privmsg(line)

            except Exception as e:
                self._connected = False
                if not self._stop_event.is_set():
                    print(f"[Twitch Chat] Connection error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
            finally:
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                self._connected = False

    def _parse_privmsg(self, line: str):
        try:
            sender = ""
            raw_color = ""
            content = ""
            if line.startswith("@"):
                tag_str, rest = line[1:].split(" :", 1)
                for item in tag_str.split(";"):
                    if item.startswith("display-name="):
                        sender = item.split("=", 1)[1]
                    elif item.startswith("color="):
                        raw_color = item.split("=", 1)[1]
                if " :" in rest:
                    prefix_part, content = rest.split(" :", 1)
                    if not sender:
                        sender = prefix_part.split("!", 1)[0].lstrip(":")
                else:
                    content = rest.split("PRIVMSG", 1)[-1].strip()
            else:
                clean_line = line.lstrip(":")
                if " :" in clean_line:
                    prefix_part, content = clean_line.split(" :", 1)
                    sender = prefix_part.split("!", 1)[0]

            sender = sender.strip() or "viewer"
            content = content.strip()
            user_color = get_twitch_color(sender, raw_color)

            if content:
                msg_obj = TwitchChatMessage(
                    sender=sender,
                    text=content,
                    color=user_color,
                    timestamp=time.strftime("%H:%M:%S"),
                    raw=line,
                )
                with self._lock:
                    self._messages.append(msg_obj)
                safe_telemetry_dispatch(record_chat_received)
        except Exception:
            pass

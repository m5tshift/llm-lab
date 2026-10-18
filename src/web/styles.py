"""Web UI styling, HTML helpers, and Twitch chat markup."""


def make_chat_status_html(status_text: str, color: str = "#a1a1aa", max_height: int = 340) -> str:
    """Renders a styled dark-mode status message card inside the Twitch chat widget."""
    return (
        f'<div style="background-color: #18181b; border: 1px solid #27272a; '
        f"border-radius: 8px; padding: 20px; height: {max_height}px; box-sizing: border-box; "
        f"color: {color}; font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
        f"font-size: 13.5px; display: flex; flex-direction: column; align-items: center; justify-content: center; "
        f'text-align: center; gap: 8px;">'
        f'<div style="font-weight: 600;">{status_text}</div>'
        f"</div>"
    )


def make_status_banner_html(status_text: str, color: str = "#a1a1aa") -> str:
    """Renders a compact notification banner card without massive empty vertical padding."""
    return (
        f'<div style="background-color: #18181b; border: 1px solid #27272a; '
        f"border-radius: 8px; padding: 12px 18px; box-sizing: border-box; "
        f"color: {color}; font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
        f"font-size: 13.5px; text-align: center; font-weight: 500; min-height: 48px; "
        f'display: flex; align-items: center; justify-content: center;">'
        f"{status_text}"
        f"</div>"
    )


CUSTOM_CSS = """
/* Centered layout & dark mode styling for Twitch Multimodal Studio */
body {
    display: flex !important;
    justify-content: center !important;
    background-color: #0b0f19 !important;
}

.gradio-container {
    max-width: 1360px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 24px 20px !important;
}

/* Center Tab navigation bar */
.tab-nav {
    display: flex !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    border-bottom: 1px solid #27272a !important;
    margin-bottom: 24px !important;
}

.tab-nav button {
    font-size: 14.5px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    border-radius: 6px !important;
    transition: all 0.15s ease-in-out !important;
}

/* Scrollbar styling for Twitch chat */
#twitch-live-chat-box::-webkit-scrollbar {
    width: 6px;
}
#twitch-live-chat-box::-webkit-scrollbar-track {
    background: #18181b;
}
#twitch-live-chat-box::-webkit-scrollbar-thumb {
    background: #3f3f46;
    border-radius: 3px;
}
"""

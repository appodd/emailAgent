from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    imap_host: str
    imap_port: int
    imap_user: str
    imap_password: str
    mailbox: str

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    time_window_hours: int = 72
    sim_threshold: float = 0.55
    state_path: str = "imap_state.json"
    request_timeout: int = 60


def get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    if val is None:
        if default is None:
            raise RuntimeError(f"Missing environment variable: {name}")
        return default
    return val


def load_config() -> Config:
    load_dotenv()

    imap_host = get_env("IMAP_HOST")
    imap_port = int(get_env("IMAP_PORT", "993"))
    imap_user = get_env("IMAP_USER")
    imap_password = get_env("IMAP_PASSWORD")
    mailbox = get_env("MAILBOX", "INBOX")

    deepseek_api_key = get_env("DEEPSEEK_API_KEY")
    deepseek_base_url = get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model = get_env("DEEPSEEK_MODEL", "deepseek-chat")

    time_window_hours = int(get_env("TIME_WINDOW_HOURS", "72"))
    sim_threshold = float(get_env("SIM_THRESHOLD", "0.55"))
    state_path = get_env("STATE_PATH", "imap_state.json")
    request_timeout = int(get_env("REQUEST_TIMEOUT", "60"))

    return Config(
        imap_host=imap_host,
        imap_port=imap_port,
        imap_user=imap_user,
        imap_password=imap_password,
        mailbox=mailbox,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_model=deepseek_model,
        time_window_hours=time_window_hours,
        sim_threshold=sim_threshold,
        state_path=state_path,
        request_timeout=request_timeout,
    )


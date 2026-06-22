"""
watchlist.py
============
Helpers para gestionar la cartera de jugadores en session_state.
"""

from __future__ import annotations

def _ensure_watchlist(session_state):
    if "watchlist" not in session_state:
        session_state.watchlist = []

def add_to_watchlist(player_name: str, session_state):
    _ensure_watchlist(session_state)
    if player_name and player_name not in session_state.watchlist:
        session_state.watchlist.append(player_name)

def remove_from_watchlist(player_name: str, session_state):
    _ensure_watchlist(session_state)
    if player_name in session_state.watchlist:
        session_state.watchlist.remove(player_name)

def is_in_watchlist(player_name: str, session_state) -> bool:
    _ensure_watchlist(session_state)
    return player_name in session_state.watchlist
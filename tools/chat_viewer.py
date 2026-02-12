#!/usr/bin/env python3
"""
Chat transcript viewer for inspecting focal-opposing conversations
where participants still expected commonality.

Usage:
    python tools/chat_viewer.py
    python tools/chat_viewer.py --filter opposing-commonality
    python tools/chat_viewer.py --filter all
"""

import argparse
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

LIKERT_LABELS = {
    1: "Definitely not",
    2: "Probably not",
    3: "Unsure",
    4: "Probably yes",
    5: "Definitely yes",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load messages and response data, returning (messages, chat_responses)."""
    msgs = pd.read_csv(DATA_DIR / "messages.csv")
    resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    chat = resp[resp["experiment"] == "chat"].copy()
    return msgs, chat


def get_dyad_info(chat: pd.DataFrame, group_id: str) -> list[dict]:
    """Get per-participant info for a dyad."""
    group = chat[chat["groupId"] == group_id]
    participants = []
    for pid in group["pid"].unique():
        p = group[group["pid"] == pid]
        focal = p[p["question_type"] == "observed"]
        if len(focal) == 0:
            continue
        focal = focal.iloc[0]
        participants.append(
            {
                "pid": pid,
                "focal_question": focal["matchedQuestion"],
                "focal_domain": focal["matchedDomain"],
                "self_response": int(focal["preChatResponse"]),
                "perceived_partner": (
                    int(focal["postChatResponse"])
                    if pd.notna(focal["postChatResponse"])
                    else None
                ),
                "commonality_judgment": int(focal["participant_binary_prediction"]),
                "stance": focal["stance"],
            }
        )
    return participants


def get_filtered_groups(
    msgs: pd.DataFrame, chat: pd.DataFrame, filter_mode: str
) -> list[str]:
    """Get group IDs matching the filter criteria."""
    all_groups = sorted(set(msgs["group_id"]) & set(chat["groupId"]))

    if filter_mode == "all":
        return all_groups

    if filter_mode == "opposing-commonality":
        # Opposing stance dyads where at least one participant expected commonality
        matching = []
        for gid in all_groups:
            info = get_dyad_info(chat, gid)
            if any(
                p["stance"] == "opposing" and p["commonality_judgment"] == 1
                for p in info
            ):
                matching.append(gid)
        return matching

    if filter_mode == "opposing":
        matching = []
        for gid in all_groups:
            info = get_dyad_info(chat, gid)
            if any(p["stance"] == "opposing" for p in info):
                matching.append(gid)
        return matching

    return all_groups


class ChatViewer:
    def __init__(
        self, root: tk.Tk, msgs: pd.DataFrame, chat: pd.DataFrame, groups: list[str]
    ):
        self.root = root
        self.msgs = msgs
        self.chat = chat
        self.groups = groups
        self.current_idx = 0

        self.root.title("Chat Transcript Viewer")
        self.root.geometry("800x700")
        self.root.configure(bg="#1a1a1a")

        # Dark theme style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            ".", background="#1a1a1a", foreground="white", fieldbackground="#1a1a1a"
        )
        style.configure("TFrame", background="#1a1a1a")
        style.configure("TLabel", background="#1a1a1a", foreground="white")
        style.configure("TLabelframe", background="#1a1a1a", foreground="white")
        style.configure("TLabelframe.Label", background="#1a1a1a", foreground="white")
        style.configure("TButton", background="#333333", foreground="white")

        self._build_ui()
        self._show_current()

    def _build_ui(self):
        # Top nav bar
        nav = ttk.Frame(self.root)
        nav.pack(fill="x", padx=10, pady=(10, 5))

        self.prev_btn = ttk.Button(nav, text="< Prev", command=self._prev)
        self.prev_btn.pack(side="left")

        self.counter_label = ttk.Label(nav, text="", font=("Helvetica Neue", 12))
        self.counter_label.pack(side="left", padx=20)

        self.next_btn = ttk.Button(nav, text="Next >", command=self._next)
        self.next_btn.pack(side="left")

        # Group ID label
        self.group_label = ttk.Label(
            nav, text="", font=("Helvetica Neue", 10, "italic")
        )
        self.group_label.pack(side="right")

        # Info panel (focal question + participant info)
        info_frame = ttk.LabelFrame(self.root, text="Dyad Info", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        self.focal_label = ttk.Label(
            info_frame, text="", font=("Helvetica Neue", 11, "bold"), wraplength=750
        )
        self.focal_label.pack(anchor="w")

        self.domain_label = ttk.Label(info_frame, text="", font=("Helvetica Neue", 10))
        self.domain_label.pack(anchor="w")

        self.participant_frame = ttk.Frame(info_frame)
        self.participant_frame.pack(fill="x", pady=(5, 0))

        # Chat transcript
        chat_frame = ttk.LabelFrame(self.root, text="Conversation", padding=10)
        chat_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            font=("Helvetica Neue", 12),
            bg="#1a1a1a",
            fg="white",
            relief="flat",
            padx=10,
            pady=10,
            spacing3=4,
            insertbackground="white",
        )
        self.chat_text.pack(fill="both", expand=True)

        # Configure tags for styling
        self.chat_text.tag_configure(
            "cat", foreground="#FF6B9D", font=("Helvetica Neue", 12, "bold")
        )
        self.chat_text.tag_configure(
            "dog", foreground="#7EB3FF", font=("Helvetica Neue", 12, "bold")
        )
        self.chat_text.tag_configure(
            "msg", foreground="white", font=("Helvetica Neue", 12)
        )

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.chat_text.configure(yscrollcommand=scrollbar.set)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())

    def _show_current(self):
        if not self.groups:
            return

        gid = self.groups[self.current_idx]
        self.counter_label.config(text=f"{self.current_idx + 1} / {len(self.groups)}")
        self.group_label.config(text=gid)

        # Get participant info
        info = get_dyad_info(self.chat, gid)

        # Focal question (same for both participants in a dyad)
        if info:
            self.focal_label.config(text=f"Focal: {info[0]['focal_question']}")
            self.domain_label.config(
                text=f"Domain: {info[0]['focal_domain']}  |  Stance: {info[0]['stance']}"
            )

        # Build author → pid map from messages
        group_msgs = self.msgs[self.msgs["group_id"] == gid].sort_values(
            "absolute_timestamp"
        )
        author_pid = group_msgs.groupby("author")["prolific_id"].first().to_dict()

        # Clear and rebuild participant info cards
        for widget in self.participant_frame.winfo_children():
            widget.destroy()

        for p in info:
            # Find which emoji this participant is
            emoji = next((e for e, pid in author_pid.items() if pid == p["pid"]), "?")
            color = "#DC267F" if emoji == "\U0001f431" else "#648FFF"

            card = ttk.Frame(self.participant_frame)
            card.pack(side="left", fill="x", expand=True, padx=(0, 10))

            emoji_color = "#FF6B9D" if emoji == "\U0001f431" else "#7EB3FF"

            header = tk.Label(
                card,
                text=f"{emoji} {p['pid'][:8]}...",
                font=("Helvetica Neue", 10, "bold"),
                fg=emoji_color,
                bg="#1a1a1a",
            )
            header.pack(anchor="w")

            self_resp = LIKERT_LABELS.get(p["self_response"], str(p["self_response"]))
            perceived = (
                LIKERT_LABELS.get(p["perceived_partner"], "N/A")
                if p["perceived_partner"]
                else "N/A"
            )
            commonality = "Yes" if p["commonality_judgment"] == 1 else "No"
            commonality_color = (
                "#2ecc71" if p["commonality_judgment"] == 1 else "#e74c3c"
            )

            tk.Label(
                card,
                text=f"Self: {self_resp} ({p['self_response']})",
                font=("Helvetica Neue", 9),
                fg="white",
                bg="#1a1a1a",
            ).pack(anchor="w")
            tk.Label(
                card,
                text=f"Perceived partner: {perceived} ({p['perceived_partner']})",
                font=("Helvetica Neue", 9),
                fg="white",
                bg="#1a1a1a",
            ).pack(anchor="w")

            commonality_lbl = tk.Label(
                card,
                text=f"Expected commonality: {commonality}",
                font=("Helvetica Neue", 9, "bold"),
                fg=commonality_color,
                bg="#1a1a1a",
            )
            commonality_lbl.pack(anchor="w")

        # Render messages
        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", "end")

        for _, m in group_msgs.iterrows():
            author = m["author"]
            tag = "cat" if author == "\U0001f431" else "dog"
            self.chat_text.insert("end", f"{author}  ", tag)
            self.chat_text.insert("end", f"{m['message_string']}\n", "msg")

        self.chat_text.config(state="disabled")

        # Update button states
        self.prev_btn.config(state="normal" if self.current_idx > 0 else "disabled")
        self.next_btn.config(
            state="normal" if self.current_idx < len(self.groups) - 1 else "disabled"
        )

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _next(self):
        if self.current_idx < len(self.groups) - 1:
            self.current_idx += 1
            self._show_current()


def main():
    parser = argparse.ArgumentParser(description="Chat transcript viewer")
    parser.add_argument(
        "--filter",
        choices=["all", "opposing", "opposing-commonality"],
        default="opposing-commonality",
        help="Filter dyads: 'opposing-commonality' (default) shows opposing-stance dyads "
        "where at least one participant expected focal commonality",
    )
    args = parser.parse_args()

    print("Loading data...")
    msgs, chat = load_data()

    print(f"Finding groups (filter={args.filter})...")
    groups = get_filtered_groups(msgs, chat, args.filter)
    print(f"Found {len(groups)} dyads matching filter")

    root = tk.Tk()
    ChatViewer(root, msgs, chat, groups)
    root.mainloop()


if __name__ == "__main__":
    main()

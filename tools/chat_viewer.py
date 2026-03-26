#!/usr/bin/env python3
"""
Chat transcript viewer for inspecting chat dyad conversations.

Usage:
    python tools/chat_viewer.py
"""

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

FONT_SM = ("Helvetica Neue", 13)
FONT_MD = ("Helvetica Neue", 14)
FONT_MD_BOLD = ("Helvetica Neue", 14, "bold")
FONT_LG = ("Helvetica Neue", 15)
FONT_LG_BOLD = ("Helvetica Neue", 15, "bold")
FONT_CHAT = ("Helvetica Neue", 15)
FONT_CHAT_BOLD = ("Helvetica Neue", 15, "bold")

BG = "#1a1a1a"
FG = "white"
DIM = "#999999"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load messages and response data, returning (messages, chat_responses)."""
    msgs = pd.read_csv(DATA_DIR / "messages.csv")
    resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    chat = resp[resp["experiment"] == "chat"].copy()
    return msgs, chat


def get_dyad_info(chat: pd.DataFrame, group_id: str) -> list[dict]:
    """Get per-participant info for a dyad, including partner's true response."""
    group = chat[chat["groupId"] == group_id]
    participants = []
    for pid in group["pid"].unique():
        p = group[group["pid"] == pid]
        focal = p[p["question_type"] == "observed"]
        if len(focal) == 0:
            continue
        focal = focal.iloc[0]

        perceived_partner = (
            int(focal["postChatResponse"])
            if pd.notna(focal["postChatResponse"])
            else None
        )
        partner_true = (
            int(focal["partner_response"])
            if pd.notna(focal["partner_response"])
            else None
        )

        # Prediction error: |perceived - true|
        prediction_error = None
        if perceived_partner is not None and partner_true is not None:
            prediction_error = abs(perceived_partner - partner_true)

        # Perceived distance: |perceived_partner - self|
        self_response = int(focal["preChatResponse"])
        perceived_distance = None
        if perceived_partner is not None:
            perceived_distance = abs(perceived_partner - self_response)

        participants.append(
            {
                "pid": pid,
                "focal_question": focal["matchedQuestion"],
                "focal_domain": focal["matchedDomain"],
                "self_response": self_response,
                "perceived_partner": perceived_partner,
                "partner_true": partner_true,
                "prediction_error": prediction_error,
                "perceived_distance": perceived_distance,
                "commonality_judgment": int(focal["participant_binary_prediction"]),
                "stance": focal["stance"],
            }
        )
    return participants


def load_llm_annotations() -> dict[str, dict]:
    """Load LLM common ground annotations, returning {group_id: annotation_dict}.

    Uses the aggregated per-bin commonality timecourse (modal vote across 3 runs).
    Builds a dyad-level summary: whether stance was revealed, whether post-stance
    shared reality was ever found, and the broadest topic scope reached.
    Returns empty dict if the annotation file doesn't exist.
    """
    agg_path = (
        DATA_DIR / "llm_results" / "perbin_commonality_timecourse_15s_aggregated.csv"
    )
    if not agg_path.exists():
        return {}

    df = pd.read_csv(agg_path)

    # Filter to post-stance bins only
    post = df[df["focal_stance_revealed"] == True]

    scope_priority = {
        "same_values": 4,
        "related_subtopic": 3,
        "different_topic": 2,
        "rapport_only": 1,
        "none": 0,
    }
    priority_to_scope = {v: k for k, v in scope_priority.items()}

    result: dict[str, dict] = {}
    for gid, group in df.groupby("group_id"):
        # Whether stance was ever revealed
        focal_revealed = bool(group["focal_stance_revealed"].any())

        # Post-stance shared reality (from post-stance bins only)
        post_group = post[post["group_id"] == gid]
        if len(post_group) > 0:
            ever_sr = bool(post_group["post_stance_shared_reality"].any())
            best_priority = (
                post_group["post_stance_topic_scope"]
                .map(scope_priority)
                .max()
            )
            best_scope = priority_to_scope.get(best_priority, "none")
        else:
            ever_sr = False
            best_scope = "none"

        # Initial alignment (from first bin where it's set)
        alignments = group["initial_alignment"].dropna()
        initial_alignment = alignments.iloc[0] if len(alignments) > 0 else None

        result[gid] = {
            "focal_stance_revealed": focal_revealed,
            "post_stance_cg": ever_sr,
            "cg_type": best_scope,
            "initial_alignment": initial_alignment,
        }

    return result


def build_dyad_metadata(
    msgs: pd.DataFrame, chat: pd.DataFrame
) -> list[dict]:
    """Build metadata for all dyads: stance, prediction errors, commonality."""
    all_groups = sorted(set(msgs["group_id"]) & set(chat["groupId"]))

    # Load LLM annotations if available
    llm = load_llm_annotations()

    dyads = []
    for gid in all_groups:
        info = get_dyad_info(chat, gid)
        if not info:
            continue

        errors = [p["prediction_error"] for p in info if p["prediction_error"] is not None]
        distances = [p["perceived_distance"] for p in info if p["perceived_distance"] is not None]
        commonality_count = sum(1 for p in info if p["commonality_judgment"] == 1)

        # LLM annotation fields
        annot = llm.get(gid, {})
        focal_stance_revealed = annot.get("focal_stance_revealed", None)
        post_stance_cg = annot.get("post_stance_cg", None)
        cg_type = annot.get("cg_type", None)
        initial_alignment = annot.get("initial_alignment", None)

        dyads.append(
            {
                "group_id": gid,
                "stance": info[0]["stance"],
                "info": info,
                "min_error": min(errors) if errors else None,
                "max_error": max(errors) if errors else None,
                "min_distance": min(distances) if distances else None,
                "max_distance": max(distances) if distances else None,
                "commonality_count": commonality_count,
                "n_participants": len(info),
                # LLM fields
                "focal_stance_revealed": focal_stance_revealed,
                "post_stance_cg": post_stance_cg,
                "cg_type": cg_type,
                "initial_alignment": initial_alignment,
            }
        )

    return dyads


class ChatViewer:
    def __init__(
        self,
        root: tk.Tk,
        msgs: pd.DataFrame,
        chat: pd.DataFrame,
        dyads: list[dict],
    ):
        self.root = root
        self.msgs = msgs
        self.chat = chat
        self.all_dyads = dyads
        self.filtered_groups: list[str] = [d["group_id"] for d in dyads]
        self.current_idx = 0

        # Filter state
        self.stance_var = tk.StringVar(value="all")
        self.commonality_var = tk.StringVar(value="any")
        self.max_error_var = tk.IntVar(value=4)  # max prediction error threshold
        self.error_filter_var = tk.StringVar(value="off")  # off / at_most / at_least
        self.distance_val_var = tk.IntVar(value=4)
        self.distance_filter_var = tk.StringVar(value="off")
        self.domain_var = tk.StringVar(value="All")
        self.question_var = tk.StringVar(value="All")
        self.stance_revealed_var = tk.StringVar(value="any")
        self.llm_cg_var = tk.StringVar(value="any")

        # Collect unique domains and questions
        self.all_domains = sorted({
            p["focal_domain"] for d in dyads for p in d["info"] if p["focal_domain"]
        })
        self.all_questions = sorted({
            p["focal_question"] for d in dyads for p in d["info"] if p["focal_question"]
        })

        self.root.title("Chat Transcript Viewer")
        self.root.geometry("950x1080")
        self.root.configure(bg=BG)

        # Dark theme style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=BG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=FG)
        style.configure("TLabelframe.Label", background=BG, foreground=FG, font=FONT_SM)
        style.configure("TButton", background="#333333", foreground=FG)
        style.configure(
            "Active.TButton",
            background="#555555",
            foreground=FG,
        )
        style.map(
            "Toggle.TButton",
            background=[("active", "#555555"), ("!active", "#333333")],
        )
        style.configure("Toggle.TButton", background="#333333", foreground=FG, font=FONT_SM)
        style.configure(
            "ToggleActive.TButton",
            background="#4a6fa5",
            foreground=FG,
            font=FONT_SM,
        )
        style.map(
            "ToggleActive.TButton",
            background=[("active", "#5a7fb5"), ("!active", "#4a6fa5")],
        )

        # Radiobutton styles
        style.configure(
            "Dark.TRadiobutton",
            background=BG,
            foreground=FG,
            font=FONT_SM,
            indicatorcolor="#333333",
        )
        style.map(
            "Dark.TRadiobutton",
            indicatorcolor=[("selected", "#4a6fa5")],
        )

        # Scale style
        style.configure(
            "Dark.Horizontal.TScale",
            background=BG,
            troughcolor="#333333",
        )

        # Combobox style
        style.configure(
            "Dark.TCombobox",
            fieldbackground="#333333",
            background="#333333",
            foreground=FG,
            font=FONT_SM,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#333333")],
            foreground=[("readonly", FG)],
        )

        self._build_ui()
        self._apply_filters()
        self._show_current()

    def _build_ui(self):
        # ── Filter panel ──
        filter_frame = ttk.LabelFrame(self.root, text="Filters", padding=8)
        filter_frame.pack(fill="x", padx=10, pady=(10, 2))

        # Row 1: Stance toggle
        row1 = ttk.Frame(filter_frame)
        row1.pack(fill="x", pady=(0, 4))

        ttk.Label(row1, text="Stance:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.stance_buttons: dict[str, ttk.Button] = {}
        for stance in ["all", "shared", "opposing"]:
            btn = ttk.Button(
                row1,
                text=stance.title(),
                command=lambda s=stance: self._set_toggle("stance", s),
                style="ToggleActive.TButton" if stance == "all" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.stance_buttons[stance] = btn

        # Row 2: Commonality toggle
        row2 = ttk.Frame(filter_frame)
        row2.pack(fill="x", pady=4)

        ttk.Label(row2, text="Focal commonality:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.commonality_buttons: dict[str, ttk.Button] = {}
        for mode, label in [
            ("any", "Any"),
            ("at_least_one", "1+ expects"),
            ("both", "Both expect"),
            ("neither", "Neither expects"),
        ]:
            btn = ttk.Button(
                row2,
                text=label,
                command=lambda m=mode: self._set_toggle("commonality", m),
                style="ToggleActive.TButton" if mode == "any" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.commonality_buttons[mode] = btn

        # Row 3: Prediction error filter
        row3 = ttk.Frame(filter_frame)
        row3.pack(fill="x", pady=4)

        ttk.Label(row3, text="Prediction error:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.error_buttons: dict[str, ttk.Button] = {}
        for mode, label in [
            ("off", "Off"),
            ("at_most", "At most"),
            ("at_least", "At least"),
        ]:
            btn = ttk.Button(
                row3,
                text=label,
                command=lambda m=mode: self._set_toggle("error_filter", m),
                style="ToggleActive.TButton" if mode == "off" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.error_buttons[mode] = btn

        # Error threshold scale
        self.error_val_label = ttk.Label(row3, text="0", font=FONT_MD_BOLD)

        self.error_scale = tk.Scale(
            row3,
            from_=0,
            to=4,
            orient="horizontal",
            bg=BG,
            fg=FG,
            highlightbackground=BG,
            troughcolor="#333333",
            font=FONT_SM,
            length=150,
            showvalue=False,
            command=self._on_error_scale,
        )
        self.error_scale.set(0)
        self.error_scale.pack(side="left", padx=(10, 4))
        self.error_val_label.pack(side="left")

        # Row 4: Perceived distance filter
        row4 = ttk.Frame(filter_frame)
        row4.pack(fill="x", pady=4)

        ttk.Label(row4, text="Perceived distance:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.distance_buttons: dict[str, ttk.Button] = {}
        for mode, label in [
            ("off", "Off"),
            ("at_most", "At most"),
            ("at_least", "At least"),
        ]:
            btn = ttk.Button(
                row4,
                text=label,
                command=lambda m=mode: self._set_toggle("distance_filter", m),
                style="ToggleActive.TButton" if mode == "off" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.distance_buttons[mode] = btn

        self.distance_val_label = ttk.Label(row4, text="0", font=FONT_MD_BOLD)

        self.distance_scale = tk.Scale(
            row4,
            from_=0,
            to=4,
            orient="horizontal",
            bg=BG,
            fg=FG,
            highlightbackground=BG,
            troughcolor="#333333",
            font=FONT_SM,
            length=150,
            showvalue=False,
            command=self._on_distance_scale,
        )
        self.distance_scale.set(0)
        self.distance_scale.pack(side="left", padx=(10, 4))
        self.distance_val_label.pack(side="left")

        # Row 5: Stance revealed filter (LLM)
        row5 = ttk.Frame(filter_frame)
        row5.pack(fill="x", pady=4)

        ttk.Label(row5, text="Stance revealed:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.revealed_buttons: dict[str, ttk.Button] = {}
        for mode, label in [
            ("any", "Any"),
            ("yes", "Yes"),
            ("no", "Never revealed"),
        ]:
            btn = ttk.Button(
                row5,
                text=label,
                command=lambda m=mode: self._set_toggle("stance_revealed", m),
                style="ToggleActive.TButton" if mode == "any" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.revealed_buttons[mode] = btn

        # Row 6: LLM common ground filter
        row6 = ttk.Frame(filter_frame)
        row6.pack(fill="x", pady=4)

        ttk.Label(row6, text="LLM common ground:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 8)
        )

        self.llm_cg_buttons: dict[str, ttk.Button] = {}
        for mode, label in [
            ("any", "Any"),
            ("yes", "Found CG"),
            ("no", "No CG"),
        ]:
            btn = ttk.Button(
                row6,
                text=label,
                command=lambda m=mode: self._set_toggle("llm_cg", m),
                style="ToggleActive.TButton" if mode == "any" else "Toggle.TButton",
            )
            btn.pack(side="left", padx=2)
            self.llm_cg_buttons[mode] = btn

        # Row 7: Domain and question dropdowns
        row7 = ttk.Frame(filter_frame)
        row7.pack(fill="x", pady=4)

        ttk.Label(row7, text="Domain:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 4)
        )

        self.domain_combo = ttk.Combobox(
            row7,
            textvariable=self.domain_var,
            values=["All"] + self.all_domains,
            state="readonly",
            style="Dark.TCombobox",
            width=12,
        )
        self.domain_combo.pack(side="left", padx=(0, 16))
        self.domain_combo.bind("<<ComboboxSelected>>", self._on_dropdown_change)

        ttk.Label(row7, text="Question:", font=FONT_MD_BOLD).pack(
            side="left", padx=(0, 4)
        )

        self.question_combo = ttk.Combobox(
            row7,
            textvariable=self.question_var,
            values=["All"] + self.all_questions,
            state="readonly",
            style="Dark.TCombobox",
            width=50,
        )
        self.question_combo.pack(side="left", fill="x", expand=True)
        self.question_combo.bind("<<ComboboxSelected>>", self._on_dropdown_change)

        # Match count
        self.match_label = ttk.Label(
            filter_frame, text="", font=FONT_SM, foreground=DIM
        )
        self.match_label.pack(anchor="w", pady=(4, 0))

        # ── Nav bar ──
        nav = ttk.Frame(self.root)
        nav.pack(fill="x", padx=10, pady=(5, 5))

        self.prev_btn = ttk.Button(nav, text="< Prev", command=self._prev)
        self.prev_btn.pack(side="left")

        self.counter_label = ttk.Label(nav, text="", font=FONT_MD)
        self.counter_label.pack(side="left", padx=20)

        self.next_btn = ttk.Button(nav, text="Next >", command=self._next)
        self.next_btn.pack(side="left")

        self.group_label = ttk.Label(
            nav, text="", font=("Helvetica Neue", 12, "italic")
        )
        self.group_label.pack(side="right")

        # ── Info panel ──
        info_frame = ttk.LabelFrame(self.root, text="Dyad Info", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        self.focal_label = ttk.Label(
            info_frame, text="", font=FONT_LG_BOLD, wraplength=900
        )
        self.focal_label.pack(anchor="w")

        self.domain_label = ttk.Label(info_frame, text="", font=FONT_MD)
        self.domain_label.pack(anchor="w")

        self.participant_frame = ttk.Frame(info_frame)
        self.participant_frame.pack(fill="x", pady=(5, 0))

        # ── Chat transcript ──
        chat_frame = ttk.LabelFrame(self.root, text="Conversation", padding=10)
        chat_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            font=FONT_CHAT,
            bg=BG,
            fg=FG,
            relief="flat",
            padx=10,
            pady=10,
            spacing3=6,
            insertbackground=FG,
        )
        self.chat_text.pack(fill="both", expand=True, side="left")

        self.chat_text.tag_configure("cat", foreground="#FF6B9D", font=FONT_CHAT_BOLD)
        self.chat_text.tag_configure("dog", foreground="#7EB3FF", font=FONT_CHAT_BOLD)
        self.chat_text.tag_configure("msg", foreground=FG, font=FONT_CHAT)

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.chat_text.configure(yscrollcommand=scrollbar.set)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())

    def _set_toggle(self, group: str, value: str):
        """Handle toggle button click for any filter group."""
        if group == "stance":
            self.stance_var.set(value)
            buttons = self.stance_buttons
        elif group == "commonality":
            self.commonality_var.set(value)
            buttons = self.commonality_buttons
        elif group == "error_filter":
            self.error_filter_var.set(value)
            buttons = self.error_buttons
        elif group == "distance_filter":
            self.distance_filter_var.set(value)
            buttons = self.distance_buttons
        elif group == "stance_revealed":
            self.stance_revealed_var.set(value)
            buttons = self.revealed_buttons
        elif group == "llm_cg":
            self.llm_cg_var.set(value)
            buttons = self.llm_cg_buttons
        else:
            return

        for k, btn in buttons.items():
            btn.configure(
                style="ToggleActive.TButton" if k == value else "Toggle.TButton"
            )

        self._apply_filters()
        self._show_current()

    def _on_error_scale(self, val: str):
        """Handle error threshold scale change."""
        self.max_error_var.set(int(float(val)))
        self.error_val_label.config(text=str(int(float(val))))
        if self.error_filter_var.get() != "off":
            self._apply_filters()
            self._show_current()

    def _on_distance_scale(self, val: str):
        """Handle distance threshold scale change."""
        self.distance_val_var.set(int(float(val)))
        self.distance_val_label.config(text=str(int(float(val))))
        if self.distance_filter_var.get() != "off":
            self._apply_filters()
            self._show_current()

    def _on_dropdown_change(self, event=None):
        """Handle domain/question dropdown change."""
        # When domain changes, update question dropdown to show only matching questions
        domain = self.domain_var.get()
        if domain == "All":
            self.question_combo["values"] = ["All"] + self.all_questions
        else:
            domain_questions = sorted({
                p["focal_question"]
                for d in self.all_dyads
                for p in d["info"]
                if p["focal_domain"] == domain and p["focal_question"]
            })
            self.question_combo["values"] = ["All"] + domain_questions
            # Reset question if current selection not in new domain
            if self.question_var.get() not in (["All"] + domain_questions):
                self.question_var.set("All")

        self._apply_filters()
        self._show_current()

    def _apply_filters(self):
        """Apply all filters and rebuild the filtered group list."""
        stance = self.stance_var.get()
        commonality = self.commonality_var.get()
        error_mode = self.error_filter_var.get()
        error_thresh = self.max_error_var.get()
        dist_mode = self.distance_filter_var.get()
        dist_thresh = self.distance_val_var.get()
        domain = self.domain_var.get()
        question = self.question_var.get()
        revealed_mode = self.stance_revealed_var.get()
        llm_cg_mode = self.llm_cg_var.get()

        filtered = []
        for d in self.all_dyads:
            # Stance filter
            if stance != "all" and d["stance"] != stance:
                continue

            # Domain filter
            if domain != "All":
                if not any(p["focal_domain"] == domain for p in d["info"]):
                    continue

            # Question filter
            if question != "All":
                if not any(p["focal_question"] == question for p in d["info"]):
                    continue

            # Commonality filter
            if commonality == "at_least_one" and d["commonality_count"] < 1:
                continue
            if commonality == "both" and d["commonality_count"] < 2:
                continue
            if commonality == "neither" and d["commonality_count"] != 0:
                continue

            # Prediction error filter (at least one participant matches)
            if error_mode != "off":
                errors = [
                    p["prediction_error"]
                    for p in d["info"]
                    if p["prediction_error"] is not None
                ]
                if not errors:
                    continue
                if error_mode == "at_most" and not any(e <= error_thresh for e in errors):
                    continue
                if error_mode == "at_least" and not any(e >= error_thresh for e in errors):
                    continue

            # Perceived distance filter (at least one participant matches)
            if dist_mode != "off":
                distances = [
                    p["perceived_distance"]
                    for p in d["info"]
                    if p["perceived_distance"] is not None
                ]
                if not distances:
                    continue
                if dist_mode == "at_most" and not any(x <= dist_thresh for x in distances):
                    continue
                if dist_mode == "at_least" and not any(x >= dist_thresh for x in distances):
                    continue

            # Stance revealed filter (LLM annotation)
            if revealed_mode != "any" and d["focal_stance_revealed"] is not None:
                if revealed_mode == "yes" and not d["focal_stance_revealed"]:
                    continue
                if revealed_mode == "no" and d["focal_stance_revealed"]:
                    continue

            # LLM common ground filter
            if llm_cg_mode != "any" and d["post_stance_cg"] is not None:
                if llm_cg_mode == "yes" and not d["post_stance_cg"]:
                    continue
                if llm_cg_mode == "no" and d["post_stance_cg"]:
                    continue

            filtered.append(d["group_id"])

        self.filtered_groups = filtered
        self.current_idx = 0
        self.match_label.config(
            text=f"{len(filtered)} dyads match filters"
        )

    def _show_current(self):
        if not self.filtered_groups:
            self.counter_label.config(text="0 / 0")
            self.group_label.config(text="")
            self.focal_label.config(text="No dyads match these filters")
            self.domain_label.config(text="")
            for widget in self.participant_frame.winfo_children():
                widget.destroy()
            self.chat_text.config(state="normal")
            self.chat_text.delete("1.0", "end")
            self.chat_text.config(state="disabled")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            return

        gid = self.filtered_groups[self.current_idx]
        self.counter_label.config(
            text=f"{self.current_idx + 1} / {len(self.filtered_groups)}"
        )
        self.group_label.config(text=gid)

        # Get participant info
        info = get_dyad_info(self.chat, gid)

        # Focal question and LLM annotation
        dyad_data = next((d for d in self.all_dyads if d["group_id"] == gid), None)
        if info:
            self.focal_label.config(text=f"Focal: {info[0]['focal_question']}")
            stance_text = f"Domain: {info[0]['focal_domain']}  |  Stance: {info[0]['stance']}"
            if dyad_data:
                rev = dyad_data.get("focal_stance_revealed")
                cg = dyad_data.get("post_stance_cg")
                cg_type = dyad_data.get("cg_type")
                align = dyad_data.get("initial_alignment")
                llm_parts = []
                if rev is not None:
                    llm_parts.append(f"Revealed: {'Yes' if rev else 'No'}")
                if align:
                    llm_parts.append(f"Alignment: {align}")
                if cg is not None:
                    cg_str = f"CG: {'Yes' if cg else 'No'}"
                    if cg and cg_type and cg_type != "none":
                        cg_str += f" ({cg_type.replace('_', ' ')})"
                    llm_parts.append(cg_str)
                if llm_parts:
                    stance_text += "  |  LLM: " + ", ".join(llm_parts)
            self.domain_label.config(text=stance_text)

        # Build author → pid map from messages
        group_msgs = self.msgs[self.msgs["group_id"] == gid].sort_values(
            "absolute_timestamp"
        )
        author_pid = group_msgs.groupby("author")["prolific_id"].first().to_dict()

        # Clear and rebuild participant info cards
        for widget in self.participant_frame.winfo_children():
            widget.destroy()

        for p in info:
            emoji = next((e for e, pid in author_pid.items() if pid == p["pid"]), "?")
            emoji_color = "#FF6B9D" if emoji == "\U0001f431" else "#7EB3FF"

            self_resp = LIKERT_LABELS.get(p["self_response"], str(p["self_response"]))
            perceived = (
                LIKERT_LABELS.get(p["perceived_partner"], "N/A")
                if p["perceived_partner"]
                else "N/A"
            )
            partner_true = (
                LIKERT_LABELS.get(p["partner_true"], "N/A")
                if p["partner_true"]
                else "N/A"
            )
            commonality = "Yes" if p["commonality_judgment"] == 1 else "No"
            commonality_color = (
                "#2ecc71" if p["commonality_judgment"] == 1 else "#e74c3c"
            )

            err_text = (
                f"Prediction error: {p['prediction_error']}"
                if p["prediction_error"] is not None
                else "Prediction error: N/A"
            )
            err_color = (
                "#e74c3c"
                if p["prediction_error"] is not None and p["prediction_error"] >= 2
                else "#f39c12"
                if p["prediction_error"] is not None and p["prediction_error"] == 1
                else "#2ecc71"
                if p["prediction_error"] is not None
                else DIM
            )

            pdist = p["perceived_distance"]
            pdist_text = (
                f"Perceived distance: {pdist}"
                if pdist is not None
                else "Perceived distance: N/A"
            )

            # Build card as a single selectable Text widget
            card = tk.Text(
                self.participant_frame,
                wrap="word",
                font=FONT_SM,
                bg=BG,
                fg=FG,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                padx=4,
                pady=2,
                height=7,
                width=40,
                cursor="arrow",
            )
            card.pack(side="left", fill="x", expand=True, padx=(0, 10))

            # Configure tags for colored text
            card.tag_configure("header", foreground=emoji_color, font=FONT_MD_BOLD)
            card.tag_configure("dim", foreground=DIM, font=FONT_SM)
            card.tag_configure("err", foreground=err_color, font=FONT_SM)
            card.tag_configure("commonality", foreground=commonality_color, font=("Helvetica Neue", 13, "bold"))

            card.insert("end", f"{emoji} {p['pid']}\n", "header")
            card.insert("end", f"Self: {self_resp} ({p['self_response']})\n")
            card.insert("end", f"Partner true: {partner_true} ({p['partner_true']})\n", "dim")
            card.insert("end", f"Perceived partner: {perceived} ({p['perceived_partner']})\n")
            card.insert("end", f"{err_text}\n", "err")
            card.insert("end", f"{pdist_text}\n")
            card.insert("end", f"Expected commonality: {commonality}", "commonality")

            card.config(state="disabled")

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
            state=(
                "normal"
                if self.current_idx < len(self.filtered_groups) - 1
                else "disabled"
            )
        )

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _next(self):
        if self.current_idx < len(self.filtered_groups) - 1:
            self.current_idx += 1
            self._show_current()


def main():
    print("Loading data...")
    msgs, chat = load_data()

    print("Building dyad metadata...")
    dyads = build_dyad_metadata(msgs, chat)
    print(f"  {len(dyads)} total dyads")

    root = tk.Tk()
    ChatViewer(root, msgs, chat, dyads)
    root.mainloop()


if __name__ == "__main__":
    main()

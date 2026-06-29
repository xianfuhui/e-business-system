from flask import Flask, request, jsonify
from pyngrok import ngrok
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import base64
from io import BytesIO

import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.visualization.bpmn import visualizer as bpmn_vis

import numpy as np
import base64
import textwrap
from collections import Counter
import matplotlib.pyplot as plt
import polars as pl

from google import genai
from google.genai import types

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Only run this block for Gemini Developer API
client = genai.Client(api_key='AIzaSyD0ZHsh3xKrdpxLGW6O_cO4YofHk2CJzFQ')

def llm_analyze(data: dict):
#     prompt = f"""
# You are a process mining expert.

# Analyze this data:
# {data}

# Return:
# - key insights
# - bottlenecks reason
# - improvement suggestions
# """
    prompt = f"""
Analyze this data:
{data}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        error_msg = str(e)

        import re
        match = re.search(r"'message': '([^']+)'", error_msg)

        if match:
            return match.group(1)

        return error_msg

def build_sequences(df):
    sequences = []
    for _, group in df.groupby(CASE_COL):
        seq = group.sort_values(TIME_COL)[ACTIVITY_COL].tolist()
        sequences.append(seq)
    return sequences


def create_samples(sequences):
    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    return X, y


def encode_data(X, y):
    global le

    le = LabelEncoder()
    all_tokens = [a for seq in X for a in seq] + y
    le.fit(all_tokens)

    X_enc = [le.transform(seq) for seq in X]
    y_enc = le.transform(y)

    X_pad = pad_sequences(X_enc, padding="pre")

    return X_pad, y_enc, len(le.classes_)


def build_model(vocab_size, max_len):
    inputs = layers.Input(shape=(max_len,))

    x = layers.Embedding(vocab_size, 64, mask_zero=True)(inputs)

    pos = tf.range(start=0, limit=max_len, delta=1)
    pos_embed = layers.Embedding(max_len, 64)(pos)
    x = x + pos_embed

    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

CASE_COL = "user_session"
ACTIVITY_COL = "event_type"
TIME_COL = "event_time"

def build_ecommerce_dashboard(df):

    # ========================================================
    # IMPORTS
    # ========================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ========================================================
    # GLOBAL PLOT STYLE
    # ========================================================
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(style="whitegrid")

    TITLE_SIZE = 15
    LABEL_SIZE = 12

    COLOR_SESSION = "#3498db"
    COLOR_PURCHASE = "#e67e22"
    COLOR_SUCCESS = "#27ae60"
    COLOR_DANGER = "#e74c3c"
    COLOR_DARK = "#34495e"
    COLOR_PURPLE = "#8e44ad"
    COLOR_INFO = "#16a085"

    # ========================================================
    # PREPROCESSING
    # ========================================================
    df = df.copy()

    df["event_time"] = pd.to_datetime(df["event_time"])
    df["hour"] = df["event_time"].dt.hour

    df["brand"] = df["brand"].fillna("Unknown")
    df["category_code"] = df["category_code"].fillna("Unknown")

    price_bins = [
        0,
        50,
        100,
        200,
        500,
        1000,
        np.inf
    ]

    price_labels = [
        "0-50",
        "50-100",
        "100-200",
        "200-500",
        "500-1000",
        "1000+"
    ]

    df["price_bin"] = pd.cut(
        df["price"],
        bins=price_bins,
        labels=price_labels
    )

    purchase_df = (
        df[
            df["event_type"] == "purchase"
        ]
        .copy()
    )

    images = {}

    # ========================================================
    # KPI
    # ========================================================
    rows_count = len(df)

    total_sessions = (
        df["user_session"]
        .nunique()
    )

    total_users = (
        df["user_id"]
        .nunique()
    )

    purchase_sessions = (
        purchase_df["user_session"]
        .nunique()
    )

    total_categories = (
        df["category_code"]
        .nunique()
    )

    total_brands = (
        df["brand"]
        .nunique()
    )

    total_event_types = (
        df["event_type"]
        .nunique()
    )

    total_products = (
        df["product_id"]
        .nunique()
    )

    # ========================================================
    # TRAFFIC & PURCHASE ANALYSIS
    # ========================================================
    BIN_SIZE = 50
    MAX_PRICE = 2000

    # ========================================================
    # SESSION DISTRIBUTION BY PRICE
    # ========================================================

    session_price = df.copy()

    session_price["price_bin"] = (
        np.floor(
            session_price["price"] / BIN_SIZE
        ).astype(int)
        * BIN_SIZE
    )

    session_price = session_price[
        session_price["price_bin"] <= MAX_PRICE
    ]

    session_price["price_label"] = (
        "$"
        + session_price["price_bin"].astype(str)
        + "-$"
        + (
            session_price["price_bin"]
            + BIN_SIZE
            - 1
        ).astype(str)
    )

    session_price = (
        session_price
        .groupby(
            ["price_bin", "price_label"]
        )["user_session"]
        .nunique()
        .reset_index(name="session_count")
        .sort_values("price_bin")
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    sns.barplot(
        data=session_price,
        x="price_label",
        y="session_count",
        color=COLOR_SESSION,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    for p in ax.patches:
        h = p.get_height()

        if h > 0:
            ax.annotate(
                f"{int(h):,}",
                (
                    p.get_x() + p.get_width()/2,
                    h
                ),
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_title(
        "Session Distribution by Product Price",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Product Price Range (USD)"
    )

    ax.set_ylabel(
        "Unique Sessions"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    # Nếu nhiều cột quá thì ẩn bớt nhãn
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)

    plt.tight_layout()

    images["session_price_dist"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # SESSION HEATMAP
    # ========================================================
    heatmap_df = (
        df.assign(
            price_bin=(
                np.floor(df["price"] / BIN_SIZE)
                * BIN_SIZE
            )
        )
    )

    heatmap_df = heatmap_df[
        heatmap_df["price_bin"]
        <= MAX_PRICE
    ]

    heatmap_df = (
        heatmap_df
        .groupby(
            ["price_bin", "hour"]
        )["user_session"]
        .nunique()
        .reset_index(
            name="session_count"
        )
    )

    pivot_heatmap = (
        heatmap_df
        .pivot(
            index="price_bin",
            columns="hour",
            values="session_count"
        )
        .fillna(0)
        .sort_index(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )

    sns.heatmap(
        pivot_heatmap,
        cmap="Blues",
        linewidths=.1,
        linecolor="white",
        cbar_kws={
            "label":
            "Unique Sessions"
        },
        ax=ax
    )

    ax.set_title(
        "Session Heatmap by Product Price and Hour",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Hour of Day",
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        "Product Price (USD)",
        fontsize=LABEL_SIZE
    )

    plt.tight_layout()

    images["session_heatmap"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # PURCHASE DISTRIBUTION BY PRICE
    # ========================================================

    purchase_price = purchase_df.copy()

    purchase_price["price_bin"] = (
        np.floor(
            purchase_price["price"] / BIN_SIZE
        ).astype(int)
        * BIN_SIZE
    )

    purchase_price = purchase_price[
        purchase_price["price_bin"] <= MAX_PRICE
    ]

    purchase_price["price_label"] = (
        "$"
        + purchase_price["price_bin"].astype(str)
        + "-$"
        + (
            purchase_price["price_bin"]
            + BIN_SIZE
            - 1
        ).astype(str)
    )

    purchase_price = (
        purchase_price
        .groupby(
            ["price_bin", "price_label"]
        )["user_session"]
        .nunique()
        .reset_index(name="session_count")
        .sort_values("price_bin")
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    sns.barplot(
        data=purchase_price,
        x="price_label",
        y="session_count",
        color=COLOR_PURCHASE,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    for p in ax.patches:
        h = p.get_height()

        if h > 0:
            ax.annotate(
                f"{int(h):,}",
                (
                    p.get_x() + p.get_width()/2,
                    h
                ),
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_title(
        "Purchase Session Distribution by Product Price",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Product Price Range (USD)"
    )

    ax.set_ylabel(
        "Unique Purchase Sessions"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)

    plt.tight_layout()

    images["purchase_price_dist"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # PURCHASE HEATMAP
    # ========================================================
    purchase_heatmap_df = (
        purchase_df.assign(
            price_bin=(
                np.floor(
                    purchase_df["price"] / BIN_SIZE
                ) * BIN_SIZE
            )
        )
    )

    purchase_heatmap_df = (
        purchase_heatmap_df[
            purchase_heatmap_df["price_bin"]
            <= MAX_PRICE
        ]
    )

    purchase_heatmap_df = (
        purchase_heatmap_df
        .groupby(
            ["price_bin", "hour"]
        )["user_session"]
        .nunique()
        .reset_index(
            name="session_count"
        )
    )

    purchase_heatmap = (
        purchase_heatmap_df
        .pivot(
            index="price_bin",
            columns="hour",
            values="session_count"
        )
        .fillna(0)
        .sort_index(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )

    sns.heatmap(
        purchase_heatmap,
        cmap="OrRd",
        linewidths=.1,
        linecolor="white",
        cbar_kws={
            "label":
            "Purchase Sessions"
        },
        ax=ax
    )

    ax.set_title(
        "Purchase Heatmap by Product Price and Hour",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Hour of Day"
    )

    ax.set_ylabel(
        "Product Price (USD)"
    )

    plt.tight_layout()

    images["purchase_heatmap"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # BRAND ANALYSIS
    # ========================================================
    brand_sessions = (
        df.groupby("brand")
        ["user_session"]
        .nunique()
    )

    brand_purchase = (
        purchase_df.groupby("brand")
        ["user_session"]
        .nunique()
    )

    brand_compare = pd.concat(
        [
            brand_sessions.rename(
                "Sessions"
            ),
            brand_purchase.rename(
                "Purchase"
            )
        ],
        axis=1
    ).fillna(0)

    brand_compare = (
        brand_compare
        .sort_values(
            "Sessions",
            ascending=False
        )
        .head(15)
    )

    brand_compare["CR_%"] = (
        brand_compare["Purchase"]
        /
        brand_compare["Sessions"]
        * 100
    ).fillna(0)

    df_melted = (
        brand_compare
        .reset_index()
        .melt(
            id_vars=[
                "brand",
                "CR_%"
            ],
            value_vars=[
                "Sessions",
                "Purchase"
            ],
            var_name="Type",
            value_name="Count"
        )
    )

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    sns.barplot(
        data=df_melted,
        x="brand",
        y="Count",
        hue="Type",
        palette=[
            COLOR_DARK,
            COLOR_DANGER
        ],
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    brand_count = len(
        brand_compare
    )

    for i in range(
        brand_count
    ):
        purchase_bar = (
            ax.patches[
                brand_count + i
            ]
        )

        cr = (
            brand_compare
            .iloc[i]["CR_%"]
        )

        height = (
            purchase_bar
            .get_height()
        )

        if height > 0:
            ax.annotate(
                f"{cr:.1f}%",
                (
                    purchase_bar.get_x()
                    +
                    purchase_bar.get_width()/2,
                    height
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )

    ax.set_title(
        "Top 15 Brands: Sessions vs Purchase Sessions",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Brand"
    )

    ax.set_ylabel(
        "Number of Sessions"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    images["brand_compare"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # CUSTOMER JOURNEY
    # ========================================================
    session_category = (
        df.groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    purchase_category = (
        purchase_df.groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    # ========================================================
    # PURCHASE RATE
    # ========================================================
    purchase_rate_df = pd.concat(
        [
            session_category.rename(
                "total_sessions"
            ),
            purchase_category.rename(
                "purchase_sessions"
            )
        ],
        axis=1
    ).fillna(0)

    purchase_rate_df[
        "non_purchase_sessions"
    ] = (
        purchase_rate_df[
            "total_sessions"
        ]
        -
        purchase_rate_df[
            "purchase_sessions"
        ]
    )

    purchase_rate_df[
        "purchase_rate"
    ] = (
        purchase_rate_df[
            "purchase_sessions"
        ]
        /
        purchase_rate_df[
            "total_sessions"
        ]
        * 100
    ).fillna(0)

    purchase_rate = (
        purchase_rate_df[
            "purchase_rate"
        ]
    )

    top_purchase = (
        purchase_rate_df
        .sort_values(
            "total_sessions",
            ascending=False
        )
        .head(20)
    )

    fig, ax1 = plt.subplots(
        figsize=(15, 7)
    )

    x = np.arange(
        len(top_purchase)
    )

    ax1.bar(
        x,
        top_purchase[
            "non_purchase_sessions"
        ],
        color=COLOR_DARK,
        label="Non-Purchase"
    )

    ax1.bar(
        x,
        top_purchase[
            "purchase_sessions"
        ],
        bottom=top_purchase[
            "non_purchase_sessions"
        ],
        color=COLOR_SUCCESS,
        label="Purchase"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        top_purchase[
            "purchase_rate"
        ],
        color=COLOR_DANGER,
        marker="o",
        linewidth=3
    )

    for i, v in enumerate(
        top_purchase[
            "purchase_rate"
        ]
    ):
        ax2.annotate(
            f"{v:.1f}%",
            (i, v),
            ha="center",
            fontsize=8
        )

    ax1.set_xticks(x)

    ax1.set_xticklabels(
        top_purchase.index,
        rotation=45,
        ha="right"
    )

    ax1.set_title(
        "Purchase Rate by Category",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax1.set_ylabel(
        "Number of Sessions"
    )

    ax2.set_ylabel(
        "Purchase Rate (%)"
    )

    ax1.legend()

    plt.tight_layout()

    images["purchase_rate"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # ABANDONMENT RATE
    # ========================================================
    cart_sessions = (
        df[
            df["event_type"] == "cart"
        ]
        .groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    purchase_sessions_cat = (
        purchase_df
        .groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    abandon_df = pd.concat(
        [
            cart_sessions.rename(
                "cart_sessions"
            ),
            purchase_sessions_cat.rename(
                "purchase_sessions"
            )
        ],
        axis=1
    ).fillna(0)

    abandon_df[
        "abandoned_sessions"
    ] = (
        abandon_df[
            "cart_sessions"
        ]
        -
        abandon_df[
            "purchase_sessions"
        ]
    )

    abandon_df[
        "abandonment_rate"
    ] = (
        abandon_df[
            "abandoned_sessions"
        ]
        /
        abandon_df[
            "cart_sessions"
        ]
        * 100
    ).fillna(0)

    abandon_rate = (
        abandon_df[
            "abandonment_rate"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    (
        abandon_df[
            "abandonment_rate"
        ]
        .sort_values(
            ascending=False
        )
        .head(20)
        .plot(
            kind="bar",
            color=COLOR_DANGER,
            edgecolor="black",
            ax=ax
        )
    )

    ax.set_title(
        "Cart Abandonment Rate by Category",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Abandonment Rate (%)"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    images["abandonment_rate"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)
    # ========================================================
    # WINDOW SHOPPER RATE
    # ========================================================
    session_events = (
        df.groupby(
            "user_session"
        )["event_type"]
        .apply(set)
    )

    window_sessions = (
        session_events[
            session_events.apply(
                lambda x: x == {"view"}
            )
        ]
        .index
    )

    window_rate = (
        len(window_sessions)
        /
        total_sessions
        * 100
    )

    window_df = (
        df[
            df["user_session"]
            .isin(window_sessions)
        ]
        .groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    view_df = (
        df[
            df["event_type"]
            == "view"
        ]
        .groupby(
            "category_code"
        )["user_session"]
        .nunique()
    )

    window_rate_df = pd.concat(
        [
            view_df.rename(
                "view_sessions"
            ),
            window_df.rename(
                "window_sessions"
            )
        ],
        axis=1
    ).fillna(0)

    window_rate_df[
        "window_rate"
    ] = (
        window_rate_df[
            "window_sessions"
        ]
        /
        window_rate_df[
            "view_sessions"
        ]
        * 100
    ).fillna(0)

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    (
        window_rate_df[
            "window_rate"
        ]
        .sort_values(
            ascending=False
        )
        .head(20)
        .plot(
            kind="bar",
            color=COLOR_PURPLE,
            edgecolor="black",
            ax=ax
        )
    )

    ax.set_title(
        "Window Shopper Rate by Category",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Window Shopper Rate (%)"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    images["window_shopper"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # REMOVE CART THEN PURCHASE
    # ========================================================
    df_sorted = (
        df.sort_values(
            [
                "user_session",
                "event_time"
            ]
        )
    )

    remove_purchase_sessions = []

    for session_id, group in (
        df_sorted.groupby(
            "user_session"
        )
    ):

        remove_rows = (
            group[
                group["event_type"]
                ==
                "remove_from_cart"
            ]
        )

        purchase_rows = (
            group[
                group["event_type"]
                ==
                "purchase"
            ]
        )

        if (
            len(remove_rows) == 0
            or
            len(purchase_rows) == 0
        ):
            continue

        if (
            purchase_rows[
                "event_time"
            ].min()
            >
            remove_rows[
                "event_time"
            ].min()
        ):
            remove_purchase_sessions.append(
                group[
                    "event_time"
                ]
                .min()
                .date()
            )

    remove_purchase_daily = (
        pd.Series(
            remove_purchase_sessions
        )
        .value_counts()
        .sort_index()
    )

    remove_purchase_rate = (
        len(remove_purchase_sessions)
        /
        max(
            1,
            (
                df["event_type"]
                ==
                "remove_from_cart"
            ).sum()
        )
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    if len(remove_purchase_daily) > 0:
        remove_purchase_daily.plot(
            ax=ax,
            marker="o",
            linewidth=3,
            color=COLOR_INFO
        )

    ax.set_title(
        "Sessions That Removed Items but Still Purchased",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Number of Sessions"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    images["remove_purchase"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # RETURNING / LOYAL USERS
    # ========================================================
    session_start = (
        df.groupby(
            [
                "user_id",
                "user_session"
            ]
        )["event_time"]
        .min()
        .reset_index()
    )

    session_start = (
        session_start
        .sort_values(
            [
                "user_id",
                "event_time"
            ]
        )
    )

    session_start[
        "session_order"
    ] = (
        session_start
        .groupby("user_id")
        .cumcount()
        + 1
    )

    session_start[
        "is_returning"
    ] = (
        session_start[
            "session_order"
        ] > 1
    )

    daily_returning = (
        session_start
        .groupby(
            session_start[
                "event_time"
            ].dt.date
        )[
            "is_returning"
        ]
        .mean()
        * 100
    )

    user_sessions = (
        df.groupby(
            "user_id"
        )["user_session"]
        .nunique()
    )

    returning_rate = (
        (user_sessions > 1)
        .mean()
        * 100
    )

    loyal_rate = (
        (user_sessions > 3)
        .mean()
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    daily_returning.plot(
        ax=ax,
        marker="o",
        linewidth=3,
        color=COLOR_SUCCESS
    )

    ax.set_title(
        "Daily Returning Session Rate",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Returning Rate (%)"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    images["loyal_returning"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # SESSION DISTRIBUTION BY HOUR AND PRICE
    # ========================================================
    session_hour_price = (
        df.groupby(
            [
                "hour",
                "price_bin"
            ],
            observed=False
        )["user_session"]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    session_hour_price.plot(
        kind="bar",
        colormap="Blues",
        ax=ax
    )

    ax.set_title(
        "Session Distribution by Hour and Price",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Hour of Day"
    )

    ax.set_ylabel(
        "Unique Sessions"
    )

    plt.tight_layout()

    images["session_hour_price"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # PURCHASE DISTRIBUTION BY HOUR AND PRICE
    # ========================================================
    purchase_hour_price = (
        purchase_df.groupby(
            [
                "hour",
                "price_bin"
            ],
            observed=False
        )["user_session"]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    purchase_hour_price.plot(
        kind="bar",
        colormap="Oranges",
        ax=ax
    )

    ax.set_title(
        "Purchase Distribution by Hour and Price",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Hour of Day"
    )

    ax.set_ylabel(
        "Purchase Sessions"
    )

    plt.tight_layout()

    images["purchase_hour_price"] = (
        fig_to_base64(fig)
    )

    plt.close(fig)

    # ========================================================
    # DEBUG
    # ========================================================
    print(
        "E-commerce images:",
        sorted(images.keys())
    )

    # ========================================================
    # RETURN
    # ========================================================
    return {
        "kpis": {
            "rows": int(rows_count),
            "sessions": int(total_sessions),
            "users": int(total_users),
            "purchase_sessions": int(purchase_sessions),
            "products": int(total_products),
            "categories": int(total_categories),
            "brands": int(total_brands),
            "event_types": int(total_event_types)
        },

        "metrics": {
            "purchase_rate_avg":
                float(
                    purchase_rate.mean()
                ),

            "abandonment_rate_avg":
                float(
                    abandon_rate.mean()
                ),

            "window_shopper_rate":
                float(
                    window_rate
                ),

            "remove_cart_purchase_rate":
                float(
                    remove_purchase_rate
                ),

            "returning_rate":
                float(
                    returning_rate
                ),

            "loyal_rate":
                float(
                    loyal_rate
                )
        },

        "images": images
    }

def process_pipeline(df):
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df = df.sort_values([CASE_COL, TIME_COL])

    # ===== Helper =====
    def wrap_text(text, width=15):
        return "\n".join(textwrap.wrap(str(text), width))

    # =====================
    # GRAPH MINING
    # =====================
    G = nx.DiGraph()

    for _, group in df.groupby(CASE_COL):
        acts = group[ACTIVITY_COL].tolist()
        for i in range(len(acts)-1):
            u, v = acts[i], acts[i+1]
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    # ----- Graph -----
    fig1 = plt.figure(figsize=(14,6))
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, with_labels=True,
            node_color="#87CEEB",
            node_size=3000,
            font_size=9)

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        font_size=8
    )

    plt.title("Process Flow Graph")
    plt.tight_layout()

    graph_img = fig_to_base64(fig1)
    plt.close()

    # ----- Degree -----
    deg = nx.degree_centrality(G)
    sorted_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]

    fig2 = plt.figure(figsize=(12,6))

    labels = [wrap_text(x[0]) for x in sorted_deg]
    values = [x[1] for x in sorted_deg]

    plt.bar(labels, values)
    plt.xticks(rotation=0)
    plt.tight_layout()

    deg_img = fig_to_base64(fig2)
    plt.close()

    # ----- Betweenness -----
    bc = nx.betweenness_centrality(G)
    sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]

    fig3 = plt.figure(figsize=(12,6))

    labels = [wrap_text(x[0]) for x in sorted_bc]
    values = [x[1] for x in sorted_bc]

    plt.bar(labels, values)
    plt.xticks(rotation=0)
    plt.tight_layout()

    bc_img = fig_to_base64(fig3)
    plt.close()

    # ----- Paths -----
    paths = [
        " -> ".join(group[ACTIVITY_COL].tolist())
        for _, group in df.groupby(CASE_COL)
    ]

    counter = Counter(paths)
    top_paths = counter.most_common(5)

    fig4 = plt.figure(figsize=(12,8))

    labels = [wrap_text(x[0], 40) for x in top_paths]
    values = [x[1] for x in top_paths]

    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    paths_img = fig_to_base64(fig4)
    plt.close()

    # =====================
    # PROCESS MINING
    # =====================
    df_pm = df.rename(columns={
        CASE_COL: "case:concept:name",
        ACTIVITY_COL: "concept:name",
        TIME_COL: "time:timestamp"
    })

    log = pm4py.format_dataframe(
        df_pm,
        case_id='case:concept:name',
        activity_key='concept:name',
        timestamp_key='time:timestamp'
    )

    # Petri
    net, im, fm = alpha_miner.apply(log)
    gviz = pn_vis.apply(net, im, fm)
    pn_vis.save(gviz, "petri.png")

    with open("petri.png", "rb") as f:
        petri_img = base64.b64encode(f.read()).decode()

    # BPMN
    tree = inductive_miner.apply(log)
    bpmn = pt_converter.apply(tree, variant=pt_converter.Variants.TO_BPMN)

    gviz_bpmn = bpmn_vis.apply(bpmn)
    bpmn_vis.save(gviz_bpmn, "bpmn.png")

    with open("bpmn.png", "rb") as f:
        bpmn_img = base64.b64encode(f.read()).decode()

    # =====================
    # TRAIN TRANSFORMER
    # =====================
    global model, le, MAX_LEN

    if model is None:
        sequences = build_sequences(df)
        X, y = create_samples(sequences)

        X_pad, y_enc, vocab_size = encode_data(X, y)

        MAX_LEN = X_pad.shape[1]

        model = build_model(vocab_size, MAX_LEN)

        print("🔥 Training Transformer...")
        model.fit(X_pad, y_enc, epochs=5, batch_size=32, verbose=0)
        print("✅ Done")

    # =====================
    # OUTPUT
    # =====================
    safe_df = df.replace({np.nan: None})

    result = {
        "event_log": safe_df.to_dict(orient="records"),
        "important_steps": sorted_deg[:3],
        "bottlenecks": sorted_bc[:3],
        "top_paths": top_paths,
        "images": {
            "graph": graph_img,
            "degree": deg_img,
            "betweenness": bc_img,
            "paths": paths_img,
            "petri": petri_img,
            "bpmn": bpmn_img
        }
    }

    ecommerce = build_ecommerce_dashboard(df)
    result["ecommerce"] = ecommerce

    # LLM (context nhẹ)
    llm_context = {
        "important_steps": sorted_deg[:5],
        "bottlenecks": sorted_bc[:5],
        "top_paths": top_paths,
        "kpis": ecommerce["kpis"],
        "metrics": ecommerce["metrics"]
    }

    result["llm_analysis"] = llm_analyze(llm_context)

    return result

GLOBAL_CONTEXT = {}

model = None
le = None
MAX_LEN = None

app = Flask(__name__)

@app.route("/")
def home():
    return "API is running!"

import os

@app.route("/process", methods=["POST"])
def process():
    try:
        filename = request.json["filename"]

        path = os.path.join(
            r"C:\Users\tphuy\OneDrive\Documents\dataset",
            filename
        )

        print("Reading file:", path)
        print("Exists:", os.path.exists(path))

        if not os.path.exists(path):
            return jsonify({
                "error": f"File not found: {path}"
            }), 404

        df = pl.read_csv(path)
        df = df.to_pandas()

        df = df.where(pd.notnull(df), None)

        df[TIME_COL] = pd.to_datetime(
            df[TIME_COL],
            utc=True
        )

        result = process_pipeline(df)

        GLOBAL_CONTEXT["data"] = result

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/llm", methods=["POST"])
def run_llm():
    data = GLOBAL_CONTEXT.get("data", {})

    llm_context = {
        "important_steps": data.get("important_steps", []),
        "bottlenecks": data.get("bottlenecks", []),
        "top_paths": data.get("top_paths", [])
    }

    result = llm_analyze(llm_context)

    return jsonify({"llm_analysis": result})

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    data = GLOBAL_CONTEXT.get("data", {})

    context = {
        "important_steps": data.get("important_steps", []),
        "bottlenecks": data.get("bottlenecks", []),
        "top_paths": data.get("top_paths", []),
        "ecommerce_kpis": data.get("ecommerce", {}).get("kpis", {}),
        "ecommerce_metrics": data.get("ecommerce", {}).get("metrics", {})
    }

    prompt = f"""
You are a process mining expert chatbot.

Context data:
{context}

User question:
{user_msg}

Answer clearly based on the data.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify({"reply": response.text})

    except Exception as e:
        error_msg = str(e)

        import re
        match = re.search(r"'message': '([^']+)'", error_msg)

        if match:
            return jsonify({"reply": match.group(1)})

        return jsonify({"reply": error_msg})

@app.route("/predict_next", methods=["POST"])
def predict_next():
    global model, le, MAX_LEN

    data = request.json
    sequence = data.get("sequence", [])

    if model is None:
        return jsonify({"suggestions": []})

    if not sequence:
        return jsonify({"suggestions": []})

    try:
        seq_encoded = le.transform(sequence)
    except:
        return jsonify({"suggestions": []})

    seq_padded = pad_sequences([seq_encoded], maxlen=MAX_LEN, padding="pre")

    preds = model.predict(seq_padded, verbose=0)[0]

    top_k = preds.argsort()[-3:][::-1]

    suggestions = [
        {
            "activity": le.inverse_transform([i])[0],
            "prob": float(preds[i])
        }
        for i in top_k
    ]

    return jsonify({"suggestions": suggestions})

app.run(host="0.0.0.0", port=5000)
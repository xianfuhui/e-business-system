import os
import gc
import re
import base64
import textwrap
from io import BytesIO
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import polars as pl
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.visualization.bpmn import visualizer as bpmn_vis

from google import genai

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = r"C:\Users\tphuy\OneDrive\Documents\dataset"

CASE_COL = "user_session"
ACTIVITY_COL = "event_type"
TIME_COL = "event_time"

# Hard caps so a 5GB file doesn't blow up RAM / pm4py / matplotlib
MAX_ROWS_FOR_PROCESS_MINING = 300_000   # rows sampled (by full sessions) for alpha/inductive miner + graph viz
MAX_SESSIONS_FOR_TRAINING = 200_000     # sessions used to build transformer training sequences
TRAINING_BATCH_SIZE = 256
TRAINING_EPOCHS = 3
MAX_EVENT_LOG_ROWS_IN_RESPONSE = 1500    # small sample only, for the client-side table/graph — not the full dataset
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "process_mining_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}
MAX_UPLOAD_BYTES = 6 * 1024 * 1024 * 1024  # 6GB headroom; update the "Max 10 MB" hint text in the HTML too

# Only needed columns are read from the CSV — change this if your schema differs
NEEDED_COLUMNS = [
    "event_time", "event_type", "product_id", "category_code",
    "category_id", "brand", "price", "user_id", "user_session"
]

# API key must come from environment, never hardcoded
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def sanitize_for_json(obj):
    """
    Recursively replace NaN/Infinity (not valid JSON, but Python's json module
    emits them as bare literals by default) with 0, and convert numpy scalar
    types to native Python types. Call this right before jsonify(...) on any
    payload built from pandas/numpy computations.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    buf.close()
    return data


def llm_analyze(data: dict):
    if client is None:
        return "LLM disabled: set GEMINI_API_KEY environment variable to enable."

    prompt = f"Analyze this data:\n{data}"
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        match = re.search(r"'message': '([^']+)'", error_msg)
        return match.group(1) if match else error_msg


# ============================================================
# STREAMING / LAZY DATA LOADING (this is the part that actually
# lets a 5GB CSV work on a normal machine)
# ============================================================
def load_dataset_lazy(path: str) -> pl.LazyFrame:
    """
    Build a Polars LazyFrame over the CSV. Nothing is read into memory
    until .collect() / .collect(streaming=True) is called downstream.
    Dtypes are narrowed to cut memory roughly in half vs default pandas read.
    """
    # NOTE: category_id arrives as scientific notation text (e.g. "2.05E+18"),
    # so it must be read as Float64 first, never Int64 directly (that would
    # fail to parse). event_time has a trailing " UTC" in the string, so we
    # parse it with an explicit format instead of relying on auto-detection.
    schema_overrides = {
        "price": pl.Float32,
        "user_id": pl.Int64,
        "category_id": pl.Float64,
    }

    lf = (
        pl.scan_csv(
            path,
            try_parse_dates=False,   # parse manually below, faster + more predictable
            schema_overrides=schema_overrides,
        )
        .select([c for c in NEEDED_COLUMNS])
        .with_columns([
            pl.col(TIME_COL)
              .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False)
              .alias(TIME_COL),
            pl.col("brand").fill_null("Unknown"),
            pl.col("category_code").fill_null("Unknown"),
            pl.col("category_id").cast(pl.Int64, strict=False),
        ])
        .filter(pl.col(TIME_COL).is_not_null())
    )
    return lf


def sample_sessions(lf: pl.LazyFrame, max_rows: int) -> pl.DataFrame:
    """
    Sample by whole sessions (not raw rows) so process/sequence mining
    still sees coherent case histories, rather than cutting sessions in half.
    Uses streaming collect so Polars processes the 5GB file in chunks.
    """
    session_ids = (
        lf.select(CASE_COL).unique().collect(streaming=True)[CASE_COL]
    )

    if len(session_ids) == 0:
        return lf.collect(streaming=True).head(0)

    # Estimate how many sessions we need to hit ~max_rows
    total_rows = lf.select(pl.len()).collect(streaming=True).item()
    if total_rows <= max_rows:
        return lf.collect(streaming=True)

    frac = max_rows / total_rows
    n_sessions = max(1, int(len(session_ids) * frac))
    chosen = session_ids.sample(n=min(n_sessions, len(session_ids)), seed=42)

    sampled = (
        lf.filter(pl.col(CASE_COL).is_in(chosen))
        .collect(streaming=True)
    )
    return sampled


# ============================================================
# TRANSFORMER NEXT-ACTIVITY MODEL — streamed instead of building
# the full X/y list for the whole dataset in RAM
# ============================================================
def fit_label_encoder_streaming(lf: pl.LazyFrame) -> LabelEncoder:
    activities = (
        lf.select(ACTIVITY_COL).unique().collect(streaming=True)[ACTIVITY_COL].to_list()
    )
    le = LabelEncoder()
    le.fit(activities)
    return le


def sequence_generator(df_sessions: pl.DataFrame, le: LabelEncoder, max_len: int):
    """
    Yields (X_padded, y) one sample at a time from a (sampled, in-memory)
    set of sessions. Used to build a tf.data.Dataset so the model never
    needs the full list of training pairs materialized at once.
    """
    pdf = df_sessions.to_pandas()
    for _, group in pdf.groupby(CASE_COL):
        seq = group.sort_values(TIME_COL)[ACTIVITY_COL].tolist()
        if len(seq) < 2:
            continue
        enc = le.transform(seq) + 1  # +1: reserve 0 exclusively for padding, never a real class
        for i in range(1, len(enc)):
            x = enc[:i][-max_len:]
            x_pad = np.zeros(max_len, dtype=np.int32)
            x_pad[-len(x):] = x
            yield x_pad, enc[i]


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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_transformer_streaming(lf: pl.LazyFrame, le: LabelEncoder, max_len: int = 20):
    sampled = sample_sessions(lf, MAX_SESSIONS_FOR_TRAINING)
    vocab_size = len(le.classes_) + 1  # +1 for the reserved padding index (0)

    output_signature = (
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(sampled, le, max_len),
        output_signature=output_signature,
    ).shuffle(2048).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model(vocab_size, max_len)
    print("Training transformer (streamed)...")
    model.fit(ds, epochs=TRAINING_EPOCHS, verbose=0)
    print("Training done")

    del sampled
    gc.collect()
    return model, max_len


# ============================================================
# GRAPH / PROCESS MINING — run on a bounded sample, never the
# raw 5GB frame
# ============================================================
def wrap_text(text, width=15):
    return "\n".join(textwrap.wrap(str(text), width))


def build_graph_mining(df: pd.DataFrame):
    G = nx.DiGraph()
    for _, group in df.groupby(CASE_COL):
        acts = group.sort_values(TIME_COL)[ACTIVITY_COL].tolist()
        for i in range(len(acts) - 1):
            u, v = acts[i], acts[i + 1]
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    fig1 = plt.figure(figsize=(14, 6))
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_color="#87CEEB", node_size=3000, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=8)
    plt.title("Process Flow Graph (sampled)")
    plt.tight_layout()
    graph_img = fig_to_base64(fig1)
    plt.close(fig1)

    deg = nx.degree_centrality(G)
    sorted_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]

    fig2 = plt.figure(figsize=(12, 6))
    labels = [wrap_text(x[0]) for x in sorted_deg]
    values = [x[1] for x in sorted_deg]
    plt.bar(labels, values)
    plt.xticks(rotation=0)
    plt.tight_layout()
    deg_img = fig_to_base64(fig2)
    plt.close(fig2)

    bc = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))  # approximate BC for speed
    sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]

    fig3 = plt.figure(figsize=(12, 6))
    labels = [wrap_text(x[0]) for x in sorted_bc]
    values = [x[1] for x in sorted_bc]
    plt.bar(labels, values)
    plt.xticks(rotation=0)
    plt.tight_layout()
    bc_img = fig_to_base64(fig3)
    plt.close(fig3)

    paths = [
        " -> ".join(group.sort_values(TIME_COL)[ACTIVITY_COL].tolist())
        for _, group in df.groupby(CASE_COL)
    ]
    counter = Counter(paths)
    top_paths = counter.most_common(5)

    fig4 = plt.figure(figsize=(12, 8))
    labels = [wrap_text(x[0], 40) for x in top_paths]
    values = [x[1] for x in top_paths]
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    paths_img = fig_to_base64(fig4)
    plt.close(fig4)

    del G
    gc.collect()

    return {
        "images": {"graph": graph_img, "degree": deg_img, "betweenness": bc_img, "paths": paths_img},
        "important_steps": sorted_deg[:3],
        "bottlenecks": sorted_bc[:3],
        "top_paths": top_paths,
    }


def build_pm4py_models(df: pd.DataFrame):
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

    net, im, fm = alpha_miner.apply(log)
    gviz = pn_vis.apply(net, im, fm)
    pn_vis.save(gviz, "petri.png")
    with open("petri.png", "rb") as f:
        petri_img = base64.b64encode(f.read()).decode()

    tree = inductive_miner.apply(log)
    bpmn = pt_converter.apply(tree, variant=pt_converter.Variants.TO_BPMN)
    gviz_bpmn = bpmn_vis.apply(bpmn)
    bpmn_vis.save(gviz_bpmn, "bpmn.png")
    with open("bpmn.png", "rb") as f:
        bpmn_img = base64.b64encode(f.read()).decode()

    del log, net, im, fm, tree, bpmn
    gc.collect()

    return {"petri": petri_img, "bpmn": bpmn_img}


# ============================================================
# ECOMMERCE DASHBOARD — same logic as before, but only ever
# receives a bounded sample (see process_pipeline)
# ============================================================
def build_ecommerce_dashboard(df: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(style="whitegrid")

    TITLE_SIZE, LABEL_SIZE = 15, 12
    COLOR_SESSION, COLOR_PURCHASE = "#3498db", "#e67e22"
    COLOR_SUCCESS, COLOR_DANGER = "#27ae60", "#e74c3c"
    COLOR_DARK, COLOR_PURPLE, COLOR_INFO = "#34495e", "#8e44ad", "#16a085"

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["hour"] = df["event_time"].dt.hour
    df["brand"] = df["brand"].fillna("Unknown")
    df["category_code"] = df["category_code"].fillna("Unknown")

    purchase_df = df[df["event_type"] == "purchase"].copy()
    images = {}

    kpis = {
        "rows": int(len(df)),
        "sessions": int(df["user_session"].nunique()),
        "users": int(df["user_id"].nunique()),
        "purchase_sessions": int(purchase_df["user_session"].nunique()),
        "products": int(df["product_id"].nunique()),
        "categories": int(df["category_code"].nunique()),
        "brands": int(df["brand"].nunique()),
        "event_types": int(df["event_type"].nunique()),
    }

    BIN_SIZE, MAX_PRICE = 50, 2000

    def price_bin_chart(source_df, color, title, ylabel, key):
        tmp = source_df.copy()
        tmp["price_bin"] = (np.floor(tmp["price"] / BIN_SIZE).astype(int) * BIN_SIZE)
        tmp = tmp[tmp["price_bin"] <= MAX_PRICE]
        tmp["price_label"] = "$" + tmp["price_bin"].astype(str) + "-$" + (tmp["price_bin"] + BIN_SIZE - 1).astype(str)
        agg = tmp.groupby(["price_bin", "price_label"])["user_session"].nunique().reset_index(name="session_count").sort_values("price_bin")

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data=agg, x="price_label", y="session_count", color=color, edgecolor="black", linewidth=0.5, ax=ax)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 != 0:
                label.set_visible(False)
        plt.tight_layout()
        images[key] = fig_to_base64(fig)
        plt.close(fig)
        del tmp, agg

    price_bin_chart(df, COLOR_SESSION, "Session Distribution by Product Price", "Unique Sessions", "session_price_dist")
    price_bin_chart(purchase_df, COLOR_PURCHASE, "Purchase Session Distribution by Product Price", "Unique Purchase Sessions", "purchase_price_dist")

    def heatmap_chart(source_df, cmap, title, key):
        tmp = source_df.assign(price_bin=(np.floor(source_df["price"] / BIN_SIZE) * BIN_SIZE))
        tmp = tmp[tmp["price_bin"] <= MAX_PRICE]
        agg = tmp.groupby(["price_bin", "hour"])["user_session"].nunique().reset_index(name="session_count")
        pivot = agg.pivot(index="price_bin", columns="hour", values="session_count").fillna(0).sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(16, 9))
        sns.heatmap(pivot, cmap=cmap, linewidths=.1, linecolor="white", ax=ax)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
        plt.tight_layout()
        images[key] = fig_to_base64(fig)
        plt.close(fig)
        del tmp, agg, pivot

    heatmap_chart(df, "Blues", "Session Heatmap by Product Price and Hour", "session_heatmap")
    heatmap_chart(purchase_df, "OrRd", "Purchase Heatmap by Product Price and Hour", "purchase_heatmap")

    # Session / Purchase Distribution by Hour & Price (stacked bar by price_bin per hour)
    price_bins = [0, 50, 100, 200, 500, 1000, np.inf]
    price_labels = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]

    def hour_price_chart(source_df, cmap, title, key):
        tmp = source_df.copy()
        tmp["price_bin_label"] = pd.cut(tmp["price"], bins=price_bins, labels=price_labels)
        agg = (
            tmp.groupby(["hour", "price_bin_label"], observed=False)["user_session"]
            .nunique()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(15, 6))
        agg.plot(kind="bar", colormap=cmap, ax=ax)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Unique Sessions")
        plt.tight_layout()
        images[key] = fig_to_base64(fig)
        plt.close(fig)
        del tmp, agg

    hour_price_chart(df, "Blues", "Session Distribution by Hour and Price", "session_hour_price")
    hour_price_chart(purchase_df, "Oranges", "Purchase Distribution by Hour and Price", "purchase_hour_price")

    # Brand comparison
    brand_sessions = df.groupby("brand")["user_session"].nunique()
    brand_purchase = purchase_df.groupby("brand")["user_session"].nunique()
    brand_compare = pd.concat([brand_sessions.rename("Sessions"), brand_purchase.rename("Purchase")], axis=1).fillna(0)
    brand_compare = brand_compare.sort_values("Sessions", ascending=False).head(15)
    brand_compare["CR_%"] = (brand_compare["Purchase"] / brand_compare["Sessions"] * 100).replace([np.inf, -np.inf], 0).fillna(0)

    df_melted = brand_compare.reset_index().melt(id_vars=["brand", "CR_%"], value_vars=["Sessions", "Purchase"], var_name="Type", value_name="Count")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=df_melted, x="brand", y="Count", hue="Type", palette=[COLOR_DARK, COLOR_DANGER], edgecolor="black", linewidth=0.5, ax=ax)
    ax.set_title("Top 15 Brands: Sessions vs Purchase Sessions", fontsize=TITLE_SIZE, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    images["brand_compare"] = fig_to_base64(fig)
    plt.close(fig)

    # Purchase rate / abandonment / window shopper / returning users
    session_category = df.groupby("category_code")["user_session"].nunique()
    purchase_category = purchase_df.groupby("category_code")["user_session"].nunique()
    purchase_rate_df = pd.concat([session_category.rename("total_sessions"), purchase_category.rename("purchase_sessions")], axis=1).fillna(0)
    purchase_rate_df["non_purchase_sessions"] = purchase_rate_df["total_sessions"] - purchase_rate_df["purchase_sessions"]
    purchase_rate_df["purchase_rate"] = (purchase_rate_df["purchase_sessions"] / purchase_rate_df["total_sessions"] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    purchase_rate_avg = float(purchase_rate_df["purchase_rate"].mean())

    top_purchase = purchase_rate_df.sort_values("total_sessions", ascending=False).head(20)
    fig, ax1 = plt.subplots(figsize=(15, 7))
    x = np.arange(len(top_purchase))
    ax1.bar(x, top_purchase["non_purchase_sessions"], color=COLOR_DARK, label="Non-Purchase")
    ax1.bar(x, top_purchase["purchase_sessions"], bottom=top_purchase["non_purchase_sessions"], color=COLOR_SUCCESS, label="Purchase")
    ax2 = ax1.twinx()
    ax2.plot(x, top_purchase["purchase_rate"], color=COLOR_DANGER, marker="o", linewidth=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_purchase.index, rotation=45, ha="right")
    ax1.set_title("Purchase Rate by Category", fontsize=TITLE_SIZE, fontweight="bold")
    ax1.set_ylabel("Number of Sessions")
    ax2.set_ylabel("Purchase Rate (%)")
    ax1.legend()
    plt.tight_layout()
    images["purchase_rate"] = fig_to_base64(fig)
    plt.close(fig)
    del top_purchase

    cart_sessions = df[df["event_type"] == "cart"].groupby("category_code")["user_session"].nunique()
    purchase_sessions_cat = purchase_df.groupby("category_code")["user_session"].nunique()
    abandon_df = pd.concat([cart_sessions.rename("cart_sessions"), purchase_sessions_cat.rename("purchase_sessions")], axis=1).fillna(0)
    abandon_df["abandonment_rate"] = ((abandon_df["cart_sessions"] - abandon_df["purchase_sessions"]) / abandon_df["cart_sessions"] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    abandonment_rate_avg = float(abandon_df["abandonment_rate"].mean())

    fig, ax = plt.subplots(figsize=(13, 6))
    abandon_df["abandonment_rate"].sort_values(ascending=False).head(20).plot(kind="bar", color=COLOR_DANGER, edgecolor="black", ax=ax)
    ax.set_title("Cart Abandonment Rate by Category", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Abandonment Rate (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    images["abandonment_rate"] = fig_to_base64(fig)
    plt.close(fig)

    session_events = df.groupby("user_session")["event_type"].apply(set)
    window_sessions = session_events[session_events.apply(lambda x: x == {"view"})].index
    window_rate = float(len(window_sessions) / max(1, kpis["sessions"]) * 100)

    view_df = df[df["event_type"] == "view"].groupby("category_code")["user_session"].nunique()
    window_df = df[df["user_session"].isin(window_sessions)].groupby("category_code")["user_session"].nunique()
    window_rate_df = pd.concat([view_df.rename("view_sessions"), window_df.rename("window_sessions")], axis=1).fillna(0)
    window_rate_df["window_rate"] = (window_rate_df["window_sessions"] / window_rate_df["view_sessions"] * 100).replace([np.inf, -np.inf], 0).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    window_rate_df["window_rate"].sort_values(ascending=False).head(20).plot(kind="bar", color=COLOR_PURPLE, edgecolor="black", ax=ax)
    ax.set_title("Window Shopper Rate by Category", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Window Shopper Rate (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    images["window_shopper"] = fig_to_base64(fig)
    plt.close(fig)
    del window_rate_df

    user_sessions = df.groupby("user_id")["user_session"].nunique()
    returning_rate = float((user_sessions > 1).mean() * 100)
    loyal_rate = float((user_sessions > 3).mean() * 100)

    session_start = df.groupby(["user_id", "user_session"])["event_time"].min().reset_index()
    session_start = session_start.sort_values(["user_id", "event_time"])
    session_start["session_order"] = session_start.groupby("user_id").cumcount() + 1
    session_start["is_returning"] = session_start["session_order"] > 1
    daily_returning = session_start.groupby(session_start["event_time"].dt.date)["is_returning"].mean() * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    daily_returning.plot(ax=ax, marker="o", linewidth=3, color=COLOR_SUCCESS)
    ax.set_title("Daily Returning Session Rate", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Returning Rate (%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    images["loyal_returning"] = fig_to_base64(fig)
    plt.close(fig)
    del session_start, daily_returning

    # Remove-from-cart-then-still-purchased trend (cheap on a sampled df)
    df_sorted = df.sort_values(["user_session", "event_time"])
    remove_purchase_dates = []
    for _, group in df_sorted.groupby("user_session"):
        remove_rows = group[group["event_type"] == "remove_from_cart"]
        purchase_rows = group[group["event_type"] == "purchase"]
        if len(remove_rows) == 0 or len(purchase_rows) == 0:
            continue
        if purchase_rows["event_time"].min() > remove_rows["event_time"].min():
            remove_purchase_dates.append(group["event_time"].min().date())

    remove_purchase_daily = pd.Series(remove_purchase_dates).value_counts().sort_index()
    remove_total = max(1, int((df["event_type"] == "remove_from_cart").sum()))
    remove_purchase_rate = float(len(remove_purchase_dates) / remove_total * 100)

    fig, ax = plt.subplots(figsize=(13, 6))
    if len(remove_purchase_daily) > 0:
        remove_purchase_daily.plot(ax=ax, marker="o", linewidth=3, color=COLOR_INFO)
    ax.set_title("Sessions That Removed Items but Still Purchased", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Number of Sessions")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    images["remove_purchase"] = fig_to_base64(fig)
    plt.close(fig)
    del df_sorted

    metrics = {
        "purchase_rate_avg": purchase_rate_avg,
        "abandonment_rate_avg": abandonment_rate_avg,
        "window_shopper_rate": window_rate,
        "remove_cart_purchase_rate": remove_purchase_rate,
        "returning_rate": returning_rate,
        "loyal_rate": loyal_rate,
    }

    del df, purchase_df
    gc.collect()

    return {"kpis": kpis, "metrics": metrics, "images": images}


# ============================================================
# MAIN PIPELINE
# ============================================================
def build_event_log_sample(df: pd.DataFrame, max_rows: int):
    """
    Small JSON-safe sample for the browser table + vis-network graph.
    Picks whole sessions (capped) rather than an arbitrary row slice so the
    client-side graph rendering (which groups by user_session) still makes sense.
    """
    if max_rows <= 0 or df.empty:
        return []

    session_order = df.drop_duplicates(CASE_COL)[CASE_COL].tolist()
    rows_acc = []
    seen_rows = 0
    for sid in session_order:
        if seen_rows >= max_rows:
            break
        chunk = df[df[CASE_COL] == sid].sort_values(TIME_COL)
        rows_acc.append(chunk)
        seen_rows += len(chunk)

    sample = pd.concat(rows_acc).head(max_rows) if rows_acc else df.head(0)
    sample = sample.copy()
    sample[TIME_COL] = sample[TIME_COL].astype(str)
    sample = sample.replace({np.nan: None})
    return sample.to_dict(orient="records")


def process_pipeline(path: str):
    lf = load_dataset_lazy(path)

    # Fit encoder on full activity vocabulary (cheap: just unique values, streamed)
    le = fit_label_encoder_streaming(lf)

    # Bounded sample for everything visual / graph / process mining
    sample_df = sample_sessions(lf, MAX_ROWS_FOR_PROCESS_MINING).to_pandas()

    event_log_sample = build_event_log_sample(sample_df, MAX_EVENT_LOG_ROWS_IN_RESPONSE)

    graph_result = build_graph_mining(sample_df)
    pm4py_result = build_pm4py_models(sample_df)
    ecommerce_result = build_ecommerce_dashboard(sample_df)

    model, max_len = train_transformer_streaming(lf, le)

    result = {
        "sample_size_used": int(len(sample_df)),
        "event_log": event_log_sample,  # small sample only — never the full 5GB log
        "important_steps": graph_result["important_steps"],
        "bottlenecks": graph_result["bottlenecks"],
        "top_paths": graph_result["top_paths"],
        "images": {**graph_result["images"], **pm4py_result},
        "ecommerce": ecommerce_result,
    }

    llm_context = {
        "important_steps": graph_result["important_steps"],
        "bottlenecks": graph_result["bottlenecks"],
        "top_paths": graph_result["top_paths"],
        "kpis": ecommerce_result["kpis"],
        "metrics": ecommerce_result["metrics"],
    }
    result["llm_analysis"] = llm_analyze(llm_context)

    del sample_df
    gc.collect()

    return result, model, le, max_len


# ============================================================
# FLASK APP
# ============================================================
GLOBAL_CONTEXT = {}
model = None
le = None
MAX_LEN = None

app = Flask(__name__)

# The HTML dashboard is expected to be served separately (e.g. by Spring Boot)
# and call this Flask API cross-origin, so CORS must be enabled.
# Lock allowed origins down to your actual Spring Boot host in production —
# origins="*" is only fine for local dev.
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return "API is running!"


# ------------------------------------------------------------------
# Matches: const formData = new FormData(e.target); fetch('/api/run', ...)
# The HTML form sends the file as multipart/form-data under field "file".
# ------------------------------------------------------------------
@app.route("/api/run", methods=["POST"])
def api_run():
    global model, le, MAX_LEN
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(f.filename):
            return jsonify({"error": "Only .csv files are supported"}), 400

        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        f.save(path)

        print("Saved upload to:", path)

        result, model, le, MAX_LEN = process_pipeline(path)
        result = sanitize_for_json(result)
        GLOBAL_CONTEXT["data"] = result

        return jsonify(result)

    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500
    finally:
        # Don't keep uploaded CSVs lying around — they can be multi-GB
        try:
            if "path" in dir() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# Kept for non-browser / scripted use: same logic, takes a server-side path by name
@app.route("/process", methods=["POST"])
def process_by_filename():
    global model, le, MAX_LEN
    try:
        filename = request.json["filename"]
        path = os.path.join(DATASET_DIR, filename)

        if not os.path.exists(path):
            return jsonify({"error": f"File not found: {path}"}), 404

        result, model, le, MAX_LEN = process_pipeline(path)
        result = sanitize_for_json(result)
        GLOBAL_CONTEXT["data"] = result

        return jsonify(result)

    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/llm", methods=["POST"])
def api_llm():
    data = GLOBAL_CONTEXT.get("data", {})
    llm_context = {
        "important_steps": data.get("important_steps", []),
        "bottlenecks": data.get("bottlenecks", []),
        "top_paths": data.get("top_paths", []),
        "kpis": data.get("ecommerce", {}).get("kpis", {}),
        "metrics": data.get("ecommerce", {}).get("metrics", {}),
    }
    return jsonify({"llm_analysis": llm_analyze(llm_context)})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_msg = (request.json or {}).get("message", "")
    data = GLOBAL_CONTEXT.get("data", {})

    context = {
        "important_steps": data.get("important_steps", []),
        "bottlenecks": data.get("bottlenecks", []),
        "top_paths": data.get("top_paths", []),
        "ecommerce_kpis": data.get("ecommerce", {}).get("kpis", {}),
        "ecommerce_metrics": data.get("ecommerce", {}).get("metrics", {}),
    }
    prompt = f"You are a process mining expert chatbot.\n\nContext data:\n{context}\n\nUser question:\n{user_msg}\n\nAnswer clearly based on the data."

    if client is None:
        return jsonify({"reply": "LLM disabled: set GEMINI_API_KEY environment variable to enable."})

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return jsonify({"reply": response.text})
    except Exception as e:
        error_msg = str(e)
        match = re.search(r"'message': '([^']+)'", error_msg)
        return jsonify({"reply": match.group(1) if match else error_msg})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global model, le, MAX_LEN
    data = request.json or {}
    sequence = data.get("sequence", [])

    if model is None or le is None:
        return jsonify({"suggestions": [], "error": "Model not trained yet — run /api/run first"}), 200

    if not sequence:
        return jsonify({"suggestions": []})

    known_classes = set(le.classes_)
    # Drop activities the model never saw during training instead of failing
    # the whole request — this is the most common cause of "no suggestions".
    filtered_sequence = [s for s in sequence if s in known_classes]

    if not filtered_sequence:
        return jsonify({
            "suggestions": [],
            "error": f"None of the clicked activities were seen during training. Known activities: {sorted(known_classes)}"
        })

    try:
        seq_encoded = le.transform(filtered_sequence) + 1  # match the +1 shift used at training time
        seq_padded = pad_sequences([seq_encoded], maxlen=MAX_LEN, padding="pre")
        preds = model.predict(seq_padded, verbose=0)[0]

        # Defensive: NaN/Inf are not valid JSON (this was the actual root cause of
        # "Expected ',' or '}' ..." parse errors in the browser) — sanitize regardless.
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        preds[0] = 0.0  # index 0 = padding, never a valid "next activity" prediction

        top_k = [i for i in preds.argsort()[::-1] if i != 0][:3]

        suggestions = [
            {"activity": le.inverse_transform([i - 1])[0], "prob": float(preds[i])}
            for i in top_k
        ]
        return jsonify(sanitize_for_json({"suggestions": suggestions}))

    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"suggestions": [], "error": traceback.format_exc()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
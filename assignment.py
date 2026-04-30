# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.10.0",
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "scipy>=1.11",
#     "ipython>=8.0",
#     "drawdata==0.5.0",
#     "anywidget>=0.9",
#     "seaborn==0.13.2",
#     "altair==6.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semantic Axes: Intro Notebook

    /// Note | This notebook is NOT the deliverable.

    It is a worked example that walks through the full pipeline you are expected to build in your own submission.

    Feel free to copy the `make_axis` and `score_words` functions from
    this notebook into your submission — they are the reference
    implementation.

    ///

    We'll learn **SemAxis** (An, Kwak, and Ahn. 2019 ACL), a tool to create interpretable window into embedding space.

    > Jisun An, Haewoon Kwak, and Yong-Yeol Ahn. 2018. SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2450–2461, Melbourne, Australia. Association for Computational Linguistics.


    The original work uses word embeddings. Here we use **Sentence Transformers** to create word embeddings. Let us first load the sentence transformer model. We'll then build the concept of SemAxis and how we implement it using sentence transformers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Setup — Load the embedding model

    The sentence transformer is a mid-sized (~420 MB) pre-trained model
    that maps any text to a 768-dimensional unit vector. The first call
    downloads the weights; subsequent calls reuse them from disk.

    **Copy this cell into your own submission** — you will need the same
    model (or any other sentence transformer) to reproduce anything below.
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2 if you want faster but noisier results
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — Concepts

    Run the cells below in order. Each takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 1 — Word Embeddings

    A sentence transformer maps any text to a point in a high-dimensional
    space. Texts with similar meaning land near each other.
    """)
    return


@app.cell
def _(model):
    _words = ["Harvard University", "MIT", "Swarthmore College", "community college"]

    # This is how the embedding model turns text into vectors. Each text gets mapped to a point in a 384-dimensional space.
    _emb = model.encode(_words, normalize_embeddings=True)

    print(f"Each text → a vector of {_emb.shape[1]} numbers.\n")

    for _w, _e in zip(_words, _emb):
        print(f"  '{_w}' (first 6 dims): {_e[:6].round(3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 2 — Cosine Similarity

    Two unit vectors are **similar** when they point in the same direction.
    We measure this with a dot product (= cosine similarity for unit vectors).
    """)
    return


@app.cell
def _(model, np):
    _pairs = [
        ("Harvard University", "Yale University"),
        ("MIT", "Georgia Tech"),
        ("Harvard University", "community college"),
    ]

    for _pair in _pairs:
        _left, _right = _pair
        _emb = model.encode([_left, _right], normalize_embeddings=True)
        _sim = np.dot(_emb[0], _emb[1])
        print(f"  {_left:<20} ↔  {_right:<20}  sim = {_sim:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 3 — SemAxis

    A **semantic axis** is a direction in embedding space defined by two
    opposite word sets. Any new word can be scored by projecting its
    embedding onto that direction.

    $$
    \text{axis} = \frac{\bar{\mathbf{e}}_{+} - \bar{\mathbf{e}}_{-}}
                       {\|\bar{\mathbf{e}}_{+} - \bar{\mathbf{e}}_{-}\|}
    \qquad
    \text{score}(w) = \mathbf{e}_w \cdot \text{axis}
    $$

    Large positive → near the + pole. Large negative → near the − pole.

    /// warning | "Score ≈ 0" does NOT mean "neutral".

    A score near zero means the word's embedding is (roughly) **orthogonal**
    to the axis direction — the axis simply does not carry information
    about that word. This is very different from "the word sits exactly
    between the two poles". For a sentiment axis, for example, "orange"
    and "democracy" both score near zero, but neither is sentiment-neutral
    in any meaningful sense — they are just off-topic for the axis.
    Keep this in mind when you interpret the 2D scatter.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Try it in 2D

    The math above lives in 768-dim space, which is impossible to draw.
    The same operation in **2D** is easy to see. Use the widget below:

    - **Draw points** with up to four pens. The **first two colors are the
      poles** (color 1 = − pole, color 2 = + pole); colors 3 and 4 are
      optional "test" points that get projected onto the same axis.
    - Bold **colored arrows** show each pole centroid as a vector from the
      origin — the SemAxis lives in a vector space.
    - The **thick black arrow** is the SemAxis itself:
      $\mathbf{e}_{+} - \mathbf{e}_{-}$, the difference of the two pole
      centroids.
    - The right panel shows the 1D distribution of projections for every
      class — a jittered strip plot with a kernel-density envelope
      (violin-style). This is what `score_words` returns after reducing
      the embedding to a single axis.
    """)
    return


@app.cell
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget(width=820, height=480, point_size=12))
    widget
    return (widget,)


@app.cell(hide_code=True)
def _(pd, widget):
    # Read the drawn data reactively. Fall back to the preset when empty.
    _ = widget.value  # noqa: register widget as reactivity dependency
    try:
        drawn = widget.data_as_pandas
    except Exception:
        drawn = pd.DataFrame()
    df_demo = drawn if (not drawn.empty and drawn["color"].nunique() >= 2) else make_preset_clusters()
    plot_semaxis_2d(df_demo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Reference Implementation

    Two short functions: **build an axis** and **score words**. Copy these
    into your own submission.
    """)
    return


@app.cell
def _(np):
    def make_axis(positive_words, negative_words, embedding_model):
        """Return a unit-length semantic axis from two word sets."""

        # get the embeddings for each pole
        pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
        neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)

        # Compute the pole centroids
        # axis = 0 means "average across the rows, keep the columns (dims) intact"
        # since pos_emb is shape (num_pos_words, embedding_dim), the mean is shape (embedding_dim,)
        pole_pos = pos_emb.mean(axis=0)  # (embedding_dim,)
        pole_neg = neg_emb.mean(axis=0)  # (embedding_dim,)

        # The axis is the difference between the two centroids, normalized to unit length.
        v = pole_pos - pole_neg

        v = v / (np.linalg.norm(v) + 1e-10)  # add small epsilon to prevent division by zero

        return v / (np.linalg.norm(v) + 1e-10)

    return (make_axis,)


@app.function
def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""

    emb = embedding_model.encode(list(words), normalize_embeddings=True)

    # Projection to the axis is just a dot product (since the axis is unit-length).
    # @ is matrix multiplication in NumPy. Since `emb` is shape (num_words, embedding_dim) and `axis` is shape (embedding_dim,), the result is shape (num_words,), which is exactly what we want: one score per word.
    proj = emb @ axis

    return proj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Worked Example: World Cities

    The `data/cities.csv` file lists several hundred cities worldwide
    with their country, region, coordinates, population, founding date,
    and a GaWC business-activity rating. We will score every city along
    two semantic axes and color the scatter by **region** (continent).

    Study this example, then build your own submission using whichever
    case-study dataset (or your own data) you choose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load the dataset

    The cities live in `data/cities.csv`. We load them with **pandas**
    and set explicit dtypes up front — this is a good habit for any
    tabular pipeline:

    - It makes your column assumptions explicit and catches typos or
      format drift early.
    - `"category"` is memory-efficient for low-cardinality text columns
      (like `region`) and gives you ordered plotting for free.
    - `"Int64"` (capital `I`) is pandas' *nullable* integer type — use
      it for integer columns that have missing values, like `population`.
    - Columns with mixed formats — e.g. `founded`, which contains entries
      such as `"500s BCE"` and `"6th c."` alongside ordinary years —
      stay as `"string"` so pandas does not silently coerce them to NaN.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/universities.csv"
    ,
        dtype={
            "city": "string",
            "country": "string",
            "region": "category",
            "lat": "float64",
            "lon": "float64",
            "population": "Int64",  # nullable integer
            "founded": "string",  # mixed: "1624", "500s BCE", "6th c."
            "business_activity": "string",
        },
    )
    print(f"{len(df)} cities across {df['region'].nunique()} regions.")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Design two semantic axes

    A **good axis** is:

    - **Well-separated**: the + and − word sets should be far apart in
      embedding space (pole distance ≥ 0.3; not a strict requirement but a rule-of-thumb).
    - **Discriminative**: when projected onto your dataset it should spread
      the points out, not pile them in the middle.
    - **Orthogonal to the other axis**: the two axes should capture
      different aspects of the data.

    /// note | Why multiple words per pole — why not just one?

    A single word's embedding is **noisy**. It carries the quirks of how
    that specific word appears in the training data: rare senses,
    collocations, polysemy, branding. If you build an axis from just two
    single words, those quirks *become* the axis.

    Rule of thumb: **3–6 words per pole**. Fewer is too noisy; many more
    starts to dilute the concept by pulling in unrelated vocabulary.

    ///

    Our two axes for cities:

    - **Horizontal** — *small town / village* (−) ↔ *megacity / metropolis* (+)
    - **Vertical** — *cold / northern climate* (−) ↔ *tropical / warm climate* (+)

    These are conceptually independent: Singapore is both tropical *and* a
    megacity; Reykjavik is cold and small; Asunción is warm-ish but not a
    global megalopolis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 1 — small town / village (−) ↔ megacity / metropolis (+)

    Notice that the pole words each evoke the concept from a different
    angle. The centroid lies in the shared-meaning intersection of all of them.
    """)
    return


@app.cell
def _(make_axis, model):
    axis1_pos = [
        "elite university",
        "ivy league",
        "highly selective",
        "prestigious institution",
        "top ranked university"
    ]

    axis1_neg = [
        "community college",
        "open enrollment",
        "less selective",
        "local college",
        "accessible education"
    ]

    axis_metro = make_axis(axis1_pos, axis1_neg, model)
    return (axis_metro,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 2 — cold/northern climate (−) ↔ tropical/warm climate (+)

    The two climate poles should barely overlap in normal text, so the
    axis direction is strong and the resulting projection cleanly sorts
    cities by how their descriptions lean thermally.
    """)
    return


app._unparsable_cell(
    r"""
    axis2_pos = [
        ""research intensive",
        "doctoral programs",
        "graduate research",
        "large research university",
        "lab based study",
    ]
    axis2_neg = [
        "teaching focused",
        "undergraduate education",
        "small classes",
        "liberal arts college",
        "student centered learning",
    ]
    axis_climate = make_axis(axis2_pos, axis2_neg, model)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Score every city along both axes

    For each city name we compute its embedding and take the dot product
    with each axis. The result is two numbers per city — its `x` and `y`
    coordinates in the 2D semantic space — which we attach to the
    DataFrame.
    """)
    return


@app.cell
def _(axis_climate, axis_metro, df, model):
    x = score_words(df["name"].tolist(), axis_metro, model)
    y = score_words(df["name"].tolist(), axis_climate, model)
    df_scored = df.assign(x=x, y=y,)
    df_scored.head()
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Visualize
    """)
    return


@app.cell(hide_code=True)
def _(alt, df_scored):
    alt.Chart(df_scored).mark_circle(size=60).encode(
        x="x",
        y="y",
        color="region",
        tooltip=["name", "type", "region"]
    )

    return


@app.cell(hide_code=True)
def _(alt, color_by, df_scored, mo):
    # Okabe–Ito palette — categorical, colorblind-safe.
    REGION_COLORS = {
        "Africa": "#009E73",
        "Americas": "#0072B2",
        "Asia": "#D55E00",
        "Europe": "#E69F00",
        "Oceania": "#56B4E9",
    }

    # GaWC rating ordered from "most globally connected" to "least".
    BUSINESS_ORDER = [
        "Alpha++",
        "Alpha+",
        "Alpha",
        "Alpha-",
        "Beta+",
        "Beta",
        "Beta-",
        "Gamma+",
        "Gamma",
        "Gamma-",
        "High Sufficiency",
        "Sufficiency",
    ]

    if color_by.value == "region":
        # Categorical → qualitative palette.
        _color = alt.Color(
            "region:N",
            scale=alt.Scale(
                domain=list(REGION_COLORS.keys()),
                range=list(REGION_COLORS.values()),
            ),
            legend=alt.Legend(title="Region"),
        )
    elif color_by.value == "business_activity":
        # Ordinal → sequential palette, sorted along the order.
        _color = alt.Color(
            "business_activity:O",
            sort=BUSINESS_ORDER,
            scale=alt.Scale(domain=BUSINESS_ORDER, scheme="viridis"),
            legend=alt.Legend(title="GaWC rating"),
        )
    elif color_by.value == "population":
        # Quantitative, heavy-tailed → log-scaled sequential palette.
        _color = alt.Color(
            "population:Q",
            scale=alt.Scale(scheme="viridis", type="log"),
            legend=alt.Legend(title="Population (log)"),
        )
    else:  # abs_lat — distance to the equator
        # Sequential (one-sided) → sequential palette, no white midpoint.
        # reverse=True so low |lat| (equator, hot) → light, high |lat| (pole, cold) → dark.
        _color = alt.Color(
            "abs_lat:Q",
            scale=alt.Scale(scheme="inferno", reverse=True),
            legend=alt.Legend(title="|latitude| (° from equator)"),
        )

    chart = (
        alt.Chart(df_scored)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "x:Q",
                title="← small town / village          megacity / metropolis →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "y:Q",
                title="← cold / northern          tropical / warm →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            color=_color,
            tooltip=[
                alt.Tooltip("city:N", title="City"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("business_activity:N", title="GaWC"),
                alt.Tooltip("population:Q", title="Population", format=","),
                alt.Tooltip("lat:Q", title="Latitude", format=".2f"),
                alt.Tooltip("x:Q", title="megacity score", format=".3f"),
                alt.Tooltip("y:Q", title="tropical-warm score", format=".3f"),
            ],
        )
        .properties(
            width=720,
            height=500,
            title="World cities in a 2D semantic space (hover for details)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()  # pan + zoom
    )

    # Stack the dropdown directly above the chart so it is always visible.
    mo.vstack([color_by, chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 — Document what you see

    A good observation paragraph answers:

    ### 1. What clusters form?

    Most of the universities cluster toward the center of the graph. That tells me that, across the dataset, institutions are being described in relatively similar ways in terms of both prestige and research focus. There isn’t a strong horizontal separation, so the distinction between highly selective and less selective schools is not as pronounced in how they are represented.

    There is more separation vertically. Universities that are more research-intensive tend to sit higher, while more teaching-focused institutions fall lower. So the research vs. teaching axis creates more meaningful spread than the prestige axis.

    ---

    ### 2. Are there surprises?

    One thing that stands out is that not all prestigious universities are strongly aligned with the research-intensive side. Some of them sit closer to the center, which suggests their descriptions include both research and teaching elements rather than leaning heavily in one direction.

    Another observation is that less selective institutions are not pushed as far to the negative side as expected. This suggests there is overlap in how institutions are described, and the language used does not create a strict separation between categories.

    ---

    ### 3. What does the axis not capture?

    The axes do not capture regional differences well. Universities from different regions are mixed throughout the plot, which indicates that region is not a dominant semantic feature in how these institutions are described.

    Additionally, the axes do not reflect other important aspects such as campus environment, student experience, or specialization. These are meaningful differences between institutions, but they are not represented in this projection. This shows that while the axes provide structure, they only capture part of the overall system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ⚙ Back office

    Infrastructure cells. You do not need to read or modify these.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    from sentence_transformers import SentenceTransformer
    from drawdata import ScatterWidget

    return ScatterWidget, SentenceTransformer, alt, mo, np, pd


@app.function(hide_code=True)
def make_preset_clusters(n: int = 25, seed: int = 0):
    """Four 2-D Gaussian blobs used when the widget is empty.

    Colors match drawdata's first four pens:
      1. blue   (#1f77b4) — − pole
      2. red    (#d62728) — + pole
      3. green  (#2ca02c) — test points
      4. orange (#ff7f0e) — test points
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    clusters = [
        ("#1f77b4", [140, 260]),
        ("#d62728", [360, 140]),
        ("#2ca02c", [240, 320]),
        ("#ff7f0e", [300, 210]),
    ]

    xs, ys, cs = [], [], []
    for color, loc in clusters:
        pts = rng.normal(loc=loc, scale=[32, 28], size=(n, 2))
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
        cs.extend([color] * n)

    return pd.DataFrame({"x": xs, "y": ys, "color": cs})


@app.function(hide_code=True)
def plot_semaxis_2d(df):
    """Interactive SemAxis demo.

    df is expected to have columns x, y, color. The first two unique color
    values are treated as the negative and positive poles respectively; any
    additional colors are shown as "test" classes that get projected onto
    the same axis.

    Left panel: points + bold arrows from the origin to each pole
    centroid, plus the thick SemAxis arrow (e_+ - e_-).
    Right panel: per-class 1-D projection scores as a seaborn violin
    with an overlaid strip.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="talk", font_scale=0.85)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    colors = df["color"].unique().tolist() if len(df) else []
    if len(colors) < 2:
        for a in (ax1, ax2):
            a.text(
                0.5,
                0.5,
                "Draw points with at least two colors, or the preset will appear.",
                ha="center",
                va="center",
                transform=a.transAxes,
                color="#666",
            )
            a.set_axis_off()
        return fig

    neg_color, pos_color = colors[0], colors[1]
    neg = df.loc[df["color"] == neg_color, ["x", "y"]].to_numpy()
    pos = df.loc[df["color"] == pos_color, ["x", "y"]].to_numpy()
    pts = df[["x", "y"]].to_numpy()
    color_arr = df["color"].to_numpy()

    neg_c = neg.mean(axis=0)
    pos_c = pos.mean(axis=0)
    v = pos_c - neg_c
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-8:
        for a in (ax1, ax2):
            a.text(
                0.5,
                0.5,
                "Pole centroids coincide - move the two pole clusters apart.",
                ha="center",
                va="center",
                transform=a.transAxes,
                color="#666",
            )
            a.set_axis_off()
        return fig
    axis_unit = v / v_len

    # Projections measured from the ORIGIN (matches the real SemAxis algorithm).
    t = pts @ axis_unit

    # ---- Left: 2-D scene ----
    class_labels = [f"class {i + 1}" for i in range(len(colors))]
    df_plot = df.copy()
    df_plot["class"] = df_plot["color"].map(dict(zip(colors, class_labels)))
    palette = dict(zip(class_labels, colors))

    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="class",
        palette=palette,
        s=70,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        ax=ax1,
        legend=False,
        zorder=2,
    )

    # Thin guide lines from origin to each pole centroid.
    for center, c_rgb, lbl in [
        (neg_c, neg_color, r"$e_{-}$ (- pole centroid)"),
        (pos_c, pos_color, r"$e_{+}$ (+ pole centroid)"),
    ]:
        ax1.annotate(
            "",
            xy=center,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-",
                color=c_rgb,
                lw=1,
                alpha=0.6,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
        ax1.plot([], [], color=c_rgb, lw=1, alpha=0.6, label=lbl)

    # SemAxis arrow: e_+ - e_-.
    ax1.annotate(
        "",
        xy=pos_c,
        xytext=neg_c,
        arrowprops=dict(
            arrowstyle="-|>",
            color="#222",
            lw=2.5,
            mutation_scale=20,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )
    ax1.plot([], [], color="#222", lw=2.5, label=r"SemAxis: $e_{+} - e_{-}$")

    # Origin marker.
    ax1.scatter([0], [0], s=55, marker="x", color="#222", linewidths=2, zorder=5)
    ax1.annotate(
        "origin",
        xy=(0, 0),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=9,
        color="#555",
    )

    pad_x = max(30.0, 0.1 * (pts[:, 0].max() - pts[:, 0].min()))
    pad_y = max(30.0, 0.1 * (pts[:, 1].max() - pts[:, 1].min()))
    ax1.set_xlim(min(0.0, pts[:, 0].min()) - pad_x, max(0.0, pts[:, 0].max()) + pad_x)
    ax1.set_ylim(min(0.0, pts[:, 1].min()) - pad_y, max(0.0, pts[:, 1].max()) + pad_y)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Pole centroids are vectors; the SemAxis is their difference", pad=12)
    ax1.legend(loc="best", fontsize=9, frameon=False)
    sns.despine(ax=ax1)

    # ---- Right: 1-D projected scores per class ----
    proj_df = pd.DataFrame(
        {
            "projection": t,
            "class": pd.Categorical(
                [class_labels[colors.index(c)] for c in color_arr],
                categories=class_labels,
                ordered=True,
            ),
        }
    )

    sns.stripplot(
        data=proj_df,
        x="projection",
        y="class",
        hue="class",
        palette=palette,
        size=5,
        jitter=0.2,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=ax2,
        legend=False,
    )

    # Per-class mean as a short vertical tick.
    means = proj_df.groupby("class", observed=True)["projection"].mean()
    for i, lbl in enumerate(class_labels):
        if lbl in means.index:
            ax2.plot(
                [means[lbl], means[lbl]],
                [i - 0.28, i + 0.28],
                color="#222",
                lw=1.5,
                zorder=5,
            )

    ax2.set_xlabel(r"projection onto SemAxis $\rightarrow$")
    ax2.set_ylabel("")
    ax2.set_title("1-D projected scores per class", pad=12)
    sns.despine(ax=ax2, left=True)
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    app.run()

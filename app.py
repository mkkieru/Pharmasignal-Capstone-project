"""PharmaSignal - Streamlit app.

Run the notebook once (it creates outputs/), then:   streamlit run app.py
"""
import re

import altair as alt
import pandas as pd
import streamlit as st

import pharmasignal_core as core

st.set_page_config(page_title="PharmaSignal", page_icon="💊", layout="wide")

MIN_STREAMLIT = (1, 67)
if tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2]) < MIN_STREAMLIT or not hasattr(st, "navigation"):
    st.error(f"This app needs Streamlit {'.'.join(map(str, MIN_STREAMLIT))} or newer, but this computer has {st.__version__}.")
    st.markdown("Upgrade it in a terminal (on Windows: the **Anaconda Prompt**), then start the app again:\n\n"
                "```\npip install --upgrade streamlit\npython -m streamlit run app.py\n```")
    st.stop()

missing = core.missing_outputs()
if missing:
    st.error("The notebook outputs are missing: " + ", ".join(missing))
    st.info("Open **Drug_Recall_Intelligence_Final.ipynb** and choose **Restart & Run All** once. "
            "It creates the `outputs/` folder this app reads. Then refresh this page.")
    st.stop()


@st.cache_resource(show_spinner="Loading recall data and building the search index...")
def get_data():
    return core.load_data()


@st.cache_resource
def get_llm():
    return core.make_llm()


d = get_data()
llm = get_llm()
DISCLAIMER = ("Decision support only. Evidence comes from US FDA recall records "
              f"({d.source}); it never declares a medicine unsafe or recommends accepting or rejecting a supplier.")


# ============================================================ pages
def overview():
    st.title("💊 PharmaSignal: drug-quality recall intelligence")
    st.caption(DISCLAIMER)
    df = d.recalls
    years = sorted(int(y) for y in df.year.dropna().unique())
    c1, c2 = st.columns([2, 1])
    yr = c1.slider("Publication year", years[0], years[-1], (years[0], years[-1]))
    all_classes = sorted(df.classification.dropna().unique())
    classes = c2.multiselect("Recall class", all_classes, default=all_classes,
                             help="Class I = most serious (reasonable probability of serious harm).")
    f = df[df.year.between(*yr) & df.classification.isin(classes)]
    ev = f.drop_duplicates("event_id")

    m = st.columns(4)
    m[0].metric("Recall records", f"{len(f):,}")
    m[1].metric("Recall events", f"{ev.event_id.nunique():,}")
    m[2].metric("Recalling firms", f"{f.firm_std.nunique():,}")
    m[3].metric("Class I events", f"{ev[ev.classification == 'Class I'].event_id.nunique():,}")

    st.subheader("Recall events per year")
    per_year = ev.groupby(["year", "classification"]).size().reset_index(name="events")
    st.altair_chart(alt.Chart(per_year).mark_bar().encode(
        x=alt.X("year:O", title=None), y=alt.Y("events:Q", title="Recall events"),
        color=alt.Color("classification:N", title="Class", sort=["Class I", "Class II", "Class III"]),
        tooltip=["year", "classification", "events"]), width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Failure types")
        cats = ev.analysis_category.value_counts().reset_index()
        cats.columns = ["category", "events"]
        st.altair_chart(alt.Chart(cats).mark_bar().encode(
            x=alt.X("events:Q", title="Recall events"), y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category", "events"]), width="stretch")
    with b:
        st.subheader("Dosage forms")
        groups = ev.product_group.value_counts().reset_index()
        groups.columns = ["product_group", "events"]
        st.altair_chart(alt.Chart(groups).mark_bar().encode(
            x=alt.X("events:Q", title="Recall events"), y=alt.Y("product_group:N", sort="-x", title=None),
            tooltip=["product_group", "events"]), width="stretch")

    st.subheader("Top recalling firms")
    top = ev.groupby("firm_std").agg(events=("event_id", "nunique"),
                                     class_I=("classification", lambda s: int((s == "Class I").sum()))) \
        .sort_values("events", ascending=False).head(15).reset_index()
    st.dataframe(top, hide_index=True, width="stretch")

    with st.expander("Search recall records"):
        q = st.text_input("Product, firm or reason contains", placeholder="e.g. metformin, benzene, Teva")
        if q:
            hit = f[f.product_description.str.contains(q, case=False, na=False, regex=False)
                    | f.recalling_firm.str.contains(q, case=False, na=False, regex=False)
                    | f.reason_clean.str.contains(q, case=False, na=False, regex=False)]
            st.write(f"{len(hit):,} records")
            st.dataframe(hit.sort_values("report_date", ascending=False)[
                ["recall_number", "report_date", "classification", "recalling_firm", "analysis_category",
                 "product_group", "reason_clean", "product_description"]].head(500), hide_index=True, width="stretch")

    if len(d.evaluation):
        with st.expander("Project evaluation (from the notebook)"):
            st.dataframe(d.evaluation, hide_index=True, width="stretch")


def alerts_page():
    st.title("🚨 Anomaly alerts")
    st.caption("An alert means a month had unusually many recall events compared with the previous 12 months. "
               "It is a signal to investigate, **not** a confirmed quality failure.")
    al = d.alerts.copy()
    c1, c2, c3 = st.columns(3)
    window = c1.selectbox("Period", ["Last 6 months", "Last 12 months", "All history"], index=1)
    kinds = c2.multiselect("Series type", ["category", "product_group", "firm"], default=["category", "product_group", "firm"],
                           format_func={"category": "Failure type", "product_group": "Dosage form", "firm": "Firm"}.get)
    min_methods = c3.select_slider("Minimum methods agreeing", [1, 2, 3], value=1,
                                   help="z-score, IQR and Isolation Forest. 2+ = consensus alert.")
    if window != "All history":
        n = 6 if window == "Last 6 months" else 12
        al = al[al.month > str(d.last_month - n)]
    al = al[al.series_type.isin(kinds) & (al.n_methods >= min_methods)].sort_values(["month", "n_methods"], ascending=False)
    st.write(f"**{len(al)} alerts**, data up to {d.last_month}")
    st.dataframe(al[["month", "series_type", "series_name", "n_events", "hist_mean", "zscore", "n_methods", "why_flagged"]]
                 .rename(columns={"hist_mean": "12-month avg", "n_methods": "methods"}),
                 hide_index=True, width="stretch",
                 column_config={"12-month avg": st.column_config.NumberColumn(format="%.1f"),
                                "zscore": st.column_config.NumberColumn(format="%.1f")})

    st.subheader("Trend for one series")
    ms = d.monthly
    options = (al[["series_type", "series_name"]].drop_duplicates().apply(tuple, axis=1).tolist()
               or ms[ms.series_type != "firm"][["series_type", "series_name"]].drop_duplicates().apply(tuple, axis=1).tolist())
    pick = st.selectbox("Series", options, format_func=lambda t: f"{t[1]} ({t[0].replace('_', ' ')})")
    s = ms[(ms.series_type == pick[0]) & (ms.series_name == pick[1])].copy()
    s["date"] = pd.to_datetime(s.month)
    s["alert"] = s.n_methods.fillna(0) >= 1
    base = alt.Chart(s).encode(x=alt.X("date:T", title=None))
    chart = (base.mark_line().encode(y=alt.Y("n_events:Q", title="Recall events per month"),
                                     tooltip=["month", "n_events", alt.Tooltip("hist_mean:Q", format=".1f"), "n_methods"])
             + base.mark_line(strokeDash=[4, 4], color="gray").encode(y="hist_mean:Q")
             + base.transform_filter("datum.alert").mark_point(color="red", size=70, filled=True).encode(y="n_events:Q"))
    st.altair_chart(chart, width="stretch")
    st.caption("Line = monthly recall events; dashed = average of the previous 12 months; red dots = alert months.")

    if len(d.backtest):
        with st.expander("Back-test: would the detectors have caught 6 real crises?"):
            st.dataframe(d.method_summary, hide_index=True, width="stretch")
            st.dataframe(d.backtest, hide_index=True, width="stretch")


def tender_page():
    st.title("📋 Supplier / tender screening")
    st.caption(DISCLAIMER)
    c1, c2 = st.columns([2, 1])
    bidders_txt = c1.text_area("Bidders (one per line)",
                               "Sun Pharmaceutical Industries\nLupin Pharmaceuticals\nMarksans Pharma\nNairobi Generics Ltd", height=140)
    product = c2.text_input("Product being tendered", "metformin")
    run = c2.button("Screen bidders", type="primary", width="stretch")
    bidders = [b.strip() for b in bidders_txt.splitlines() if b.strip()]
    if run and bidders and product.strip():
        st.session_state.tender = core.tender_screen(d, bidders, product.strip())
    if "tender" in st.session_state:
        table, briefs = st.session_state.tender
        st.dataframe(table, hide_index=True, width="stretch")
        all_md = "\n\n---\n\n".join(briefs.values())
        st.download_button("Download all briefs (.md)", all_md, file_name="tender_briefs.md", mime="text/markdown")
        for bidder, md in briefs.items():
            with st.expander(f"Evidence brief: {bidder}"):
                st.markdown(md)
                st.download_button("Download this brief", md, key=f"dl-{bidder}",
                                   file_name=re.sub(r"[^a-z0-9]+", "_", bidder.lower()) + ".md", mime="text/markdown")

    st.divider()
    st.subheader("Watch list: firms with recent evidence flags")
    st.caption(f"Recent window: {d.recent_start} to {d.last_month}.")
    watch = d.fe[(d.fe.events_12m > 0) & (d.fe.n_flags > 0)].sort_values(["n_flags", "events_12m"], ascending=False)
    st.dataframe(watch[["firm_std", "events_12m", "class_I_events_12m", "prior_events_per_year", "alert_months_12m", "flags"]],
                 hide_index=True, width="stretch",
                 column_config={"prior_events_per_year": st.column_config.NumberColumn(format="%.1f")})


EXAMPLES = ["Have there been unusual increases in contamination-related recalls?",
            "Has Teva Pharmaceuticals USA had Class I recalls?",
            "What happened during the nitrosamine recalls of valsartan?",
            "What checks should KEMSA do before awarding a supply contract?",
            "What are the limitations of using FDA recall data for suppliers to Kenya?"]


def assistant_page():
    st.title("💬 Ask the assistant")
    if llm is None:
        st.warning("No Anthropic API key found, so answers show the evidence only. Add `ANTHROPIC_API_KEY=...` to a `.env` "
                   "file next to app.py and restart the app (see README).")
    else:
        st.success(f"Claude connected ({core.LLM_MODEL}). Answers use only the recall data and are fact-checked.")
    st.caption(DISCLAIMER)

    if "chat" not in st.session_state:
        st.session_state.chat = []
    cols = st.columns(len(EXAMPLES))
    clicked = next((q for c, q in zip(cols, EXAMPLES) if c.button(q, width="stretch")), None)
    question = st.chat_input("Ask about recalls, firms, products or alerts...") or clicked

    if question:
        with st.spinner("Searching the evidence" + (" and asking Claude..." if llm else "...")):
            st.session_state.chat.append((question, core.ask(d, question, llm)))
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()

    for q, res in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            for w in res["warnings"]:
                st.warning(w)
            if res["answer"]:
                st.markdown(res["answer"])
                (st.caption if res["verified"] else st.error)(res["status"])
            else:
                st.caption(res["status"])
                for title, tab in res["facts"]:
                    st.markdown(f"**{title}**")
                    st.dataframe(tab, hide_index=True, width="stretch")
                st.markdown("**Top evidence**")
                for e in res["evidence"].head(3).itertuples():
                    st.markdown(f"- `[{e.doc_id}]` {e.text[:300]}...")
            with st.expander("Sources and data used"):
                if res["answer"]:
                    for title, tab in res["facts"]:
                        st.markdown(f"**{title}**")
                        st.dataframe(tab, hide_index=True, width="stretch")
                st.dataframe(res["evidence"][["doc_id", "source", "text"]], hide_index=True, width="stretch")


pg = st.navigation([st.Page(overview, title="Overview", icon="📊", default=True),
                    st.Page(alerts_page, title="Anomaly alerts", icon="🚨", url_path="alerts"),
                    st.Page(tender_page, title="Tender screening", icon="📋", url_path="tender"),
                    st.Page(assistant_page, title="Ask the assistant", icon="💬", url_path="assistant")])
with st.sidebar:
    st.caption(f"Data: {d.source}\n\n{d.provenance['records']:,} records · {d.provenance['events']:,} events\n\n"
               f"Search: {d.embedding_method}\n\nAI: {'Claude ' + core.LLM_MODEL if llm else 'not connected (evidence only)'}")
pg.run()

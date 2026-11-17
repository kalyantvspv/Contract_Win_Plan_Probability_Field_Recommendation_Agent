"""
Aurevia Win-Plan Intelligence — interactive app for field reps
Run:  streamlit run app.py
"""
import os, re, json, time, math, pickle, difflib, hashlib, datetime
import numpy as np, pandas as pd, joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────── paths (auto-detect) ──────────────────
_cwd = os.getcwd(); _base = "."
for _p in [_cwd, os.path.dirname(_cwd), os.path.dirname(os.path.dirname(_cwd))]:
    if os.path.isdir(os.path.join(_p, "data", "raw")):
        _base = os.path.relpath(_p, _cwd); break
RAW   = os.path.join(_base, "data", "raw") + "/"
INTER = os.path.join(_base, "data", "intermediate") + "/"
PROC  = os.path.join(_base, "data", "processed") + "/"
MLFLOW_DB = "sqlite:///" + os.path.join(os.path.abspath(_base), "mlflow.db")

# ─────────────── config ───────────────────────────────
WINDOW_DAYS  = 365

def _secret(key, default=None):
    """Read from Streamlit secrets (cloud) or environment variables (local/Docker)."""
    try:
        if key in st.secrets: return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# LLM provider: any OpenAI-compatible endpoint. Change in secrets, no code edits needed.
PROVIDERS = {
    "anthropic":  ("https://api.anthropic.com/v1/",                         "claude-sonnet-5-5"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "groq":       ("https://api.groq.com/openai/v1",                            "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1",                              "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama":     ("http://localhost:11434/v1",                                 "llama3.1"),
}
LLM_PROVIDER = _secret("LLM_PROVIDER", "gemini").lower()
LLM_BASE_URL = _secret("LLM_BASE_URL", PROVIDERS.get(LLM_PROVIDER, PROVIDERS["gemini"])[0])
LLM_MODEL    = _secret("LLM_MODEL",    PROVIDERS.get(LLM_PROVIDER, PROVIDERS["gemini"])[1])
LLM_API_KEY  = _secret("LLM_API_KEY", "ollama" if LLM_PROVIDER == "ollama" else None)

# ─────────────── page setup ───────────────────────────
st.set_page_config(page_title="Aurevia Intelligence", page_icon="💉", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main > div { max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 1.6rem; }
.mc { background: var(--secondary-background-color, #f8f9fb); color: var(--text-color, #111827); border: 1px solid var(--border-color, #e2e6ea); border-radius: 8px; padding: 14px 18px; text-align: center; }
.mc .v { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.mc .l { font-size: 0.75rem; opacity: 0.72; margin-top: 4px; }
.t-ar { color: #dc2626; } .t-co { color: #d97706; } .t-lw { color: #059669; }
.ptag { display:inline-block; background:var(--secondary-background-color, #e0f2fe); color:var(--primary-color, #0369a1); font-size:0.72rem; font-weight:600; padding:2px 7px; border-radius:4px; margin-right:5px; }
.etag { display:inline-block; background:var(--secondary-background-color, #f3f4f6); color:var(--text-color, #6b7280); font-size:0.68rem; padding:1px 5px; border-radius:3px; margin-right:3px; font-family:monospace; }
</style>""", unsafe_allow_html=True)

TEMOJI = {"At risk": "🔴", "Contested": "🟡", "Likely win": "🟢"}
TCLS   = {"At risk": "t-ar", "Contested": "t-co", "Likely win": "t-lw"}
BLUE, GREY, RED = "#0b4a77", "#9aa5b1", "#c0392b"

def mcard(val, label, cls=""):
    return f'<div class="mc"><div class="v {cls}">{val}</div><div class="l">{label}</div></div>'

# ─────────────── optional password gate ───────────────
_pw = _secret("APP_PASSWORD")
if _pw and not st.session_state.get("authed"):
    st.markdown("# 💉 Aurevia Intelligence")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == _pw: st.session_state["authed"] = True; st.rerun()
        else: st.error("Incorrect password")
    st.stop()

# ══════════════════════ DATA ══════════════════════════
@st.cache_data(show_spinner="Loading data…")
def load_data():
    interactions = pd.read_csv(RAW + "interactions.csv", parse_dates=["interaction_date"]).drop_duplicates("interaction_id")
    interactions["duration_minutes"] = interactions["duration_minutes"].clip(lower=0)
    notes_raw = pd.read_csv(RAW + "interaction_notes.csv", parse_dates=["note_date"]).drop_duplicates("note_id")
    def mask(t):
        t = re.sub(r"\bDr\.\s+[A-Z][a-z]+", "[CONTACT]", str(t))
        return re.sub(r"\b(Michelle|Sarah)\b", "[CONTACT]", t).strip()
    notes = notes_raw.assign(text=notes_raw.note_text.fillna("").map(mask))
    notes = notes.merge(interactions[["interaction_id","interaction_date","interaction_type",
                        "product_focus","rep_id","issue_flag_raised"]], on="interaction_id", how="left")
    customers = pd.read_csv(RAW + "customers.csv")
    preds     = pd.read_csv(PROC + "predictions_2025_26.csv")
    mt        = pd.read_csv(PROC + "model_table.csv", parse_dates=["offer_date"])
    ch        = pd.read_csv(RAW + "contracts_history.csv", parse_dates=["offer_date","outcome_date"])
    model_sel = pd.read_csv(PROC + "model_selection.csv") if os.path.exists(PROC + "model_selection.csv") else None

    # ---- build deals table: merge carefully to avoid _x/_y collisions ----
    score_only = ["product_id","offer_date","comp_family","sim_comp_price","sim_list_price","is_incumbent",
                  "has_buying_group","relationship_years","region"]
    score = mt[mt.dataset=="score"][["opportunity_id","customer_id"] + score_only]
    cust_only = ["bed_count","annual_flu_doses_est","clinical_committee_influence","state"]
    cust = customers[["customer_id"] + cust_only].copy()

    deals = preds.merge(score, on=["opportunity_id","customer_id"], how="left")
    deals = deals.merge(cust, on="customer_id", how="left")
    deals = deals.set_index("opportunity_id", drop=False)

    rep_of = interactions.groupby("customer_id")["rep_id"].agg(lambda s: s.mode().iloc[0])
    deals["rep_id"] = deals["customer_id"].map(rep_of)
    deals["value_at_risk"] = (1 - deals.win_probability) * deals.annual_flu_doses_est * deals.sim_offer_price

    history = mt[mt.dataset=="history"].copy()
    history = history.merge(ch[["opportunity_id","outcome","final_negotiated_price_per_dose",
                  "num_negotiation_rounds","competitor_product"]], on="opportunity_id", how="left")
    history["won"] = history.target.astype(int)
    history["concession_pct"] = (1 - history.final_negotiated_price_per_dose / history.sim_offer_price) * 100
    as_of = interactions.interaction_date.max()
    model_art = joblib.load(PROC + "final_model.joblib") if os.path.exists(PROC + "final_model.joblib") else {}
    return interactions, notes, customers, deals, history, ch, as_of, model_sel, model_art

@st.cache_resource(show_spinner="Building note index…")
def build_index(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True).fit(texts)
    return vec, vec.transform(texts).toarray().astype("float32")

interactions, notes, customers, deals, history, contracts_hist, AS_OF, model_sel, model_art = load_data()
unique_texts = sorted(notes.text.unique()); text_to_row = {t:i for i,t in enumerate(unique_texts)}
notes["erow"] = notes.text.map(text_to_row)
tfidf_vec, tfidf_E = build_index(unique_texts)

# ══════════════════════ TOOLS ═════════════════════════
def jsonable(o):
    if isinstance(o, dict): return {str(k):jsonable(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float,np.floating)): return None if (math.isnan(o) or math.isinf(o)) else round(float(o),4)
    if isinstance(o, pd.Timestamp): return o.strftime("%Y-%m-%d")
    return o

def search_notes_fn(cid, query, pid, k=5):
    d = notes[(notes.customer_id==cid)&(notes.interaction_date<=AS_OF)
              &(notes.interaction_date>=AS_OF-pd.Timedelta(days=WINDOW_DAYS))
              &(notes.product_focus.isin(["PORTFOLIO",pid]))].copy()
    if d.empty: return []
    v = tfidf_vec.transform([query]).toarray()[0].astype("float32"); v /= (np.linalg.norm(v)+1e-9)
    d["score"] = tfidf_E[d.erow.values] @ v + 0.10*np.exp(-((AS_OF-d.interaction_date).dt.days.values)/90.0)
    d = d.sort_values("score",ascending=False).drop_duplicates("text").head(k)
    return [dict(note_id=r.note_id, date=f"{r.interaction_date:%Y-%m-%d}", channel=r.interaction_type,
                 product=r.product_focus, text=r.text, relevance=round(float(r.score),3)) for r in d.itertuples()]

def get_deal_brief(oid):
    d = deals.loc[oid]; wif = {k:d[f"p_{k}"] for k in ["-5.0%","-2.5%","+0.0%","+2.5%","+5.0%"]}
    return jsonable({"opportunity_id":oid,"customer":d.customer_name,"customer_type":d.customer_type,
        "region":d.region,"product":d.product_name,"incumbent_supplier":d.incumbent_supplier,
        "main_competitor_family":d.comp_family,"offer_date":d.offer_date,
        "our_price_per_dose":d.sim_offer_price,"competitor_price_per_dose":d.sim_comp_price,
        "price_vs_competitor_pct":d.price_vs_comp_pct,
        "account":{"beds":d.bed_count,"est_annual_doses":d.annual_flu_doses_est,
                    "relationship_years":d.relationship_years,"in_buying_group":bool(d.has_buying_group)},
        "model":{"win_probability":d.win_probability,"risk_tier":str(d.risk_tier),
            "drivers_for_win":d.drivers_for_win,"drivers_against_win":d.drivers_against_win,
            "what_if":wif,"gain_from_5pct_cut":d.gain_from_5pct_cut}})

def get_recent_interactions(oid, n=8):
    d = deals.loc[oid]
    x = interactions[(interactions.customer_id==d.customer_id)&(interactions.interaction_date<=AS_OF)
                     &(interactions.interaction_date>=AS_OF-pd.Timedelta(days=WINDOW_DAYS))].sort_values("interaction_date")
    if x.empty: return {"n_interactions_12m":0,"message":"No interactions on record."}
    return jsonable({"n_interactions_12m":len(x),
        "n_last_90d":len(x[x.interaction_date>AS_OF-pd.Timedelta(days=90)]),
        "days_since_last_contact":(AS_OF-x.interaction_date.max()).days,
        "issue_flags_12m":int(x.issue_flag_raised.sum()),
        "clinical_committee_12m":int((x.interaction_type=="clinical_committee_presentation").sum()),
        "latest":[dict(date=r.interaction_date,type=r.interaction_type,product=r.product_focus,
                       minutes=r.duration_minutes,issue_flag=bool(r.issue_flag_raised)) for r in x.tail(n).itertuples()]})

def search_account_notes(oid, query, k=5):
    d = deals.loc[oid]; hits = search_notes_fn(d.customer_id, query, d.product_id, int(k))
    return {"query":query,"n_results":len(hits),"notes":hits} if hits else {"query":query,"n_results":0,"message":"No matching notes."}

def get_account_history(oid):
    d = deals.loc[oid]; h = history[history.customer_id==d.customer_id].sort_values(["season","product_name"])
    if h.empty: return {"n_past_deals":0,"message":"New account."}
    rows = [dict(season=r.season,product=r.product_name,outcome=r.outcome,price_vs_comp_pct=r.price_vs_comp_pct,
                 incumbent_then=r.incumbent_supplier,competitor=r.competitor_product,
                 rounds=r.num_negotiation_rounds,concession_pct=(r.concession_pct if r.won else None)) for r in h.itertuples()]
    return jsonable({"n_past_deals":len(h),"n_won":int(h.won.sum()),"past_deals":rows[-8:]})

def get_portfolio_view(oid):
    d = deals.loc[oid]; o = deals[(deals.customer_id==d.customer_id)&(deals.opportunity_id!=oid)]
    return jsonable({"n_other_open_deals":len(o),
        "other_open_deals":[dict(opportunity_id=r.opportunity_id,product=r.product_name,
            win_probability=r.win_probability,risk_tier=str(r.risk_tier),
            price_vs_competitor_pct=r.price_vs_comp_pct,gain_from_5pct_cut=r.gain_from_5pct_cut) for r in o.itertuples()]})

def get_similar_past_deals(oid):
    d = deals.loc[oid]; h = history
    for label, cond in [
        ("same incumbency+type+product ±3pts",(h.is_incumbent==d.is_incumbent)&(h.customer_type==d.customer_type)&(h.product_id==d.product_id)),
        ("same incumbency+type ±3pts",(h.is_incumbent==d.is_incumbent)&(h.customer_type==d.customer_type)),
        ("same incumbency ±3pts",(h.is_incumbent==d.is_incumbent))]:
        s = h[cond&((h.price_vs_comp_pct-d.price_vs_comp_pct).abs()<=3)]
        if len(s)>=15: break
    won = s[s.won==1]
    return jsonable({"definition":label,"n_past_deals":len(s),"win_rate":s.target.mean() if len(s) else None,
        "overall_win_rate":h.target.mean(),
        "typical_when_won":{"avg_rounds":won.num_negotiation_rounds.mean(),"avg_concession_pct":won.concession_pct.mean()}})

TOOLS = {"get_deal_brief":get_deal_brief,"get_recent_interactions":get_recent_interactions,
         "search_account_notes":search_account_notes,"get_account_history":get_account_history,
         "get_portfolio_view":get_portfolio_view,"get_similar_past_deals":get_similar_past_deals}

# ══════════════════════ PLAYBOOK & AGENT ══════════════
PLAYBOOK = {
    "EXEC_SPONSOR":"Executive-to-executive outreach to the decision makers.",
    "PRICE_VALUE_DEFENCE":"Answer price pressure with a total-value case before moving price.",
    "TARGETED_CONCESSION":"Small, conditional price move tied to volume or multi-product commitment.",
    "CLINICAL_EDUCATION":"Clinical-committee session or data review to reinforce clinical preference.",
    "SERVICE_RECOVERY":"Resolve open service/supply/billing issues with named owner and dated commitments.",
    "PORTFOLIO_BUNDLE":"Offer a cross-product package instead of discounting a single product.",
    "COMPETITOR_COUNTER":"Factual comparison against the named competitor's claims or offers.",
    "ENGAGEMENT_BOOST":"Raise contact cadence and add stakeholders — multi-thread the account.",
    "EXPANSION_PLAY":"Account signals growth: propose expanded volume or additional sites.",
    "HOLD_AND_PROTECT":"Strong position: hold price, confirm commitments, avoid unnecessary concessions.",
}

SUBMIT_SCHEMA = {"type":"object","required":["headline","situation","risk_signals","recommended_actions",
    "price_guidance","talking_points","confidence"],"properties":{
    "headline":{"type":"string"},"situation":{"type":"string"},
    "risk_signals":{"type":"array","items":{"type":"object","required":["signal","evidence"],
        "properties":{"signal":{"type":"string"},"evidence":{"type":"array","items":{"type":"string"}}}}},
    "recommended_actions":{"type":"array","minItems":1,"maxItems":5,"items":{"type":"object",
        "required":["priority","play","action","why","evidence","timing"],"properties":{
        "priority":{"type":"integer"},"play":{"type":"string","enum":list(PLAYBOOK)},
        "action":{"type":"string"},"why":{"type":"string"},
        "evidence":{"type":"array","items":{"type":"string"}},"timing":{"type":"string"}}}},
    "price_guidance":{"type":"object","required":["stance","detail"],"properties":{
        "stance":{"type":"string","enum":["hold price","small concession may help","do not discount"]},
        "detail":{"type":"string"},"max_concession_pct":{"type":"number"}}},
    "talking_points":{"type":"array","items":{"type":"string"},"maxItems":4},
    "questions_to_ask":{"type":"array","items":{"type":"string"},"maxItems":3},
    "confidence":{"type":"string","enum":["low","medium","high"]},
    "data_gaps":{"type":"array","items":{"type":"string"}}}}

TOOL_CONFIG = {"tools":[
    {"toolSpec":{"name":"get_deal_brief","description":"Model win probability, drivers, price what-if.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"}},"required":["opportunity_id"]}}}},
    {"toolSpec":{"name":"get_recent_interactions","description":"Contact cadence, issue flags, latest interactions.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"}},"required":["opportunity_id"]}}}},
    {"toolSpec":{"name":"search_account_notes","description":"Semantic search over call notes. Use specific queries; call more than once.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"},"query":{"type":"string"},"k":{"type":"integer"}},"required":["opportunity_id","query"]}}}},
    {"toolSpec":{"name":"get_account_history","description":"Past deals: outcome, price, rounds, concession.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"}},"required":["opportunity_id"]}}}},
    {"toolSpec":{"name":"get_portfolio_view","description":"Other open 2025-26 deals at the same account.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"}},"required":["opportunity_id"]}}}},
    {"toolSpec":{"name":"get_similar_past_deals","description":"Win rate and negotiation stats of comparable past deals.",
        "inputSchema":{"json":{"type":"object","properties":{"opportunity_id":{"type":"string"}},"required":["opportunity_id"]}}}},
    {"toolSpec":{"name":"submit_recommendation","description":"Submit the final win plan.",
        "inputSchema":{"json":SUBMIT_SCHEMA}}}]}

SYSTEM_PROMPT = (
    "You are a commercial coach for Aurevia Pharmaceutical field reps "
    "(flu vaccines: Vaxora HD, Vaxora Standard, Immunexa).\n"
    "Goal: give the rep a short, specific plan that raises the chance of winning "
    "this contract without giving away margin unnecessarily.\n\n"
    "1. Call get_deal_brief, get_recent_interactions, get_account_history, "
    "get_portfolio_view and get_similar_past_deals first, then search_account_notes "
    "at least twice on DIFFERENT topics.\n"
    "2. Use ONLY facts returned by the tools. Cite evidence: note ids, or "
    "'model', 'interactions', 'similar_deals', 'account_history', 'portfolio'.\n"
    "3. Choose actions only from this playbook:\n"
    + "\n".join(f"   - {k}: {v}" for k,v in PLAYBOOK.items()) + "\n"
    "4. Price guidance is advisory: never outside -5%..+5%. "
    "Prefer non-price levers when the account is not price-responsive.\n"
    "5. Account history is CONTEXT, not a forecast. The model probability is an "
    "offer-time estimate; newer notes can override it.\n"
    "6. Be concise: at most 5 actions. Finish by calling submit_recommendation."
)

# Same tool specs as before, converted to the OpenAI function-calling format
OA_TOOLS = [{"type":"function","function":{"name":t["toolSpec"]["name"],
             "description":t["toolSpec"]["description"],
             "parameters":t["toolSpec"]["inputSchema"]["json"]}} for t in TOOL_CONFIG["tools"]]

@st.cache_resource(show_spinner=False)
def get_llm():
    from openai import OpenAI
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set. Add it in Streamlit secrets / environment variables.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120, max_retries=5)

def run_agent(oid, user_question=None, status_container=None, max_steps=10):
    """Tool-use loop: the LLM calls tools, we execute them, repeat until submit_recommendation."""
    client = get_llm()
    focus = f"\nThe rep's question: {user_question}" if user_question else ""
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":f"Prepare the win plan for opportunity_id={oid}.{focus}"}]
    seen_ids, trace = set(), []
    for step in range(max_steps):
        if status_container:
            status_container.update(label=f"Agent thinking… (step {step+1})", state="running")
        resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, tools=OA_TOOLS,
                                              tool_choice="auto", max_tokens=3000)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        calls = msg.tool_calls or []
        if not calls:
            messages.append({"role":"user","content":"Please call submit_recommendation now."})
            continue
        final = None
        for tc in calls:
            name = tc.function.name
            try: args = json.loads(tc.function.arguments or "{}")
            except Exception: args = {}
            if name == "submit_recommendation":
                final = args
                messages.append({"role":"tool","tool_call_id":tc.id,"content":"received"}); continue
            try:
                call_args = dict(args); call_args.pop("opportunity_id", None)
                out = jsonable(TOOLS[name](oid, **call_args)); status = "success"
                seen_ids |= {n["note_id"] for n in out.get("notes",[])}
                if status_container:
                    q = call_args.get("query","")
                    status_container.update(label=f"Called {name}" + (f": {q[:40]}" if q else ""), state="running")
            except Exception as e:
                out, status = {"error":str(e)}, "error"
            trace.append((name, {k:v for k,v in args.items() if k!="opportunity_id"}, status))
            messages.append({"role":"tool","tool_call_id":tc.id,"content":json.dumps(out)})
        if final is not None:
            issues = [f"Unknown play: {a.get('play')}" for a in final.get("recommended_actions",[]) if a.get("play") not in PLAYBOOK]
            return {"opportunity_id":oid,"plan":final,"qa_issues":issues,"trace":trace,"notes_seen":sorted(seen_ids)}
    return {"opportunity_id":oid,"plan":None,"qa_issues":["Agent did not finish"],"trace":[],"notes_seen":[]}

# ══════════════════════ MLFLOW ════════════════════════
def log_model_to_mlflow():
    import mlflow, mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment("aurevia-response-model")
    art = model_art
    if not art: return None
    existing = mlflow.search_runs(experiment_names=["aurevia-response-model"],
                                  filter_string=f"tags.model_version = '{art.get('model_version','')}'")
    if len(existing): return existing.iloc[0].run_id
    with mlflow.start_run(run_name=art.get("model_version","run")) as run:
        mlflow.set_tag("model_version", art.get("model_version",""))
        mlflow.set_tag("data_version", art.get("data_version",""))
        mlflow.log_param("n_features", len(art.get("features",[])))
        mlflow.log_param("features", json.dumps(art.get("features",[])))
        mlflow.log_param("cutoff", art.get("cutoff"))
        mlflow.log_param("tier_bounds", str(art.get("tier_bounds")))
        if model_sel is not None:
            for _, r in model_sel.iterrows():
                tag = r["model"].replace(" ","_").replace("(","").replace(")","")
                mlflow.log_metric(f"cv_auc_{tag}", r["auc"])
                mlflow.log_metric(f"cv_prauc_{tag}", r["pr_auc"])
            best = model_sel[model_sel.model != "baseline (prior rate)"].iloc[0]
            mlflow.log_metric("best_cv_auc", best.auc)
        try: mlflow.sklearn.log_model(art["model"], "model")
        except Exception: pass
        mlflow.log_dict({"features":art.get("features",[]),"model_version":art.get("model_version"),
                         "data_version":art.get("data_version")}, "metadata.json")
    return run.info.run_id

# ══════════════════════ RENDER PLAN ═══════════════════
def render_plan(result):
    r = result["plan"]
    st.markdown(f"**{r.get('headline','')}**")
    st.markdown(r.get("situation",""))
    if r.get("risk_signals"):
        st.markdown("##### Risk signals")
        for s in r["risk_signals"]:
            ev = " ".join(f'<span class="etag">{e}</span>' for e in s.get("evidence",[]))
            st.markdown(f"- {s['signal']} {ev}", unsafe_allow_html=True)
    st.markdown("##### Recommended actions")
    for a in sorted(r.get("recommended_actions",[]), key=lambda x:x.get("priority",99)):
        ev = " ".join(f'<span class="etag">{e}</span>' for e in a.get("evidence",[]))
        st.markdown(
            f'**{a.get("priority","")}.** <span class="ptag">{a.get("play","").replace("_"," ")}</span> '
            f'{a.get("action","")}\n\n'
            f'*Why:* {a.get("why","")} · *When:* {a.get("timing","")} {ev}',
            unsafe_allow_html=True)
    pg = r.get("price_guidance",{})
    mc = pg.get("max_concession_pct")
    st.info(f"**{pg.get('stance','').capitalize()}**"
            + (f" · up to {mc:.1f}%" if mc else "")
            + f"  \n{pg.get('detail','')}", icon="💰")
    c1, c2 = st.columns(2)
    with c1:
        if r.get("talking_points"):
            st.markdown("##### Talking points")
            for t in r["talking_points"]: st.markdown(f"- {t}")
    with c2:
        if r.get("questions_to_ask"):
            st.markdown("##### Questions to ask")
            for q in r["questions_to_ask"]: st.markdown(f"- {q}")
    conf = r.get("confidence","")
    gaps = r.get("data_gaps",[])
    st.caption(f"Confidence: {conf}" + (f" · Data gaps: {', '.join(gaps)}" if gaps else ""))
    if result.get("qa_issues"):
        st.warning("QA flags: " + "; ".join(result["qa_issues"]))
    with st.expander(f"Notes reviewed ({len(result.get('notes_seen',[]))})"):
        seen = notes[notes.note_id.isin(result.get("notes_seen",[]))].sort_values("interaction_date")
        if seen.empty: st.caption("No notes retrieved.")
        else:
            for r_ in seen.itertuples():
                st.markdown(f"`{r_.note_id}` {r_.interaction_date:%Y-%m-%d}: {r_.text}")
    with st.expander("Agent tool trace"):
        for name, args, status in result.get("trace",[]):
            q = args.get("query","")
            st.code(f"{name}({q})" if q else name, language=None)

# ══════════════════════ PAGES ═════════════════════════
def _pool():
    rep = st.session_state.get("rep_id")
    return deals[deals.rep_id==rep] if rep else deals

# ────────── PORTFOLIO ──────────
def page_portfolio():
    st.markdown("# 📊 My Portfolio")
    pool = _pool()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Open deals", len(pool))
    c2.metric("Avg win prob", f"{pool.win_probability.mean():.0%}")
    c3.metric("At risk", int((pool.risk_tier=='At risk').sum()))
    c4.metric("Value at risk", f"${pool.value_at_risk.sum():,.0f}")

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("##### Tier breakdown")
        for tier in ["At risk","Contested","Likely win"]:
            n = (pool.risk_tier==tier).sum()
            st.markdown(f"{TEMOJI.get(tier,'')} **{tier}:** {n} deals ({n/max(len(pool),1):.0%})")
    with col2:
        fig, ax = plt.subplots(figsize=(6,2.8))
        for tier, color in [("At risk","#dc2626"),("Contested","#d97706"),("Likely win","#059669")]:
            sub = pool[pool.risk_tier==tier]
            if len(sub):
                ax.scatter(sub.price_vs_comp_pct, sub.win_probability*100, c=color,
                           s=sub.annual_flu_doses_est/200, alpha=0.6, label=f"{tier} ({len(sub)})",
                           edgecolors="white", linewidth=0.3)
        ax.set(xlabel="Price vs competitor %", ylabel="Win probability %",
               title="Price position vs win probability (size = volume)")
        ax.legend(fontsize=7, loc="upper right")
        ax.axhline(50,color="#e5e7eb",lw=0.8); ax.axvline(0,color="#e5e7eb",lw=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("##### All deals")
    view_cols = ["customer_name","product_name","win_probability","risk_tier","price_vs_comp_pct",
                 "gain_from_5pct_cut","recommendation","incumbent_supplier","customer_type"]
    sort = st.selectbox("Sort by", ["value_at_risk","win_probability","price_vs_comp_pct","customer_name"])
    tbl = pool.sort_values(sort, ascending=(sort=="customer_name"))[view_cols].round(3)
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.markdown("##### Jump to a deal")
    opp_list = pool.sort_values("value_at_risk", ascending=False)
    labels = [f"{r.customer_name} — {r.product_name} ({r.win_probability:.0%})" for _,r in opp_list.iterrows()]
    choice = st.selectbox("Select deal", labels, index=0, key="port_jump")
    idx = labels.index(choice)
    if st.button("Open deal detail →", key="port_go"):
        st.session_state["selected_opp"] = opp_list.iloc[idx].opportunity_id
        st.session_state["page"] = "Deal Detail"; st.rerun()

# ────────── CUSTOMER VIEW ──────────
def page_customer():
    st.markdown("# 🏥 Customer View")
    pool = _pool()
    cust_names = sorted(pool.customer_name.unique())
    selected_cust = st.selectbox("Select customer", cust_names, index=0)
    cust_deals = pool[pool.customer_name==selected_cust]
    d0 = cust_deals.iloc[0]

    parts = [str(d0.customer_type), str(d0.region)]
    if pd.notna(d0.get("state")): parts.append(str(d0.state))
    parts += [f"{int(d0.bed_count):,} beds", f"{int(d0.annual_flu_doses_est):,} est. doses",
              f"Relationship: {int(d0.relationship_years)} yrs",
              "In buying group" if d0.has_buying_group else "Not in buying group"]
    if pd.notna(d0.get("clinical_committee_influence")):
        parts.append(f"Clinical committee: {d0.clinical_committee_influence}")
    st.caption(" · ".join(parts))

    st.markdown("##### Open 2025-26 deals")
    for _, r in cust_deals.sort_values("win_probability").iterrows():
        tier = str(r.risk_tier); emoji = TEMOJI.get(tier,"")
        c1,c2,c3,c4,c5 = st.columns([3,2,2,2,2])
        c1.markdown(f"**{r.product_name}** {emoji} {tier}")
        c2.markdown(f"Win: **{r.win_probability:.0%}**")
        c3.markdown(f"Price gap: **{r.price_vs_comp_pct:+.1f}%**")
        c4.markdown(f"5% cut: **{r.gain_from_5pct_cut:+.1%}**")
        if c5.button("View →", key=f"cv_{r.opportunity_id}"):
            st.session_state["selected_opp"] = r.opportunity_id
            st.session_state["page"] = "Deal Detail"; st.rerun()

    # account history chart
    cid = d0.customer_id
    h = history[history.customer_id==cid].sort_values(["season","product_name"])
    if len(h):
        st.markdown("##### Account history")
        fig, ax = plt.subplots(figsize=(7,2.5))
        for pid in h.product_name.unique():
            g = h[h.product_name==pid]
            marker = "^" if "HD" in pid else ("s" if "Imm" in pid else "o")
            colors = ["#059669" if w else "#dc2626" for w in g.won]
            ax.scatter(g.season, g.price_vs_comp_pct, c=colors, marker=marker, s=60, label=pid,
                       edgecolors="white", linewidth=0.5)
        ax.set(ylabel="Price vs competitor %", title="Past outcomes (green=won, red=lost)")
        ax.axhline(0,color="#e5e7eb",lw=0.8); ax.legend(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(h[["season","product_name","outcome","price_vs_comp_pct","incumbent_supplier",
                         "competitor_product","num_negotiation_rounds","concession_pct"]].round(1),
                      hide_index=True, use_container_width=True)

    # recent interactions
    ix = interactions[(interactions.customer_id==cid)
                      &(interactions.interaction_date>=AS_OF-pd.Timedelta(days=WINDOW_DAYS))].sort_values("interaction_date",ascending=False)
    if len(ix):
        st.markdown(f"##### Recent interactions ({len(ix)} in last 12 months)")
        st.dataframe(ix[["interaction_date","interaction_type","product_focus","duration_minutes",
                          "issue_flag_raised","rep_id"]].head(15), hide_index=True, use_container_width=True)

# ────────── DEAL DETAIL ──────────
def page_deal_detail():
    oid = st.session_state.get("selected_opp")
    if not oid or oid not in deals.index:
        st.info("Select a deal from the Portfolio or Customer page.")
        return
    d = deals.loc[oid]; tier = str(d.risk_tier)
    st.markdown(f"# {d.customer_name} — {d.product_name}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(mcard(f"{d.win_probability:.0%}", "Win probability"), unsafe_allow_html=True)
    c2.markdown(mcard(f"{TEMOJI.get(tier,'')} {tier}", "Risk tier", TCLS.get(tier,"")), unsafe_allow_html=True)
    c3.markdown(mcard(f"{d.price_vs_comp_pct:+.1f}%", "vs competitor"), unsafe_allow_html=True)
    c4.markdown(mcard(f"{d.gain_from_5pct_cut:+.1%}", "5% cut gain"), unsafe_allow_html=True)
    c5.markdown(mcard(f"${d.sim_offer_price:.2f}", "Offer $/dose"), unsafe_allow_html=True)
    st.caption(f"{d.customer_type} · {d.region} · Incumbent: {d.incumbent_supplier} · "
               f"Competitor: {d.comp_family} · Rep: {d.rep_id} · Offer: {d.offer_date}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price What-If","📋 Drivers & History","📝 Notes Search","🤖 AI Win Plan"])

    with tab1:
        moves = [-5.0,-2.5,0.0,2.5,5.0]
        probs = [d[f"p_{m:+.1f}%"] for m in moves]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(moves, [p*100 for p in probs], "o-", color=BLUE, linewidth=2, markersize=8)
        ax.axhline(d.win_probability*100, color=GREY, ls="--", lw=0.8)
        ax.set(xlabel="Price change %", ylabel="Win probability %", title="What-if: win probability by price change")
        for m, p in zip(moves, probs):
            ax.annotate(f"{p:.0%}", (m, p*100), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Historical association only — a person approves every price.")
        sim = get_similar_past_deals(oid)
        st.info(f"**Similar past deals** ({sim['definition']}): {sim['n_past_deals']} deals, "
                f"**{sim['win_rate']:.0%}** win rate. Won deals took ~{sim['typical_when_won']['avg_rounds']:.1f} rounds, "
                f"~{sim['typical_when_won']['avg_concession_pct']:.1f}% concession.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Drivers **for** a win"); st.success(d.drivers_for_win or "none identified")
            st.markdown("##### Drivers **against** a win"); st.error(d.drivers_against_win or "none identified")
        with c2:
            ah = get_account_history(oid)
            if ah.get("past_deals"):
                st.markdown(f"##### Account history ({ah['n_past_deals']} deals, {ah['n_won']} won)")
                st.dataframe(pd.DataFrame(ah["past_deals"]).round(1), hide_index=True, use_container_width=True)
            port = get_portfolio_view(oid)
            if port["other_open_deals"]:
                st.markdown("##### Other open deals")
                st.dataframe(pd.DataFrame(port["other_open_deals"]).round(3), hide_index=True, use_container_width=True)

    with tab3:
        query = st.text_input("Search this account's call notes", value="competitor price switching", key=f"ns_{oid}")
        if query:
            hits = search_notes_fn(d.customer_id, query, d.product_id, k=8)
            if hits:
                for h in hits:
                    st.markdown(f"`{h['note_id']}` {h['date']} ({h['channel']}, {h['product']}): {h['text']}  \n"
                                f"*relevance: {h['relevance']:.2f}*")
            else:
                st.caption("No matching notes.")

    with tab4:
        extra_q = st.text_input("Optional question for the AI coach",
                                placeholder="e.g. How do I win without cutting the price?", key=f"aq_{oid}")
        if st.button("🤖 Generate AI win plan", key=f"gen_{oid}", type="primary"):
            with st.status("Agent working…", expanded=True) as status:
                try:
                    result = run_agent(oid, user_question=extra_q or None, status_container=status)
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Agent failed: {type(e).__name__}: {e}")
                    result = None
            if result and result.get("plan"):
                render_plan(result)
            elif result:
                st.warning("Agent did not produce a plan. " + "; ".join(result.get("qa_issues",[])))

# ────────── MODEL REGISTRY ──────────
def page_model_registry():
    st.markdown("# 🧪 Model Registry (MLflow)")
    try:
        import mlflow
    except ImportError:
        mlflow = None
    if mlflow is None:
        st.info("MLflow is not installed in this deployment, showing saved model metadata only.")
        if not model_art: st.warning("No model artifacts found. Run Notebook 2 first."); return
        c1,c2,c3 = st.columns(3)
        c1.metric("Model version", model_art.get("model_version",""))
        c2.metric("Data version", model_art.get("data_version",""))
        c3.metric("Features", len(model_art.get("features",[])))
        if model_sel is not None: st.dataframe(model_sel.round(3), hide_index=True, use_container_width=True)
        st.code(", ".join(model_art.get("features",[])))
        return
    with st.spinner("Logging model to MLflow…"):
        run_id = log_model_to_mlflow()
    if not run_id:
        st.warning("No model artifacts found. Run Notebook 2 first."); return
    mlflow.set_tracking_uri(MLFLOW_DB)
    run = mlflow.get_run(run_id)

    c1,c2,c3 = st.columns(3)
    c1.metric("Model version", run.data.tags.get("model_version",""))
    c2.metric("Data version", run.data.tags.get("data_version",""))
    c3.metric("Features", run.data.params.get("n_features",""))

    if model_sel is not None:
        st.markdown("##### Cross-validated AUC by candidate")
        fig, ax = plt.subplots(figsize=(7,3.5))
        s = model_sel.sort_values("auc")
        best_name = s.iloc[-1].model
        colors = [RED if m==best_name else BLUE for m in s.model]
        ax.barh(s.model, s.auc, color=colors)
        ax.set(xlabel="Grouped-CV ROC-AUC", title="Model comparison (red = selected)")
        ax.set_xlim(0.45, max(0.75, s.auc.max()+0.02))
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(model_sel.round(3), hide_index=True, use_container_width=True)

    st.markdown("##### Parameters")
    st.json({k:v for k,v in run.data.params.items() if k!="features"})
    st.markdown("##### Features")
    feats = json.loads(run.data.params.get("features","[]"))
    st.code(", ".join(feats))

    st.markdown("##### Scored output summary (2025-26)")
    fig, axes = plt.subplots(1,3, figsize=(12,3))
    axes[0].hist(deals.win_probability, bins=20, color=BLUE, edgecolor="white")
    axes[0].set(title="Win probability distribution", xlabel="Probability", ylabel="Count")
    for tier, color in [("At risk","#dc2626"),("Contested","#d97706"),("Likely win","#059669")]:
        axes[1].bar(tier, (deals.risk_tier==tier).sum(), color=color)
    axes[1].set(title="Risk tier counts")
    deals.groupby("product_name")["win_probability"].mean().plot.bar(ax=axes[2], color=BLUE)
    axes[2].set(title="Avg win prob by product", ylabel="Probability")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(f"Run ID: `{run_id}` · Tracking: `{MLFLOW_DB}`")
    st.caption("Full MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open proxy on port 5000.")

# ══════════════════════ SIDEBAR ═══════════════════════
with st.sidebar:
    st.markdown("## 💉 Aurevia Intelligence")
    st.caption(f"Data as of {AS_OF:%Y-%m-%d} · {len(deals)} open deals")

    reps = sorted(deals.rep_id.dropna().unique())
    rep_choice = st.selectbox("I am", ["All reps"] + list(reps), index=0)
    st.session_state["rep_id"] = None if rep_choice=="All reps" else rep_choice

    st.divider()
    pages = ["Portfolio","Customer View","Deal Detail","Model Registry"]
    icons = ["📊","🏥","📋","🧪"]
    for p, icon in zip(pages, icons):
        is_current = st.session_state.get("page")==p
        if st.button(f"{icon}  {p}", key=f"nav_{p}", use_container_width=True,
                     type="primary" if is_current else "secondary"):
            st.session_state["page"] = p; st.rerun()

    if st.session_state.get("rep_id"):
        st.divider()
        st.markdown("##### ⚠️ Top at-risk deals")
        pool = deals[deals.rep_id==st.session_state["rep_id"]]
        at_risk = pool[pool.risk_tier=="At risk"].sort_values("value_at_risk",ascending=False).head(5)
        for _, r in at_risk.iterrows():
            if st.button(f"🔴 {r.customer_name} — {r.product_name} ({r.win_probability:.0%})",
                         key=f"ar_{r.opportunity_id}", use_container_width=True):
                st.session_state["selected_opp"] = r.opportunity_id
                st.session_state["page"] = "Deal Detail"; st.rerun()

# ══════════════════════ ROUTER ════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "Portfolio"
{"Portfolio":page_portfolio, "Customer View":page_customer,
 "Deal Detail":page_deal_detail, "Model Registry":page_model_registry}[st.session_state["page"]]()
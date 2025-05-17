import streamlit as st
import pandas as pd
import gspread
import json
import re
from google.oauth2.service_account import Credentials
from datetime import datetime
import random

# ———————————————
# 1) Google‐Sheets writer (unchanged)
# ———————————————
def write_to_gsheet(data):
    creds_dict = st.secrets["gcp_service_account"]
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1rKyqOc9OULQaeAFtbqaCumpwM1xlzhxxeQYikeHaSoc").sheet1

    header = list(data[0].keys())
    if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
        sheet.append_row(header)

    for row in data:
        sheet.append_row([row.get(col, "") for col in header])


# ———————————————
# 2) Load full DF
# ———————————————
@st.cache_data
def load_data(path="edit_df.csv"):
    return pd.read_csv(path).sample(frac=1, random_state=42).reset_index(drop=True)


# ———————————————
# 3) Build samples carrying every column
# ———————————————
def prepare_evaluation_samples(df, n_samples=30):
    # keep all original columns, plus row_number & sample_uid
    df2 = df.copy().reset_index().rename(columns={"index":"row_number"})
    df2["sample_uid"] = df2["row_number"].apply(lambda i: f"item_{i}")
    # shuffle & take top-n
    sampled = df2.sample(n=n_samples, random_state=42).reset_index(drop=True)
    return sampled.to_dict(orient="records")


# ———————————————
# 4) Streamlit app flow
# ———————————————
st.title("Edit-Alignment Human Evaluation")

if "samples" not in st.session_state:
    df = load_data()
    st.session_state.samples = prepare_evaluation_samples(df)
    st.session_state.current_idx = 0
    st.session_state.responses = {}
    st.session_state.submitted = False

samples = st.session_state.samples
idx = st.session_state.current_idx
sample = samples[idx]
is_last = idx == len(samples) - 1

# collect user ID
if "user_id" not in st.session_state:
    st.markdown("### Please enter a nickname to begin:")
    uid = st.text_input("", key="user_id_input")
    if st.button("Continue"):
        if uid.strip():
            st.session_state.user_id = uid.strip()
            st.rerun()
        else:
            st.warning("Nickname cannot be empty.")
    st.stop()

st.markdown(f"**User:** `{st.session_state.user_id}`")
st.markdown("---")

if not is_last:
    st.markdown(f"### Item {idx+1} of {len(samples)}")

    # 1) Edit prompt
    st.markdown("**1) Edit Prompt**")
    prompt_html = f"""
    <div style="
        background-color: #f0f8ff;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #cce5ff;
        ">
    {sample["edit_prompt"]}
    </div>
    """
    st.markdown(prompt_html, unsafe_allow_html=True)

    # 2) Annotator edits (diff-style)
    st.markdown("**2) Annotator’s Ground-Truth Edits**")
    with st.expander("Show Annotator Edits", expanded=False):
        st.code(sample["annotator_edits"], language="diff")

    # 3) Model prediction (diff-style)
    st.markdown("**3) Model’s Original Prediction**")
    with st.expander("Show Model Prediction", expanded=False):
        st.code(sample["model_prediction"], language="diff")

    # 4) GPT-4 output (parsed JSON)
    st.markdown("**4) GPT-4’s Full Output**")
    raw = sample["gpt4o_full_output"]

    # 1) try to extract the JSON payload between ```json … ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    json_str = m.group(1) if m else raw


    # 1) extract JSON between ```json … ``` if present
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    json_str = m.group(1) if m else raw

    # 2) parse & display
    try:
        parsed = json.loads(json_str)
        rating = int(parsed.get("rating", 0))
        rationale = parsed.get("rationale", "").strip()

        # 3) map rating 1–5 into t∈[0,1]
        t = max(0, min((rating - 1) / 4, 1))
        r = int(255 * (1 - t))
        g = int(255 * t)
        hex_color = f"#{r:02x}{g:02x}00"

        # 4) show the colored rating
        st.markdown(
            f"<div style='font-size:28px; font-weight:bold; color:{hex_color};'>"
            f"🔢 GPT-4 Rating: {rating}"
            "</div>",
            unsafe_allow_html=True
        )

        # 5) show the rationale
        st.markdown("**💡 GPT-4 Rationale:**")
        rationale_html = f"""
        <div style="
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #ffeeba;
            ">
        {rationale}
        </div>
        """
        st.markdown(rationale_html, unsafe_allow_html=True)

    except json.JSONDecodeError:
        # fallback: raw code block if parsing fails
        st.warning("⚠️ Could not parse GPT-4 output as JSON. Showing raw:")
        st.code(raw, language="json")
    st.markdown("### Does GPT-4’s output correctly reflect how much the model applied the edit?")

    # judgment buttons
    cols = st.columns(3)
    for choice, col in zip(["Yes", "No", "Maybe"], cols):
        if col.button(choice, key=f"btn_{sample['sample_uid']}_{choice}"):
            entry = sample.copy()  # includes all original columns
            entry.update({
                "user_id":       st.session_state.user_id,
                "user_judgment": choice,
                "timestamp":     datetime.utcnow().isoformat(),
            })
            st.session_state.responses[sample["sample_uid"]] = entry
            st.session_state.current_idx += 1
            st.rerun()

    # navigation
    back, skip, restart = st.columns([1,1,1])
    with back:
        if idx > 0 and st.button("◀️ Back"):
            st.session_state.current_idx -= 1
            st.rerun()
    with skip:
        if st.button("⏭️ Skip"):
            st.session_state.current_idx += 1
            st.rerun()
    with restart:
        if st.button("🔄 Restart"):
            df = load_data()
            st.session_state.samples = prepare_evaluation_samples(df)
            st.session_state.current_idx = 0
            st.session_state.responses = {}
            st.session_state.submitted = False
            st.rerun()

    st.progress((idx+1)/len(samples), text=f"{idx+1}/{len(samples)}")

else:
    st.markdown("## Final Step: Submit Your Judgments")
    if not st.session_state.submitted:
        if st.button("✅ Submit All to Google Sheets"):
            with st.spinner("Submitting..."):
                df_out = pd.DataFrame(st.session_state.responses.values())
                write_to_gsheet(df_out.to_dict(orient="records"))
                st.session_state.submitted = True
                st.success("All done! You may now close this tab.")
    else:
        st.success("✅ Already submitted. Thank you!")
        st.stop()

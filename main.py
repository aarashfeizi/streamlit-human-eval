import os
import streamlit as st
import pandas as pd
import gspread
from PIL import Image
import requests
import json
import re
from google.oauth2.service_account import Credentials
from datetime import datetime
import random
from io import BytesIO

st.cache_data.clear()
N_SAMPLES = 10

if st.session_state.get("submitted", False):
    for k in ["samples", "current_idx", "responses", "submitted"]:
        st.session_state.pop(k, None)
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
    header = [h for h in header if h not in ['reference', 'model_prediction_code']]
    print(header)
    print(data)
    print(data[0])
    print(data[0].values())
    if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
        sheet.append_row(header)

    for row in data:
        sheet.append_row([row.get(col, "") for col in header])


# ———————————————
# 2) Load full DF
# ———————————————
@st.cache_data
def load_data(path="edit_df.csv"):
    return pd.read_csv(path)


# ———————————————
# 3) Build samples carrying every column
# ———————————————
def prepare_evaluation_samples(df, n_samples=30):
    df2 = df.copy().reset_index().rename(columns={"index":"row_number"})
    df2["sample_uid"] = df2["row_number"].apply(lambda i: f"item_{i}")
    # Now draw n_samples at random (no fixed seed):
    sampled = df2.sample(n=n_samples).reset_index(drop=True)
    return sampled.to_dict(orient="records")


# ———————————————
# 4) Streamlit app flow
# ———————————————
st.title("Edit-Alignment Human Evaluation")

if "samples" not in st.session_state:
    df = load_data()
    st.session_state.samples = prepare_evaluation_samples(df, n_samples=N_SAMPLES)
    st.session_state.current_idx = 0
    st.session_state.responses = {}
    st.session_state.submitted = False

samples = st.session_state.samples
idx = st.session_state.current_idx
is_last = (idx == len(samples))

if not is_last:
    sample = samples[idx]
else:
    sample = None

# collect user ID
if "user_id" not in st.session_state:
    st.markdown(
        """
        ### Welcome to the Code-Edit Human Study!

        Thank you for taking the time to help us evaluate model edits.

        **Before you begin:**
        - You’ll see a series of code-edit examples. You can answer each one or skip if you’re unsure.
        - **Your progress is not saved automatically.** After you finish (or skip) all items, be sure to click the **Submit Responses** button at the end.

        Let’s get started!
        """
    )
    st.markdown("### Please enter a nickname to begin:")
    uid = st.text_input(
    "Enter a nickname", 
    key="user_id_input", 
    label_visibility="collapsed", 
    placeholder="Your nickname…"
    )
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
    # … inside your `if not is_last:` block, right after:
    st.markdown(f"### Item {idx+1} of {len(samples)}")

    # # Create a 2-column layout: image on the left, all the text on the right
    # col_img, col_content = st.columns([1, 4])

    # 0) Display the image URL in the left column
    # with col_img:
    
    img = Image.open(os.path.join("images", sample["image_filename"]))
    st.image(img, use_container_width=True)
    # url = sample["image_url"]
    # try:
    #     resp = requests.get(sample["image_url"], timeout=5)
    #     resp.raise_for_status()
    #     img = Image.open(BytesIO(resp.content))
    #     st.image(img, use_container_width=True)
    # except Exception as e:
    #     st.markdown(
    #             f"""
    #     <details style="font-size:10px; color:#666; margin-top:4px;">
    #     <summary style="cursor: pointer;">Image could not be loaded. :(</summary>
    #     <pre style="font-size:9px; color:#444; white-space: pre-wrap;">{e}</pre>
    #     </details>
    #     """,
    #             unsafe_allow_html=True,
    #         )
        # with st.expander("Show error details", expanded=False):
        #     st.code(str(e))

    # 1) Move everything that follows into the right column…
    # with col_content:
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

    # 2) Annotator edits (diff‐style)
    st.markdown("**2) Annotator’s Ground-Truth Edits**")
    with st.expander("Show Ground Truth Edits", expanded=False):
        st.code(sample["reference"], language="diff")

    # 3) Model prediction (diff‐style)
    st.markdown("**3) Model’s Original Prediction**")
    with st.expander("Show Model Prediction", expanded=False):
        st.code(sample["model_prediction_code"], language="diff")

    # 4) GPT-4 output (parsed JSON with colored rating & rationale box)
    st.markdown("**4) GPT-4’s Full Output**")
    raw = sample["gpt4o_full_output"]

    # extract JSON block
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    json_str = m.group(1) if m else raw

    try:
        parsed = json.loads(json_str)
        rating = int(parsed.get("rating", 0))
        rationale = parsed.get("rationale", "").strip()

        # gradient color
        t = max(0, min((rating - 1) / 4, 1))
        r = int(255 * (1 - t)); g = int(255 * t)
        hex_color = f"#{r:02x}{g:02x}00"

        st.markdown(
            f"<div style='font-size:28px; font-weight:bold; color:{hex_color};'>"
            f"🔢 GPT-4 Rating: {rating}"
            "</div>",
            unsafe_allow_html=True
        )

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
            st.session_state.samples = prepare_evaluation_samples(df, n_samples=N_SAMPLES)
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

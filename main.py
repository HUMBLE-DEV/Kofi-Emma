"""
ScholarGraph — Streamlit Frontend Client

Provides a ChatGPT-like chat interface for students researching with the
LangGraph agent. Authentication is handled via the FastAPI backend API.
Users log in / sign up through the sidebar; their JWT tokens are stored in
st.session_state and sent as Bearer tokens on every API request.

All agent work happens behind the authenticated FastAPI backend — this
frontend never imports the LangGraph agent directly, so every chat turn is
subject to JWT validation and per-student conversation isolation.

Run with:  streamlit run main.py
"""

import json
import os
import uuid
from urllib.parse import unquote

import requests
import streamlit as st


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api/v1"

ALLOWED_DOC_EXTENSIONS = ["pdf", "docx", "txt", "md"]
MAX_UPLOAD_MB = 20
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]


# =====================================================================
# Session state
# =====================================================================

def init_session():
    """Initialise the session keys the app relies on."""
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("refresh_token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("message_history", [])
    st.session_state.setdefault("chat_thread", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("show_forgot_password", False)
    st.session_state.setdefault("current_page", "chat")
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("search_topic", "")


# =====================================================================
# Auth helpers
# =====================================================================

def _auth_headers():
    """Return the Authorization header for the current user (if logged in)."""
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api(method: str, path: str, **kwargs):
    """Make an authenticated API call. If the access token expired, refresh it once and retry."""
    extra_headers = kwargs.pop("headers", {})
    kwargs.setdefault("timeout", 60)

    def send():
        return requests.request(
            method, f"{API_BASE}{path}", headers={**_auth_headers(), **extra_headers}, **kwargs
        )

    resp = send()
    if resp.status_code == 401 and refresh_access():
        for entry in (kwargs.get("files") or {}).values():  # rewind uploaded files before resending
            try:
                entry[1].seek(0)
            except (AttributeError, IndexError):
                pass
        resp = send()
    return resp


def _format_login_error(detail) -> str:
    """Extract a user-friendly error message from a FastAPI validation error detail."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        # FastAPI validation errors often have 'msg' or 'ctx.reason'
        msg = detail.get("msg") or detail.get("ctx", {}).get("reason")
        if msg:
            return msg
        # Fall back to any other meaningful field
        for key in ("msg", "detail", "summary"):
            if key in detail and detail[key]:
                return str(detail[key])
        # Last resort: stringify the dict
        return str(detail)
    # Fallback
    return str(detail)


def forgot_password(email: str):
    """Request a password reset link for the given email address."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/forgot_password",
            json={"email": email},
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return False

    if resp.ok:
        st.success("If an account with this email exists, you will receive a reset link shortly.")
        return True
    else:
        try:
            detail = resp.json().get("detail", "Unknown error")
        except ValueError:
            detail = "Unknown error"
        st.error(f"Password reset request failed: {_format_login_error(detail)}")
        return False


def validate_reset_token(token: str):
    """Validate a password reset token via the backend GET endpoint."""
    try:
        resp = requests.get(
            f"{API_BASE}/auth/reset_password",
            params={"token": token},
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return None

    if resp.ok:
        return resp.json()
    return None


def reset_password(token: str, new_password: str, confirm_password: str):
    """Reset password via the backend POST endpoint."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/reset_password",
            params={"token": token},
            json={"new_password": new_password, "confirm_password": confirm_password},
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return False

    if resp.ok:
        st.success("Password reset successfully. You can now log in.")
        return True
    else:
        try:
            detail = resp.json().get("detail", "Unknown error")
        except ValueError:
            detail = "Unknown error"
        st.error(f"Password reset failed: {_format_login_error(detail)}")
        return False


def login(email: str, password: str):
    """Authenticate against the FastAPI backend and store the tokens."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": email, "password": password},
            timeout=10,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return

    if resp.ok:
        data = resp.json()
        st.session_state["access_token"] = data["access_token"]
        st.session_state["refresh_token"] = data["refresh_token"]
        st.session_state["user"] = data["user"]
        st.session_state["authenticated"] = True
        st.session_state["message_history"] = []
        st.session_state["chat_thread"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()
    else:
        try:
            detail = resp.json().get("detail", "Login failed")
        except ValueError:
            detail = "Login failed"
        st.error(f"Login failed: {_format_login_error(detail)}")


def signup(email: str, username: str, password: str):
    """Register a new account via the FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/signup",
            json={"email": email, "username": username, "password": password},
            timeout=15,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return

    if resp.ok:
        st.success("Account created! Please check your email to verify the account.")
    else:
        try:
            detail = resp.json().get("detail", "Unknown error")
        except ValueError:
            detail = "Unknown error"
        st.error(f"Signup failed: {_format_login_error(detail)}")


def refresh_access():
    """Refresh the access token using the stored refresh token."""
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False
    try:
        resp = requests.get(
            f"{API_BASE}/auth/refresh",
            headers={"Authorization": f"Bearer {refresh_token}"},
            timeout=10,
        )
    except requests.RequestException:
        logout(remote=False)
        return False

    if resp.ok:
        st.session_state["access_token"] = resp.json()["access_token"]
        return True

    # If refresh fails, force a clean logout
    logout(remote=False)
    return False


def logout(remote: bool = True):
    """Revoke the token server-side and clear all authenticated state."""
    if remote and st.session_state.get("access_token"):
        try:
            requests.get(
                f"{API_BASE}/auth/logout",
                headers=_auth_headers(),
                timeout=10,
            )
        except requests.RequestException:
            pass  # Local logout must succeed even if the backend is down.

    st.session_state.pop("access_token", None)
    st.session_state.pop("refresh_token", None)
    st.session_state.pop("user", None)
    st.session_state["authenticated"] = False
    st.session_state["message_history"] = []
    st.session_state["chat_thread"] = []
    st.rerun()


def render_auth_sidebar():
    """Render the login/signup/logout controls in the sidebar."""
    if st.session_state.get("authenticated"):
        user = st.session_state["user"]
        st.sidebar.success(f"👤 {user.get('username', user.get('email'))}")
        st.sidebar.caption(f"Role: {user.get('role', 'student')}")
        if st.sidebar.button("Logout", key="logout_btn"):
            logout()
        return

    with st.sidebar:
        st.title("ScholarGraph 🔐")
        st.caption("Research Assistant — secure access required")

        # Check for reset token in query params (from email link)
        reset_token = None
        query_params = st.query_params
        if isinstance(query_params, dict) and "token" in query_params:
            raw_token = query_params.get("token")
            reset_token = raw_token[0] if isinstance(raw_token, list) else raw_token
        elif hasattr(query_params, 'get'):
            reset_token = query_params.get("token")

        # URL-decode the token since it was URL-encoded in the email link
        if reset_token:
            reset_token = unquote(reset_token)

        if reset_token:
            # User came from password reset email link - show reset form
            st.success("Password Reset Link Received")
            st.caption("Enter a new password below to complete the reset.")

            with st.form(key="reset_password_form", clear_on_submit=True):
                new_password = st.text_input("New Password", type="password", key="reset_new_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reset_confirm_password")
                submit_reset = st.form_submit_button("Reset Password", use_container_width=True)

                if submit_reset:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters")
                    if reset_password(reset_token, new_password, confirm_password):
                        # Clear the token and show login form
                        st.query_params.clear()
                        st.rerun()

            if st.button("Back to Login"):
                st.query_params.clear()
                st.rerun()

            st.stop()

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form(key="login_form", clear_on_submit=True):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submit_login = st.form_submit_button("Log In", use_container_width=True)

                if submit_login:
                    login(email, password)

            st.divider()
            st.caption("If you don't have an account, you can sign up above.")
            if st.button("Forgot your password?", key="forgot_password_btn"):
                st.session_state["show_forgot_password"] = True

            if st.session_state.get("show_forgot_password"):
                with st.expander("Reset Password", expanded=True):
                    st.caption(
                        "Enter your email below and we will send you a password reset link."
                    )
                    fp_email = st.text_input("Email", key="forgot_password_email")
                    if st.button("Send Reset Link", key="forgot_password_submit", use_container_width=True):
                        forgot_password(fp_email)
                    if st.button("Cancel", key="forgot_password_cancel"):
                        st.session_state["show_forgot_password"] = False
                        st.rerun()

        with tab_signup:
            with st.form(key="signup_form", clear_on_submit=True):
                s_email = st.text_input("Email", key="signup_email")
                s_username = st.text_input("Name", key="signup_username")
                s_password = st.text_input("Password", type="password", key="signup_password")
                s_confirm = st.text_input("Confirm", type="password", key="signup_confirm")
                submit_signup = st.form_submit_button("Create Account", use_container_width=True)

                if submit_signup:
                    if s_password != s_confirm:
                        st.error("Passwords do not match")
                    elif len(s_password) < 8:
                        st.error("Password must be at least 8 characters")
                    else:
                        signup(s_email, s_username, s_password)


# =====================================================================
# Chat logic (all calls go through the authenticated backend)
# =====================================================================

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)


def reset_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = str(uuid.uuid4())
    add_thread(st.session_state["thread_id"])


def load_conversations():
    """Fetch the signed-in student's conversation list from the backend."""
    try:
        resp = _api("GET", "/conversations")
    except requests.RequestException as exc:
        st.sidebar.error(f"Backend unreachable: {exc}")
        return []
    if not resp.ok:
        return []
    return resp.json().get("conversations", [])


def load_models():
    """Model names come from the backend (/models). Fall back to a short list if it is unreachable."""
    if st.session_state.get("model_options"):
        return st.session_state["model_options"], st.session_state.get("model_default")
    try:
        resp = _api("GET", "/models", timeout=10)
        if resp.ok:
            body = resp.json()
            st.session_state["model_options"] = body["models"]
            st.session_state["model_default"] = body["default"]
            return body["models"], body["default"]
    except (requests.RequestException, KeyError, ValueError):
        pass
    return FALLBACK_MODELS, FALLBACK_MODELS[0]


def load_documents(thread_id: str):
    try:
        resp = _api("GET", "/documents", params={"thread_id": thread_id}, timeout=15)
    except requests.RequestException:
        return None
    return resp.json().get("documents", []) if resp.ok else None


def delete_document(document_id: str) -> bool:
    try:
        return _api("DELETE", f"/documents/{document_id}", timeout=30).ok
    except requests.RequestException:
        return False


def delete_conversation(thread_id: str):
    """Delete a conversation via the backend."""
    resp = _api("DELETE", f"/conversations/{thread_id}")
    if resp.ok:
        st.toast("🗑️ Conversation deleted.")
        return True
    st.error("Failed to delete conversation. Please try again.")
    return False


def load_conversation(thread_id):
    """Fetch the message history for one thread."""
    try:
        resp = _api("GET", f"/history/{thread_id}")
    except requests.RequestException:
        return []
    if not resp.ok:
        return []
    return resp.json().get("messages", [])


def upload_document(uploaded_file, thread_id: str):
    """Send a document to the backend /upload endpoint for RAG processing."""
    if uploaded_file is None:
        return {"success": False, "message": "No file selected."}

    # Build multipart form upload for the backend
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "application/octet-stream")}
    data = {"thread_id": thread_id}

    try:
        resp = _api("POST", "/upload", files=files, data=data, timeout=120)
    except requests.RequestException as exc:
        return {"success": False, "message": f"Could not reach the backend: {exc}"}

    if resp.ok:
        try:
            return resp.json()
        except ValueError:
            return {"success": True, "message": "Document uploaded successfully."}

    try:
        detail = resp.json().get("message", resp.json().get("detail", "Upload failed"))
    except ValueError:
        detail = f"Upload failed with status {resp.status_code}"
    return {"success": False, "message": detail}


def search_papers(topic: str, from_year: int | None = None, to_year: int | None = None, limit: int = 8):
    """Search OpenAlex for papers on a topic, with optional year filtering."""
    try:
        data = {"topic": topic, "limit": str(limit)}
        if from_year:
            data["from_year"] = str(from_year)
        if to_year:
            data["to_year"] = str(to_year)
        resp = _api("POST", "/papers/search", data=data, timeout=30)
    except requests.RequestException as exc:
        return {"success": False, "message": f"Could not reach the backend: {exc}"}

    if resp.ok:
        try:
            return {"success": True, "papers": resp.json().get("papers", [])}
        except ValueError:
            return {"success": True, "papers": []}
    try:
        detail = resp.json().get("detail", "Search failed")
    except ValueError:
        detail = f"Search failed with status {resp.status_code}"
    return {"success": False, "message": detail}


def save_paper(paper: dict):
    """Save an OpenAlex paper result to the student's account.

    Saved papers are scoped to the user, not to a chat thread, so they are
    available in every conversation.
    """
    try:
        resp = _api(
            "POST",
            "/papers/save",
            json={"paper": paper},
            timeout=30,
        )
    except requests.RequestException as exc:
        return {"success": False, "message": f"Could not reach the backend: {exc}"}

    if resp.ok:
        try:
            return resp.json()
        except ValueError:
            return {"success": True, "message": "Paper saved."}
    try:
        detail = resp.json().get("detail", "Save failed")
    except ValueError:
        detail = f"Save failed with status {resp.status_code}"
    return {"success": False, "message": detail}


def load_saved_papers():
    """Fetch the OpenAlex papers the student has saved to their account."""
    try:
        resp = _api("GET", "/papers", timeout=15)
    except requests.RequestException:
        return []
    if not resp.ok:
        return []
    return resp.json().get("papers", [])


def delete_saved_paper(paper_id: str) -> bool:
    try:
        return _api("DELETE", f"/papers/{paper_id}", timeout=30).ok
    except requests.RequestException:
        return False


def stream_response(user_input, active_thread, model=None):
    """Stream the assistant's reply from the backend's SSE endpoint."""

    if not user_input or not user_input.strip():
        yield "⚠️ Please type a message first."
        return
    
    payload = {"message": user_input, "thread_id": active_thread}
    if model:
        payload["model"] = model


    def open_stream():
        return requests.post(
            f"{API_BASE}/chat/stream", headers=_auth_headers(), json=payload, stream=True, timeout=300
        )

    try:
        resp = open_stream()
        if resp.status_code == 401 and refresh_access():  # access token expired: refresh and retry once
            resp.close()
            resp = open_stream()
    except requests.RequestException:
        yield "⚠️ Could not reach the server. Please check your connection and try again."
        return

    with resp:
        if resp.status_code != 200:
            try:
                body = resp.json()
                detail = body.get("detail") or body.get("error") or "Request failed"
            except ValueError:
                detail = f"Request failed with status {resp.status_code}"
            yield f"⚠️ {detail}"
            return

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            body = raw_line[len("data:"):].strip()
            if not body:
                continue
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue

            if event.get("token"):
                yield event["token"]
            elif event.get("error"):
                yield f"\n\n⚠️ {event['error']}"
            elif event.get("done"):
                break


STATUS_LABELS = {
    "queued": "⏳ Waiting to start",
    "processing": "⚙️ Processing",
    "ready": "✅ Ready",
    "failed": "❌ Not usable",
}


@st.fragment(run_every="5s")
def render_documents_panel(thread_id: str):
    """Shows each file's real status and refreshes itself every few seconds."""
    docs = load_documents(thread_id)
    if docs is None:
        st.caption("Could not load documents right now.")
        return
    if not docs:
        st.caption("No documents in this chat yet.")
        return
    for doc in docs:
        label = STATUS_LABELS.get(doc["status"], doc["status"])
        if doc.get("retrying"):
            label += " (retrying)"
        col_text, col_del = st.columns([0.8, 0.2])
        with col_text:
            st.write(f"**{doc['filename']}**")
            st.caption(label)
            if doc["status"] == "failed" and doc.get("reason"):
                st.caption(f"Reason: {doc['reason']}")
        with col_del:
            if st.button("🗑️", key=f"del_doc_{doc['id']}", help="Remove this document"):
                if delete_document(doc["id"]):
                    st.rerun()
                else:
                    st.error("Could not remove it. Try again.")


# =====================================================================
# App
# =====================================================================

st.set_page_config(page_title="ScholarGraph", page_icon="🎓")

init_session()

# ── Auth gate ──
render_auth_sidebar()

if not st.session_state.get("authenticated"):
    st.title("🎓 ScholarGraph")
    st.info("Please log in or create an account in the sidebar to start researching.")
    st.stop()


def render_paper_search_page():
    """Full-page Paper Search interface using OpenAlex."""
    st.title("🔍 Paper Search — OpenAlex")
    st.caption(
        "Find peer-reviewed papers on a topic, browse results, and save them to your "
        "current conversation so the agent can use them for literature reviews, snapshots "
        "and gap analysis."
    )

    # Back to chat button
    if st.button("← Back to Chat", key="back_to_chat_from_paper_search"):
        st.session_state["current_page"] = "chat"
        st.rerun()

    st.divider()

    with st.form(key="paper_search_form", clear_on_submit=False):
        search_topic = st.text_input(
            "Topic",
            key="paper_search_topic_page",
            placeholder="e.g. microplastics marine benthic ecosystems",
        )
        col_year_from, col_year_to = st.columns(2)
        with col_year_from:
            year_from = st.number_input(
                "From year", min_value=1900, max_value=2026, value=None, step=1, key="paper_search_year_from_page"
            )
        with col_year_to:
            year_to = st.number_input(
                "To year", min_value=1900, max_value=2026, value=None, step=1, key="paper_search_year_to_page"
            )
        search_limit = st.slider("Results", min_value=1, max_value=20, value=8, key="paper_search_limit_page")
        search_submit = st.form_submit_button("Search", use_container_width=True)

    if search_submit:
        if not search_topic or not search_topic.strip():
            st.error("Please enter a search topic.")
        else:
            result = search_papers(
                search_topic.strip(),
                from_year=int(year_from) if year_from else None,
                to_year=int(year_to) if year_to else None,
                limit=int(search_limit),
            )
            if not result.get("success"):
                st.error(f"❌ {result.get('message', 'Search failed')}")
            else:
                st.session_state["search_results"] = result.get("papers", [])
                st.session_state["search_topic"] = search_topic.strip()
                st.rerun()

    # Show search results with checkboxes for multi-select
    search_results = st.session_state.get("search_results") or []
    if search_results:
        st.markdown("---")
        st.subheader(f"Results ({len(search_results)})")

        # Build the selection set from checkbox states
        selected_papers: set = set()

        for i, paper in enumerate(search_results, start=1):
            openalex_id = paper.get("id") or str(i)
            with st.expander(f"{i}. {paper.get('title', 'Untitled')}", expanded=False):
                col_cb, col_info = st.columns([0.1, 0.9])
                with col_cb:
                    if st.checkbox("Select", key=f"cb_paper_{openalex_id}"):
                        selected_papers.add(openalex_id)
                with col_info:
                    st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                    st.write(f"**Year:** {paper.get('year', 'n.d.')}")
                    if paper.get("doi"):
                        st.write(f"**DOI:** {paper.get('doi')}")
                    abstract = paper.get("abstract") or ""
                    if abstract:
                        st.write(f"**Abstract:** {abstract[:800]}{'...' if len(abstract) > 800 else ''}")

        # Bulk save button
        selected_count = len(selected_papers)
        if selected_count:
            if st.button(
                f"💾 Save {selected_count} selected paper(s)",
                key="save_selected_papers",
                use_container_width=True,
                type="primary",
            ):
                saved_ok = 0
                failed = 0
                for paper in search_results:
                    openalex_id = paper.get("id") or str(paper.get("title", ""))
                    if openalex_id not in selected_papers:
                        continue
                    save_result = save_paper(paper)
                    if save_result.get("success"):
                        saved_ok += 1
                    else:
                        failed += 1
                if saved_ok:
                    st.success(f"✅ Saved {saved_ok} paper(s).")
                    st.session_state["search_results"] = []
                if failed:
                    st.error(f"❌ {failed} paper(s) failed to save.")
                st.rerun()

    st.divider()
    st.subheader("🔖 Saved papers in this chat")
    saved_papers = load_saved_papers()
    if not saved_papers:
        st.caption("No saved papers yet. Search and save some above.")
    else:
        for paper in saved_papers:
            col_info, col_del = st.columns([0.85, 0.15])
            with col_info:
                st.write(f"**{paper.get('title', 'Untitled')}**")
                st.caption(
                    f"{paper.get('authors', 'Unknown')} • {paper.get('year', 'n.d.')}"
                    + (f" • DOI: {paper['doi']}" if paper.get("doi") else "")
                )
            with col_del:
                if st.button(
                    "🗑️", key=f"del_saved_page_{paper.get('id')}", help="Remove this paper from this chat"
                ):
                    if delete_saved_paper(str(paper.get("id"))):
                        st.rerun()
                    else:
                        st.error("Could not remove it. Try again.")


# ── Authenticated app ──
add_thread(st.session_state["thread_id"])

st.title("🎓 ScholarGraph your Research friend like no other")

# ── Sidebar threading ──
with st.sidebar:
    st.markdown("---")
    st.subheader("Conversations")

    if st.button("New Chat", key="new_chat"):
        reset_chat()
        st.rerun()

    for convo in load_conversations():
        thread_id = convo.get("thread_id", "")
        title = convo.get("title") or "Untitled"

        col_conv, col_del = st.columns([0.85, 0.15])
        with col_conv:
            if st.button(
                f"💬 {title}",
                key=f"convo_{thread_id}",
                use_container_width=True,
                type="primary" if thread_id == st.session_state["thread_id"] else "secondary",
            ):
                st.session_state["thread_id"] = thread_id
                add_thread(thread_id)
                st.session_state["message_history"] = load_conversation(thread_id)
                st.rerun()
        with col_del:
            with st.popover(
                "🗑️",
                key=f"del_popover_{thread_id}",
                use_container_width=True,
                help=f"Delete conversation: {title}",
            ):
                st.caption(f"Delete '{title}'? This cannot be undone.")
                st.divider()
                if st.button(
                    "Confirm delete",
                    key=f"confirm_del_{thread_id}",
                    use_container_width=True,
                ):
                    delete_conversation(thread_id)
                    if thread_id == st.session_state["thread_id"]:
                        reset_chat()
                    st.rerun()

    st.markdown("---")
    st.subheader("🧠 Model")
    st.caption("Select which LLM powers this conversation.")
    model_options, model_default = load_models()
    if st.session_state.get("selected_model") not in model_options:
        st.session_state["selected_model"] = model_default or model_options[0]
    st.session_state["selected_model"] = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(st.session_state["selected_model"]),
        key="model_selector",
        help="Switches the underlying LLM for all subsequent messages in this conversation.",
    )

    st.markdown("---")
    st.subheader("🔍 Paper Search")
    st.caption(
        "Find peer-reviewed papers on a topic and save them to this conversation. "
        "The agent can then use them for literature reviews, snapshots and gap analysis."
    )
    if st.button(
        "🔍 Open Paper Search",
        key="nav_paper_search",
        use_container_width=True,
        type="primary" if st.session_state.get("current_page") == "paper_search" else "secondary",
    ):
        st.session_state["current_page"] = "paper_search"
        st.rerun()

    st.markdown("---")
    st.subheader("📎 Upload Documents")
    st.caption(
        f"Upload research papers ({', '.join(ALLOWED_DOC_EXTENSIONS)}), up to {MAX_UPLOAD_MB} MB each. "
        "They are added to this conversation's knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=ALLOWED_DOC_EXTENSIONS,
        accept_multiple_files=True,
        key="document_uploader",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected:")
        for f in uploaded_files:
            st.write(f"• {f.name} ({len(f.getvalue()) / 1024:.1f} KB)")

        if st.button("Upload all to this conversation", key="upload_document_btn", type="primary"):
            progress = st.progress(0.0, text="Uploading documents...")
            total = len(uploaded_files)
            uploaded_ok = 0
            for i, f in enumerate(uploaded_files, start=1):
                progress.progress(i / total, text=f"Uploading {f.name} ({i}/{total})...")
                if len(f.getvalue()) > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"❌ {f.name}: larger than {MAX_UPLOAD_MB} MB.")
                    continue
                result = upload_document(f, st.session_state["thread_id"])
                if result.get("success"):
                    uploaded_ok += 1
                else:
                    st.error(f"❌ {f.name}: {result.get('message', 'Upload failed')}")
            progress.empty()
            if uploaded_ok:
                st.toast(f"📤 {uploaded_ok} of {total} uploaded. Watch the status below.")
                st.rerun()

    st.markdown("---")
    st.subheader("📄 Documents in this chat")
    render_documents_panel(st.session_state["thread_id"])

    # ── Saved papers panel ──
    st.markdown("---")
    st.subheader("🔖 Saved papers")
    saved_papers = load_saved_papers()
    if not saved_papers:
        st.caption("No saved papers yet. Search and save some above.")
    else:
        for paper in saved_papers:
            col_info, col_del = st.columns([0.85, 0.15])
            with col_info:
                st.write(f"**{paper.get('title', 'Untitled')}**")
                st.caption(
                    f"{paper.get('authors', 'Unknown')} • {paper.get('year', 'n.d.')}"
                    + (f" • DOI: {paper['doi']}" if paper.get("doi") else "")
                )
            with col_del:
                if st.button(
                    "🗑️", key=f"del_saved_{paper.get('id')}", help="Remove this paper from this chat"
                ):
                    if delete_saved_paper(str(paper.get("id"))):
                        st.rerun()
                    else:
                        st.error("Could not remove it. Try again.")

# ── Page router ──
if st.session_state.get("current_page") == "paper_search":
    render_paper_search_page()
else:
    # ── Message history ──
    if not st.session_state["message_history"]:
        st.info(
            "Try: *\"Write a literature review on my uploaded papers\"*, "
            "*\"What are the research gaps?\"*, or *\"Give me a study snapshot of paper-name\"*. "
            "Research answers are drafts: always check them against the original papers."
        )
    retry_message = None
    for index, message in enumerate(st.session_state["message_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # If the assistant hit an error (rate limit, quota, etc.), offer a retry.
            if message["role"] == "assistant" and (
                "Something went wrong" in message["content"]
                or "daily quota has been exceeded" in message["content"]
                or "service is busy right now" in message["content"]
            ):
                prev_user_msg = ""
                if index > 0:
                    prev = st.session_state["message_history"][index - 1]
                    if prev["role"] == "user":
                        prev_user_msg = prev["content"]
                if st.button(
                    "🔄 Try Again",
                    key=f"retry_{index}",
                    use_container_width=True,
                ):
                    retry_message = prev_user_msg

    # ── Chat input ──
    user_input = st.chat_input("Ask about your research topic...")

    # A retry click resends the last user message that preceded an error response.
    if retry_message:
        user_input = retry_message

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            ai_message = st.write_stream(
                stream_response(user_input, st.session_state["thread_id"], model=st.session_state["selected_model"])
            )
        if ai_message:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )
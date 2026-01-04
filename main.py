"""
Candilyzer: AI-powered candidate analyzer for elite technical hiring.

This Streamlit application leverages the Agno AI Agent Orchestration Framework to conduct
forensic-level multi-candidate and single-candidate analysis using verified GitHub and LinkedIn data.

Agents are powered by Nebius and enhanced with Agno's GitHubTools, ExaTools,
and ReasoningTools — enabling strict, professional-grade hiring decisions with full traceability.
"""

import re
import yaml
import io
import streamlit as st

from agno.agent import Agent
from agno.models.nebius import Nebius
from agno.tools.github import GithubTools
from agno.tools.exa import ExaTools
from agno.tools.reasoning import ReasoningTools

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Set wide layout
st.set_page_config(layout="wide")

# Function to extract text from resume files
def extract_resume_text(uploaded_file):
    """Extract text from uploaded resume file (PDF, DOCX, or TXT)"""
    text = ""
    file_type = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_type == 'pdf':
            if not PDF_AVAILABLE:
                return None, "PyPDF2 library is not installed. Please install it: pip install PyPDF2"
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif file_type in ['docx', 'doc']:
            if not DOCX_AVAILABLE:
                return None, "python-docx library is not installed. Please install it: pip install python-docx"
            doc = Document(io.BytesIO(uploaded_file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
                
        elif file_type == 'txt':
            text = str(uploaded_file.read(), "utf-8")
            
        else:
            return None, f"Unsupported file type: {file_type}. Supported formats: PDF, DOCX, TXT"
            
        return text.strip(), None
        
    except Exception as e:
        return None, f"Error extracting text from file: {str(e)}"

# Load YAML prompts
@st.cache_data
def load_yaml(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("❌ YAML prompt file not found.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"❌ YAML parsing error: {e}")
        st.stop()

data = load_yaml("hiring_prompts.yaml")
description_multi = data.get("description_for_multi_candidates", "")
instructions_multi = data.get("instructions_for_multi_candidates", "")
description_single = data.get("description_for_single_candidate", "")
instructions_single = data.get("instructions_for_single_candidate", "")

# Header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="font-size: 2.8rem;">🧠 Candilyzer</h1>
        <p style="font-size:1.1rem;">Elite GitHub + LinkedIn Candidate Analyzer for Tech Hiring</p>
    </div>
""", unsafe_allow_html=True)

# Session state init
for key in ["Nebius_api_key",  "model_id", "github_api_key", "exa_api_key"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# Sidebar
st.sidebar.title("🔑 API Keys & Navigation")
st.sidebar.markdown("### Enter API Keys")
st.session_state.Nebius_api_key = st.sidebar.text_input("Nebius API Key", value=st.session_state.Nebius_api_key, type="password")
st.session_state.model_id = st.sidebar.text_input("Model ID", value=st.session_state.model_id, placeholder="e.g. openai/gpt-oss-20b or openai/deepseek-chat")
st.sidebar.caption("💡 Format: provider/model-name (e.g., openai/gpt-oss-20b)")
st.session_state.github_api_key = st.sidebar.text_input("GitHub API Key", value=st.session_state.github_api_key, type="password")
st.session_state.exa_api_key = st.sidebar.text_input("Exa API Key", value=st.session_state.exa_api_key, type="password")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select Page", ("Multi-Candidate Analyzer", "Single Candidate Analyzer"))

# ---------------- Multi-Candidate Analyzer ---------------- #
if page == "Multi-Candidate Analyzer":
    st.header("Multi-Candidate Analyzer 🕵️‍♂️")
    st.markdown("Enter GitHub usernames and/or LinkedIn profiles (one per line) and a target job role. You can use either or both.")

    with st.form("multi_candidate_form"):
        col1, col2 = st.columns(2)
        with col1:
            github_usernames = st.text_area("GitHub Usernames (one per line)", placeholder="username1\nusername2\n...", help="Enter GitHub usernames, one per line")
        with col2:
            linkedin_profiles = st.text_area("LinkedIn Profiles (one per line)", placeholder="https://linkedin.com/in/profile1\nhttps://linkedin.com/in/profile2\n...", help="Enter LinkedIn profile URLs, one per line")
        job_role = st.text_input("Target Job Role", placeholder="e.g. Backend Engineer")
        resume_files = st.file_uploader(
            "Upload Resume Files (Optional)", 
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Upload resume files (PDF, DOCX, or TXT) for additional analysis"
        )
        submit = st.form_submit_button("Analyze Candidates")

    if submit:
        if not job_role:
            st.error("❌ Please enter a job role.")
        elif not all([st.session_state.Nebius_api_key, st.session_state.exa_api_key, st.session_state.model_id]):
            st.error("❌ Please enter Nebius API key, Exa API key, and model info in the sidebar.")
        else:
            github_list = [u.strip() for u in github_usernames.split("\n") if u.strip()] if github_usernames else []
            linkedin_list = [l.strip() for l in linkedin_profiles.split("\n") if l.strip()] if linkedin_profiles else []
            
            # Extract text from uploaded resume files
            resume_texts = []
            if resume_files:
                for resume_file in resume_files:
                    text, error = extract_resume_text(resume_file)
                    if error:
                        st.warning(f"⚠️ Could not extract text from {resume_file.name}: {error}")
                    elif text:
                        resume_texts.append(f"Resume ({resume_file.name}):\n{text}")
            
            if not github_list and not linkedin_list and not resume_texts:
                st.error("❌ Please enter at least one GitHub username, LinkedIn profile, or upload resume files.")
            else:
                # Check GitHub API key only if GitHub usernames are provided
                if github_list and not st.session_state.github_api_key:
                    st.error("❌ Please enter GitHub API key in the sidebar (required for GitHub analysis).")
                else:
                    # Validate model ID format
                    model_id = st.session_state.model_id.strip()
                    if not model_id or len(model_id.split('/')) < 2:
                        st.warning(f"⚠️ Model ID format might be incorrect. Expected format: 'provider/model-name' (e.g., 'openai/gpt-oss-20b')")
                    
                    # Build tools list
                    tools_list = [
                        ReasoningTools(enable_think=True, instructions="Strict candidate evaluation", add_instructions=True),
                    ]
                    
                    # Add GithubTools only if GitHub usernames are provided
                    if github_list:
                        tools_list.append(GithubTools(access_token=st.session_state.github_api_key))
                    
                    # Add ExaTools with appropriate domains
                    domains = []
                    if github_list:
                        domains.append("github.com")
                    if linkedin_list:
                        domains.append("linkedin.com")
                    
                    tools_list.append(ExaTools(
                        api_key=st.session_state.exa_api_key,
                        include_domains=domains if domains else ["github.com", "linkedin.com"],
                        type="keyword"
                    ))
                    
                    try:
                        model = Nebius(
                            id=model_id,
                            api_key=st.session_state.Nebius_api_key,
                        )
                    except Exception as e:
                        st.error(f"❌ Error initializing model: {str(e)}")
                        st.info(f"💡 **Model ID:** {model_id}")
                        st.stop()
                    
                    agent = Agent(
                        description=description_multi,
                        instructions=instructions_multi,
                        model=model,
                        name="StrictCandidateEvaluator",
                        tools=tools_list,
                        markdown=True
                    )

                    st.markdown("### 🔎 Evaluation in Progress...")
                    try:
                        with st.spinner("Running detailed analysis..."):
                            # Build query with GitHub, LinkedIn, and resume data
                            query_parts = []
                            if github_list:
                                query_parts.append(f"GitHub: {', '.join(github_list)}")
                            if linkedin_list:
                                query_parts.append(f"LinkedIn: {', '.join(linkedin_list)}")
                            if resume_texts:
                                query_parts.append("\n\n".join(resume_texts))
                            
                            query = f"Evaluate candidates for role '{job_role}'. {', '.join(query_parts)}"
                            stream = agent.run(query, stream=True)

                            output = ""
                            block = st.empty()
                            for chunk in stream:
                                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                                    output += chunk.content
                                    block.markdown(output, unsafe_allow_html=True)
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "unknown model" in error_msg or "model" in error_msg and "not found" in error_msg:
                            st.error(f"❌ Unknown model error. Please check your Model ID: '{st.session_state.model_id}'")
                            st.info("💡 **Tip:** Model ID should be in format: `provider/model-name` (e.g., `openai/gpt-oss-20b` or `openai/deepseek-chat`). Check your Nebius Token Factory dashboard for available models.")
                        elif "api" in error_msg and "key" in error_msg:
                            st.error("❌ API Key error. Please verify your Nebius API key is correct.")
                        else:
                            st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)

# ---------------- Single Candidate Analyzer ---------------- #
elif page == "Single Candidate Analyzer":
    st.header("Single Candidate Analyzer")
    st.markdown("Analyze GitHub and/or LinkedIn profile for a role. You can use either or both.")

    with st.form("single_candidate_form"):
        col1, col2 = st.columns(2)
        with col1:
            github_username = st.text_input("GitHub Username (Optional)", placeholder="e.g. Toufiq", help="Enter GitHub username or leave empty")
            linkedin_url = st.text_input("LinkedIn Profile (Optional)", placeholder="https://linkedin.com/in/...", help="Enter LinkedIn profile URL or leave empty")
        with col2:
            job_role = st.text_input("Job Role", placeholder="e.g. ML Engineer")
        resume_file = st.file_uploader(
            "Upload Resume File (Optional)",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload a resume file (PDF, DOCX, or TXT) for additional analysis"
        )
        submit_button = st.form_submit_button("Analyze Candidate 🔥")

    if submit_button:
        if not job_role:
            st.error("❌ Job role is required.")
        else:
            # Extract text from uploaded resume file
            resume_text = None
            if resume_file:
                text, error = extract_resume_text(resume_file)
                if error:
                    st.warning(f"⚠️ Could not extract text from {resume_file.name}: {error}")
                elif text:
                    resume_text = f"Resume:\n{text}"
            
            if not github_username and not linkedin_url and not resume_text:
                st.error("❌ Please enter at least one GitHub username, LinkedIn profile, or upload a resume file.")
            elif not all([st.session_state.Nebius_api_key, st.session_state.exa_api_key, st.session_state.model_id]):
                st.error("❌ Please enter Nebius API key, Exa API key, and model info.")
            else:
                # Check GitHub API key only if GitHub username is provided
                if github_username and not st.session_state.github_api_key:
                    st.error("❌ Please enter GitHub API key in the sidebar (required for GitHub analysis).")
                else:
                    try:
                        # Build tools list
                        tools_list = [
                            ReasoningTools(enable_think=True, add_instructions=True),
                        ]
                        
                        # Add GithubTools only if GitHub username is provided
                        if github_username:
                            tools_list.append(GithubTools(access_token=st.session_state.github_api_key))
                        
                        # Add ExaTools with appropriate domains
                        domains = []
                        if github_username:
                            domains.append("github.com")
                        if linkedin_url:
                            domains.append("linkedin.com")
                        
                        tools_list.append(ExaTools(
                            api_key=st.session_state.exa_api_key,
                            include_domains=domains if domains else ["github.com", "linkedin.com"],
                            type="keyword",
                            text_length_limit=2000,
                            show_results=True
                        ))
                        
                        # Validate model ID format
                        model_id = st.session_state.model_id.strip()
                        if not model_id or len(model_id.split('/')) < 2:
                            st.warning(f"⚠️ Model ID format might be incorrect. Expected format: 'provider/model-name' (e.g., 'openai/gpt-oss-20b')")
                        
                        try:
                            model = Nebius(
                                id=model_id,
                                api_key=st.session_state.Nebius_api_key,
                            )
                        except Exception as e:
                            st.error(f"❌ Error initializing model: {str(e)}")
                            st.info(f"💡 **Model ID:** {model_id}")
                            st.stop()
                        
                        agent = Agent(
                            model=model,
                            name="Candilyzer",
                            tools=tools_list,
                            description=description_single,
                            instructions=instructions_single,
                            markdown=True,
                            add_datetime_to_context=True
                        )

                        st.markdown("### 🤖 AI Evaluation in Progress...")
                        try:
                            with st.spinner("Analyzing candidate..."):
                                # Build input text with available information
                                input_parts = [f"Role: {job_role}"]
                                if github_username:
                                    input_parts.append(f"GitHub: {github_username}")
                                if linkedin_url:
                                    input_parts.append(f"LinkedIn: {linkedin_url}")
                                if resume_text:
                                    input_parts.append(resume_text)
                                
                                input_text = "\n\n".join(input_parts) if resume_text in input_parts else ", ".join(input_parts)

                                response_stream = agent.run(
                                    f"Analyze candidate for {input_text}. Provide score and detailed report.",
                                    stream=True
                                )

                            full_response = ""
                            placeholder = st.empty()
                            for chunk in response_stream:
                                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                                    full_response += chunk.content
                                    placeholder.markdown(full_response, unsafe_allow_html=True)

                            match = re.search(r"\b([1-9]?\d|100)/100\b", full_response)
                            if match:
                                score = int(match.group(1))
                                st.success(f"🎯 Candidate Score: {score}/100")
                        except Exception as e:
                            error_msg = str(e).lower()
                            if "unknown model" in error_msg or "model" in error_msg and "not found" in error_msg:
                                st.error(f"❌ Unknown model error. Please check your Model ID: '{st.session_state.model_id}'")
                                st.info("💡 **Tip:** Model ID should be in format: `provider/model-name` (e.g., `openai/gpt-oss-20b` or `openai/deepseek-chat`). Check your Nebius Token Factory dashboard for available models.")
                            elif "api" in error_msg and "key" in error_msg:
                                st.error("❌ API Key error. Please verify your Nebius API key is correct.")
                            else:
                                st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)

                    except (ValueError, KeyError, ConnectionError) as e:
                        st.error(f"❌ Known error: {e}")
                    except Exception as e:
                        st.error("❌ Unexpected error occurred.")
                        st.exception(e)


"""
Streamlit Application for Excel RAG System
- File upload for Excel
- Preview merged data
- Save/Update Excel store
- Incremental and full vector index updates
- Manual row entry
- Query interface with answer and sources
"""
import streamlit as st
import pandas as pd
from typing import Optional
import os

# Import our modules
from excel_handler import (
    process_uploaded_file,
    load_existing_store,
    save_to_excel_store,
    append_row_to_store,
    delete_row_from_store,
    add_row_ids,
    normalize_column_name,
    dataframe_to_documents,
    get_row_text_for_embedding
)
from vector_store import get_vector_store
from rag_pipeline import get_rag_pipeline, RAGPipeline
from llm_client import get_llm_client
import config

# Page configuration
st.set_page_config(
    page_title="Excel RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stats-box {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize a DataFrame for Streamlit display by converting all columns to strings.
    This prevents Arrow serialization errors from mixed types.
    """
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str).replace('nan', '')
    return display_df


def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_sheets' not in st.session_state:
        st.session_state.uploaded_sheets = None
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    if 'gemini_initialized' not in st.session_state:
        st.session_state.gemini_initialized = False
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Auto-initialize pipeline if API key is in environment and not yet initialized
    if not st.session_state.gemini_initialized:
        api_key = os.getenv("NEBIUS_API_KEY", "")
        if api_key:
            try:
                pipeline = initialize_pipeline(api_key)
                if pipeline:
                    st.session_state.gemini_initialized = True
            except Exception:
                pass  # Will show manual init option in sidebar


def check_api_key() -> bool:
    """Check if Nebius API key is configured."""
    api_key = os.getenv("NEBIUS_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get('nebius_api_key', "")
    return bool(api_key)


def initialize_pipeline(api_key: Optional[str] = None) -> Optional[RAGPipeline]:
    """Initialize the RAG pipeline with API key."""
    try:
        if api_key:
            os.environ["NEBIUS_API_KEY"] = api_key
        
        # Reset singleton instances to use new API key
        import llm_client
        llm_client._client_instance = None
        
        import rag_pipeline
        rag_pipeline._pipeline_instance = None
        
        pipeline = get_rag_pipeline()
        st.session_state.gemini_initialized = True
        st.session_state.rag_pipeline = pipeline
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None


def render_sidebar():
    """Render the sidebar with configuration and stats."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show connection status
        if st.session_state.gemini_initialized:
            st.success("✅ Nebius LLM Connected")
        else:
            st.warning("⚠️ Not Connected")
            
            # Only show API key input if not already in environment
            env_key = os.getenv("NEBIUS_API_KEY", "")
            if not env_key:
                api_key = st.text_input(
                    "Nebius API Key",
                    type="password",
                    help="Enter your Nebius API key (or set NEBIUS_API_KEY in .env)"
                )
                if api_key:
                    st.session_state.nebius_api_key = api_key
            else:
                api_key = env_key
            
            if st.button("🔑 Connect", width='stretch'):
                with st.spinner("Initializing..."):
                    pipeline = initialize_pipeline(api_key)
                    if pipeline:
                        st.success("Pipeline initialized!")
                        st.rerun()
        
        st.divider()
        
        # Model configuration
        st.subheader("Model Settings")
        chat_model = st.text_input(
            "Chat Model",
            value=config.LLM_CHAT_MODEL,
            help="Nebius chat model name"
        )
        embed_model = st.text_input(
            "Embedding Model", 
            value=config.LLM_EMBEDDING_MODEL,
            help="Nebius embedding model name"
        )
        
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=20,
            value=config.TOP_K_RESULTS,
            help="Number of results to retrieve for RAG"
        )
        st.session_state.top_k = top_k
        
        st.divider()
        
        # Index statistics
        st.subheader("📈 Index Statistics")
        try:
            vector_store = get_vector_store()
            stats = vector_store.get_collection_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('document_count', 0))
            with col2:
                existing_df = load_existing_store()
                excel_count = len(existing_df) if existing_df is not None else 0
                st.metric("Excel Rows", excel_count)
            
            if stats.get('error'):
                st.warning(f"Index warning: {stats['error']}")
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")
        
        st.divider()
        
        # Quick actions
        st.subheader("🔧 Quick Actions")
        
        if st.button("🔄 Sync Index with Excel", width='stretch'):
            if not st.session_state.gemini_initialized:
                st.error("Please initialize the pipeline first!")
            else:
                with st.spinner("Syncing..."):
                    try:
                        pipeline = st.session_state.rag_pipeline
                        result = pipeline.sync_index_with_store()
                        st.success(f"Synced! Added: {result['added']}, Removed: {result['removed']}")
                    except Exception as e:
                        st.error(f"Sync failed: {str(e)}")
        
        if st.button("🗑️ Clear Vector Index", width='stretch'):
            if st.session_state.get('confirm_clear'):
                try:
                    vector_store = get_vector_store()
                    vector_store.clear_collection()
                    st.success("Index cleared!")
                    st.session_state.confirm_clear = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear: {str(e)}")
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing the index")


def render_upload_section():
    """Render the file upload section."""
    st.header("📤 Upload Excel File")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx)",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with one or more sheets"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            try:
                sheets, merged_df = process_uploaded_file(uploaded_file)
                st.session_state.uploaded_sheets = sheets
                st.session_state.merged_df = merged_df
                
                st.success(f"✅ Loaded {len(sheets)} sheet(s) with {len(merged_df)} total rows")
                
                # Show sheet names
                st.info(f"Sheets found: {', '.join(sheets.keys())}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
    
    # Preview merged data
    if st.session_state.merged_df is not None:
        st.subheader("📋 Preview: Merged Data (All)")
        
        df = st.session_state.merged_df
        
        # Column info
        with st.expander("Column Information", expanded=False):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Sample Value': [str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(col_info, width='stretch')
        
        # Data preview
        st.dataframe(sanitize_df_for_display(df.head(100)), width='stretch', height=400)
        
        st.caption(f"Showing first 100 rows of {len(df)} total rows")


def render_data_management():
    """Render the data management section."""
    st.header("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save/Update Excel Store", width='stretch', type="primary"):
            if st.session_state.merged_df is None:
                st.error("No data to save! Please upload a file first.")
            else:
                with st.spinner("Saving to Excel store..."):
                    try:
                        save_to_excel_store(
                            st.session_state.merged_df,
                            keep_original_sheets=True,
                            original_sheets=st.session_state.uploaded_sheets
                        )
                        st.success(f"✅ Saved to {config.EXCEL_STORE_PATH}")
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")
    
    with col2:
        if st.button("🔄 Update Vector Index (Incremental)", width='stretch'):
            if not st.session_state.gemini_initialized:
                st.error("Please initialize the pipeline first!")
            else:
                with st.spinner("Updating index incrementally..."):
                    try:
                        pipeline = st.session_state.rag_pipeline
                        added, skipped = pipeline.update_index_incremental()
                        st.success(f"✅ Added: {added} new rows, Skipped: {skipped} existing rows")
                    except Exception as e:
                        st.error(f"Error updating index: {str(e)}")
    
    with col3:
        if st.button("🔨 Rebuild Index (Full)", width='stretch'):
            if not st.session_state.gemini_initialized:
                st.error("Please initialize the pipeline first!")
            else:
                with st.spinner("Rebuilding entire index..."):
                    try:
                        pipeline = st.session_state.rag_pipeline
                        count = pipeline.rebuild_index()
                        st.success(f"✅ Rebuilt index with {count} documents")
                    except Exception as e:
                        st.error(f"Error rebuilding index: {str(e)}")
    
    # Show existing store
    st.subheader("📁 Current Excel Store")
    
    existing_df = load_existing_store()
    if existing_df is not None:
        st.dataframe(sanitize_df_for_display(existing_df.head(50)), width='stretch', height=300)
        st.caption(f"Showing first 50 rows of {len(existing_df)} total rows in store")
    else:
        st.info("No existing Excel store found. Upload and save data to create one.")


def render_manual_entry():
    """Render the manual row entry form."""
    st.header("✏️ Manual Row Entry")
    
    # Get columns from existing store or uploaded data
    existing_df = load_existing_store()
    if existing_df is not None:
        available_columns = [col for col in existing_df.columns if col != 'row_id']
    elif st.session_state.merged_df is not None:
        available_columns = [col for col in st.session_state.merged_df.columns if col != 'row_id']
    else:
        st.warning("No data schema available. Please upload or save data first.")
        return
    
    st.write("Enter values for the new row:")
    
    # Create input fields for each column
    row_data = {}
    
    # Create columns for better layout
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, col_name in enumerate(available_columns):
        col_idx = i % num_cols
        with cols[col_idx]:
            # Determine input type based on column name
            if 'date' in col_name.lower():
                value = st.text_input(col_name, key=f"manual_{col_name}", 
                                     help="Enter date (e.g., 2024-01-15)")
            elif 'year' in col_name.lower():
                value = st.number_input(col_name, key=f"manual_{col_name}",
                                       min_value=1900, max_value=2100, step=1)
            else:
                value = st.text_input(col_name, key=f"manual_{col_name}")
            
            if value:
                row_data[col_name] = value
    
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("➕ Add Row", type="primary", width='stretch'):
            if not row_data:
                st.error("Please fill in at least one field!")
            elif not st.session_state.gemini_initialized:
                st.error("Please initialize the pipeline first!")
            else:
                with st.spinner("Adding row..."):
                    try:
                        # Append to Excel store
                        success, row_id, updated_df = append_row_to_store(row_data)
                        
                        if not success:
                            st.warning(f"Row already exists with ID: {row_id}")
                        else:
                            # Add to vector index
                            pipeline = st.session_state.rag_pipeline
                            new_row = updated_df[updated_df['row_id'] == row_id].iloc[0]
                            pipeline.add_single_row_to_index(row_id, new_row)
                            
                            st.success(f"✅ Row added successfully! ID: {row_id}")
                            
                            # Clear inputs by rerunning
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error adding row: {str(e)}")
    
    with col2:
        st.info("💡 The row will be automatically added to both the Excel store and the vector index.")


def render_query_interface():
    """Render the query interface."""
    st.header("🔍 Query Your Data")
    
    if not st.session_state.gemini_initialized:
        st.warning("⚠️ Please initialize the pipeline in the sidebar first!")
        return
    
    # Check if there's data in the index
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        if stats.get('document_count', 0) == 0:
            st.warning("⚠️ The vector index is empty. Please upload data and update the index first.")
            return
    except Exception as e:
        st.warning(f"Could not check index status: {str(e)}")
    
    # Query input
    question = st.text_area(
        "Ask a question about your data:",
        height=100,
        placeholder="e.g., What publications were made in 2023? Who are the authors in the computer science department?"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        search_clicked = st.button("🔎 Search", type="primary", width='stretch')
    
    with col2:
        show_sources = st.checkbox("Show retrieved sources", value=True)
    
    if search_clicked and question:
        with st.spinner("Searching and generating answer..."):
            try:
                pipeline = st.session_state.rag_pipeline
                top_k = st.session_state.get('top_k', config.TOP_K_RESULTS)
                
                result = pipeline.query(
                    question=question,
                    top_k=top_k,
                    return_sources=show_sources
                )
                
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(result['answer'])
                
                # Add to history
                st.session_state.query_history.append({
                    'question': question,
                    'answer': result['answer']
                })
                
                # Display sources
                if show_sources and result.get('sources'):
                    st.subheader("📚 Retrieved Sources")
                    
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i} (Relevance: {1 - source.get('distance', 0):.2%})"):
                            st.text(source.get('text', 'No text available'))
                            
                            if source.get('metadata'):
                                st.caption("Metadata:")
                                st.json(source['metadata'])
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                st.markdown(f"**Q{i}:** {item['question']}")
                st.markdown(f"**A{i}:** {item['answer'][:200]}...")
                st.divider()


def render_data_explorer():
    """Render a data exploration section."""
    st.header("🔬 Data Explorer")
    
    existing_df = load_existing_store()
    if existing_df is None:
        st.info("No data available. Please upload and save data first.")
        return
    
    # Search/filter
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
    
    with col2:
        search_col = st.selectbox(
            "Column",
            options=['All Columns'] + list(existing_df.columns)
        )
    
    # Filter data
    if search_term:
        if search_col == 'All Columns':
            mask = existing_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
        else:
            mask = existing_df[search_col].astype(str).str.contains(
                search_term, case=False, na=False
            )
        filtered_df = existing_df[mask]
    else:
        filtered_df = existing_df
    
    st.dataframe(sanitize_df_for_display(filtered_df), width='stretch', height=400)
    st.caption(f"Showing {len(filtered_df)} of {len(existing_df)} rows")
    
    # Column statistics
    with st.expander("📊 Column Statistics"):
        for col in existing_df.columns:
            if col != 'row_id':
                st.write(f"**{col}:** {existing_df[col].nunique()} unique values")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # App title
    st.title("📊 Excel RAG System")
    st.markdown("*Query your Excel data using natural language powered by Google Gemini*")
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "💾 Data Management", 
        "✏️ Manual Entry",
        "🔍 Query",
        "🔬 Explorer"
    ])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_data_management()
    
    with tab3:
        render_manual_entry()
    
    with tab4:
        render_query_interface()
    
    with tab5:
        render_data_explorer()
    
    # Footer
    st.divider()
    st.caption("Excel RAG System | Powered by ChromaDB & Google Gemini")


if __name__ == "__main__":
    main()

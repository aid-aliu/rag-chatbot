import streamlit as st
import pandas as pd


def render_dashboard():
    """
    Renders the Observability Dashboard tab.
    Reads data directly from st.session_state.metrics.
    """
    st.header("📊 Observability Dashboard")
    st.markdown("Monitor performance, costs, and user feedback in real-time.")

    # Ensure metrics exist (safety check)
    if "metrics" not in st.session_state:
        st.warning("No metrics data found. Start a conversation to generate data.")
        return

    metrics = st.session_state.metrics

    if len(metrics["queries"]) > 0:
        # 1. Prepare Data
        df = pd.DataFrame({
            "Query": metrics["queries"],
            "Latency (s)": metrics["latencies"],
            "Est. Tokens": metrics["tokens_generated"]
        })

        # 2. Top-Level KPIs
        # Calculate averages and totals
        total_chats = len(df)
        total_tokens = int(df["Est. Tokens"].sum())
        avg_latency = df["Latency (s)"].mean()
        net_score = metrics["feedback_score"]

        # Display Metrics in 4 columns
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Queries", total_chats)
        k2.metric("Total Tokens", total_tokens)
        k3.metric("Avg Latency", f"{avg_latency:.2f}s")
        k4.metric("Feedback Score", net_score)

        st.divider()

        # 3. Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Latency Trend")
            st.caption("Time taken (seconds) per query")
            # Line chart is great for seeing spikes over time
            st.line_chart(df["Latency (s)"])

        with col2:
            st.subheader("🪙 Token Usage")
            st.caption("Estimated length of responses")
            # Bar chart shows which questions consumed the most resources
            st.bar_chart(df["Est. Tokens"])

        st.divider()

        # 4. Raw Data Table
        with st.expander("📄 View Raw Interaction Logs", expanded=True):
            st.dataframe(df, use_container_width=True)

    else:
        # Empty State
        st.info("👋 No data yet! Go to the 'Chat' tab and ask a question to see live metrics here.")
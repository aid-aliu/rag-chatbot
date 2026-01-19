import streamlit as st
import pandas as pd


def render_dashboard():
    st.header("📊 Observability Dashboard")
    st.markdown("Monitor performance, costs, and user feedback in real-time.")

    if "metrics" not in st.session_state:
        st.warning("No metrics data found. Start a conversation to generate data.")
        return

    metrics = st.session_state.metrics

    if len(metrics["queries"]) > 0:
        try:
            # Ensure all arrays are the same length before creating DataFrame
            min_len = min(len(metrics["queries"]), len(metrics["latencies"]), len(metrics["tokens_generated"]))

            df = pd.DataFrame(
                {
                    "Query": metrics["queries"][:min_len],
                    "Latency (s)": metrics["latencies"][:min_len],
                    "Est. Tokens": metrics["tokens_generated"][:min_len],
                }
            )
        except ValueError as e:
            st.error(f"Metrics data mismatch error: {e}")
            return

        total_chats = len(df)
        total_tokens = int(df["Est. Tokens"].sum())
        avg_latency = df["Latency (s)"].mean()
        net_score = metrics.get("feedback_score", 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Queries", total_chats)
        k2.metric("Total Tokens", total_tokens)
        k3.metric("Avg Latency", f"{avg_latency:.2f}s")
        k4.metric("Feedback Score", net_score)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Latency Trend")
            st.caption("Time taken (seconds) per query")
            if not df.empty:
                st.line_chart(df["Latency (s)"])

        with col2:
            st.subheader("🪙 Token Usage")
            st.caption("Estimated length of responses")
            if not df.empty:
                st.bar_chart(df["Est. Tokens"])

        st.divider()

        with st.expander("📄 View Raw Interaction Logs", expanded=True):
            st.dataframe(df, use_container_width=True)

    else:
        st.info("👋 No data yet! Go to the 'Chat' tab and ask a question to see live metrics here.")
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import plotly.graph_objects as go

st.title("Decline Curve Analysis App")

# ── Initialize session state ─────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None
if 't' not in st.session_state:
    st.session_state.t = None
if 'q' not in st.session_state:
    st.session_state.q = None
if 'forecast_days' not in st.session_state:
    st.session_state.forecast_days = 365
if 'econ_limit' not in st.session_state:
    st.session_state.econ_limit = 50.0

# ── Upload CSV ────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
st.caption("⚠️ Date column should be in MM/DD/YYYY format e.g. 1/13/2013")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

    # ── Column selection ──────────────────────────
    cols = df.columns.tolist()
    time_col = st.selectbox("Select TIME column", cols)
    rate_col = st.selectbox("Select RATE column", cols)

    # ── Convert date to T_DAYS ────────────────────
    if df[time_col].dtype == 'object':
        df[time_col] = pd.to_datetime(
            df[time_col],
            format='mixed',
            dayfirst=False
        )

    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df['T_DAYS'] = (df[time_col] - df[time_col].min()).dt.days
        time_col = 'T_DAYS'
        st.success("Date converted to T_DAYS automatically.")

    # ── Convert rate to numeric & remove zeros ────
    df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
    df = df[df[rate_col] > 0].copy()

    # ── Smoothing selector ────────────────────────
    st.subheader("Data Smoothing")
    smoothing = st.selectbox(
        "Select smoothing method:",
        options=[
            "None (raw data)",
            "7-day rolling average",
            "15-day rolling average",
            "30-day rolling average",
            "Monthly average",
            "Median filter (removes spikes)"
        ]
    )

    df['RATE_ORIGINAL'] = df[rate_col].copy()

    if smoothing == "7-day rolling average":
        df[rate_col] = df[rate_col].rolling(
            window=7, center=True, min_periods=1
        ).mean()
        st.success("Applied 7-day rolling average")

    elif smoothing == "15-day rolling average":
        df[rate_col] = df[rate_col].rolling(
            window=15, center=True, min_periods=1
        ).mean()
        st.success("Applied 15-day rolling average")

    elif smoothing == "30-day rolling average":
        df[rate_col] = df[rate_col].rolling(
            window=30, center=True, min_periods=1
        ).mean()
        st.success("Applied 30-day rolling average")

    elif smoothing == "Monthly average":
        df['MONTH_GROUP'] = df[time_col] // 30
        df[rate_col] = df.groupby(
            'MONTH_GROUP'
        )[rate_col].transform('mean')
        st.success("Applied monthly averaging")

    elif smoothing == "Median filter (removes spikes)":
        df[rate_col] = medfilt(
            df[rate_col].values,
            kernel_size=15
        )
        st.success("Applied median filter")

    # ── Full production plot ──────────────────────
    st.subheader("Full Production Profile")
    fig_full = go.Figure()

    if smoothing != "None (raw data)":
        fig_full.add_trace(go.Scatter(
            x=df[time_col],
            y=df['RATE_ORIGINAL'],
            mode='lines',
            name='Original data',
            line=dict(color='gray', width=1),
            opacity=0.4
        ))

    fig_full.add_trace(go.Scatter(
        x=df[time_col],
        y=df[rate_col],
        mode='lines+markers',
        name='Oil Rate',
        marker=dict(size=4),
        line=dict(color='steelblue')
    ))
    fig_full.update_layout(
        title='Full Production Profile — hover to find peak day',
        xaxis_title='Time (days)',
        yaxis_title='Oil Rate (Sm³/day)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_full, use_container_width=True)

    # ── Semilog Plot ──────────────────────────────
    st.subheader("Semilog Plot (log q vs t)")

    fig_semi_full = go.Figure()
    fig_semi_full.add_trace(go.Scatter(
        x=df[time_col],
        y=df[rate_col],
        mode='lines+markers',
        name='Oil Rate',
        marker=dict(size=4),
        line=dict(color='steelblue')
    ))
    fig_semi_full.update_layout(
        title='Semilog Production Profile',
        xaxis_title='Time (days)',
        yaxis_title='log Oil Rate (Sm³/day)',
        yaxis_type='log',
        hovermode='x unified'
    )
    st.plotly_chart(fig_semi_full, use_container_width=True)

    # ── Decline period slider ─────────────────────
    st.subheader("Select Decline Period")
    min_day = int(df[time_col].min())
    max_day = int(df[time_col].max())

    decline_start, decline_end = st.select_slider(
        "Select start and end of decline period",
        options=list(range(min_day, max_day + 1)),
        value=(min_day + (max_day - min_day) // 2, max_day)
    )
    st.info(f"Decline period: day {decline_start} to day {decline_end}")

    preview = df[
        (df[time_col] >= decline_start) &
        (df[time_col] <= decline_end)
    ]
    st.write(f"Points in selected range: {len(preview)}")

    # ── Forecast slider ───────────────────────────
    forecast_days = st.slider(
        "Forecast days ahead",
        min_value=30,
        max_value=3650,
        value=730
    )

    # ── Fit button ────────────────────────────────
    if st.button("Fit Decline Curves"):

        df_clean = df[
            (df[time_col] >= decline_start) &
            (df[time_col] <= decline_end)
        ].copy()

        df_clean = df_clean[[time_col, rate_col]].dropna()
        df_clean = df_clean[df_clean[rate_col] > 0]

        if len(df_clean) == 0:
            st.error("No valid data in selected period.")
            st.stop()

        if len(df_clean) < 5:
            st.error(f"Only {len(df_clean)} rows. Need at least 5.")
            st.stop()

        t = df_clean[time_col].values.astype(float)
        q = df_clean[rate_col].values.astype(float)
        t = t - t[0]

        st.success(
            f"Fitting on {len(t)} points | "
            f"Time: 0 to {max(t):.0f} days"
        )

        # ── Equations ────────────────────────────
        def exponential(t, qi, Di):
            return qi * np.exp(-Di * t)

        def hyperbolic(t, qi, Di, b):
            return qi / (1 + b * Di * t) ** (1 / b)

        def harmonic(t, qi, Di):
            return qi / (1 + Di * t)

        # ── Fit all 3 ────────────────────────────
        results = {}

        try:
            popt_exp, pcov_exp = curve_fit(
                exponential, t, q,
                p0=[q[0], 0.001],
                bounds=([0, 0.0001], [np.inf, 0.1]),
                maxfev=10000
            )
            q_pred = exponential(t, *popt_exp)
            ss_res = np.sum((q - q_pred) ** 2)
            ss_tot = np.sum((q - np.mean(q)) ** 2)
            results['Exponential'] = {
                'params': popt_exp,
                'pcov':   pcov_exp,
                'r2': 1 - ss_res / ss_tot,
                'func': exponential,
                'label': f"qi={popt_exp[0]:.1f}, "
                         f"Di={popt_exp[1]:.5f}"
            }
        except Exception:
            st.warning("Exponential fit failed.")

        try:
            popt_hyp, pcov_hyp = curve_fit(
                hyperbolic, t, q,
                p0=[q[0], 0.001, 0.5],
                bounds=([0, 0.0001, 0.01], [np.inf, 0.1, 2]),
                maxfev=10000
            )
            q_pred = hyperbolic(t, *popt_hyp)
            ss_res = np.sum((q - q_pred) ** 2)
            ss_tot = np.sum((q - np.mean(q)) ** 2)
            results['Hyperbolic'] = {
                'params': popt_hyp,
                'pcov':   pcov_hyp,
                'r2': 1 - ss_res / ss_tot,
                'func': hyperbolic,
                'label': f"qi={popt_hyp[0]:.1f}, "
                         f"Di={popt_hyp[1]:.5f}, "
                         f"b={popt_hyp[2]:.3f}"
            }
        except Exception:
            st.warning("Hyperbolic fit failed.")

        try:
            popt_har, pcov_har = curve_fit(
                harmonic, t, q,
                p0=[q[0], 0.001],
                bounds=([0, 0.0001], [np.inf, 0.1]),
                maxfev=10000
            )
            q_pred = harmonic(t, *popt_har)
            ss_res = np.sum((q - q_pred) ** 2)
            ss_tot = np.sum((q - np.mean(q)) ** 2)
            results['Harmonic'] = {
                'params': popt_har,
                'pcov':   pcov_har,
                'r2': 1 - ss_res / ss_tot,
                'func': harmonic,
                'label': f"qi={popt_har[0]:.1f}, "
                         f"Di={popt_har[1]:.5f}"
            }
        except Exception:
            st.warning("Harmonic fit failed.")

        if not results:
            st.error("All fits failed. Check your data.")
            st.stop()

        # ── Save to session state ─────────────────
        st.session_state.results       = results
        st.session_state.t             = t
        st.session_state.q             = q
        st.session_state.forecast_days = forecast_days

    # ── Show results ──────────────────────────────
    if st.session_state.results is not None:

        results       = st.session_state.results
        t             = st.session_state.t
        q             = st.session_state.q
        forecast_days = st.session_state.forecast_days

        # ── Comparison table ──────────────────────
        st.subheader("Decline Model Comparison")
        best_model = max(results, key=lambda x: results[x]['r2'])

        comparison_data = {
            'Model': [],
            'R²': [],
            'Parameters': [],
            'Recommendation': []
        }
        for name, res in results.items():
            comparison_data['Model'].append(name)
            comparison_data['R²'].append(f"{res['r2']:.4f}")
            comparison_data['Parameters'].append(res['label'])
            comparison_data['Recommendation'].append(
                "✅ Best fit" if name == best_model else ""
            )
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True
        )

        # ── Model selector ────────────────────────
        st.subheader("Select Model for Forecast")
        model_options = list(results.keys())
        default_index = model_options.index(best_model)

        selected_model = st.radio(
            "Choose decline model (highest R² recommended):",
            options=model_options,
            index=default_index,
            horizontal=True
        )

        chosen        = results[selected_model]
        chosen_func   = chosen['func']
        chosen_params = chosen['params']
        chosen_pcov   = chosen.get('pcov', None)

        # ── Metrics ───────────────────────────────
        st.subheader(f"Results — {selected_model} Decline")

        if selected_model == 'Hyperbolic':
            qi_val, Di_val, b_val = chosen_params
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("qi (Sm³/day)", f"{qi_val:.1f}")
            col2.metric("Di (per day)", f"{Di_val:.5f}")
            col3.metric("b factor",     f"{b_val:.3f}")
            col4.metric("R²",           f"{chosen['r2']:.4f}")
        else:
            qi_val, Di_val = chosen_params
            b_val = None
            col1, col2, col3 = st.columns(3)
            col1.metric("qi (Sm³/day)", f"{qi_val:.1f}")
            col2.metric("Di (per day)", f"{Di_val:.5f}")
            col3.metric("R²",           f"{chosen['r2']:.4f}")

        # ── EUR ───────────────────────────────────
        t_max = max(t) + forecast_days

        if selected_model == 'Exponential':
            eur = (qi_val / Di_val) * \
                  (1 - np.exp(-Di_val * t_max))
        elif selected_model == 'Hyperbolic':
            if abs(b_val - 1.0) < 1e-6:
                eur = (qi_val / Di_val) * \
                      np.log(1 + Di_val * t_max)
                st.info("b ≈ 1.0 → using harmonic EUR formula")
            else:
                eur = (qi_val / ((1 - b_val) * Di_val)) * \
                      (1 - (1 + b_val * Di_val * t_max) **
                       ((b_val - 1) / b_val))
        else:
            eur = (qi_val / Di_val) * \
                  np.log(1 + Di_val * t_max)

        st.metric("EUR (Sm³)", f"{eur:,.0f}")

        # ── What to show selector ─────────────────
        st.subheader("Select What to Show")
        options = st.multiselect(
            "Choose what you want to see:",
            options=[
                "📈 Decline Curve Plot",
                "🔴 Abandonment Prediction",
                "📊 Cumulative Production Plot",
                "🎲 Probabilistic Forecast (P10/P50/P90)"
            ],
            default=["📈 Decline Curve Plot"]
        )

        # ── Colors ───────────────────────────────
        colors = {
            'Exponential': 'green',
            'Hyperbolic':  'orangered',
            'Harmonic':    'yellow'
        }

        # ── x axis limit ─────────────────────────
        x_max = max(t) + forecast_days

        # ── Abandonment variables default ─────────
        t_abandon  = None
        years      = 0
        months     = 0
        days_r     = 0
        econ_limit = st.session_state.econ_limit

        # ── Abandonment Prediction ────────────────
        if "🔴 Abandonment Prediction" in options:

            st.subheader("Abandonment Prediction")

            col_a, col_b = st.columns(2)
            with col_a:
                econ_limit_bbl = st.number_input(
                    "Abandonment rate limit (bbl/day)",
                    min_value=1.0,
                    max_value=5000.0,
                    value=50.0,
                    step=5.0
                )
            econ_limit = econ_limit_bbl * 0.158987
            with col_b:
                st.metric(
                    "Equivalent Sm³/day",
                    f"{econ_limit:.2f}"
                )

            st.info(
                f"Well abandoned when rate drops below "
                f"{econ_limit_bbl:.0f} bbl/day "
                f"= {econ_limit:.2f} Sm³/day"
            )

            st.session_state.econ_limit = econ_limit

            def calc_abandonment_time(qi, Di, b,
                                       q_limit, model):
                try:
                    if q_limit >= qi:
                        return None
                    if model == 'Exponential':
                        t_ab = -np.log(q_limit / qi) / Di
                    elif model == 'Hyperbolic':
                        if abs(b - 1.0) < 1e-6:
                            t_ab = (qi / q_limit - 1) / Di
                        else:
                            t_ab = ((qi / q_limit) ** b - 1) \
                                   / (b * Di)
                    else:
                        t_ab = (qi / q_limit - 1) / Di
                    return float(t_ab) if t_ab > 0 else None
                except Exception:
                    return None

            if selected_model == 'Hyperbolic':
                t_abandon = calc_abandonment_time(
                    qi_val, Di_val, b_val,
                    econ_limit, selected_model
                )
            else:
                t_abandon = calc_abandonment_time(
                    qi_val, Di_val, None,
                    econ_limit, selected_model
                )

            if t_abandon is not None:
                years  = int(t_abandon // 365)
                months = int((t_abandon % 365) // 30)
                days_r = int(t_abandon % 30)

                x_max = t_abandon * 1.1

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Abandonment Time",
                    f"{years}y {months}m {days_r}d"
                )
                col2.metric(
                    "Days from decline start",
                    f"{t_abandon:.0f} days"
                )
                col3.metric(
                    "Rate at abandonment",
                    f"{econ_limit_bbl:.1f} bbl/day"
                )

                if selected_model == 'Exponential':
                    eur_ab = (qi_val / Di_val) * \
                             (1 - np.exp(-Di_val * t_abandon))
                elif selected_model == 'Hyperbolic':
                    if abs(b_val - 1.0) < 1e-6:
                        eur_ab = (qi_val / Di_val) * \
                                 np.log(1 + Di_val * t_abandon)
                    else:
                        eur_ab = (qi_val / ((1-b_val)*Di_val)) * \
                                 (1-(1+b_val*Di_val*t_abandon) **
                                  ((b_val-1)/b_val))
                else:
                    eur_ab = (qi_val / Di_val) * \
                             np.log(1 + Di_val * t_abandon)

                st.metric(
                    "EUR at abandonment (Sm³)",
                    f"{eur_ab:,.0f}"
                )

            else:
                st.warning(
                    f"Rate limit {econ_limit_bbl:.1f} bbl/day "
                    f"is higher than initial rate. "
                    f"Well already below limit!"
                )

        # ── Decline Curve Plot ────────────────────
        if "📈 Decline Curve Plot" in options:

            st.subheader("Decline Curve Plot")

            t_fit      = np.linspace(0, max(t), 300)
            t_forecast = np.linspace(max(t), x_max, 300)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=t, y=q,
                mode='markers',
                name='Actual data',
                marker=dict(color='steelblue', size=6)
            ))

            for name, res in results.items():
                fig.add_trace(go.Scatter(
                    x=t_fit,
                    y=res['func'](t_fit, *res['params']),
                    mode='lines',
                    name=f"{name} (R²={res['r2']:.4f})",
                    line=dict(
                        color=colors[name],
                        width=3 if name == selected_model else 1,
                        dash='solid'
                        if name == selected_model else 'dot'
                    )
                ))

            fig.add_trace(go.Scatter(
                x=t_forecast,
                y=chosen_func(t_forecast, *chosen_params),
                mode='lines',
                name=f'{selected_model} Forecast',
                line=dict(
                    color=colors[selected_model],
                    width=2,
                    dash='dash'
                )
            ))

            fig.add_vline(
                x=max(t),
                line_dash='dot',
                line_color='gray',
                annotation_text='Forecast start'
            )

            if "🔴 Abandonment Prediction" in options:
                fig.add_hline(
                    y=econ_limit,
                    line_dash='dash',
                    line_color='red',
                    annotation_text=(
                        f'Rate limit '
                        f'({econ_limit/0.158987:.0f} bbl/day)'
                    ),
                    annotation_position='top right'
                )
                if t_abandon is not None:
                    fig.add_vline(
                        x=t_abandon,
                        line_dash='dash',
                        line_color='red',
                        annotation_text=(
                            f'Abandonment '
                            f'({years}y {months}m)'
                        )
                    )

            fig.update_layout(
                title=f'DCA — {selected_model}',
                xaxis_title='Time (days from decline start)',
                yaxis_title='Oil Rate (Sm³/day)',
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.2),
                xaxis=dict(range=[0, x_max]),
                yaxis=dict(range=[0, max(q) * 1.1])
            )

            st.plotly_chart(fig, use_container_width=True)

        # ── Cumulative Production Plot ────────────
        if "📊 Cumulative Production Plot" in options:

            st.subheader("Cumulative Production")

            Np_actual = np.cumsum(q)

            t_fit_days = np.arange(0, int(max(t)) + 1)
            q_fit_vals = chosen_func(
                t_fit_days, *chosen_params
            )
            Np_fit = np.cumsum(q_fit_vals)

            t_fore_days = np.arange(
                int(max(t)) + 1, int(x_max) + 1
            )
            q_fore_vals = chosen_func(
                t_fore_days, *chosen_params
            )
            Np_forecast = Np_actual[-1] + \
                          np.cumsum(q_fore_vals)

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Cumulative produced (Sm³)",
                f"{Np_actual[-1]:,.0f}"
            )
            col2.metric(
                "Cumulative produced (bbl)",
                f"{Np_actual[-1]/0.158987:,.0f}"
            )
            col3.metric(
                "Forecast cumulative (Sm³)",
                f"{Np_forecast[-1]:,.0f}"
            )

            fig_cum = go.Figure()

            fig_cum.add_trace(go.Scatter(
                x=t,
                y=Np_actual,
                mode='lines+markers',
                name='Actual cumulative',
                marker=dict(size=4),
                line=dict(color='steelblue', width=2)
            ))

            fig_cum.add_trace(go.Scatter(
                x=t_fit_days,
                y=Np_fit,
                mode='lines',
                name=f'{selected_model} fit',
                line=dict(
                    color=colors[selected_model],
                    width=2
                )
            ))

            fig_cum.add_trace(go.Scatter(
                x=t_fore_days,
                y=Np_forecast,
                mode='lines',
                name='Forecast cumulative',
                line=dict(
                    color=colors[selected_model],
                    width=2,
                    dash='dash'
                )
            ))

            fig_cum.add_vline(
                x=max(t),
                line_dash='dot',
                line_color='gray',
                annotation_text='Forecast start'
            )

            if "🔴 Abandonment Prediction" in options \
                    and t_abandon is not None:
                fig_cum.add_vline(
                    x=t_abandon,
                    line_dash='dash',
                    line_color='red',
                    annotation_text=(
                        f'Abandonment '
                        f'({years}y {months}m)'
                    )
                )

            fig_cum.update_layout(
                title=f'Cumulative Production — '
                      f'{selected_model}',
                xaxis_title='Time (days from decline start)',
                yaxis_title='Cumulative Oil (Sm³)',
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.2),
                xaxis=dict(range=[0, x_max]),
                yaxis=dict(
                    range=[0, Np_forecast[-1] * 1.1]
                )
            )

            st.plotly_chart(fig_cum, use_container_width=True)

        # ── Probabilistic Forecast P10/P50/P90 ───
        if "🎲 Probabilistic Forecast (P10/P50/P90)" in options:

            st.subheader("Probabilistic Forecast (P10/P50/P90)")


            n_samples = 1000

            if chosen_pcov is not None and \
               not np.any(np.isinf(chosen_pcov)) and \
               not np.any(np.isnan(chosen_pcov)):

                np.random.seed(42)

                try:
                    param_samples = \
                        np.random.multivariate_normal(
                            mean=list(chosen_params),
                            cov=chosen_pcov,
                            size=n_samples
                        )
                except Exception:
                    param_samples = None
                    st.warning(
                        "Could not sample parameters. "
                        "Covariance matrix invalid."
                    )

                if param_samples is not None:

                    # calculate EUR for each sample
                    eur_samples  = []
                    valid_params = []

                    for params in param_samples:
                        try:
                            if selected_model == 'Exponential':
                                qi_s, Di_s = params
                                if qi_s <= 0 or Di_s <= 0:
                                    continue
                                eur_s = (qi_s / Di_s) * \
                                        (1 - np.exp(
                                            -Di_s * t_max
                                        ))

                            elif selected_model == 'Hyperbolic':
                                qi_s, Di_s, b_s = params
                                if qi_s <= 0 or Di_s <= 0 \
                                        or b_s <= 0 or b_s >= 2:
                                    continue
                                if abs(b_s - 1.0) < 1e-6:
                                    eur_s = (qi_s / Di_s) * \
                                            np.log(
                                                1 + Di_s * t_max
                                            )
                                else:
                                    eur_s = (
                                        qi_s /
                                        ((1-b_s) * Di_s)
                                    ) * (1-(
                                        1 + b_s*Di_s*t_max
                                    )**((b_s-1)/b_s))

                            else:  # Harmonic
                                qi_s, Di_s = params
                                if qi_s <= 0 or Di_s <= 0:
                                    continue
                                eur_s = (qi_s / Di_s) * \
                                        np.log(
                                            1 + Di_s * t_max
                                        )

                            if eur_s > 0:
                                eur_samples.append(eur_s)
                                valid_params.append(params)

                        except Exception:
                            continue

                    if len(eur_samples) > 10:

                        eur_samples  = np.array(eur_samples)
                        valid_params = np.array(valid_params)

                        # calculate percentiles
                        # P10 = optimistic = 90th percentile
                        # P50 = base case  = 50th percentile
                        # P90 = pessimistic= 10th percentile
                        p10_eur = np.percentile(eur_samples, 90)
                        p50_eur = np.percentile(eur_samples, 50)
                        p90_eur = np.percentile(eur_samples, 10)

                        # show EUR metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "P90 EUR — Pessimistic (Sm³)",
                            f"{p90_eur:,.0f}"
                        )
                        col2.metric(
                            "P50 EUR — Base Case (Sm³)",
                            f"{p50_eur:,.0f}"
                        )
                        col3.metric(
                            "P10 EUR — Optimistic (Sm³)",
                            f"{p10_eur:,.0f}"
                        )

                        # find params closest to each percentile
                        def get_params_for_eur(target):
                            idx = np.argmin(
                                np.abs(eur_samples - target)
                            )
                            return valid_params[idx]

                        p10_params = get_params_for_eur(p10_eur)
                        p50_params = get_params_for_eur(p50_eur)
                        p90_params = get_params_for_eur(p90_eur)

                        # generate rate profiles
                        t_all = np.linspace(0, x_max, 500)

                        p10_rates = chosen_func(
                            t_all, *p10_params
                        )
                        p50_rates = chosen_func(
                            t_all, *p50_params
                        )
                        p90_rates = chosen_func(
                            t_all, *p90_params
                        )

                        # ── Probabilistic plot ────────
                        fig_prob = go.Figure()

                        # actual data
                        fig_prob.add_trace(go.Scatter(
                            x=t, y=q,
                            mode='markers',
                            name='Actual data',
                            marker=dict(
                                color='steelblue', size=5
                            )
                        ))

                        # shaded band P10 to P90
                        fig_prob.add_trace(go.Scatter(
                            x=np.concatenate(
                                [t_all, t_all[::-1]]
                            ),
                            y=np.concatenate(
                                [p10_rates, p90_rates[::-1]]
                            ),
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.15)',
                            line=dict(
                                color='rgba(255,255,255,0)'
                            ),
                            name='P10–P90 uncertainty band',
                            showlegend=True
                        ))

                        # P90 pessimistic
                        fig_prob.add_trace(go.Scatter(
                            x=t_all,
                            y=p90_rates,
                            mode='lines',
                            name=f'P90 Pessimistic '
                                 f'(EUR={p90_eur:,.0f} Sm³)',
                            line=dict(
                                color='red',
                                width=2,
                                dash='dash'
                            )
                        ))

                        # P50 base case
                        fig_prob.add_trace(go.Scatter(
                            x=t_all,
                            y=p50_rates,
                            mode='lines',
                            name=f'P50 Base Case '
                                 f'(EUR={p50_eur:,.0f} Sm³)',
                            line=dict(
                                color='orange',
                                width=2,
                                dash='solid'
                            )
                        ))

                        # P10 optimistic
                        fig_prob.add_trace(go.Scatter(
                            x=t_all,
                            y=p10_rates,
                            mode='lines',
                            name=f'P10 Optimistic '
                                 f'(EUR={p10_eur:,.0f} Sm³)',
                            line=dict(
                                color='green',
                                width=2,
                                dash='dash'
                            )
                        ))

                        # forecast start vline
                        fig_prob.add_vline(
                            x=max(t),
                            line_dash='dot',
                            line_color='gray',
                            annotation_text='Forecast start'
                        )

                        fig_prob.update_layout(
                            title=f'Probabilistic Forecast '
                                  f'— {selected_model} | '
                                  f'P10 / P50 / P90',
                            xaxis_title='Time (days from '
                                        'decline start)',
                            yaxis_title='Oil Rate (Sm³/day)',
                            hovermode='x unified',
                            legend=dict(
                                orientation='h', y=-0.2
                            ),
                            xaxis=dict(range=[0, x_max]),
                            yaxis=dict(
                                range=[0, max(q) * 1.1]
                            )
                        )

                        st.plotly_chart(
                            fig_prob,
                            use_container_width=True
                        )

                        # ── EUR histogram ─────────────
                        st.subheader("EUR Distribution")
                        fig_hist = go.Figure()

                        fig_hist.add_trace(go.Histogram(
                            x=eur_samples,
                            nbinsx=50,
                            name='EUR distribution',
                            marker_color='steelblue',
                            opacity=0.7
                        ))

                        fig_hist.add_vline(
                            x=p90_eur,
                            line_dash='dash',
                            line_color='red',
                            annotation_text='P90'
                        )
                        fig_hist.add_vline(
                            x=p50_eur,
                            line_dash='dash',
                            line_color='orange',
                            annotation_text='P50'
                        )
                        fig_hist.add_vline(
                            x=p10_eur,
                            line_dash='dash',
                            line_color='green',
                            annotation_text='P10'
                        )

                        fig_hist.update_layout(
                            title='EUR Probability '
                                  'Distribution '
                                  '(Monte Carlo 1000 samples)',
                            xaxis_title='EUR (Sm³)',
                            yaxis_title='Count',
                            hovermode='x unified'
                        )

                        st.plotly_chart(
                            fig_hist,
                            use_container_width=True
                        )

                    else:
                        st.warning(
                            "Not enough valid samples. "
                            "Try different decline period."
                        )

            else:
                st.warning(
                    "Probabilistic forecast not available. "
                    "Covariance matrix could not be computed."
                )
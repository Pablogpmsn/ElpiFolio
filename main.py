import streamlit as st
import pandas as pd
import numpy as np
import itertools
import math
from scipy import stats
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster import hierarchy as hc
from scipy.spatial import distance as ssd

st.set_page_config(layout="wide")
st.title('Herramientas de Análisis de Portafolios de Trading')


# --- FUNCIÓN DE CARGA OPTIMIZADA (MODIFICADA) ---
@st.cache_data
def load_and_clean_data(uploaded_files):
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, sep=';', low_memory=False)
        df.columns = df.columns.str.strip()
        if 'Result name' not in df.columns:
            df['Result name'] = uploaded_file.name
        if 'Close time' in df.columns:
            df['Close time'] = pd.to_datetime(df['Close time'], errors='coerce')

        cols_to_numeric = ['Profit/Loss', 'MAE ($)', 'Size', 'Balance']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        keep_cols = [c for c in ['Close time', 'Profit/Loss', 'Result name'] if c in df.columns]
        df.dropna(subset=keep_cols, inplace=True)

        if 'Size' in df.columns:
            df = df[df['Size'] > 0]

        dfs.append(df)

    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# --- MOTOR DE SIMULACIÓN (INTERÉS COMPUESTO) ---
def run_portfolio_simulation(trades_df, initial_capital, risk_percent, r_multiple_cap=0, fixed_risk_amount=0,
                             riesgo_historico_por_trade=0):
    if trades_df.empty or 'Close time' not in trades_df.columns or trades_df['Close time'].isna().all():
        return pd.Series([initial_capital], index=[pd.Timestamp.min]), initial_capital, 0

    trades_df = trades_df.sort_values(by='Close time').copy()

    if riesgo_historico_por_trade > 0:
        if 'Profit/Loss' not in trades_df.columns:
            st.error("Para el motor de Riesgo Histórico, se necesita la columna 'Profit/Loss'.")
            return pd.Series([initial_capital]), initial_capital, 0
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / riesgo_historico_por_trade
    else:
        if 'MAE ($)' not in trades_df.columns or 'Profit/Loss' not in trades_df.columns:
            st.error("Para el motor de Riesgo por MAE, se necesitan las columnas 'MAE ($)' y 'Profit/Loss'.")
            return pd.Series([initial_capital]), initial_capital, 0
        trades_df = trades_df[trades_df['MAE ($)'].abs() > 0.0001].copy()
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / trades_df['MAE ($)'].abs()

    trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    trades_df.dropna(subset=['R_multiple'], inplace=True)

    if trades_df.empty: return pd.Series([initial_capital], index=[pd.Timestamp.min]), initial_capital, 0

    if r_multiple_cap > 0:
        trades_df['R_multiple'] = np.where(trades_df['R_multiple'] > r_multiple_cap, r_multiple_cap,
                                           trades_df['R_multiple'])

    balance = initial_capital
    peak_balance = initial_capital
    max_drawdown_pct = 0.0
    equity_curve_list = [initial_capital]

    for _, trade in trades_df.iterrows():
        if balance <= 0:
            equity_curve_list.append(0)
            continue

        base_risk_amount = fixed_risk_amount if fixed_risk_amount > 0 else balance * (risk_percent / 100)
        risk_for_this_trade_usd = min(base_risk_amount, balance)
        simulated_profit = risk_for_this_trade_usd * trade['R_multiple']
        balance += simulated_profit
        peak_balance = max(peak_balance, balance)

        if peak_balance > 0:
            current_drawdown_pct = ((peak_balance - balance) / peak_balance) * 100
            max_drawdown_pct = max(max_drawdown_pct, current_drawdown_pct)

        equity_curve_list.append(balance)

    equity_series = pd.Series(equity_curve_list)
    final_balance = equity_series.iloc[-1]

    start_time = trades_df['Close time'].min() if not trades_df.empty else pd.Timestamp.now()
    plot_index = pd.to_datetime([start_time - pd.Timedelta(seconds=1)] + list(trades_df['Close time']))
    equity_series.index = plot_index

    return equity_series, final_balance, max_drawdown_pct


# --- FUNCIÓN PARA CÁLCULO HISTÓRICO AISLADO Y COMBINADO (RECONSTRUYE LA CURVA DE BALANCE) ---
def calculate_isolated_historical_performance(trades_df):
    if trades_df.empty or len(trades_df) < 2 or 'Close time' not in trades_df.columns or trades_df[
        'Close time'].isna().all():
        return pd.Series(), 0, 0, 0, False

    df = trades_df.sort_values(by='Close time').copy()

    if 'Balance' not in df.columns or 'Profit/Loss' not in df.columns or df['Balance'].isna().any():
        return pd.Series(), 0, 0, 0, False

    first_trade = df.iloc[0]
    start_balance = first_trade['Balance'] - first_trade['Profit/Loss']

    if start_balance <= 0: return pd.Series(), 0, 0, 0, False

    cumulative_profit = df['Profit/Loss'].cumsum()
    isolated_equity_curve = start_balance + cumulative_profit

    equity_curve_for_dd = pd.concat([pd.Series([start_balance]), isolated_equity_curve], ignore_index=True)

    final_balance = equity_curve_for_dd.iloc[-1]

    peak_equity = equity_curve_for_dd.cummax()
    drawdown_series = (peak_equity - equity_curve_for_dd) / peak_equity
    max_drawdown_pct = drawdown_series.max() * 100 if peak_equity.max() > 0 else 0.0

    plot_index = pd.to_datetime([df['Close time'].min() - pd.Timedelta(seconds=1)] + list(df['Close time']))
    equity_curve_with_time = pd.Series(equity_curve_for_dd.values, index=plot_index, name="Capital")

    return equity_curve_with_time, start_balance, final_balance, max_drawdown_pct, True


# --- FUNCIÓN PARA CALCULAR LA CONTRIBUCIÓN MEDIA A LOS MESES DE PÉRDIDAS ---
def calculate_average_contribution_to_losing_months(df, group_by_cols):
    if df.empty or 'Profit/Loss' not in df.columns:
        return {}

    df_copy = df.copy()
    if 'Close time' not in df_copy.columns or df_copy['Close time'].isna().all():
        return {}
    df_copy['YearMonth'] = df_copy['Close time'].dt.to_period('M')

    monthly_pnl = df_copy.groupby('YearMonth')['Profit/Loss'].sum()
    losing_months_pnl = monthly_pnl[monthly_pnl < 0]

    if losing_months_pnl.empty:
        return {}

    grouped_pnl = df_copy.groupby(['YearMonth'] + group_by_cols)['Profit/Loss'].sum()

    contributions = {}
    for month, total_loss in losing_months_pnl.items():
        if month in grouped_pnl.index:
            pnl_in_month = grouped_pnl.loc[month]
            for group_key, group_pnl in pnl_in_month.items():
                percentage = (group_pnl / total_loss) * 100
                if group_key not in contributions:
                    contributions[group_key] = []
                contributions[group_key].append(percentage)

    avg_contributions = {key: np.mean(val) for key, val in contributions.items()}
    return avg_contributions


# --- OTRAS FUNCIONES DE CÁLCULO ---
def calculate_drawdown_stats(equity_curve, drawdown_threshold_pct=2.0):
    if equity_curve.empty or len(equity_curve) < 2:
        return {'avg_recovery_time': pd.Timedelta(0), 'max_recovery_time': pd.Timedelta(0),
                'avg_time_between_dds': pd.Timedelta(0)}

    high_water_mark = equity_curve.cummax()
    drawdown_series = (high_water_mark - equity_curve) / high_water_mark

    in_drawdown = False
    drawdowns = []
    current_dd_start_date = None

    for i in range(len(equity_curve)):
        date = equity_curve.index[i]
        value = equity_curve.iloc[i]

        is_in_dd_currently = value < high_water_mark.iloc[i]

        if not in_drawdown and is_in_dd_currently:
            in_drawdown = True
            current_dd_start_date = date

        elif in_drawdown and not is_in_dd_currently:
            in_drawdown = False
            max_dd_in_period = drawdown_series.loc[current_dd_start_date:date].max()
            if max_dd_in_period * 100 >= drawdown_threshold_pct:
                drawdowns.append({
                    'start_date': current_dd_start_date,
                    'end_date': date,
                    'duration': date - current_dd_start_date
                })

    if not drawdowns:
        return {'avg_recovery_time': pd.Timedelta(0), 'max_recovery_time': pd.Timedelta(0),
                'avg_time_between_dds': pd.Timedelta(0)}

    recovery_times = [d['duration'] for d in drawdowns]
    avg_recovery_time = sum(recovery_times, pd.Timedelta(0)) / len(recovery_times)
    max_recovery_time = max(recovery_times)

    start_dates = [d['start_date'] for d in drawdowns]
    time_between_dds_list = []
    if len(start_dates) > 1:
        for i in range(1, len(start_dates)):
            time_between_dds_list.append(start_dates[i] - start_dates[i - 1])

    avg_time_between_dds = sum(time_between_dds_list, pd.Timedelta(0)) / len(
        time_between_dds_list) if time_between_dds_list else pd.Timedelta(0)

    return {
        'avg_recovery_time': avg_recovery_time,
        'max_recovery_time': max_recovery_time,
        'avg_time_between_dds': avg_time_between_dds,
    }


def calculate_new_high_stats(equity_curve):
    if equity_curve.empty or len(equity_curve) < 2:
        return {'avg_time_to_new_high': pd.Timedelta(0), 'max_time_to_new_high': pd.Timedelta(0)}

    high_water_mark = equity_curve.cummax()

    new_high_dates = equity_curve.index[high_water_mark > high_water_mark.shift(1).fillna(0)]

    if len(new_high_dates) < 2:
        return {'avg_time_to_new_high': pd.Timedelta(0), 'max_time_to_new_high': pd.Timedelta(0)}

    time_between_highs = (new_high_dates[1:] - new_high_dates[:-1])

    if time_between_highs.empty:
        return {'avg_time_to_new_high': pd.Timedelta(0), 'max_time_to_new_high': pd.Timedelta(0)}

    avg_time = sum(time_between_highs, pd.Timedelta(0)) / len(time_between_highs)
    max_time = max(time_between_highs)

    return {
        'avg_time_to_new_high': avg_time,
        'max_time_to_new_high': max_time
    }


def calculate_cagr(start_balance, end_balance, trades_df):
    if trades_df.empty or start_balance <= 0 or 'Close time' not in trades_df.columns or trades_df[
        'Close time'].isna().all(): return 0.0
    start_date, end_date = trades_df['Close time'].min(), trades_df['Close time'].max()
    num_days = (end_date - start_date).days
    if num_days < 1: return ((end_balance / start_balance) - 1) * 100
    num_years = num_days / 365.25
    if num_years == 0: return 0.0
    cagr = ((end_balance / start_balance) ** (1 / num_years)) - 1
    return cagr * 100


def calculate_mar(cagr, max_drawdown_pct):
    if max_drawdown_pct <= 0: return np.inf
    return cagr / max_drawdown_pct


def get_mar_for_portfolio(_trades_df, calculation_mode, initial_capital, risk_per_trade_pct, r_multiple_cap,
                          fixed_risk_amount, riesgo_historico_por_trade):
    mar = 0
    if _trades_df.empty or len(_trades_df) < 2: return 0
    if calculation_mode == 'Simulación (Interés Compuesto)':
        _, final_balance, dd = run_portfolio_simulation(_trades_df, initial_capital, risk_per_trade_pct, r_multiple_cap,
                                                        fixed_risk_amount, riesgo_historico_por_trade)
        if initial_capital > 0 and final_balance > 0:
            cagr = calculate_cagr(initial_capital, final_balance, _trades_df)
            mar = calculate_mar(cagr, dd)
    else:
        _, start, end, dd, can_calc = calculate_isolated_historical_performance(_trades_df)
        if can_calc and start > 0 and end > 0:
            cagr = calculate_cagr(start, end, _trades_df)
            mar = calculate_mar(cagr, dd)
    return mar if np.isfinite(mar) else 0


@st.cache_data
def get_strategy_returns_for_asset(_asset_df):
    if _asset_df.empty or 'Result name' not in _asset_df.columns or 'Close time' not in _asset_df.columns: return pd.DataFrame()
    all_returns = []
    df_indexed = _asset_df.set_index('Close time').sort_index()
    for strategy in df_indexed['Result name'].unique():
        strategy_df = df_indexed[df_indexed['Result name'] == strategy]
        if not strategy_df.empty:
            equity_curve = 10000 + strategy_df['Profit/Loss'].cumsum()
            daily_returns = equity_curve.resample('D').last().ffill().pct_change().fillna(0)
            daily_returns.name = strategy
            all_returns.append(daily_returns)
    if not all_returns: return pd.DataFrame()
    return pd.concat(all_returns, axis=1).fillna(0)


def calculate_historical_monthly_performance(trades_df, initial_capital, risk_per_trade_pct, r_multiple_cap=0,
                                             fixed_risk_amount=0, riesgo_historico_por_trade=0):
    if trades_df.empty: return 0, (0, 0), 0
    trades_df['YearMonth'] = trades_df['Close time'].dt.to_period('M')
    monthly_pnl, total_months = [], trades_df['YearMonth'].nunique()
    if total_months == 0: return 0, (0, 0), 0
    for month in trades_df['YearMonth'].unique():
        month_trades = trades_df[trades_df['YearMonth'] == month]
        _, final_balance, _ = run_portfolio_simulation(month_trades, initial_capital, risk_per_trade_pct,
                                                       r_multiple_cap, fixed_risk_amount, riesgo_historico_por_trade)
        pnl = final_balance - initial_capital
        monthly_pnl.append(pnl)
    pnl_array = np.array(monthly_pnl)
    prob_negative_month = (np.sum(pnl_array < 0) / total_months) * 100
    winning_months = pnl_array[pnl_array > 0];
    losing_months = pnl_array[pnl_array < 0]
    gain_p5 = np.percentile(winning_months, 5) if len(winning_months) > 0 else 0
    loss_ci_95 = (0, 0)
    if len(losing_months) > 1:
        mean_loss, std_err_loss = np.mean(losing_months), stats.sem(losing_months)
        loss_ci_95 = stats.t.interval(0.95, len(losing_months) - 1, loc=mean_loss, scale=std_err_loss)
    elif len(losing_months) == 1:
        loss_ci_95 = (losing_months[0], losing_months[0])
    return gain_p5, loss_ci_95, prob_negative_month


def calculate_monthly_recovery_dd(portfolio_trades, initial_capital, risk_per_trade_pct, r_multiple_cap=0,
                                  fixed_risk_amount=0, riesgo_historico_por_trade=0):
    if portfolio_trades.empty: return 0
    trades_by_month = portfolio_trades.groupby(pd.Grouper(key='Close time', freq='M'))
    max_recoverable_dd = 0
    for _, month_trades in trades_by_month:
        if month_trades.empty or month_trades['Profit/Loss'].sum() <= 0: continue
        _, _, monthly_max_dd = run_portfolio_simulation(month_trades.copy(), initial_capital, risk_per_trade_pct,
                                                        r_multiple_cap, fixed_risk_amount, riesgo_historico_por_trade)
        max_recoverable_dd = max(max_recoverable_dd, monthly_max_dd)
    return max_recoverable_dd

def get_strategy_level_stats(trades_df):
    """
    Calcula métricas básicas por estrategia (Result name) usando el histórico aislado de cada una.
    Devuelve un DataFrame con:
      - Result name
      - CAGR
      - MaxDD
      - MAR
      - NumTrades
    """
    if trades_df.empty or 'Result name' not in trades_df.columns:
        return pd.DataFrame()

    stats = []

    for strat_name, df in trades_df.groupby('Result name'):
        # Curva histórica aislada de la estrategia
        equity_curve, start_balance, end_balance, max_dd, can_calc = calculate_isolated_historical_performance(df)

        if not can_calc or start_balance <= 0 or end_balance <= 0:
            continue

        cagr = calculate_cagr(start_balance, end_balance, df)
        mar = calculate_mar(cagr, max_dd)

        stats.append({
            "Result name": strat_name,
            "CAGR": cagr,
            "MaxDD": max_dd,
            "MAR": mar,
            "NumTrades": len(df)
        })

    if not stats:
        return pd.DataFrame()

    return pd.DataFrame(stats)

def calculate_ror(edge, capital_units):
    if edge <= 0: return 100.0
    try:
        base = (1 - edge) / (1 + edge)
        return (base ** capital_units) * 100
    except (OverflowError, ZeroDivisionError):
        return 0.0


def calculate_risk_metrics(trades_df, risk_per_trade_pct_param):
    if trades_df.empty: return {}
    win_trades, loss_trades = trades_df[trades_df['Profit/Loss'] > 0], trades_df[trades_df['Profit/Loss'] < 0]
    win_rate = (len(win_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    payoff_ratio = (win_trades['Profit/Loss'].mean() / abs(loss_trades['Profit/Loss'].mean())) if len(
        loss_trades) > 0 and len(win_trades) > 0 else 0
    profit_factor = (win_trades['Profit/Loss'].sum() / abs(loss_trades['Profit/Loss'].sum())) if len(
        loss_trades) > 0 else float('inf')
    edge = (win_rate / 100 * payoff_ratio) - ((100 - win_rate) / 100)
    capital_units = 1 / (risk_per_trade_pct_param / 100) if risk_per_trade_pct_param > 0 else float('inf')
    risk_of_ruin = calculate_ror(edge, capital_units)
    avg_time_per_trade = pd.Timedelta(0)
    if 'Close time' in trades_df.columns and not trades_df['Close time'].isna().all() and len(trades_df) > 1:
        trades_df_sorted = trades_df.sort_values(by='Close time')
        total_duration = trades_df_sorted['Close time'].max() - trades_df_sorted['Close time'].min()
        avg_time_per_trade = total_duration / (len(trades_df_sorted) - 1)
    return {"Win Rate (%)": win_rate, "Profit Factor": profit_factor, "Payoff Ratio": payoff_ratio,
            "Riesgo de Ruina Teórico (%)": risk_of_ruin, "Edge": edge, "Capital Units": capital_units,
            "Avg Time per Trade": avg_time_per_trade}


def run_monte_carlo_simulation_vectorized(trades_df, initial_capital, risk_per_trade_pct, ruin_level_pct,
                                          num_simulations=2000, r_multiple_cap=0, fixed_risk_amount=0,
                                          riesgo_historico_por_trade=0):
    trades_df = trades_df.copy()
    if riesgo_historico_por_trade > 0:
        if 'Profit/Loss' not in trades_df.columns:
            st.warning("No se puede ejecutar Monte Carlo sin la columna 'Profit/Loss'.");
            return 0, 0, 0
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / riesgo_historico_por_trade
    else:
        if 'MAE ($)' not in trades_df.columns or 'MAE ($)'.abs().sum() == 0:
            st.warning("No se puede ejecutar Monte Carlo sin datos de MAE ($) válidos.");
            return 0, 0, 0
        trades_df = trades_df[trades_df['MAE ($)'].abs() > 0.0001].copy()
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / trades_df['MAE ($)'].abs()
    trades_df.replace([np.inf, -np.inf], np.nan, inplace=True);
    trades_df.dropna(subset=['R_multiple'], inplace=True)
    trade_outcomes_R = trades_df['R_multiple'].values
    if len(trade_outcomes_R) == 0: return 0, 0, 0
    risk_fraction = (fixed_risk_amount / initial_capital) if fixed_risk_amount > 0 and initial_capital > 0 else (
                risk_per_trade_pct / 100)
    ruin_threshold = initial_capital * (1 - ruin_level_pct / 100)
    max_trades = len(trade_outcomes_R) * 2
    random_R_matrix = np.random.choice(trade_outcomes_R, size=(max_trades, num_simulations), replace=True)
    balances = np.zeros((max_trades + 1, num_simulations));
    balances[0] = initial_capital
    for t in range(1, max_trades + 1):
        pnl = balances[t - 1] * risk_fraction * random_R_matrix[t - 1]
        balances[t] = balances[t - 1] + pnl
        balances[t][balances[t] < 0] = 0
    ruined_mask = np.any(balances <= ruin_threshold, axis=0)
    prob_of_ruin = np.mean(ruined_mask) * 100
    lower_bound, upper_bound, ruined_sims_count = 0, 0, np.sum(ruined_mask)
    if ruined_sims_count > 30:
        ruined_at_step = np.argmax(balances[:, ruined_mask] <= ruin_threshold, axis=0)
        mean_lifespan, std_dev = np.mean(ruined_at_step), np.std(ruined_at_step)
        margin_of_error = 1.96 * (std_dev / np.sqrt(ruined_sims_count))
        lower_bound, upper_bound = max(0, mean_lifespan - margin_of_error), mean_lifespan + margin_of_error
    return prob_of_ruin, lower_bound, upper_bound


def calculate_trade_frequency(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0

    df_sorted = trades_df.sort_values('Close time')
    duration_days = (df_sorted['Close time'].iloc[-1] - df_sorted['Close time'].iloc[0]).days

    if duration_days < 1:
        return 0

    num_trades = len(trades_df)
    trades_per_month = (num_trades / duration_days) * 30.44
    return trades_per_month


def run_growth_projection_mc(trades_df, initial_capital, risk_per_trade_pct, num_simulations, num_trades_to_project,
                             r_multiple_cap=0, fixed_risk_amount=0, riesgo_historico_por_trade=0):
    trades_df = trades_df.copy()
    if riesgo_historico_por_trade > 0:
        if 'Profit/Loss' not in trades_df.columns:
            st.warning("No se puede ejecutar la Proyección sin la columna 'Profit/Loss'.")
            return np.array([initial_capital] * num_simulations)
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / riesgo_historico_por_trade
    else:
        if 'MAE ($)' not in trades_df.columns or 'Profit/Loss' not in trades_df.columns or trades_df[
            'MAE ($)'].abs().sum() == 0:
            st.warning("No se puede ejecutar la Proyección sin datos de MAE ($) y Profit/Loss válidos.")
            return np.array([initial_capital] * num_simulations)
        trades_df = trades_df[trades_df['MAE ($)'].abs() > 0.0001].copy()
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / trades_df['MAE ($)'].abs()

    trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    trades_df.dropna(subset=['R_multiple'], inplace=True)

    if r_multiple_cap > 0:
        trades_df['R_multiple'] = np.where(trades_df['R_multiple'] > r_multiple_cap, r_multiple_cap,
                                           trades_df['R_multiple'])

    trade_outcomes_R = trades_df['R_multiple'].values
    if len(trade_outcomes_R) == 0:
        return np.array([initial_capital] * num_simulations)

    use_fixed_risk = fixed_risk_amount > 0
    risk_value = fixed_risk_amount if use_fixed_risk else (risk_per_trade_pct / 100)

    random_R_matrix = np.random.choice(trade_outcomes_R, size=(num_trades_to_project, num_simulations), replace=True)

    balances = np.zeros((num_trades_to_project + 1, num_simulations))
    balances[0] = initial_capital

    for t in range(1, num_trades_to_project + 1):
        if use_fixed_risk:
            risk_amount_for_trade = np.minimum(risk_value, balances[t - 1])
            pnl = risk_amount_for_trade * random_R_matrix[t - 1]
        else:
            pnl = balances[t - 1] * risk_value * random_R_matrix[t - 1]

        balances[t] = balances[t - 1] + pnl
        balances[t][balances[t] < 0] = 0

    return balances[-1]


def format_timedelta(td):
    if pd.isna(td) or td.total_seconds() == 0: return "N/A"
    days = td.days
    if days > 365: return f"{days / 365.25:.1f} Años"
    if days > 60: return f"{days / 30.44:.1f} Meses"
    if days > 14: return f"{days / 7:.1f} Semanas"
    return f"{days:.1f} Días"


def get_portfolio_metrics_final(_data, combo_tuple, _calculation_mode, _initial_capital, _risk_per_trade_pct,
                                _r_multiple_cap, _fixed_risk_amount, _riesgo_historico_por_trade):
    portfolio_trades = _data[_data['Result name'].isin(list(combo_tuple))]
    mar_ratio, dd, profit = 0, 100, -100
    if portfolio_trades.empty: return mar_ratio, dd, profit
    if _calculation_mode == 'Simulación (Interés Compuesto)':
        _, final_balance, dd_sim = run_portfolio_simulation(portfolio_trades, _initial_capital, _risk_per_trade_pct,
                                                            _r_multiple_cap, _fixed_risk_amount,
                                                            _riesgo_historico_por_trade)
        if final_balance > _initial_capital * 0.1:
            cagr = calculate_cagr(_initial_capital, final_balance, portfolio_trades)
            mar_ratio, dd = calculate_mar(cagr, dd_sim), dd_sim
            profit = ((final_balance / _initial_capital) - 1) * 100
    else:
        _, start, end, dd_hist, can_calc = calculate_isolated_historical_performance(portfolio_trades)
        if can_calc and start > 0:
            cagr = calculate_cagr(start, end, portfolio_trades)
            mar_ratio, dd = calculate_mar(cagr, dd_hist), dd_hist
            profit = ((end / start) - 1) * 100 if start > 0 else 0
    return mar_ratio, dd, profit


@st.cache_data
def get_portfolio_metrics_cached(_data_subset_tuple, _columns_list, combo_tuple, _calculation_mode, _initial_capital,
                                 _risk_per_trade_pct, _r_multiple_cap, _fixed_risk_amount, _riesgo_historico_por_trade):
    _data = pd.DataFrame.from_records(_data_subset_tuple, columns=_columns_list)
    _data['Close time'] = pd.to_datetime(_data['Close time'])
    return get_portfolio_metrics_final(_data, combo_tuple, _calculation_mode, _initial_capital, _risk_per_trade_pct,
                                       _r_multiple_cap, _fixed_risk_amount, _riesgo_historico_por_trade)


# --- INICIO: FUNCIONES PARA HIERARCHICAL RISK PARITY (HRP) Y OPTIMAL F ---
def get_daily_returns(trades_df):
    if trades_df.empty or 'Result name' not in trades_df.columns:
        return pd.DataFrame()
    all_returns = []
    df_indexed = trades_df.set_index('Close time').sort_index()
    for strategy in df_indexed['Result name'].unique():
        strategy_df = df_indexed[df_indexed['Result name'] == strategy]
        if not strategy_df.empty:
            equity_curve = 1 + strategy_df['Profit/Loss'].cumsum()
            daily_returns = equity_curve.resample('D').last().ffill().pct_change().fillna(0)
            daily_returns.name = strategy
            all_returns.append(daily_returns)
    if not all_returns: return pd.DataFrame()
    return pd.concat(all_returns, axis=1).fillna(0)


def get_ivp(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov, c_items):
    cov_ = cov.loc[c_items, c_items]
    w_ = get_ivp(cov_).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var


def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def recursive_bisection_alloc(cov, assets):
    w = pd.Series(1, index=assets)
    c_items = [assets]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w


def get_hrp_weights(returns_df):
    if returns_df.empty or len(returns_df.columns) < 2: return None
    cov, corr = returns_df.cov(), returns_df.corr()
    dist = np.sqrt((1 - corr) / 2)
    link = hc.linkage(ssd.squareform(dist), 'single')
    sort_ix = get_quasi_diag(link)
    sorted_assets = corr.index[sort_ix].tolist()
    hrp_w = recursive_bisection_alloc(cov, sorted_assets)
    return hrp_w.sort_index()


def create_hrp_resampled_portfolio(champion_trades_df, hrp_weights):
    if champion_trades_df.empty or hrp_weights is None or hrp_weights.empty: return pd.DataFrame()
    trades_by_strategy = {
        strategy: df_strat.sort_values('Close time') for strategy, df_strat in champion_trades_df.groupby('Result name')
    }
    valid_strategies = [s for s in hrp_weights.index if s in trades_by_strategy and not trades_by_strategy[s].empty]
    if not valid_strategies: return pd.DataFrame()
    weights_series = hrp_weights[valid_strategies]
    weights_series /= weights_series.sum()
    strategy_names = weights_series.index.tolist()
    strategy_weights = weights_series.values
    total_trades = len(champion_trades_df)
    chosen_strategies = np.random.choice(strategy_names, size=total_trades, p=strategy_weights, replace=True)
    trade_counters = {strategy: 0 for strategy in strategy_names}
    new_portfolio_trades = []
    original_trades_sorted = champion_trades_df.sort_values('Close time')
    for i in range(total_trades):
        strategy_to_pick = chosen_strategies[i]
        strategy_trade_list = trades_by_strategy[strategy_to_pick]
        trade_idx = trade_counters[strategy_to_pick]
        trade_to_add = strategy_trade_list.iloc[trade_idx % len(strategy_trade_list)].copy()
        trade_to_add['Close time'] = original_trades_sorted.iloc[i]['Close time']
        new_portfolio_trades.append(trade_to_add)
        trade_counters[strategy_to_pick] += 1
    if not new_portfolio_trades: return pd.DataFrame()
    return pd.DataFrame(new_portfolio_trades).sort_values('Close time').reset_index(drop=True)


def calculate_optimal_f(trades_df, riesgo_historico_por_trade):
    if trades_df.empty or riesgo_historico_por_trade <= 0: return 0.0
    if 'Profit/Loss' not in trades_df.columns: return 0.0

    r_multiples = trades_df['Profit/Loss'] / riesgo_historico_por_trade
    r_multiples = r_multiples.dropna()
    if r_multiples.empty: return 0.0

    best_f, max_twr = 0, 1.0

    for f_int in range(1, 201):
        f = f_int / 200.0
        growth_factors = 1 + f * r_multiples
        if (growth_factors <= 0).any():
            twr = 0
        else:
            log_twr = np.sum(np.log(growth_factors))
            twr = np.exp(log_twr)
        if twr > max_twr:
            max_twr = twr
            best_f = f
    return best_f


# --- FIN: FUNCIONES PARA HIERARCHICAL RISK PARITY (HRP) Y OPTIMAL F ---

# --- INICIO: NUEVAS FUNCIONES PARA CALIBRACIÓN DE RIESGO ---
def run_operational_simulation_with_rebalancing(
        trades_df, initial_capital, risk_percent, hrp_weights,
        r_multiple_cap=0, riesgo_historico_por_trade=0, fixed_risk_amount=0
):
    """
    Motor de simulación de alta fidelidad que replica la operativa real con "cajas de capital"
    y rebalanceo semanal.
    """
    if trades_df.empty or 'Close time' not in trades_df.columns or trades_df[
        'Close time'].isna().all() or hrp_weights is None:
        return pd.Series([initial_capital]), initial_capital, 0

    # 1. Preparar trades y calcular R-múltiples
    trades_df = trades_df.sort_values(by='Close time').copy()
    if riesgo_historico_por_trade > 0:
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / riesgo_historico_por_trade
    else:
        if 'MAE ($)' not in trades_df.columns or 'MAE ($)'.abs().sum() < 0.0001:
            return pd.Series([initial_capital]), initial_capital, 0
        trades_df = trades_df[trades_df['MAE ($)'].abs() > 0.0001].copy()
        trades_df['R_multiple'] = trades_df['Profit/Loss'] / trades_df['MAE ($)'].abs()

    trades_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    trades_df.dropna(subset=['R_multiple'], inplace=True)
    if r_multiple_cap > 0:
        trades_df['R_multiple'] = np.where(trades_df['R_multiple'] > r_multiple_cap, r_multiple_cap,
                                           trades_df['R_multiple'])
    if trades_df.empty:
        return pd.Series([initial_capital]), initial_capital, 0

    # 2. Inicializar cajas de capital, riesgos y curva de equity
    strategy_names = hrp_weights.index.tolist()
    capital_boxes = {name: initial_capital * weight for name, weight in hrp_weights.items()}
    weekly_risk_euros = {}

    total_balance = initial_capital
    peak_balance = initial_capital
    max_drawdown_pct = 0.0

    equity_points = [{'time': trades_df['Close time'].iloc[0] - pd.Timedelta(seconds=1), 'balance': initial_capital}]

    current_week_identifier = None

    # 3. Bucle principal a través de los trades
    for _, trade in trades_df.iterrows():
        trade_week_identifier = (trade['Close time'].year, trade['Close time'].isocalendar().week)

        # 4. Comprobación de Rebalanceo Semanal
        if trade_week_identifier != current_week_identifier:
            current_week_identifier = trade_week_identifier
            for strat_name in strategy_names:
                box_capital = capital_boxes.get(strat_name, 0)
                # La lógica de riesgo debe coincidir con la simulación principal
                risk_amount = fixed_risk_amount if fixed_risk_amount > 0 else box_capital * (risk_percent / 100)
                weekly_risk_euros[strat_name] = risk_amount if risk_amount > 0 else 0

        # 5. Procesar el trade
        trade_strategy = trade['Result name']
        if trade_strategy not in strategy_names:
            continue

        risk_for_this_trade_usd = weekly_risk_euros.get(trade_strategy, 0)

        if risk_for_this_trade_usd > capital_boxes.get(trade_strategy, 0):
            risk_for_this_trade_usd = capital_boxes.get(trade_strategy, 0)

        simulated_profit = risk_for_this_trade_usd * trade['R_multiple']

        # Actualizar balances
        capital_boxes[trade_strategy] += simulated_profit
        total_balance += simulated_profit

        if total_balance <= 0: total_balance = 0
        if capital_boxes[trade_strategy] < 0: capital_boxes[trade_strategy] = 0

        peak_balance = max(peak_balance, total_balance)

        if peak_balance > 0:
            current_drawdown_pct = ((peak_balance - total_balance) / peak_balance) * 100
            max_drawdown_pct = max(max_drawdown_pct, current_drawdown_pct)

        equity_points.append({'time': trade['Close time'], 'balance': total_balance})

    # 6. Finalizar la curva de equity
    if not equity_points:
        return pd.Series([initial_capital]), initial_capital, 0

    equity_df = pd.DataFrame(equity_points).drop_duplicates(subset='time', keep='last').set_index('time')
    equity_series = equity_df['balance']
    final_balance = equity_series.iloc[-1] if not equity_series.empty else initial_capital

    return equity_series, final_balance, max_drawdown_pct


def run_walk_forward_simulation(
        trades_df_is, trades_df_oos, hrp_weights, initial_capital,
        optimal_f_fraction, optimal_f_window_years,
        riesgo_historico_por_trade, r_multiple_cap
):
    """
    Ejecuta una simulación Walk-Forward utilizando el motor de simulación operativa de alta fidelidad.
    """
    if trades_df_oos.empty:
        return 0, 100, 0, pd.Series([initial_capital]), []

    trades_df_oos_sorted = trades_df_oos.sort_values('Close time').copy()
    trades_df_oos_sorted['Year'] = trades_df_oos_sorted['Close time'].dt.year
    oos_years = sorted(trades_df_oos_sorted['Year'].unique())

    all_oos_equity_points = [
        {'time': trades_df_oos_sorted['Close time'].iloc[0] - pd.Timedelta(seconds=1), 'balance': initial_capital}]
    current_balance = initial_capital
    annual_risk_data = []

    if trades_df_is.empty or 'Close time' not in trades_df_is.columns:
        return 0, 100, 0, pd.Series([initial_capital]), []

    current_lookback_data = trades_df_is.copy()

    for year in oos_years:
        oos_year_df = trades_df_oos_sorted[trades_df_oos_sorted['Year'] == year]
        if oos_year_df.empty:
            continue

        resampled_lookback = create_hrp_resampled_portfolio(current_lookback_data, hrp_weights)
        optimal_f = calculate_optimal_f(resampled_lookback, riesgo_historico_por_trade)
        risk_to_apply = optimal_f * optimal_f_fraction

        annual_risk_data.append({
            "Año de Aplicación": year,
            "Optimal F Calculado": optimal_f,
            "Riesgo Aplicado (%)": risk_to_apply * 100
        })

        _, final_balance, _, equity_points_year = run_operational_simulation_with_rebalancing(
            trades_df=oos_year_df,
            initial_capital=current_balance,
            risk_percent=risk_to_apply * 100,
            hrp_weights=hrp_weights,
            r_multiple_cap=r_multiple_cap,
            riesgo_historico_por_trade=riesgo_historico_por_trade
        )

        if equity_points_year:
            all_oos_equity_points.extend(equity_points_year[1:])

        current_balance = final_balance if final_balance > 0 else 0

        current_lookback_data = pd.concat([current_lookback_data, oos_year_df]).sort_values('Close time')
        min_date_in_window = current_lookback_data['Close time'].max() - pd.DateOffset(years=optimal_f_window_years)
        current_lookback_data = current_lookback_data[current_lookback_data['Close time'] >= min_date_in_window]

    if len(all_oos_equity_points) <= 1:
        return 0, 100, 0, pd.Series([initial_capital]), []

    equity_df = pd.DataFrame(all_oos_equity_points).drop_duplicates(subset='time', keep='last').set_index('time')
    full_equity_curve = equity_df['balance'].sort_index()

    final_balance_total = full_equity_curve.iloc[-1]
    cagr = calculate_cagr(initial_capital, final_balance_total, trades_df_oos)
    peak = full_equity_curve.cummax()
    max_dd = (((peak - full_equity_curve) / peak).max() * 100) if peak.max() > 0 else 0
    mar = calculate_mar(cagr, max_dd) if cagr is not None else 0

    return cagr, max_dd, mar, full_equity_curve, annual_risk_data


def find_optimal_mar_fraction(
        trades_df_is, trades_df_oos, hrp_weights, initial_capital,
        optimal_f_window_years, riesgo_historico_por_trade, r_multiple_cap,
        max_f_fraction_percentage
):
    """
    Encuentra la mejor fracción de Optimal F usando la simulación operativa de alta fidelidad.
    """
    upper_bound = (max_f_fraction_percentage / 100.0) + 0.01
    fractions_to_test = np.arange(0.05, upper_bound, 0.05)
    results = []

    st.write("Calibrando en datos In-Sample con simulación operativa para encontrar la fracción óptima...")
    progress_bar = st.progress(0)

    for i, fraction in enumerate(fractions_to_test):
        cagr, max_dd, mar, _, _ = run_walk_forward_simulation(
            trades_df_is, trades_df_is, hrp_weights, initial_capital,
            fraction, optimal_f_window_years, riesgo_historico_por_trade, r_multiple_cap
        )
        results.append({
            "Fracción de F Aplicada": f"{fraction * 100:.0f}%",
            "CAGR (IS)": cagr,
            "Max Drawdown (IS) (%)": max_dd,
            "MAR Ratio (IS)": mar
        })
        progress_bar.progress((i + 1) / len(fractions_to_test))

    results_df = pd.DataFrame(results)

    if results_df.empty or "MAR Ratio (IS)" not in results_df.columns or results_df[
        "MAR Ratio (IS)"].isnull().all() or (results_df["MAR Ratio (IS)"] == 0).all():
        st.error("No se pudieron generar resultados válidos durante la calibración.")
        return None, None, None, None

    best_is_row = results_df.loc[results_df["MAR Ratio (IS)"].idxmax()]
    recommended_fraction_str = best_is_row["Fracción de F Aplicada"]
    recommended_fraction_val = float(recommended_fraction_str.replace('%', '')) / 100.0

    st.success("Calibración completada.")

    st.write(
        f"Validando la fracción recomendada ({recommended_fraction_str}) en datos Out-of-Sample con simulación operativa...")
    oos_cagr, oos_dd, oos_mar, _, oos_annual_risk_data = run_walk_forward_simulation(
        trades_df_is, trades_df_oos, hrp_weights, initial_capital,
        recommended_fraction_val, optimal_f_window_years, riesgo_historico_por_trade, r_multiple_cap
    )

    validation_results = {
        "CAGR (OOS)": oos_cagr,
        "Max Drawdown (OOS) (%)": oos_dd,
        "MAR Ratio (OOS)": oos_mar
    }

    return results_df, best_is_row, validation_results, oos_annual_risk_data


# --- FIN: NUEVAS FUNCIONES ---


# --- INTERFAZ DE USUARIO ---
with st.sidebar:
    st.header("1. Parámetros de Simulación")
    initial_capital = st.number_input("Capital Inicial ($)", 1.0, value=10000.0, format="%.2f",
                                      help="Capital inicial para la SIMULACIÓN de Monte Carlo y otras proyecciones.")
    st.header("2. Motor de Riesgo (para Simulación)")
    riesgo_historico_por_trade = st.number_input("Riesgo Histórico Asumido por Trade ($)", min_value=0.0, value=100.0,
                                                 step=10.0,
                                                 help="Si > 0, se usa para calcular todos los R-múltiples. Si es 0, se usará el MAE.")
    risk_per_trade_pct = st.number_input("Riesgo sobre Capital (%)", min_value=0.0, max_value=100.0, value=1.0,
                                         step=0.1,
                                         help="Riesgo a aplicar en la simulación como % del capital. Usado si 'Riesgo Fijo por Trade' es 0.")
    fixed_risk_amount = st.number_input("Riesgo Fijo por Trade ($)", min_value=0.0, value=0.0, step=10.0, format="%.2f",
                                        help="Si > 0, la simulación arriesgará esta cantidad en cada trade (ignora el % de riesgo).")
    st.header("3. Archivo(s) CSV")
    uploaded_files = st.file_uploader("Sube uno o varios archivos CSV", type="csv", accept_multiple_files=True)
    st.header("5. Validación Fuera de Muestra (OOS)")
    oos_percentage = st.slider("% de datos a reservar para validación", 0, 50, 20, 5,
                               help="Reserva el X% más reciente de los datos para validar la robustez. El optimizador no verá estos datos.")

if not uploaded_files:
    st.info("Por favor, sube uno o varios archivos CSV para comenzar el análisis.")
else:
    if 'last_uploaded_files' not in st.session_state:
        st.session_state.last_uploaded_files = []

    current_file_names = [f.name for f in uploaded_files]

    if set(st.session_state.last_uploaded_files) != set(current_file_names):
        if 'portfolio_trades' in st.session_state:
            del st.session_state.portfolio_trades
        if 'strategy_selector' in st.session_state:
            del st.session_state.strategy_selector
        st.session_state.last_uploaded_files = current_file_names

    try:
        data_full = load_and_clean_data(uploaded_files)
        if data_full.empty:
            st.warning("No se encontraron trades válidos. Revisa el formato y los datos.")
        else:
            if 'Profit/Loss' in data_full.columns and 'MAE ($)' in data_full.columns:
                temp_data_for_rcap = data_full[data_full['MAE ($)'].abs() > 0.0001].copy()
                temp_data_for_rcap['R_multiple_calc'] = temp_data_for_rcap['Profit/Loss'] / temp_data_for_rcap[
                    'MAE ($)'].abs()
                temp_data_for_rcap.replace([np.inf, -np.inf], np.nan, inplace=True);
                temp_data_for_rcap.dropna(subset=['R_multiple_calc'], inplace=True)
                winning_r_multiples = temp_data_for_rcap[temp_data_for_rcap['R_multiple_calc'] > 0]['R_multiple_calc']
                dynamic_r_cap = winning_r_multiples.quantile(0.98) if not winning_r_multiples.empty else 15.0
            else:
                dynamic_r_cap = 15.0
            st.success(f"{len(data_full)} trades y {data_full['Result name'].nunique()} estrategias cargadas.")
            with st.sidebar:
                st.header("4. Límite de Ganancia (para Simulación)")
                limit_type = st.radio("Tipo de Límite", ('Automático (Recomendado)', 'Manual'),
                                      key="limit_type_selector",
                                      help=f"Automático usa un límite de {dynamic_r_cap:.2f}R (percentil 98) para controlar outliers.")
                r_multiple_cap = dynamic_r_cap if limit_type == 'Automático (Recomendado)' else st.number_input(
                    "Límite R-Multiple Manual (0 para desactivar)", min_value=0.0, value=15.0, step=1.0, format="%.2f")
                if limit_type == 'Automático (Recomendado)': st.info(f"Límite fijado en {dynamic_r_cap:.2f}R")

            data_full_sorted = data_full.sort_values('Close time').reset_index(drop=True)
            if oos_percentage > 0:
                split_index = int(len(data_full_sorted) * (1 - oos_percentage / 100))
                training_data, validation_data = data_full_sorted.iloc[:split_index].copy(), data_full_sorted.iloc[
                                                                                             split_index:].copy()
                data_for_analysis = training_data
            else:
                training_data, validation_data = data_full_sorted, pd.DataFrame()
                data_for_analysis = data_full_sorted

            tab1, tab2, tab3, tab4 = st.tabs(
                ["🏗️ Constructor de Portafolios", "🔨 Optimizador de Portafolios", "🔬 Análisis de Riesgo",
                 "📉 Análisis de Debilidades"])

            calculation_mode = st.radio("Selecciona el modo de cálculo para las métricas y gráficos principales:",
                                        ('Simulación (Interés Compuesto)', 'Histórico (Datos del CSV)'), index=1,
                                        horizontal=True, key='calc_mode')
            st.divider()

            with tab1:
                st.header("Construcción Manual de Portafolios")
                if oos_percentage > 0: st.warning(
                    f"Modo OOS activado: Se está usando el {100 - oos_percentage}% más antiguo de los datos para el análisis In-Sample.")

                estrategias_unicas = list(data_for_analysis['Result name'].unique())

                if 'strategy_selector' not in st.session_state:
                    st.session_state.strategy_selector = estrategias_unicas


                def manual_portfolio_update():
                    seleccion = st.session_state.strategy_selector
                    if seleccion:
                        st.session_state.portfolio_trades = data_for_analysis[
                            data_for_analysis['Result name'].isin(seleccion)]
                    else:
                        st.session_state.portfolio_trades = pd.DataFrame()


                st.multiselect(
                    "Selecciona las estrategias:",
                    options=estrategias_unicas,
                    key='strategy_selector',
                    on_change=manual_portfolio_update
                )

                if 'portfolio_trades' not in st.session_state:
                    manual_portfolio_update()

                if 'portfolio_trades' in st.session_state and not st.session_state.portfolio_trades.empty:
                    portafolio_a_mostrar = st.session_state.portfolio_trades
                    if calculation_mode == 'Simulación (Interés Compuesto)':
                        st.subheader("Resultados de la Simulación (con Interés Compuesto)")
                        equity_curve_sim, final_balance_sim, dd_pct_compuesto = run_portfolio_simulation(
                            portafolio_a_mostrar, initial_capital, risk_per_trade_pct, r_multiple_cap,
                            fixed_risk_amount, riesgo_historico_por_trade)
                        cagr_sim, mar_sim = calculate_cagr(initial_capital, final_balance_sim,
                                                           portafolio_a_mostrar), calculate_mar(
                            calculate_cagr(initial_capital, final_balance_sim, portafolio_a_mostrar), dd_pct_compuesto)
                        sim_col1, sim_col2, sim_col3 = st.columns(3)
                        sim_col1.metric("CAGR Simulado (%)", f"{cagr_sim:.2f}%");
                        sim_col2.metric("Máx. Drawdown Simulado (%)", f"{dd_pct_compuesto:.2f}%");
                        sim_col3.metric("MAR Ratio Simulado", f"{mar_sim:.2f}")
                        fig_sim = px.line(equity_curve_sim, title="Curva de Capital Simulada",
                                          labels={"value": "Capital", "index": "Fecha"})
                        st.plotly_chart(fig_sim.update_layout(showlegend=False), use_container_width=True,
                                        key="sim_chart_tab1")
                    else:
                        st.subheader("Resultados Históricos (Basados en el archivo CSV)")
                        equity_curve_hist, start_hist, end_hist, dd_hist, can_calc_hist = calculate_isolated_historical_performance(
                            portafolio_a_mostrar)
                        if can_calc_hist:
                            cagr_hist, mar_ratio_hist = calculate_cagr(start_hist, end_hist,
                                                                       portafolio_a_mostrar), calculate_mar(
                                calculate_cagr(start_hist, end_hist, portafolio_a_mostrar), dd_hist)
                            hist_col1, hist_col2, hist_col3 = st.columns(3)
                            hist_col1.metric("CAGR (Histórico) (%)", f"{cagr_hist:.2f}%");
                            hist_col2.metric("Máx. Drawdown (Histórico) %", f"{dd_hist:.2f}%");
                            hist_col3.metric("MAR Ratio (Histórico)", f"{mar_ratio_hist:.2f}")
                            fig_hist = px.line(equity_curve_hist, title="Curva de Capital Histórica",
                                               labels={"value": "Capital", "index": "Fecha"})
                            st.plotly_chart(fig_hist.update_layout(showlegend=False), use_container_width=True,
                                            key="hist_chart_tab1")
                        else:
                            st.error("Error: No se pueden calcular las métricas históricas.", icon="🚨")
                else:
                    st.warning("Selecciona al menos una estrategia.")

            with tab2:
                st.header("Optimizador Inteligente de Portafolios")
                st.markdown(
                    "Encuentra el número ideal de estrategias y la combinación perfecta usando métodos de búsqueda avanzados.")

                # --- FASE 0: FILTRO DE ESTRATEGIAS ANTES DEL OPTIMIZADOR ---
                strategy_stats = get_strategy_level_stats(training_data)

                st.subheader("FASE 0: Filtro de calidad individual (antes de optimizar)")

                if strategy_stats.empty:
                    st.warning("No se pudieron calcular estadísticas por estrategia (quizá faltan columnas o datos).")
                    filtered_stats = pd.DataFrame()
                else:
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

                    min_trades = col_f1.number_input(
                        "Mín. nº de trades", min_value=10, max_value=2000, value=100, step=10
                    )
                    max_dd_ind = col_f2.number_input(
                        "Máx. DD individual (%)", min_value=1.0, max_value=80.0, value=3.0, step=1.0
                    )
                    min_mar_ind = col_f3.number_input(
                        "Mín. MAR individual", min_value=0.0, max_value=8.0, value=0.30, step=0.05
                    )
                    max_candidates = col_f4.number_input(
                        "Máx. estrategias a enviar al optimizador",
                        min_value=1, max_value=400, value=80, step=5
                    )

                    # FILTRO PRINCIPAL
                    filtered_stats = strategy_stats[
                        (strategy_stats["NumTrades"] >= min_trades) &
                        (strategy_stats["MaxDD"] <= max_dd_ind) &
                        (strategy_stats["MAR"] >= min_mar_ind)
                        ].copy()

                    # Orden por MAR y limitar a número máximo de estrategias
                    filtered_stats = filtered_stats.sort_values("MAR", ascending=False).head(max_candidates)

                    st.info(
                        f"De {strategy_stats['Result name'].nunique()} estrategias totales, "
                        f"**{len(filtered_stats)}** pasan los filtros y se enviarán al optimizador."
                    )

                    with st.expander("Ver tabla de estrategias filtradas (Top por MAR)"):
                        st.dataframe(filtered_stats, use_container_width=True)


                mode_display = "Simulación (Interés Compuesto)" if calculation_mode == 'Simulación (Interés Compuesto)' else "Histórico (Datos del CSV)"
                st.info(f"Modo de Optimización actual: **{mode_display}**")

                st.subheader("1. Parámetros de Búsqueda")

                max_size = len(data_for_analysis['Result name'].unique())
                portfolio_size_range = st.slider(
                    "Rango de Tamaños de Portafolio a Probar:",
                    min_value=2, max_value=max_size, value=(2, min(20, max_size)), key="optimizer_range_slider")

                metric_option = st.selectbox("Métrica para optimizar:", ["MAR Ratio", "Menor Max DD", "Mayor Profit"],
                                             key="optimizer_metric")

                search_method = st.radio("Método de Búsqueda:",
                                         ["Algoritmo Greedy (Rápido)", "Algoritmo Evolutivo (Avanzado)",
                                          "Búsqueda Aleatoria (Básico)"], key="search_method")

                st.subheader("2. Configuración del Método")

                if search_method == "Algoritmo Greedy (Rápido)":
                    greedy_type = st.radio("Tipo de Algoritmo Greedy:",
                                           ["Eliminación (Hacia Atrás)", "Construcción (Hacia Adelante)"],
                                           horizontal=True)
                elif search_method == "Algoritmo Evolutivo (Avanzado)":
                    c1, c2, c3 = st.columns(3)
                    population_size = c1.number_input("Tamaño de Población", 50, 500, 100)
                    generations = c2.number_input("Número de Generaciones", 10, 200, 50)
                    mutation_rate = c3.slider("Tasa de Mutación", 0.01, 0.3, 0.1)
                else:
                    max_combinations = st.number_input("Límite de Combinaciones a Probar (por cada tamaño):", 10, 50000,
                                                       1000, key="optimizer_combinations")

                if st.button("🚀 Iniciar Búsqueda Exhaustiva"):
                    st.cache_data.clear()
                    if 'optimizer_results' in st.session_state: del st.session_state.optimizer_results
                    if 'hrp_results' in st.session_state: del st.session_state.hrp_results

                    all_strategies_list = list(training_data['Result name'].unique())
                    champions_by_size = []
                    sizes_to_test = range(portfolio_size_range[1], portfolio_size_range[0] - 1, -1)

                    status_text = st.empty()
                    progress_bar = st.progress(0)

                    with st.spinner("Iniciando búsqueda..."):
                        sort_key, reverse_sort = ('mar_ratio', True)
                        if metric_option == "Menor Max DD":
                            sort_key, reverse_sort = 'drawdown', False
                        elif metric_option == "Mayor Profit":
                            sort_key, reverse_sort = 'profit', True
                        metric_index = ['mar_ratio', 'drawdown', 'profit'].index(sort_key)

                        training_data_tuple = tuple(training_data.to_records(index=False))
                        training_columns = training_data.columns.tolist()


                        def evaluate_combo(combo):
                            res = get_portfolio_metrics_cached(training_data_tuple, training_columns,
                                                               tuple(sorted(combo)), calculation_mode, initial_capital,
                                                               risk_per_trade_pct, r_multiple_cap, fixed_risk_amount,
                                                               riesgo_historico_por_trade)
                            return res[metric_index]


                        total_steps = len(sizes_to_test)

                        for i, size in enumerate(sizes_to_test):
                            status_text.text(
                                f"Paso {i + 1}/{total_steps}: Optimizando para tamaño de portafolio = {size}...")
                            best_combo_for_size, convergence_data = [], []

                            if search_method == "Búsqueda Aleatoria (Básico)":
                                all_combos = list(itertools.combinations(all_strategies_list, size))
                                if len(all_combos) > 0:
                                    combos_to_test = [all_combos[i] for i in np.random.choice(len(all_combos),
                                                                                              min(max_combinations,
                                                                                                  len(all_combos)),
                                                                                              replace=False)]
                                    results = [(combo, evaluate_combo(combo)) for combo in combos_to_test]
                                    best_combo_for_size, _ = sorted(results, key=lambda x: x[1], reverse=reverse_sort)[
                                        0]

                            elif search_method == "Algoritmo Greedy (Rápido)":
                                current_portfolio, remaining = [], list(all_strategies_list)
                                for k in range(size):
                                    scores = [(strat, evaluate_combo(current_portfolio + [strat])) for strat in
                                              remaining]
                                    best_strat, _ = sorted(scores, key=lambda x: x[1], reverse=reverse_sort)[0]
                                    current_portfolio.append(best_strat)
                                    remaining.remove(best_strat)
                                best_combo_for_size = current_portfolio

                            elif search_method == "Algoritmo Evolutivo (Avanzado)":
                                all_possible_combos = list(itertools.combinations(all_strategies_list, size))
                                num_possible_combos = len(all_possible_combos)
                                if num_possible_combos < population_size:
                                    population = [list(c) for c in all_possible_combos]
                                else:
                                    indices = np.random.choice(num_possible_combos, population_size, replace=False)
                                    population = [list(all_possible_combos[i]) for i in indices]

                                for gen in range(generations):
                                    status_text.text(
                                        f"Paso {i + 1}/{total_steps} (Tamaño {size}): Generación {gen + 1}/{generations}...")
                                    scores = [(p, evaluate_combo(p)) for p in population]
                                    ranked_population = sorted(scores, key=lambda x: x[1], reverse=reverse_sort)
                                    convergence_data.append(ranked_population[0][1])

                                    parents = [p[0] for p in ranked_population[:int(population_size / 2)]]
                                    if not parents: break

                                    children = []
                                    while len(children) < population_size:
                                        p1, p2 = parents[np.random.randint(0, len(parents))], parents[
                                            np.random.randint(0, len(parents))]
                                        child = list(set(p1[:size // 2] + p2[size // 2:]))
                                        while len(child) < size:
                                            missing = [s for s in all_strategies_list if s not in child]
                                            if not missing: break
                                            child.append(np.random.choice(missing))
                                        children.append(child[:size])

                                    for child in children:
                                        if np.random.rand() < mutation_rate:
                                            idx_to_mutate = np.random.randint(0, size)
                                            missing = [s for s in all_strategies_list if s not in child]
                                            if missing: child[idx_to_mutate] = np.random.choice(missing)
                                    population = children

                                best_combo_for_size = ranked_population[0][0]

                            if best_combo_for_size:
                                mar, dd, profit = get_portfolio_metrics_cached(training_data_tuple, training_columns,
                                                                               tuple(sorted(best_combo_for_size)),
                                                                               calculation_mode, initial_capital,
                                                                               risk_per_trade_pct, r_multiple_cap,
                                                                               fixed_risk_amount,
                                                                               riesgo_historico_por_trade)
                                champions_by_size.append(
                                    {"combo": best_combo_for_size, "drawdown": dd, "profit": profit, "mar_ratio": mar,
                                     "size": size, "convergence": convergence_data})

                            progress_bar.progress((i + 1) / total_steps)

                        st.success("¡Búsqueda exhaustiva completada!")
                        if champions_by_size:
                            absolute_champion = \
                            sorted(champions_by_size, key=lambda x: x[sort_key], reverse=reverse_sort)[0]
                            st.session_state.optimizer_results = {"absolute_champion": absolute_champion,
                                                                  "champions_by_size": champions_by_size}
                        else:
                            st.warning("No se encontraron resultados válidos.")
                            if 'optimizer_results' in st.session_state: del st.session_state.optimizer_results

                if 'optimizer_results' in st.session_state:
                    res = st.session_state.optimizer_results
                    champ, champs_by_size = res['absolute_champion'], res['champions_by_size']
                    st.markdown("---");
                    st.header("🏆 Campeón Absoluto Encontrado 🏆")
                    st.metric("Tamaño Óptimo del Portafolio", f"{champ['size']} Estrategias")

                    if oos_percentage > 0 and not validation_data.empty:
                        st.subheader("Análisis de Robustez: In-Sample vs. Out-of-Sample")
                        is_mar, is_dd, is_profit = get_portfolio_metrics_final(training_data,
                                                                               tuple(sorted(champ['combo'])),
                                                                               calculation_mode, initial_capital,
                                                                               risk_per_trade_pct, r_multiple_cap,
                                                                               fixed_risk_amount,
                                                                               riesgo_historico_por_trade)
                        oos_mar, oos_dd, oos_profit = get_portfolio_metrics_final(validation_data,
                                                                                  tuple(sorted(champ['combo'])),
                                                                                  calculation_mode, initial_capital,
                                                                                  risk_per_trade_pct, r_multiple_cap,
                                                                                  fixed_risk_amount,
                                                                                  riesgo_historico_por_trade)


                        def calc_degradation(is_val, oos_val, is_lower_better=False):
                            if abs(is_val) < 1e-6: return 0
                            degradation = ((oos_val - is_val) / abs(is_val)) * 100
                            return degradation if not is_lower_better else -degradation


                        mar_deg, dd_deg = calc_degradation(is_mar, oos_mar), calc_degradation(is_dd, oos_dd,
                                                                                              is_lower_better=True)

                        oos_results_df = pd.DataFrame({
                            "Métrica": ["MAR Ratio", "Max DD (%)", "Profit (%)"],
                            "In-Sample (Entrenamiento)": [f"{is_mar:.2f}", f"{is_dd:.2f}%", f"{is_profit:.2f}%"],
                            "Out-of-Sample (Validación)": [f"{oos_mar:.2f}", f"{oos_dd:.2f}%", f"{oos_profit:.2f}%"],
                            "Degradación": [f"{mar_deg:.1f}%", f"{dd_deg:.1f}%", "N/A"]
                        })
                        st.dataframe(oos_results_df, use_container_width=True)

                        if mar_deg < -50 or dd_deg > 50:
                            st.error("🔴 PELIGRO: El portafolio está sobreajustado.")
                        elif mar_deg < -25 or dd_deg > 25:
                            st.warning("🟡 ADVERTENCIA: El portafolio muestra signos de sobreajuste.")
                        else:
                            st.success("🟢 ROBUSTO: El portafolio mantiene un rendimiento estable.")
                    else:
                        res_col1, res_col2, res_col3 = st.columns(3)
                        res_col1.metric("MAR Ratio", f"{champ['mar_ratio']:.2f}");
                        res_col2.metric("Max DD (%)", f"{champ['drawdown']:.2f}%")
                        profit_label = f"Profit ({'Sim' if calculation_mode == 'Simulación (Interés Compuesto)' else 'Hist'}) (%)"
                        res_col3.metric(profit_label, f"{champ['profit']:.2f}%")

                    with st.expander("Ver lista de estrategias del portafolio óptimo"):
                        st.json(list(champ['combo']))


                    def load_champion_portfolio():
                        champion_strategies = list(st.session_state.optimizer_results['absolute_champion']['combo'])
                        st.session_state.strategy_selector = champion_strategies
                        st.session_state.portfolio_trades = data_full[
                            data_full['Result name'].isin(champion_strategies)]


                    st.button(
                        "Cargar este portafolio en la aplicación",
                        key="load_champion",
                        on_click=load_champion_portfolio
                    )

                    st.markdown("---");
                    st.header("📊 Tabla Comparativa de Campeones por Tamaño")
                    champions_df = pd.DataFrame(champs_by_size).rename(
                        columns={"size": "Tamaño", "mar_ratio": "Mejor MAR Ratio", "drawdown": "Max DD (%)",
                                 "profit": "Mejor Profit (%)"})
                    st.dataframe(
                        champions_df[['Tamaño', 'Mejor MAR Ratio', 'Max DD (%)', 'Mejor Profit (%)']].sort_values(
                            by="Tamaño", ascending=True), use_container_width=True)

                    if search_method == "Algoritmo Evolutivo (Avanzado)":
                        st.markdown("---");
                        st.header("📈 Gráfico de Convergencia del Algoritmo")
                        with st.expander("🔍 Cómo interpretar este gráfico"):
                            st.markdown("""
                            - **Curva Ideal:** Sube rápido al principio y luego se aplana.
                            - **Se Aplana Muy Rápido:** Poca diversidad. Prueba a **aumentar el Tamaño de Población**.
                            - **Nunca se Aplana:** Búsqueda caótica. Prueba a **reducir la Tasa de Mutación**.
                            - **Se Aplana a Mitad de Camino:** Demasiadas generaciones. Puedes reducir el número.
                            """)

                        convergence_data = champ.get("convergence", [])
                        if convergence_data:
                            conv_df = pd.DataFrame(
                                {'Generación': range(1, len(convergence_data) + 1), 'Mejor Métrica': convergence_data})
                            fig_conv = px.line(conv_df, x='Generación', y='Mejor Métrica',
                                               title=f"Convergencia para Portafolio de {champ['size']} Estrategias",
                                               markers=True)
                            st.plotly_chart(fig_conv, use_container_width=True)

                    # --- INICIO: SECCIÓN DE REFINAMIENTO CON HRP ---
                    st.markdown("---");
                    st.header("Paso 3: Refinar Asignación de Riesgo con HRP (Opcional)")
                    st.info(
                        "Hierarchical Risk Parity (HRP) optimiza la asignación de capital entre las estrategias del portafolio campeón para reducir el riesgo. "
                        "No cambia las estrategias, solo su ponderación."
                    )

                    if st.button("🚀 Refinar Portafolio con Hierarchical Risk Parity (HRP)"):
                        champion_combo = tuple(sorted(res['absolute_champion']['combo']))
                        champion_trades_is = training_data[training_data['Result name'].isin(list(champion_combo))]

                        if len(champion_trades_is['Result name'].unique()) < 2:
                            st.warning(
                                "HRP requiere al menos 2 estrategias en el portafolio para poder calcular las ponderaciones.")
                        else:
                            with st.spinner("Calculando pesos HRP y simulando los portafolios refinados..."):
                                daily_returns_df = get_daily_returns(champion_trades_is)

                                if daily_returns_df.empty or daily_returns_df.shape[1] < 2:
                                    st.error(
                                        "No se pudieron calcular los retornos diarios para las estrategias. No se puede ejecutar HRP.")
                                else:
                                    hrp_weights = get_hrp_weights(daily_returns_df)

                                    # --- INICIO: LÓGICA DE COMPARACIÓN HRP ---
                                    # 1. Simulación HRP Resampling (Estadística)
                                    hrp_portfolio_resampled_is = create_hrp_resampled_portfolio(champion_trades_is,
                                                                                                hrp_weights)
                                    _, final_balance_resampled_is, dd_resampled_is = run_portfolio_simulation(
                                        hrp_portfolio_resampled_is, initial_capital, risk_per_trade_pct, r_multiple_cap,
                                        fixed_risk_amount, riesgo_historico_por_trade)
                                    cagr_resampled_is = calculate_cagr(initial_capital, final_balance_resampled_is,
                                                                       champion_trades_is)
                                    mar_resampled_is = calculate_mar(cagr_resampled_is, dd_resampled_is)
                                    profit_resampled_is = ((final_balance_resampled_is / initial_capital) - 1) * 100

                                    # 2. Simulación HRP Operativa (Alta Fidelidad)
                                    _, final_balance_op_is, dd_op_is = run_operational_simulation_with_rebalancing(
                                        champion_trades_is, initial_capital, risk_per_trade_pct, hrp_weights,
                                        r_multiple_cap, riesgo_historico_por_trade, fixed_risk_amount)
                                    cagr_op_is = calculate_cagr(initial_capital, final_balance_op_is,
                                                                champion_trades_is)
                                    mar_op_is = calculate_mar(cagr_op_is, dd_op_is)
                                    profit_op_is = ((final_balance_op_is / initial_capital) - 1) * 100

                                    hrp_comparison_data = {
                                        "Métrica": ["MAR Ratio", "Max Drawdown (%)", "Profit (%)"],
                                        "Original (Equiponderado)": [f"{is_mar:.2f}", f"{is_dd:.2f}%",
                                                                     f"{is_profit:.2f}%"],
                                        "HRP (Sim. Estadística)": [f"{mar_resampled_is:.2f}", f"{dd_resampled_is:.2f}%",
                                                                   f"{profit_resampled_is:.2f}%"],
                                        "HRP (Sim. Operativa Real)": [f"{mar_op_is:.2f}", f"{dd_op_is:.2f}%",
                                                                      f"{profit_op_is:.2f}%"],
                                    }
                                    st.session_state.hrp_comparison_is = hrp_comparison_data

                                    # 3. Misma comparación para Out-of-Sample si existe
                                    if oos_percentage > 0 and not validation_data.empty:
                                        champion_trades_oos = validation_data[
                                            validation_data['Result name'].isin(list(champion_combo))]
                                        # HRP Resampling OOS
                                        hrp_portfolio_resampled_oos = create_hrp_resampled_portfolio(
                                            champion_trades_oos, hrp_weights)
                                        _, final_balance_resampled_oos, dd_resampled_oos = run_portfolio_simulation(
                                            hrp_portfolio_resampled_oos, initial_capital, risk_per_trade_pct,
                                            r_multiple_cap, fixed_risk_amount, riesgo_historico_por_trade)
                                        cagr_resampled_oos = calculate_cagr(initial_capital,
                                                                            final_balance_resampled_oos,
                                                                            champion_trades_oos)
                                        mar_resampled_oos = calculate_mar(cagr_resampled_oos, dd_resampled_oos)
                                        profit_resampled_oos = ((
                                                                            final_balance_resampled_oos / initial_capital) - 1) * 100
                                        # HRP Operativa OOS
                                        _, final_balance_op_oos, dd_op_oos = run_operational_simulation_with_rebalancing(
                                            champion_trades_oos, initial_capital, risk_per_trade_pct, hrp_weights,
                                            r_multiple_cap, riesgo_historico_por_trade, fixed_risk_amount)
                                        cagr_op_oos = calculate_cagr(initial_capital, final_balance_op_oos,
                                                                     champion_trades_oos)
                                        mar_op_oos = calculate_mar(cagr_op_oos, dd_op_oos)
                                        profit_op_oos = ((final_balance_op_oos / initial_capital) - 1) * 100

                                        hrp_comparison_data_oos = {
                                            "Métrica": ["MAR Ratio", "Max Drawdown (%)", "Profit (%)"],
                                            "Original (Equiponderado)": [f"{oos_mar:.2f}", f"{oos_dd:.2f}%",
                                                                         f"{oos_profit:.2f}%"],
                                            "HRP (Sim. Estadística)": [f"{mar_resampled_oos:.2f}",
                                                                       f"{dd_resampled_oos:.2f}%",
                                                                       f"{profit_resampled_oos:.2f}%"],
                                            "HRP (Sim. Operativa Real)": [f"{mar_op_oos:.2f}", f"{dd_op_oos:.2f}%",
                                                                          f"{profit_op_oos:.2f}%"],
                                        }
                                        st.session_state.hrp_comparison_oos = hrp_comparison_data_oos
                                    # --- FIN: LÓGICA DE COMPARACIÓN HRP ---

                                    st.session_state.hrp_results = {"weights": hrp_weights}
                                    st.success("Análisis HRP completado.")

                    if 'hrp_results' in st.session_state:
                        hrp_res = st.session_state.hrp_results
                        hrp_weights = hrp_res["weights"]

                        st.subheader("⚖️ Asignación de Capital HRP")
                        weights_df = hrp_weights.reset_index();
                        weights_df.columns = ['Estrategia', 'Peso']
                        weights_df['Peso'] = weights_df['Peso'] * 100
                        fig_weights = px.bar(weights_df.sort_values('Peso', ascending=True), x='Peso', y='Estrategia',
                                             orientation='h', text=weights_df['Peso'].apply(lambda x: f'{x:.1f}%'))
                        st.plotly_chart(fig_weights, use_container_width=True)

                        if 'hrp_comparison_is' in st.session_state:
                            st.subheader("📊 Comparativa de Rendimiento: Equiponderado vs. HRP")
                            with st.expander("🔍 Cómo interpretar esta tabla"):
                                st.markdown("""
                                - **Original (Equiponderado):** Es el rendimiento base, sin usar los pesos HRP.
                                - **HRP (Sim. Estadística):** Es el resultado de la simulación por *resampling*. Es una prueba teórica de la eficacia de los pesos.
                                - **HRP (Sim. Operativa Real):** **ESTA ES LA MÁS IMPORTANTE.** Simula el rendimiento que habrías obtenido aplicando los pesos HRP con el método de "cajas de capital" y rebalanceo semanal. Es la que más se aproxima a la realidad.
                                """)

                            st.write("**Resultados In-Sample (Entrenamiento)**")
                            comp_is_df = pd.DataFrame(st.session_state.hrp_comparison_is).set_index("Métrica")
                            st.dataframe(comp_is_df, use_container_width=True)

                        if 'hrp_comparison_oos' in st.session_state:
                            st.write("**Resultados Out-of-Sample (Validación)**")
                            comp_oos_df = pd.DataFrame(st.session_state.hrp_comparison_oos).set_index("Métrica")
                            st.dataframe(comp_oos_df, use_container_width=True)

                        # --- INICIO: SECCIÓN DE GESTIÓN DE RIESGO DINÁMICA (OPTIMAL F) ---
                        st.markdown("---");
                        st.header("Paso 4: Simulación Walk-Forward con Gestión de Riesgo Dinámica (Optimal F)")
                        st.info(
                            "Esta herramienta utiliza el **motor de simulación operativa de alta fidelidad** para probar el rendimiento del portafolio HRP a lo largo del período OOS. "
                            "El riesgo no es fijo, sino que se recalcula anualmente basándose en el Optimal F del histórico reciente."
                        )

                        wf_col1, wf_col2 = st.columns(2)
                        with wf_col1:
                            optimal_f_fraction = st.slider(
                                "Porcentaje de Optimal F a aplicar (%)",
                                min_value=1, max_value=100, value=20, step=1,
                                help="Fracción del Optimal F a arriesgar. Usar el 100% puede ser muy agresivo. Se recomienda empezar con 20-30%.")
                        with wf_col2:
                            optimal_f_window_years = st.number_input(
                                "Años de datos para calcular Optimal F",
                                min_value=1, max_value=10, value=5, step=1,
                                help="Ventana móvil de datos históricos para recalcular el riesgo cada año.")

                        if st.button("🚀 Iniciar Simulación Walk-Forward Sencilla"):
                            if riesgo_historico_por_trade <= 0:
                                st.error(
                                    "La simulación con Optimal F requiere un 'Riesgo Histórico Asumido por Trade ($)' mayor que cero.")
                            elif oos_percentage == 0 or validation_data.empty:
                                st.error("Esta simulación requiere un set de datos de validación (Out-of-Sample).")
                            else:
                                with st.spinner("Ejecutando simulación de alta fidelidad..."):
                                    champion_combo = tuple(sorted(res['absolute_champion']['combo']))
                                    champion_trades_is = training_data[
                                        training_data['Result name'].isin(list(champion_combo))]
                                    champion_trades_oos = validation_data[
                                        validation_data['Result name'].isin(list(champion_combo))]

                                    dyn_cagr, dyn_dd, dyn_mar, dyn_equity, annual_f_data = run_walk_forward_simulation(
                                        champion_trades_is, champion_trades_oos, hrp_res["weights"], initial_capital,
                                        optimal_f_fraction / 100.0, optimal_f_window_years,
                                        riesgo_historico_por_trade, r_multiple_cap
                                    )

                                    st.subheader("Comparativa Global: Riesgo Fijo vs. Riesgo Dinámico (Optimal F)")

                                    # Métricas de Riesgo Fijo (Equiponderado, sin HRP, como baseline)
                                    _, fixed_final_balance, fixed_dd = run_portfolio_simulation(champion_trades_oos,
                                                                                                initial_capital,
                                                                                                risk_per_trade_pct,
                                                                                                r_multiple_cap,
                                                                                                fixed_risk_amount,
                                                                                                riesgo_historico_por_trade)
                                    retorno_total_fijo = ((
                                                                      fixed_final_balance / initial_capital) - 1) * 100 if initial_capital > 0 else 0
                                    multiplicador_fijo = fixed_final_balance / initial_capital if initial_capital > 0 else 0
                                    mar_adj_fijo = retorno_total_fijo / fixed_dd if fixed_dd > 0 else np.inf

                                    # Métricas de Riesgo Dinámico (Alta Fidelidad, con HRP + Optimal F)
                                    dyn_final_balance = dyn_equity.iloc[-1]
                                    retorno_total_dinamico = ((
                                                                          dyn_final_balance / initial_capital) - 1) * 100 if initial_capital > 0 else 0
                                    multiplicador_dinamico = dyn_final_balance / initial_capital if initial_capital > 0 else 0
                                    mar_adj_dinamico = retorno_total_dinamico / dyn_dd if dyn_dd > 0 else np.inf

                                    comparison_data = {
                                        "Métrica": ["Multiplicador de Capital", "Retorno Total en OOS (%)",
                                                    "Max Drawdown (%)", "MAR Ratio (Ajustado)"],
                                        "Portafolio Baseline (Sin HRP, Riesgo Fijo)": [f"{multiplicador_fijo:.2f}x",
                                                                                       f"{retorno_total_fijo:.2f}",
                                                                                       f"{fixed_dd:.2f}",
                                                                                       f"{mar_adj_fijo:.2f}"],
                                        "Portafolio Avanzado (HRP + Riesgo Dinámico)": [
                                            f"{multiplicador_dinamico:.2f}x", f"{retorno_total_dinamico:.2f}",
                                            f"{dyn_dd:.2f}", f"{mar_adj_dinamico:.2f}"]
                                    }

                                    st.dataframe(pd.DataFrame(comparison_data).set_index("Métrica"),
                                                 use_container_width=True)

                                    st.subheader("Curva de Capital con Riesgo Dinámico (Out-of-Sample)")
                                    fig_dyn_equity = px.line(dyn_equity,
                                                             title="Curva de Capital Simulada (Riesgo Dinámico)",
                                                             labels={"value": "Capital", "index": "Fecha"})
                                    st.plotly_chart(fig_dyn_equity.update_layout(showlegend=False),
                                                    use_container_width=True)

                                    if annual_f_data:
                                        st.subheader("Riesgo Anual Aplicado (Walk-Forward)")
                                        st.dataframe(
                                            pd.DataFrame(annual_f_data).set_index("Año de Aplicación").style.format({
                                                "Optimal F Calculado": "{:.2f}",
                                                "Riesgo Aplicado (%)": "{:.2f}%"
                                            }), use_container_width=True)

                        # --- INICIO: NUEVA SECCIÓN DE CALIBRACIÓN ---
                        st.markdown("---")
                        st.header("Paso 5: Calibración de Riesgo Dinámico (Búsqueda de MAR Óptimo)")
                        st.info(
                            "Este proceso utiliza la **simulación operativa de alta fidelidad** para probar múltiples fracciones de Optimal F en los datos de entrenamiento (In-Sample) "
                            "y encontrar la que produce el mejor MAR Ratio. Luego, valida esa fracción en los datos Out-of-Sample para confirmar su robustez."
                        )

                        max_f_fraction_pct = st.slider(
                            "Rango máximo de fracción de F a probar (%)",
                            min_value=10, max_value=100, value=40, step=5,
                            help="Define el límite superior para la búsqueda de la fracción óptima de F. Si el resultado es el valor máximo, considera aumentarlo."
                        )

                        if st.button("🚀 Iniciar Calibración y Validación de Riesgo"):
                            if riesgo_historico_por_trade <= 0:
                                st.error("Se requiere un 'Riesgo Histórico Asumido por Trade ($)' > 0.")
                            elif oos_percentage == 0 or validation_data.empty:
                                st.error("Se requiere un set de datos Out-of-Sample para una calibración robusta.")
                            else:
                                with st.spinner(
                                        "Ejecutando análisis de sensibilidad con simulación de alta fidelidad... Esto puede tardar varios minutos."):
                                    champion_combo = tuple(sorted(res['absolute_champion']['combo']))
                                    champion_trades_is = training_data[
                                        training_data['Result name'].isin(list(champion_combo))]
                                    champion_trades_oos = validation_data[
                                        validation_data['Result name'].isin(list(champion_combo))]

                                    sensitivity_df, best_row, validation_res, annual_f_data_validation = find_optimal_mar_fraction(
                                        champion_trades_is, champion_trades_oos, hrp_res["weights"], initial_capital,
                                        optimal_f_window_years, riesgo_historico_por_trade, r_multiple_cap,
                                        max_f_fraction_pct
                                    )

                                    if sensitivity_df is not None:
                                        st.subheader("1. Tabla de Sensibilidad (Resultados In-Sample)")
                                        st.dataframe(sensitivity_df.style.format({
                                            "CAGR (IS)": "{:,.2f}%",
                                            "Max Drawdown (IS) (%)": "{:,.2f}",
                                            "MAR Ratio (IS)": "{:,.2f}"
                                        }), use_container_width=True)

                                        st.subheader("2. Conclusión de la Calibración")
                                        st.success(
                                            f"La fracción óptima en el período de entrenamiento fue **{best_row['Fracción de F Aplicada']}**, "
                                            f"produciendo un MAR Ratio de **{best_row['MAR Ratio (IS)']:.2f}**.")
                                        st.markdown(
                                            "**Recomendación:** Analiza la tabla. Si hay una 'meseta' de buenos resultados, considera elegir un valor "
                                            "conservador dentro de esa zona en lugar del pico absoluto para mayor robustez."
                                        )

                                        st.subheader("3. Resultados de la Validación (Out-of-Sample)")
                                        st.write(
                                            f"A continuación, se muestran los resultados al aplicar la fracción recomendada "
                                            f"de **{best_row['Fracción de F Aplicada']}** a los datos de validación.")

                                        oos_df = pd.DataFrame([validation_res])
                                        st.dataframe(oos_df.style.format({
                                            "CAGR (OOS)": "{:,.2f}%",
                                            "Max Drawdown (OOS) (%)": "{:,.2f}",
                                            "MAR Ratio (OOS)": "{:,.2f}"
                                        }), use_container_width=True)

                                        degradation = ((validation_res['MAR Ratio (OOS)'] - best_row[
                                            'MAR Ratio (IS)']) / best_row['MAR Ratio (IS)']) * 100 if best_row[
                                                                                                          'MAR Ratio (IS)'] != 0 else 0
                                        if degradation < -50:
                                            st.error(
                                                f"🔴 ¡Peligro! El MAR Ratio se degradó en un {abs(degradation):.1f}%. El modelo está probablemente sobreajustado.")
                                        elif degradation < -25:
                                            st.warning(
                                                f"🟡 Advertencia: El MAR Ratio se degradó en un {abs(degradation):.1f}%. Procede con cautela.")
                                        else:
                                            st.success(
                                                f"🟢 Robusto: El rendimiento se mantuvo bien en el período de validación (Degradación del MAR: {degradation:.1f}%).")

                                        if annual_f_data_validation:
                                            st.subheader("4. Desglose del Riesgo Anual Aplicado en la Validación (OOS)")
                                            st.write(
                                                f"Estos son los valores de Optimal F calculados anualmente y el riesgo real aplicado usando la fracción recomendada de **{best_row['Fracción de F Aplicada']}**.")
                                            st.dataframe(pd.DataFrame(annual_f_data_validation).set_index(
                                                "Año de Aplicación").style.format({
                                                "Optimal F Calculado": "{:.2f}",
                                                "Riesgo Aplicado (%)": "{:.2f}%"
                                            }), use_container_width=True)
                        # --- FIN: NUEVA SECCIÓN DE CALIBRACIÓN ---

            with tab3:
                st.header("Análisis de Riesgo y Peores Escenarios")
                if 'portfolio_trades' not in st.session_state or st.session_state.portfolio_trades.empty:
                    st.info("Construye o optimiza un portafolio en las pestañas anteriores.")
                else:
                    portfolio_trades = st.session_state.portfolio_trades
                    st.success(
                        f"Analizando el portafolio de **{portfolio_trades['Result name'].nunique()} estrategias**.")
                    st.divider()

                    if calculation_mode == 'Simulación (Interés Compuesto)':
                        st.subheader("Métricas de Rendimiento (Simulación)")
                        equity_curve_sim_t3, final_balance_sim_t3, max_dd_pct_compuesto = run_portfolio_simulation(
                            portfolio_trades, initial_capital, risk_per_trade_pct, r_multiple_cap, fixed_risk_amount,
                            riesgo_historico_por_trade)
                        cagr_sim = calculate_cagr(initial_capital, final_balance_sim_t3, portfolio_trades)
                        mar_sim = calculate_mar(cagr_sim, max_dd_pct_compuesto)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("CAGR Simulado (%)", f"{cagr_sim:.2f}%")
                        col2.metric("Máx. DD Simulado (%)", f"{max_dd_pct_compuesto:.2f}%")
                        col3.metric("MAR Ratio Simulado", f"{mar_sim:.2f}")

                        st.plotly_chart(px.line(equity_curve_sim_t3, title="Curva de Capital Simulada").update_layout(
                            showlegend=False), use_container_width=True, key="sim_chart_tab3")

                        st.subheader("Análisis de Estancamiento y Recuperación")

                        dd_threshold_sim = st.slider(
                            "Umbral de Drawdown para Análisis (%)",
                            min_value=0.1, max_value=100.0, value=2.0, step=0.1,
                            key="dd_threshold_sim",
                            help="Solo las caídas superiores a este porcentaje se considerarán para el análisis de duración y frecuencia."
                        )
                        dd_stats_sim = calculate_drawdown_stats(equity_curve_sim_t3,
                                                                drawdown_threshold_pct=dd_threshold_sim)
                        new_high_stats_sim = calculate_new_high_stats(equity_curve_sim_t3)

                        dd_col1, dd_col2, dd_col3, dd_col4 = st.columns(4)
                        dd_col1.metric("Máximo Tiempo en Drawdown", format_timedelta(dd_stats_sim['max_recovery_time']))
                        dd_col2.metric("Tiempo Medio en Drawdown", format_timedelta(dd_stats_sim['avg_recovery_time']))
                        dd_col3.metric("Frecuencia Media de DDs",
                                       format_timedelta(dd_stats_sim['avg_time_between_dds']))
                        dd_col4.metric("Tiempo Medio para Nuevo Máximo",
                                       format_timedelta(new_high_stats_sim['avg_time_to_new_high']))

                        with st.expander("🔍 Cómo interpretar estas métricas"):
                            st.markdown("""
                            - **Máximo Tiempo en Drawdown:** El período más largo que el portafolio ha estado por debajo de un pico anterior de capital. Mide el peor escenario de "estancamiento".
                            - **Tiempo Medio en Drawdown:** La duración promedio de un período de drawdown. Indica cuánto dura típicamente una fase de recuperación.
                            - **Frecuencia Media de DDs:** El tiempo promedio entre el inicio de un drawdown significativo y el siguiente. Responde a: ¿cada cuánto tiempo puedo esperar que comience una caída importante?
                            - **Tiempo Medio para Nuevo Máximo:** Una vez que se alcanza un pico de capital, ¿cuánto tiempo tarda el portafolio en promedio en superar ese pico? Mide el "tiempo de estancamiento" antes de un nuevo crecimiento.
                            """)
                        st.divider()

                        st.subheader("Análisis de Volatilidad Mensual (Simulado)")
                        gain_p5, loss_ci_95, prob_loss_month = calculate_historical_monthly_performance(
                            portfolio_trades, initial_capital, risk_per_trade_pct, r_multiple_cap, fixed_risk_amount,
                            riesgo_historico_por_trade)
                        recoverable_dd = calculate_monthly_recovery_dd(portfolio_trades, initial_capital,
                                                                       risk_per_trade_pct, r_multiple_cap,
                                                                       fixed_risk_amount, riesgo_historico_por_trade)
                        row2_col1_t3, row2_col2_t3 = st.columns(2);
                        row3_col1_t3, row3_col2_t3 = st.columns(2)
                        row2_col1_t3.metric("Ganancia Mensual Esperada (P5)", f"${gain_p5:,.2f}");
                        row2_col2_t3.metric("Pérdida Mensual (IC 95%)",
                                            f"${loss_ci_95[1]:,.2f} a ${loss_ci_95[0]:,.2f}")
                        row3_col1_t3.metric("Máx. DD Mensual Recuperable (%)", f"{recoverable_dd:.2f}%");
                        row3_col2_t3.metric("Prob. Mes Negativo (%)", f"{prob_loss_month:.2f}%")

                    else:  # Modo Histórico
                        st.subheader("Métricas de Rendimiento (Histórico)")
                        equity_curve_hist_t3, start_hist_t3, end_hist_t3, dd_hist_t3, can_calc_hist_t3 = calculate_isolated_historical_performance(
                            portfolio_trades)
                        if can_calc_hist_t3:
                            cagr_hist_t3 = calculate_cagr(start_hist_t3, end_hist_t3, portfolio_trades)
                            mar_ratio_hist_t3 = calculate_mar(cagr_hist_t3, dd_hist_t3)
                            hist_col1_t3, hist_col2_t3, hist_col3_t3 = st.columns(3)
                            hist_col1_t3.metric("CAGR (Histórico) (%)", f"{cagr_hist_t3:.2f}%");
                            hist_col2_t3.metric("Máx. Drawdown (Histórico) %", f"{dd_hist_t3:.2f}%");
                            hist_col3_t3.metric("MAR Ratio (Histórico)", f"{mar_ratio_hist_t3:.2f}")

                            st.plotly_chart(
                                px.line(equity_curve_hist_t3, title="Curva de Capital Histórica").update_layout(
                                    showlegend=False), use_container_width=True, key="hist_chart_tab3")

                            st.subheader("Análisis de Estancamiento y Recuperación")

                            dd_threshold_hist = st.slider(
                                "Umbral de Drawdown para Análisis (%)",
                                min_value=0.1, max_value=100.0, value=2.0, step=0.1,
                                key="dd_threshold_hist",
                                help="Solo las caídas superiores a este porcentaje se considerarán para el análisis de duración y frecuencia."
                            )
                            dd_stats_hist = calculate_drawdown_stats(equity_curve_hist_t3,
                                                                     drawdown_threshold_pct=dd_threshold_hist)
                            new_high_stats_hist = calculate_new_high_stats(equity_curve_hist_t3)

                            dd_col1, dd_col2, dd_col3, dd_col4 = st.columns(4)
                            dd_col1.metric("Máximo Tiempo en Drawdown",
                                           format_timedelta(dd_stats_hist['max_recovery_time']))
                            dd_col2.metric("Tiempo Medio en Drawdown",
                                           format_timedelta(dd_stats_hist['avg_recovery_time']))
                            dd_col3.metric("Frecuencia Media de DDs",
                                           format_timedelta(dd_stats_hist['avg_time_between_dds']))
                            dd_col4.metric("Tiempo Medio para Nuevo Máximo",
                                           format_timedelta(new_high_stats_hist['avg_time_to_new_high']))

                            with st.expander("🔍 Cómo interpretar estas métricas"):
                                st.markdown("""
                                - **Máximo Tiempo en Drawdown:** El período más largo que el portafolio ha estado por debajo de un pico anterior de capital. Mide el peor escenario de "estancamiento".
                                - **Tiempo Medio en Drawdown:** La duración promedio de un período de drawdown. Indica cuánto dura típicamente una fase de recuperación.
                                - **Frecuencia Media de DDs:** El tiempo promedio entre el inicio de un drawdown significativo y el siguiente. Responde a: ¿cada cuánto tiempo puedo esperar que comience una caída importante?
                                - **Tiempo Medio para Nuevo Máximo:** Una vez que se alcanza un pico de capital, ¿cuánto tiempo tarda el portafolio en promedio en superar ese pico? Mide el "tiempo de estancamiento" antes de un nuevo crecimiento.
                                """)
                            st.divider()

                        else:
                            st.error("Error: No se pueden calcular las métricas históricas.", icon="🚨")

                    st.divider()

                    st.subheader("📈 Proyecciones de Crecimiento (Simulación Monte Carlo)")
                    st.info(
                        "**Nota Importante:** Las proyecciones de crecimiento **siempre** utilizan un motor de simulación con "
                        "interés compuesto, aplicando los parámetros de riesgo definidos en la barra lateral."
                    )
                    if st.button("🚀 Calcular Proyecciones de Crecimiento"):
                        num_sims_for_growth = st.session_state.get('num_sims_input', 2000)

                        with st.spinner(
                                f"Ejecutando {num_sims_for_growth:,} simulaciones para cada horizonte de tiempo..."):
                            trades_per_month = calculate_trade_frequency(portfolio_trades)

                            if trades_per_month == 0:
                                st.warning(
                                    "No hay suficientes datos históricos (se necesita más de 1 día de operativa) para calcular la frecuencia de trades y realizar proyecciones.")
                            else:
                                st.info(f"Frecuencia de trading calculada: **{trades_per_month:.1f} trades/mes**.")

                                horizons = {'6 Meses': 6, '12 Meses': 12, '24 Meses': 24}
                                projection_results = []

                                for name, months in horizons.items():
                                    num_trades_to_project = int(trades_per_month * months)
                                    if num_trades_to_project < 1: continue

                                    final_balances = run_growth_projection_mc(
                                        trades_df=portfolio_trades, initial_capital=initial_capital,
                                        risk_per_trade_pct=risk_per_trade_pct, num_simulations=num_sims_for_growth,
                                        num_trades_to_project=num_trades_to_project, r_multiple_cap=r_multiple_cap,
                                        fixed_risk_amount=fixed_risk_amount,
                                        riesgo_historico_por_trade=riesgo_historico_por_trade
                                    )

                                    if final_balances is not None and len(final_balances) > 0:
                                        p10, p50, p90 = np.percentile(final_balances, [10, 50, 90])
                                        gain_p10 = ((p10 / initial_capital) - 1) * 100 if initial_capital > 0 else 0
                                        gain_p50 = ((p50 / initial_capital) - 1) * 100 if initial_capital > 0 else 0
                                        gain_p90 = ((p90 / initial_capital) - 1) * 100 if initial_capital > 0 else 0

                                        projection_results.append({
                                            "Horizonte Temporal": name,
                                            "Pesimista (P10)": f"${p10:,.2f} ({gain_p10:+.1f}%)",
                                            "Medio (P50)": f"${p50:,.2f} ({gain_p50:+.1f}%)",
                                            "Optimista (P90)": f"${p90:,.2f} ({gain_p90:+.1f}%)",
                                        })

                                if projection_results:
                                    st.table(pd.DataFrame(projection_results).set_index("Horizonte Temporal"))
                                else:
                                    st.warning(
                                        "No se pudieron generar proyecciones. La frecuencia de trading puede ser demasiado baja.")
                    st.divider()

                    st.subheader("Métricas de Trades (Basado en datos originales)")
                    risk_pct_for_metrics = (
                                fixed_risk_amount / initial_capital * 100) if fixed_risk_amount > 0 and initial_capital > 0 else risk_per_trade_pct
                    risk_metrics = calculate_risk_metrics(portfolio_trades, risk_pct_for_metrics)
                    tm_col1, tm_col2 = st.columns(2)
                    tm_col1.metric("Profit Factor", f"{risk_metrics.get('Profit Factor', 0):.2f}");
                    tm_col2.metric("Win Rate (%)", f"{risk_metrics.get('Win Rate (%)', 0):.2f}%")
                    st.divider()

                    st.subheader("Análisis de Riesgo de Ruina Teórico (Basado en datos originales)")
                    ror_col1, ror_col2 = st.columns(2)
                    ror_col1.metric("Riesgo de Ruina Teórico (%)",
                                    f"{risk_metrics.get('Riesgo de Ruina Teórico (%)', 0):.4f}%")
                    with st.expander("🔍 Entender el Cálculo del RoR Teórico y hacer un 'Stress Test'"):
                        st.markdown(
                            f"**Edge:** `{risk_metrics.get('Edge', 0):.4f}` | **Capital Units:** `{risk_metrics.get('Capital Units', 0):.1f}`")
                        edge_real = risk_metrics.get('Edge', 0)
                        sim_edge = st.slider("Simular un 'Edge' menor:", -0.1,
                                             max(0.1, edge_real) if edge_real is not None else 0.1, edge_real, 0.005,
                                             format="%.4f", key="edge_slider")
                        sim_ror = calculate_ror(sim_edge, risk_metrics.get('Capital Units', 0))
                        ror_col2.metric("RoR con 'Edge' Simulado (%)", f"{sim_ror:.4f}%")
                    st.divider()

                    st.subheader("🔬 Simulación de Monte Carlo (Riesgo a Largo Plazo)")
                    mc_col1, mc_col2 = st.columns(2)
                    with mc_col1:
                        ruin_level = st.slider("Nivel de Ruina (Máx. DD %)", 10, 100, 50, key="ruin_level_slider")
                    with mc_col2:
                        num_sims = st.number_input("Número de Simulaciones", 100, 10000, 2000, key="num_sims_input")
                    if st.button("🚀 Ejecutar Simulación de Monte Carlo"):
                        with st.spinner(f"Ejecutando {num_sims:,} simulaciones..."):
                            prob_ruin, lower, upper = run_monte_carlo_simulation_vectorized(portfolio_trades,
                                                                                            initial_capital,
                                                                                            risk_per_trade_pct,
                                                                                            ruin_level, num_sims,
                                                                                            r_multiple_cap,
                                                                                            fixed_risk_amount,
                                                                                            riesgo_historico_por_trade)
                            res_col1, res_col2 = st.columns(2)
                            res_col1.metric("Probabilidad de Ruina (MC)", f"{prob_ruin:.2f}%",
                                            delta=f"{100 - prob_ruin:.2f}% Supervivencia", delta_color="inverse")
                            if lower > 0 and upper > 0:
                                avg_trade_time = risk_metrics.get("Avg Time per Trade")
                                life_lower, life_upper = avg_trade_time * lower, avg_trade_time * upper
                                res_col2.metric("Esperanza de Vida (95% Confianza)",
                                                f"{format_timedelta(life_lower)} - {format_timedelta(life_upper)}")
                            else:
                                res_col2.metric("Esperanza de Vida", "Muy Alta")

            with tab4:
                st.header("Análisis de Debilidades por Activo")
                if 'portfolio_trades' not in st.session_state or st.session_state.portfolio_trades.empty:
                    st.info("Construye o optimiza un portafolio para ver el análisis por activo.")
                else:
                    portfolio_trades = st.session_state.portfolio_trades
                    if 'Symbol' not in portfolio_trades.columns:
                        st.error("La columna 'Symbol' no se encuentra en el archivo CSV.")
                    else:
                        st.subheader("Métricas Históricas Aisladas y Contribución a Péridas")
                        with st.expander("🔍 ¿Cómo interpretar la 'Contribución Media a Meses de Pérdida'?"):
                            st.markdown("""
                            - **Positivo Grande (>100%):** El activo pierde más que la pérdida total del portafolio. **Muy Malo** 👎
                            - **Positivo (0%-100%):** El activo contribuye a la pérdida del portafolio. **Malo**
                            - **Negativo (< 0%):** El activo gana dinero mientras el portafolio pierde. **Excelente** 👍
                            """)

                        asset_contributions = calculate_average_contribution_to_losing_months(portfolio_trades,
                                                                                              ['Symbol'])
                        strategy_contributions = calculate_average_contribution_to_losing_months(portfolio_trades,
                                                                                                 ['Symbol',
                                                                                                  'Result name'])
                        if not asset_contributions:
                            st.info(
                                "El portafolio no ha tenido meses con pérdidas. No se puede calcular la contribución.")
                        st.markdown("---")
                        unique_assets = portfolio_trades['Symbol'].unique()
                        for asset in unique_assets:
                            st.subheader(f"Activo: {asset}")
                            asset_df = portfolio_trades[portfolio_trades['Symbol'] == asset].copy()
                            _, start_hist, end_hist, dd_hist, can_calc_hist = calculate_isolated_historical_performance(
                                asset_df)
                            if can_calc_hist:
                                cagr_hist, mar_ratio_hist = calculate_cagr(start_hist, end_hist,
                                                                           asset_df), calculate_mar(
                                    calculate_cagr(start_hist, end_hist, asset_df), dd_hist)
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("CAGR (Aislado) (%)", f"{cagr_hist:.2f}%")
                                col2.metric("Máx. DD (Aislado) %", f"{dd_hist:.2f}%")
                                col3.metric("MAR Ratio (Aislado)", f"{mar_ratio_hist:.2f}")
                                avg_contribution = asset_contributions.get(asset, 0)
                                col4.metric("Contr. Media a Meses de Pérdida", f"{avg_contribution:.1f}%")
                            else:
                                st.warning(f"No se pudieron calcular las métricas para '{asset}'.")
                            with st.expander("Ver desglose por estrategia"):
                                unique_strategies_in_asset = asset_df['Result name'].unique()
                                if len(unique_strategies_in_asset) == 0:
                                    st.info("No hay estrategias individuales para este activo.")
                                else:
                                    for strategy in unique_strategies_in_asset:
                                        st.markdown(f"**Estrategia:** `{strategy}`")
                                        strategy_df = asset_df[asset_df['Result name'] == strategy].copy()
                                        _, s_start, s_end, s_dd, s_can_calc = calculate_isolated_historical_performance(
                                            strategy_df)
                                        if s_can_calc:
                                            s_cagr, s_mar = calculate_cagr(s_start, s_end, strategy_df), calculate_mar(
                                                calculate_cagr(s_start, s_end, strategy_df), s_dd)
                                            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                                            s_col1.metric("CAGR (Aislado) (%)", f"{s_cagr:.2f}%")
                                            s_col2.metric("Máx. DD (Aislado) %", f"{s_dd:.2f}%")
                                            s_col3.metric("MAR Ratio (Aislado)", f"{s_mar:.2f}")
                                            strategy_key = (asset, strategy)
                                            avg_strat_contribution = strategy_contributions.get(strategy_key, 0)
                                            s_col4.metric("Contr. Media a Meses de Pérdida",
                                                          f"{avg_strat_contribution:.1f}%")
                                        else:
                                            st.write("Métricas no disponibles.")
                                        st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("---")

                        st.header("Análisis de Contribución y Redundancia de Estrategias")
                        st.info(f"Modo de cálculo actual: **{calculation_mode}**.")
                        if st.button("🔬 Ejecutar Análisis de Contribución y Redundancia"):
                            with st.spinner("Realizando análisis..."):
                                params = {"calculation_mode": calculation_mode, "initial_capital": initial_capital,
                                          "risk_per_trade_pct": risk_per_trade_pct, "r_multiple_cap": r_multiple_cap,
                                          "fixed_risk_amount": fixed_risk_amount,
                                          "riesgo_historico_por_trade": riesgo_historico_por_trade}
                                st.subheader("1. Ranking de Contribución de Estrategias al Portafolio")
                                portfolio_base_mar = get_mar_for_portfolio(portfolio_trades, **params)
                                st.write(f"**MAR Ratio Base del Portafolio Actual: {portfolio_base_mar:.2f}**")
                                all_strategies = portfolio_trades['Result name'].unique()
                                analysis_results = []
                                progress_bar = st.progress(0)
                                for i, strategy_name in enumerate(all_strategies):
                                    portfolio_minus_one = portfolio_trades[
                                        portfolio_trades['Result name'] != strategy_name]
                                    new_mar = get_mar_for_portfolio(portfolio_minus_one, **params)
                                    marginal_contribution = new_mar - portfolio_base_mar
                                    asset_name = \
                                    portfolio_trades[portfolio_trades['Result name'] == strategy_name]['Symbol'].iloc[0]
                                    analysis_results.append({"Estrategia": strategy_name, "Activo": asset_name,
                                                             "Contribución Marginal al MAR": marginal_contribution})
                                    progress_bar.progress((i + 1) / len(all_strategies))
                                analysis_df = pd.DataFrame(analysis_results).sort_values(
                                    by="Contribución Marginal al MAR", ascending=False)


                                def get_diagnosis(value):
                                    if value > 0.05:
                                        return "🔴 Lastre Principal"
                                    elif value > 0:
                                        return "🟠 Lastre Secundario"
                                    elif value < -0.1:
                                        return "🔵 Pilar del Portafolio"
                                    elif value < 0:
                                        return "🟢 Contribuidor Clave"
                                    else:
                                        return "🟡 Neutral"


                                analysis_df["Diagnóstico"] = analysis_df["Contribución Marginal al MAR"].apply(
                                    get_diagnosis)
                                st.dataframe(analysis_df, use_container_width=True)
                                st.markdown(
                                    "Una **contribución positiva** significa que el MAR del portafolio *mejora* si se quita la estrategia.")
                                st.subheader("2. Análisis de Redundancia por Activo")
                                for asset in unique_assets:
                                    st.markdown(f"#### Activo: {asset}")
                                    asset_df = portfolio_trades[portfolio_trades['Symbol'] == asset].copy()
                                    asset_strategies = asset_df['Result name'].unique()
                                    if len(asset_strategies) < 2:
                                        st.write("No hay suficientes estrategias para analizar la redundancia.")
                                        continue
                                    returns_df = get_strategy_returns_for_asset(asset_df)
                                    if returns_df.empty:
                                        st.write("No se pudieron calcular los retornos.")
                                        continue
                                    corr_matrix = returns_df.corr()
                                    redundant_pairs_found = False
                                    for i in range(len(corr_matrix.columns)):
                                        for j in range(i):
                                            if corr_matrix.iloc[i, j] > 0.8:
                                                redundant_pairs_found = True
                                                strat1_name, strat2_name = corr_matrix.columns[i], corr_matrix.columns[
                                                    j]
                                                strat1_contribution = \
                                                analysis_df[analysis_df['Estrategia'] == strat1_name][
                                                    "Contribución Marginal al MAR"].iloc[0]
                                                strat2_contribution = \
                                                analysis_df[analysis_df['Estrategia'] == strat2_name][
                                                    "Contribución Marginal al MAR"].iloc[0]
                                                st.markdown(
                                                    f"- **Par Redundante (Correlación: {corr_matrix.iloc[i, j]:.2f})**")
                                                st.text(f"  - {strat1_name} (Contribución: {strat1_contribution:.2f})")
                                                st.text(f"  - {strat2_name} (Contribución: {strat2_contribution:.2f})")
                                                if strat1_contribution > strat2_contribution:
                                                    st.warning(
                                                        f"  - **Recomendación:** Ambas son muy similares, pero **'{strat1_name}'** es menos efectiva.")
                                                else:
                                                    st.warning(
                                                        f"  - **Recomendación:** Ambas son muy similares, pero **'{strat2_name}'** es menos efectiva.")
                                    if not redundant_pairs_found:
                                        st.success("No se encontraron estrategias redundantes en este activo.")

    except Exception as e:
        st.error(f"Ha ocurrido un error: {e}")
        st.exception(e)

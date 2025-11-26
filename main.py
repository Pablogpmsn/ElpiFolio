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
        return pd.Series([initial_capital]), initial_capital, 0

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

    if trades_df.empty: return pd.Series([initial_capital]), initial_capital, 0

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

    plot_index = pd.to_datetime(
        [trades_df['Close time'].min() - pd.Timedelta(seconds=1)] + list(trades_df['Close time']))
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

@st.cache_data
def get_strategy_level_stats(trades_df):
    """
    Calcula métricas básicas por estrategia para poder filtrar
    antes de entrar al optimizador.
    """
    if trades_df.empty or 'Result name' not in trades_df.columns:
        return pd.DataFrame()

    stats = []
    for strat_name, df_s in trades_df.groupby('Result name'):
        # Usamos la lógica histórica ya existente
        eq, start, end, dd, ok = calculate_isolated_historical_performance(df_s)
        if not ok or start <= 0 or end <= 0:
            continue

        cagr = calculate_cagr(start, end, df_s)
        mar = calculate_mar(cagr, dd)
        n_trades = len(df_s)

        stats.append({
            "Result name": strat_name,
            "NumTrades": n_trades,
            "CAGR": cagr,
            "MaxDD": dd,
            "MAR": mar
        })

    if not stats:
        return pd.DataFrame()

    return pd.DataFrame(stats)

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
        if 'MAE ($)' not in trades_df.columns or trades_df['MAE ($)'].abs().sum() == 0:
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


# --- INICIO: FUNCIONES PARA HIERARCHICAL RISK PARITY (HRP) ---
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


# --- FIN: FUNCIONES PARA HIERARCHICAL RISK PARITY (HRP) ---


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
                training_data = data_full_sorted.iloc[:split_index].copy()
                validation_data = data_full_sorted.iloc[split_index:].copy()
                data_for_analysis = training_data
            else:
                training_data = data_full_sorted
                validation_data = pd.DataFrame()
                data_for_analysis = data_full_sorted

            # >>> NUEVO: calcular stats por estrategia para el training <<<
            strategy_stats = get_strategy_level_stats(training_data)

            # Después de esto ya siguen las pestañas:
            tab1, tab2, tab3, tab4 = st.tabs(["Constructor de Portafolios",
                                              "Optimizador de Portafolios",
                                              "Análisis de Riesgo",
                                              "Análisis de Debilidades"])

            calculation_mode = st.radio("Selecciona el modo de cálculo para las métricas y gráficos principales:",
                                        ('Simulación (Interés Compuesto)', 'Histórico (Datos del CSV)'),
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
                st.subheader("FASE 0: Filtro de calidad individual (antes de optimizar)")

                if strategy_stats.empty:
                    st.warning("No se pudieron calcular estadísticas por estrategia (quizá faltan columnas o datos).")
                    filtered_stats = pd.DataFrame()
                else:
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    min_trades = col_f1.number_input("Mín. nº de trades", min_value=10, max_value=2000, value=100,
                                                     step=10)
                    max_dd_ind = col_f2.number_input("Máx. DD individual (%)", min_value=2.0, max_value=80.0,
                                                     value=35.0, step=1.0)
                    min_mar_ind = col_f3.number_input("Mín. MAR individual", min_value=0.0, max_value=5.0, value=0.30,
                                                      step=0.05)
                    max_candidates = col_f4.number_input("Máx. estrategias a enviar al optimizador",
                                                         min_value=10, max_value=400, value=80, step=5)

                    filtered_stats = strategy_stats[
                        (strategy_stats["NumTrades"] >= min_trades) &
                        (strategy_stats["MaxDD"] <= max_dd_ind) &
                        (strategy_stats["MAR"] >= min_mar_ind)
                        ].copy()

                    # Ordenamos por MAR y nos quedamos con las mejores
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

                    if filtered_stats.empty:
                        st.error("No hay estrategias tras aplicar los filtros. Afloja un poco los criterios.")
                        all_strategies_list = []
                    else:
                        all_strategies_list = list(filtered_stats['Result name'].unique())

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
                            # --- NUEVO: calcular robustez IS vs OOS para cada campeón ---
                            if oos_percentage > 0 and not validation_data.empty:
                                def calc_degradation(is_val, oos_val, is_lower_better=False):
                                    if abs(is_val) < 1e-6:
                                        return 0.0
                                    degradation = ((oos_val - is_val) / abs(is_val)) * 100
                                    return degradation if not is_lower_better else -degradation


                                for champ in champions_by_size:
                                    combo_tuple = tuple(sorted(champ["combo"]))

                                    # métricas In-Sample (training)
                                    is_mar, is_dd, is_profit = get_portfolio_metrics_final(
                                        training_data,
                                        combo_tuple,
                                        calculation_mode,
                                        initial_capital,
                                        risk_per_trade_pct,
                                        r_multiple_cap,
                                        fixed_risk_amount,
                                        riesgo_historico_por_trade
                                    )

                                    # métricas Out-of-Sample (validation)
                                    oos_mar, oos_dd, oos_profit = get_portfolio_metrics_final(
                                        validation_data,
                                        combo_tuple,
                                        calculation_mode,
                                        initial_capital,
                                        risk_per_trade_pct,
                                        r_multiple_cap,
                                        fixed_risk_amount,
                                        riesgo_historico_por_trade
                                    )

                                    champ["is_mar"] = is_mar
                                    champ["is_dd"] = is_dd
                                    champ["is_profit"] = is_profit
                                    champ["oos_mar"] = oos_mar
                                    champ["oos_dd"] = oos_dd
                                    champ["oos_profit"] = oos_profit

                                    champ["mar_deg"] = calc_degradation(is_mar, oos_mar)
                                    champ["dd_deg"] = calc_degradation(is_dd, oos_dd, is_lower_better=True)

                                    # MISMOS UMBRALES que la luz roja de la app
                                    champ["is_robust"] = (champ["mar_deg"] >= -50) and (champ["dd_deg"] <= 50)
                            else:
                                # si no hay OOS, se consideran robustos por defecto
                                for champ in champions_by_size:
                                    champ["is_robust"] = True

                            # --- NUEVO: elegir campeón solo entre robustos (verde/amarillo) ---
                            robust_candidates = [c for c in champions_by_size if c.get("is_robust", True)]
                            if not robust_candidates:
                                # si todos son rojos, usamos todos pero luego ya lo verás en rojo
                                robust_candidates = champions_by_size

                            absolute_champion = sorted(
                                robust_candidates,
                                key=lambda x: x[sort_key],
                                reverse=reverse_sort
                            )[0]

                            st.session_state.optimizer_results = {
                                "absolute_champion": absolute_champion,
                                "champions_by_size": champions_by_size
                            }
                        else:
                            st.warning("No se encontraron resultados válidos.")
                            if 'optimizer_results' in st.session_state:
                                del st.session_state.optimizer_results

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

                    champions_df = pd.DataFrame(champs_by_size)

                    # columnas legibles
                    champions_df = champions_df.rename(columns={
                        "size": "Tamaño",
                        "mar_ratio": "MAR In-Sample (búsqueda)",
                        "drawdown": "Max DD In-Sample (%)",
                        "profit": "Profit In-Sample (%)",
                        "is_mar": "MAR IS",
                        "is_dd": "Max DD IS (%)",
                        "oos_mar": "MAR OOS",
                        "oos_dd": "Max DD OOS (%)",
                        "is_robust": "Robusto",
                    })

                    # combo en texto (para que veas qué estrategias lleva)
                    champions_df["Estrategias"] = champions_df["combo"].apply(
                        lambda c: ", ".join(c) if isinstance(c, (list, tuple)) else str(c)
                    )

                    cols_base = ["Tamaño", "MAR In-Sample (búsqueda)", "Max DD In-Sample (%)", "Profit In-Sample (%)"]
                    cols_oos = ["MAR IS", "Max DD IS (%)", "MAR OOS", "Max DD OOS (%)", "Robusto"]
                    cols_extra = [c for c in cols_oos if c in champions_df.columns]

                    cols_to_show = cols_base + cols_extra + ["Estrategias"]

                    st.dataframe(
                        champions_df[cols_to_show].sort_values(by="Tamaño", ascending=True),
                        use_container_width=True
                    )

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
                        "No cambia las estrategias, solo su ponderación. La simulación del portafolio HRP es estocástica, por lo que los resultados pueden variar ligeramente en cada ejecución."
                    )

                    if st.button("🚀 Refinar Portafolio con Hierarchical Risk Parity (HRP)"):
                        champion_combo = tuple(sorted(res['absolute_champion']['combo']))
                        champion_trades_is = training_data[training_data['Result name'].isin(list(champion_combo))]

                        if len(champion_trades_is['Result name'].unique()) < 2:
                            st.warning(
                                "HRP requiere al menos 2 estrategias en el portafolio para poder calcular las ponderaciones.")
                        else:
                            with st.spinner(
                                    "Calculando pesos HRP y simulando los portafolios refinados (In-Sample y Out-of-Sample)..."):
                                daily_returns_df = get_daily_returns(champion_trades_is)

                                if daily_returns_df.empty or daily_returns_df.shape[1] < 2:
                                    st.error(
                                        "No se pudieron calcular los retornos diarios para las estrategias. No se puede ejecutar HRP.")
                                else:
                                    hrp_weights = get_hrp_weights(daily_returns_df)
                                    hrp_portfolio_trades_is = create_hrp_resampled_portfolio(champion_trades_is,
                                                                                             hrp_weights)

                                    hrp_results_dict = {
                                        "weights": hrp_weights,
                                        "original_trades_is": champion_trades_is,
                                        "hrp_trades_is": hrp_portfolio_trades_is,
                                        "oos_results": {}
                                    }

                                    # --- NUEVA LÓGICA PARA OOS ---
                                    if oos_percentage > 0 and not validation_data.empty:
                                        champion_trades_oos = validation_data[
                                            validation_data['Result name'].isin(list(champion_combo))]
                                        if not champion_trades_oos.empty:
                                            hrp_portfolio_trades_oos = create_hrp_resampled_portfolio(
                                                champion_trades_oos, hrp_weights)
                                            if not hrp_portfolio_trades_oos.empty:
                                                hrp_results_dict["oos_results"] = {
                                                    "original_trades_oos": champion_trades_oos,
                                                    "hrp_trades_oos": hrp_portfolio_trades_oos
                                                }

                                    st.session_state.hrp_results = hrp_results_dict
                                    st.success("Análisis HRP completado.")

                    if 'hrp_results' in st.session_state:
                        hrp_res = st.session_state.hrp_results
                        hrp_weights = hrp_res["weights"]

                        params_for_metrics = {
                            "_calculation_mode": calculation_mode, "_initial_capital": initial_capital,
                            "_risk_per_trade_pct": risk_per_trade_pct, "_r_multiple_cap": r_multiple_cap,
                            "_fixed_risk_amount": fixed_risk_amount,
                            "_riesgo_historico_por_trade": riesgo_historico_por_trade
                        }


                        def format_improvement(value, is_dd=False):
                            if not np.isfinite(value): return "N/A"
                            if is_dd: value = -value
                            if value > 0:
                                return f"🟢 +{value:.1f}%"
                            elif value < 0:
                                return f"🔴 {value:.1f}%"
                            else:
                                return f"⚪️ {value:.1f}%"


                        # --- RENDERIZADO IN-SAMPLE ---
                        st.subheader("📊 Tabla Comparativa de Rendimiento (In-Sample)")
                        original_trades_is = hrp_res["original_trades_is"]
                        hrp_trades_is = hrp_res["hrp_trades_is"]

                        orig_is_mar, orig_is_dd, orig_is_profit = get_portfolio_metrics_final(original_trades_is, tuple(
                            original_trades_is['Result name'].unique()), **params_for_metrics)
                        hrp_is_mar, hrp_is_dd, hrp_is_profit = get_portfolio_metrics_final(hrp_trades_is, tuple(
                            hrp_trades_is['Result name'].unique()), **params_for_metrics)

                        is_mar_imp = ((hrp_is_mar / orig_is_mar) - 1) * 100 if orig_is_mar != 0 else float('inf')
                        is_dd_imp = ((orig_is_dd / hrp_is_dd) - 1) * 100 if hrp_is_dd != 0 else float('inf')
                        is_profit_imp = ((hrp_is_profit / orig_is_profit) - 1) * 100 if orig_is_profit != 0 else float(
                            'inf')

                        comparison_is_df = pd.DataFrame({
                            "Métrica": ["MAR Ratio", "Max Drawdown (%)", "Rentabilidad (%)"],
                            "Portafolio Original": [f"{orig_is_mar:.2f}", f"{orig_is_dd:.2f}%",
                                                    f"{orig_is_profit:.2f}%"],
                            "Portafolio Refinado (HRP)": [f"{hrp_is_mar:.2f}", f"{hrp_is_dd:.2f}%",
                                                          f"{hrp_is_profit:.2f}%"],
                            "Mejora": [format_improvement(is_mar_imp), format_improvement(-is_dd_imp),
                                       format_improvement(is_profit_imp)]
                        })
                        st.dataframe(comparison_is_df.set_index("Métrica"), use_container_width=True)

                        hrp_col1, hrp_col2 = st.columns(2)
                        with hrp_col1:
                            st.subheader("⚖️ Asignación de Capital HRP")
                            weights_df = hrp_weights.reset_index()
                            weights_df.columns = ['Estrategia', 'Peso']
                            weights_df['Peso'] = weights_df['Peso'] * 100  # convertir a %

                            # ordenamos el DF y lo guardamos (muy importante)
                            weights_sorted = weights_df.sort_values('Peso', ascending=True)

                            fig_weights = px.bar(
                                weights_sorted,
                                x='Peso',
                                y='Estrategia',
                                orientation='h',
                                text=weights_sorted['Peso'].apply(lambda x: f"{x:.1f}%"),
                            )

                            # ajusta eje y añade margen a la derecha
                            fig_weights.update_layout(
                                xaxis=dict(range=[0, weights_sorted['Peso'].max() * 1.10])
                            )
                            fig_weights.update_traces(textposition='outside')
                            st.plotly_chart(fig_weights, use_container_width=True)

                        with hrp_col2:
                            st.subheader("📈 Curvas de Capital (In-Sample)")
                            if calculation_mode == 'Simulación (Interés Compuesto)':
                                ec_orig, _, _ = run_portfolio_simulation(original_trades_is,
                                                                         **{k.lstrip('_'): v for k, v in
                                                                            params_for_metrics.items() if
                                                                            k != '_calculation_mode'})
                                ec_hrp, _, _ = run_portfolio_simulation(hrp_trades_is, **{k.lstrip('_'): v for k, v in
                                                                                          params_for_metrics.items() if
                                                                                          k != '_calculation_mode'})
                            else:
                                ec_orig, _, _, _, _ = calculate_isolated_historical_performance(original_trades_is)
                                ec_hrp, _, _, _, _ = calculate_isolated_historical_performance(hrp_trades_is)
                            fig_curves = go.Figure()
                            fig_curves.add_trace(
                                go.Scatter(x=ec_orig.index, y=ec_orig.values, mode='lines', name='Original'))
                            fig_curves.add_trace(
                                go.Scatter(x=ec_hrp.index, y=ec_hrp.values, mode='lines', name='Refinado (HRP)'))
                            st.plotly_chart(fig_curves, use_container_width=True)

                        # --- RENDERIZADO OUT-OF-SAMPLE ---
                        if hrp_res.get("oos_results"):
                            st.markdown("---")
                            st.subheader("📊 Tabla Comparativa de Rendimiento (Out-of-Sample)")
                            original_trades_oos = hrp_res["oos_results"]["original_trades_oos"]
                            hrp_trades_oos = hrp_res["oos_results"]["hrp_trades_oos"]

                            orig_oos_mar, orig_oos_dd, orig_oos_profit = get_portfolio_metrics_final(
                                original_trades_oos, tuple(original_trades_oos['Result name'].unique()),
                                **params_for_metrics)
                            hrp_oos_mar, hrp_oos_dd, hrp_oos_profit = get_portfolio_metrics_final(hrp_trades_oos, tuple(
                                hrp_trades_oos['Result name'].unique()), **params_for_metrics)

                            oos_mar_imp = ((hrp_oos_mar / orig_oos_mar) - 1) * 100 if orig_oos_mar != 0 else float(
                                'inf')
                            oos_dd_imp = ((orig_oos_dd / hrp_oos_dd) - 1) * 100 if hrp_oos_dd != 0 else float('inf')
                            oos_profit_imp = ((
                                                          hrp_oos_profit / orig_oos_profit) - 1) * 100 if orig_oos_profit != 0 else float(
                                'inf')

                            comparison_oos_df = pd.DataFrame({
                                "Métrica": ["MAR Ratio", "Max Drawdown (%)", "Rentabilidad (%)"],
                                "Portafolio Original": [f"{orig_oos_mar:.2f}", f"{orig_oos_dd:.2f}%",
                                                        f"{orig_oos_profit:.2f}%"],
                                "Portafolio Refinado (HRP)": [f"{hrp_oos_mar:.2f}", f"{hrp_oos_dd:.2f}%",
                                                              f"{hrp_oos_profit:.2f}%"],
                                "Mejora": [format_improvement(oos_mar_imp), format_improvement(-oos_dd_imp),
                                           format_improvement(oos_profit_imp)]
                            })
                            st.dataframe(comparison_oos_df.set_index("Métrica"), use_container_width=True)

                            st.subheader("📈 Curvas de Capital (Out-of-Sample)")
                            if calculation_mode == 'Simulación (Interés Compuesto)':
                                ec_orig_oos, _, _ = run_portfolio_simulation(original_trades_oos,
                                                                             **{k.lstrip('_'): v for k, v in
                                                                                params_for_metrics.items() if
                                                                                k != '_calculation_mode'})
                                ec_hrp_oos, _, _ = run_portfolio_simulation(hrp_trades_oos,
                                                                            **{k.lstrip('_'): v for k, v in
                                                                               params_for_metrics.items() if
                                                                               k != '_calculation_mode'})
                            else:
                                ec_orig_oos, _, _, _, _ = calculate_isolated_historical_performance(original_trades_oos)
                                ec_hrp_oos, _, _, _, _ = calculate_isolated_historical_performance(hrp_trades_oos)

                            fig_curves_oos = go.Figure()
                            fig_curves_oos.add_trace(go.Scatter(x=ec_orig_oos.index, y=ec_orig_oos.values, mode='lines',
                                                                name='Original (OOS)'))
                            fig_curves_oos.add_trace(go.Scatter(x=ec_hrp_oos.index, y=ec_hrp_oos.values, mode='lines',
                                                                name='Refinado HRP (OOS)'))
                            st.plotly_chart(fig_curves_oos, use_container_width=True)
                    # --- FIN: SECCIÓN DE REFINAMIENTO CON HRP ---

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
                            min_value=0.1, max_value=20.0, value=2.0, step=0.1,
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
                                min_value=0.1, max_value=20.0, value=2.0, step=0.1,
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

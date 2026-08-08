"""Synthetic CreditGuard dataset generator.

Builds customers, loan applications, point-in-time financial/credit snapshots and
12-month default labels from a documented latent-risk process (see
docs/data_generation.md), then injects a configurable rate of data quality
imperfections for Phase 3 to catch. Everything is driven by
config/data_generation.yaml -- see that file for every tunable parameter.

CLI: python -m creditguard.data.generator [--n-customers N] [--seed S]
     [--config PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from creditguard.config import get_settings
from creditguard.data.distributions import (
    poisson_capped,
    sample_beta,
    sample_categorical,
    sample_conditional_categorical,
    sigmoid,
    truncated_normal,
    zero_inflated_poisson,
)
from creditguard.data.versioning import (
    GeneratedDataset,
    make_dataset_version,
    write_dataset,
)

DAYS_PER_MONTH = 30.4375


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the data-generation YAML config into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deduplicate_dates_per_customer(
    df: pd.DataFrame, group_col: str, date_col: str, direction: int, max_iter: int = 20
) -> pd.DataFrame:
    """Nudge duplicate (group_col, date_col) pairs by one day until unique.

    `direction` is +1 or -1; use -1 for dates that must stay on-or-before a bound
    (e.g. snapshot as_of_date) and +1 otherwise, so nudging can never violate it.
    """
    df = df.copy()
    step = pd.Timedelta(days=direction)
    for _ in range(max_iter):
        dup_mask = df.duplicated(subset=[group_col, date_col], keep="first")
        if not dup_mask.any():
            break
        df.loc[dup_mask, date_col] = df.loc[dup_mask, date_col] + step
    return df


def _disjoint_samples(
    rng: np.random.Generator, n_rows: int, rates: dict[str, float]
) -> dict[str, np.ndarray]:
    """Partition a random row-index permutation into disjoint pools sized by `rates`."""
    order = rng.permutation(n_rows)
    pools: dict[str, np.ndarray] = {}
    start = 0
    for name, rate in rates.items():
        count = int(round(n_rows * rate))
        pools[name] = order[start : start + count]
        start += count
    return pools


def _calibrate_intercept(
    logit_no_intercept: np.ndarray,
    target_rate: float,
    low: float = -15.0,
    high: float = 15.0,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Bisection search for the intercept making E[sigmoid(logit)] hit `target_rate`."""
    mid = 0.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        rate = sigmoid(logit_no_intercept + mid).mean()
        if abs(rate - target_rate) < tol:
            break
        if rate < target_rate:
            low = mid
        else:
            high = mid
    return mid


def _generate_customers(
    rng: np.random.Generator, cfg: dict[str, Any], n_customers: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate the customers table plus a latent per-customer risk_propensity array.

    risk_propensity is never written to any table; it is a generation-time device
    that correlates bureau fields (previous_defaults, late/missed payments) with
    each other realistically, without itself entering the default label directly.
    """
    demo_cfg = cfg["demographics"]

    age = np.round(
        truncated_normal(
            rng,
            demo_cfg["age"]["mean"],
            demo_cfg["age"]["std"],
            demo_cfg["age"]["min"],
            demo_cfg["age"]["max"],
            size=n_customers,
        )
    ).astype(int)

    gender = sample_categorical(
        rng, demo_cfg["gender"]["categories"], demo_cfg["gender"]["probs"], n_customers
    )
    marital_status = sample_categorical(
        rng,
        demo_cfg["marital_status"]["categories"],
        demo_cfg["marital_status"]["probs"],
        n_customers,
    )
    education = sample_categorical(
        rng,
        demo_cfg["education"]["categories"],
        demo_cfg["education"]["probs"],
        n_customers,
    )
    employment_type = sample_conditional_categorical(
        rng,
        education,
        demo_cfg["employment_type"]["categories"],
        demo_cfg["employment_type"]["probs_given_education"],
    )
    city_tier = sample_categorical(
        rng,
        demo_cfg["city_tier"]["categories"],
        demo_cfg["city_tier"]["probs"],
        n_customers,
    ).astype(int)
    dependents = poisson_capped(
        rng,
        demo_cfg["dependents"]["lambda"],
        n_customers,
        max_value=demo_cfg["dependents"]["max"],
    )

    risk_propensity = rng.normal(0, 1, size=n_customers)

    emp_cfg = cfg["employment"]
    max_employment_years = np.clip(age - 18, 0, None).astype(float)
    unemployed_factor = np.where(
        employment_type == "UNEMPLOYED", emp_cfg["unemployed_years_factor"], 1.0
    )
    effective_max = max_employment_years * unemployed_factor
    employment_years = truncated_normal(
        rng,
        emp_cfg["years_fraction_of_max_mean"] * effective_max,
        emp_cfg["years_noise_std"],
        0.0,
        effective_max,
        size=n_customers,
    )

    income_cfg = cfg["income"]
    education_shift = (
        pd.Series(education).map(income_cfg["education_shift"]).to_numpy(float)
    )
    employment_shift = (
        pd.Series(employment_type)
        .map(income_cfg["employment_type_shift"])
        .to_numpy(float)
    )
    city_shift = pd.Series(city_tier).map(income_cfg["city_tier_shift"]).to_numpy(float)
    log_income = (
        income_cfg["base_lognormal_mean_log"]
        + education_shift
        + employment_shift
        + income_cfg["age_shift_per_year_over_25"] * np.clip(age - 25, 0, None)
        + city_shift
        + rng.normal(0, income_cfg["base_lognormal_sigma"], size=n_customers)
    )
    annual_income = np.clip(np.exp(log_income), income_cfg["min_annual_income"], None)
    monthly_income = np.clip(
        annual_income
        / 12
        * (1 + rng.normal(0, income_cfg["monthly_income_noise_std"], size=n_customers)),
        0,
        None,
    )

    customer_id = np.array([f"CUST-{i:09d}" for i in range(1, n_customers + 1)])

    customers_df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "dependents": dependents,
            "education": education,
            "employment_type": employment_type,
            "employment_years": employment_years,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": city_tier,
        }
    )
    return customers_df, risk_propensity


def _generate_customer_base(
    rng: np.random.Generator,
    cfg: dict[str, Any],
    customers_df: pd.DataFrame,
    risk_propensity: np.ndarray,
) -> pd.DataFrame:
    """Generate per-customer base bureau/financial traits, before per-loan snapshots."""
    n = len(customers_df)
    age = customers_df["age"].to_numpy()
    annual_income = customers_df["annual_income"].to_numpy()
    monthly_income = customers_df["monthly_income"].to_numpy()
    dependents = customers_df["dependents"].to_numpy()

    exp_cfg = cfg["expenses"]
    dti_fraction = rng.uniform(
        exp_cfg["dti_fraction_min"], exp_cfg["dti_fraction_max"], size=n
    )
    dti_fraction = (
        dti_fraction + exp_cfg["dependents_effect_per_dependent"] * dependents
    )
    dti_fraction = np.clip(dti_fraction, 0.05, 0.95)
    monthly_expenses = monthly_income * dti_fraction

    debt_cfg = cfg["existing_debt"]
    existing_loan_count = poisson_capped(
        rng, debt_cfg["loan_count_lambda"], n, max_value=debt_cfg["loan_count_max"]
    )
    emi_fraction = truncated_normal(
        rng,
        debt_cfg["emi_income_fraction_mean"],
        debt_cfg["emi_income_fraction_std"],
        0.0,
        0.6,
        size=n,
    )
    monthly_emi = (
        monthly_income
        * emi_fraction
        * (existing_loan_count > 0)
        * np.sqrt(np.clip(existing_loan_count, 1, None))
    )
    existing_loan_amount = (
        monthly_emi
        * debt_cfg["loan_amount_months_of_emi_mean"]
        * (existing_loan_count > 0)
    )

    dti = (monthly_expenses + monthly_emi) / np.clip(monthly_income, 1.0, None)

    savings_cfg = cfg["savings"]
    savings_months = truncated_normal(
        rng,
        savings_cfg["months_of_income_mean"],
        savings_cfg["months_of_income_std"],
        0.0,
        24.0,
        size=n,
    )
    savings_penalty = np.clip(
        1 - savings_cfg["dti_penalty"] * np.clip(dti - 0.4, 0, None), 0.1, 1.0
    )
    savings_balance = np.clip(
        monthly_income * savings_months * savings_penalty,
        savings_cfg["min_balance"],
        None,
    )

    assets_cfg = cfg["assets"]
    other_assets = annual_income * rng.uniform(
        0, assets_cfg["other_assets_income_multiple_max"], size=n
    )
    total_assets = savings_balance + other_assets

    bureau_cfg = cfg["credit_bureau"]
    credit_history_months = truncated_normal(
        rng,
        mean=np.clip(age - 21, 0, None) * bureau_cfg["months_per_adult_year_mean"],
        std=bureau_cfg["months_noise_std"],
        low=0,
        high=np.clip(age - 18, 0, None) * 12,
        size=n,
    )
    num_credit_accounts = 1 + poisson_capped(
        rng,
        credit_history_months
        / 24.0
        * bureau_cfg["accounts_per_24_history_months_lambda"],
        n,
        max_value=bureau_cfg["accounts_max"],
    )
    credit_limit_multiple = truncated_normal(
        rng,
        bureau_cfg["total_credit_limit_income_multiple_mean"],
        bureau_cfg["total_credit_limit_income_multiple_std"],
        0.05,
        10.0,
        size=n,
    )
    total_credit_limit = np.clip(
        annual_income * credit_limit_multiple * (1 + credit_history_months / 240.0),
        1000.0,
        None,
    )

    zero_prob = np.clip(
        bureau_cfg["previous_defaults_zero_prob_base"]
        - bureau_cfg["previous_defaults_zero_prob_risk_sensitivity"] * risk_propensity,
        0.05,
        0.99,
    )
    lam_defaults = np.clip(
        bureau_cfg["previous_defaults_lambda_base"]
        + bureau_cfg["previous_defaults_lambda_risk_sensitivity"]
        * np.clip(risk_propensity, 0, None),
        0.01,
        None,
    )
    previous_defaults = zero_inflated_poisson(rng, zero_prob, lam_defaults, n)

    late_lambda = np.clip(
        bureau_cfg["late_payments_lambda_base"]
        + bureau_cfg["late_payments_lambda_risk_sensitivity"]
        * np.clip(risk_propensity, 0, None),
        0.01,
        None,
    )
    late_payments_12m = poisson_capped(rng, late_lambda, n, max_value=24)

    missed_lambda = np.clip(
        bureau_cfg["missed_payments_lambda_base"]
        + bureau_cfg["missed_payments_lambda_risk_sensitivity"]
        * np.clip(risk_propensity, 0, None),
        0.01,
        None,
    )
    missed_payments_12m = poisson_capped(rng, missed_lambda, n, max_value=24)

    active_range = bureau_cfg["active_loans_noise_range"]
    active_loans = np.clip(
        existing_loan_count + rng.integers(-active_range, active_range + 1, size=n),
        0,
        None,
    )
    closed_loans = poisson_capped(
        rng,
        credit_history_months / bureau_cfg["closed_loans_history_divisor"],
        n,
        max_value=30,
    )

    return pd.DataFrame(
        {
            "customer_id": customers_df["customer_id"].to_numpy(),
            "monthly_expenses": monthly_expenses,
            "existing_loan_count": existing_loan_count,
            "existing_loan_amount": existing_loan_amount,
            "monthly_emi": monthly_emi,
            "savings_balance": savings_balance,
            "total_assets": total_assets,
            "credit_history_months": credit_history_months,
            "num_credit_accounts": num_credit_accounts,
            "total_credit_limit": total_credit_limit,
            "previous_defaults": previous_defaults,
            "late_payments_12m": late_payments_12m,
            "missed_payments_12m": missed_payments_12m,
            "active_loans": active_loans,
            "closed_loans": closed_loans,
        }
    )


def _generate_loan_skeleton(
    rng: np.random.Generator, cfg: dict[str, Any], customers_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate one row per loan application: customer_id, loan_id, application_date."""
    n = len(customers_df)
    vol_cfg = cfg["volume"]
    n_loans_per_customer = rng.integers(
        vol_cfg["min_loans_per_customer"], vol_cfg["max_loans_per_customer"] + 1, size=n
    )
    repeated_customer_id = np.repeat(
        customers_df["customer_id"].to_numpy(), n_loans_per_customer
    )
    total_loans = len(repeated_customer_id)

    dates_cfg = cfg["dates"]
    reference_date = pd.Timestamp(dates_cfg["reference_date"])
    window_end = reference_date - pd.DateOffset(months=dates_cfg["outcome_lag_months"])
    window_start = window_end - pd.DateOffset(months=dates_cfg["window_months"])
    window_days = (window_end - window_start).days

    offset_days = rng.integers(0, window_days + 1, size=total_loans)
    application_date = window_start + pd.to_timedelta(offset_days, unit="D")

    skeleton = pd.DataFrame(
        {"customer_id": repeated_customer_id, "application_date": application_date}
    )
    skeleton = _deduplicate_dates_per_customer(
        skeleton, "customer_id", "application_date", direction=1
    )
    skeleton = skeleton.sort_values(["customer_id", "application_date"]).reset_index(
        drop=True
    )
    skeleton["loan_id"] = [f"LOAN-{i:09d}" for i in range(1, len(skeleton) + 1)]
    return skeleton


def _apply_snapshot_noise(
    rng: np.random.Generator, wide: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Add per-loan-snapshot noise to each customer's base bureau/financial traits."""
    noise_std = cfg["credit_bureau"]["snapshot_noise_std"]
    n = len(wide)

    def snap(base_col: str) -> np.ndarray:
        base = wide[base_col].to_numpy(dtype=float)
        noisy = base * (1 + rng.normal(0, noise_std, size=n))
        return np.clip(noisy, 0, None)

    wide["monthly_income_snap"] = snap("monthly_income")
    wide["monthly_expenses_snap"] = snap("monthly_expenses")
    wide["existing_loan_amount_snap"] = snap("existing_loan_amount")
    wide["monthly_emi_snap"] = snap("monthly_emi")
    wide["savings_balance_snap"] = snap("savings_balance")
    wide["total_credit_limit_snap"] = snap("total_credit_limit")

    wide["existing_loan_count_snap"] = np.round(snap("existing_loan_count")).astype(int)
    wide["num_credit_accounts_snap"] = np.round(snap("num_credit_accounts")).astype(int)
    wide["previous_defaults_snap"] = np.round(snap("previous_defaults")).astype(int)
    wide["late_payments_12m_snap"] = np.round(snap("late_payments_12m")).astype(int)
    wide["missed_payments_12m_snap"] = np.round(snap("missed_payments_12m")).astype(int)
    wide["active_loans_snap"] = np.round(snap("active_loans")).astype(int)
    wide["closed_loans_snap"] = np.round(snap("closed_loans")).astype(int)

    latest_as_of = wide.groupby("customer_id")["as_of_date"].transform("max")
    months_before_latest = (latest_as_of - wide["as_of_date"]).dt.days / DAYS_PER_MONTH
    adjusted_history = np.clip(
        wide["credit_history_months"].to_numpy() - months_before_latest.to_numpy(),
        0,
        None,
    )
    noisy_history = adjusted_history * (1 + rng.normal(0, noise_std, size=n))
    wide["credit_history_months_snap"] = np.clip(
        np.round(noisy_history), 0, None
    ).astype(int)

    total_assets_other = np.clip(
        wide["total_assets"].to_numpy() - wide["savings_balance"].to_numpy(), 0, None
    )
    wide["total_assets_snap"] = np.clip(
        wide["savings_balance_snap"]
        + total_assets_other * (1 + rng.normal(0, noise_std, size=n)),
        0,
        None,
    )
    return wide


def _generate_loan_terms(
    rng: np.random.Generator, wide: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Generate per-loan terms: type, amount, tenure, purpose, rate, decision date."""
    n = len(wide)
    loan_cfg = cfg["loan_application"]

    loan_type = sample_categorical(
        rng, loan_cfg["loan_type"]["categories"], loan_cfg["loan_type"]["probs"], n
    )
    amount_multiple = (
        pd.Series(loan_type)
        .map(loan_cfg["amount_income_multiple"])
        .to_numpy(dtype=float)
    )
    loan_amount = np.clip(
        wide["annual_income"].to_numpy()
        * amount_multiple
        * rng.lognormal(0, loan_cfg["amount_noise_sigma"], size=n),
        1000.0,
        None,
    )

    tenure_range = pd.Series(loan_type).map(loan_cfg["tenure_months_range"])
    tenure_min = np.array([r[0] for r in tenure_range], dtype=int)
    tenure_max = np.array([r[1] for r in tenure_range], dtype=int)
    loan_tenure_months = rng.integers(tenure_min, tenure_max + 1)

    loan_purpose = sample_categorical(
        rng,
        loan_cfg["loan_purpose"]["categories"],
        loan_cfg["loan_purpose"]["probs"],
        n,
    )

    base_rate = (
        pd.Series(loan_type).map(loan_cfg["base_interest_rate"]).to_numpy(dtype=float)
    )
    interest_rate = np.clip(
        base_rate
        + loan_cfg["interest_rate_risk_sensitivity"]
        * np.clip(wide["risk_propensity"], -2, 3)
        + rng.normal(0, loan_cfg["interest_rate_noise_std"], size=n),
        1.0,
        40.0,
    )

    decision_lag = rng.integers(
        loan_cfg["decision_lag_days_min"], loan_cfg["decision_lag_days_max"] + 1, size=n
    )
    decision_date = wide["application_date"] + pd.to_timedelta(decision_lag, unit="D")

    wide["loan_type"] = loan_type
    wide["loan_amount"] = loan_amount
    wide["loan_tenure_months"] = loan_tenure_months
    wide["loan_purpose"] = loan_purpose
    wide["interest_rate"] = interest_rate
    wide["decision_date"] = decision_date
    wide["status"] = "APPROVED"
    return wide


def _compute_label(
    rng: np.random.Generator, wide: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, float]:
    """Compute the latent-risk logit, calibrate the intercept, sample default_12m."""
    n = len(wide)
    label_cfg = cfg["label"]
    coef = label_cfg["coefficients"]

    dti = (wide["monthly_expenses_snap"] + wide["monthly_emi_snap"]) / np.clip(
        wide["monthly_income_snap"], 1.0, None
    )
    savings_to_income = wide["savings_balance_snap"] / np.clip(
        wide["monthly_income_snap"], 1.0, None
    )

    bureau_cfg = cfg["credit_bureau"]
    base_util = sample_beta(
        rng, bureau_cfg["utilization_alpha"], bureau_cfg["utilization_beta"], n
    )
    high_dti_component = bureau_cfg["utilization_high_dti_shift"] * np.clip(
        dti - 0.5, 0, None
    )
    low_savings_component = bureau_cfg["utilization_low_savings_shift"] * np.clip(
        1.0 - savings_to_income / 2.0, 0, None
    )
    credit_utilization = np.clip(
        base_util + high_dti_component + low_savings_component, 0.0, 2.0
    )

    total_outstanding = credit_utilization * wide["total_credit_limit_snap"]
    total_liabilities = wide["existing_loan_amount_snap"] + total_outstanding

    wide["dti"] = dti
    wide["savings_to_income"] = savings_to_income
    wide["credit_utilization"] = credit_utilization
    wide["total_outstanding"] = total_outstanding
    wide["total_liabilities"] = total_liabilities

    loan_to_income = wide["loan_amount"] / np.clip(wide["annual_income"], 1.0, None)
    emi_to_income = wide["monthly_emi_snap"] / np.clip(
        wide["monthly_income_snap"], 1.0, None
    )
    log_monthly_income = np.log(np.clip(wide["monthly_income_snap"], 1.0, None))

    logit_no_intercept = (
        coef["dti"] * dti
        + coef["credit_utilization"] * credit_utilization
        + coef["previous_defaults"] * wide["previous_defaults_snap"]
        + coef["late_payments_12m"] * wide["late_payments_12m_snap"]
        + coef["emi_to_income"] * emi_to_income
        + coef["loan_to_income"] * loan_to_income
        - coef["log_monthly_income"] * log_monthly_income
        - coef["employment_years"] * wide["employment_years"]
        - coef["credit_history_months"] * wide["credit_history_months_snap"]
        - coef["savings_to_income"] * savings_to_income
        + label_cfg["interaction"]["dti_x_credit_utilization"]
        * dti
        * credit_utilization
        + rng.normal(0, label_cfg["noise_std"], size=n)
    ).to_numpy()

    target_rate = (
        label_cfg["target_default_rate_min"] + label_cfg["target_default_rate_max"]
    ) / 2
    b0 = _calibrate_intercept(logit_no_intercept, target_rate)
    p_default = sigmoid(logit_no_intercept + b0)
    default_12m = (rng.random(n) < p_default).astype(int)

    wide["p_default"] = p_default
    wide["default_12m"] = default_12m
    wide["outcome_observed_date"] = wide["application_date"] + pd.DateOffset(months=12)
    return wide, b0


def _inject_data_quality_issues(
    rng: np.random.Generator,
    cfg: dict[str, Any],
    customers_df: pd.DataFrame,
    financial_profiles_df: pd.DataFrame,
    credit_history_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Inject controlled bad rows and return the modified tables plus a manifest."""
    inj_cfg = cfg["data_quality_injection"]
    customers_df = customers_df.copy()
    financial_profiles_df = financial_profiles_df.copy()
    credit_history_df = credit_history_df.copy()
    manifest: dict[str, Any] = {}

    customer_pools = _disjoint_samples(
        rng,
        len(customers_df),
        {
            "duplicate_customer": inj_cfg["duplicate_customer_rate"],
            "out_of_range_age": inj_cfg["out_of_range_age_rate"],
            "negative_financial_value": inj_cfg["negative_financial_value_rate"],
            "inconsistent_income": inj_cfg["inconsistent_income_rate"],
        },
    )

    idx = customer_pools["out_of_range_age"]
    bad_ages = rng.choice([5, 12, 17, 105, 130, 150], size=len(idx))
    customers_df.iloc[idx, customers_df.columns.get_loc("age")] = bad_ages
    manifest["out_of_range_age"] = {
        "count": int(len(idx)),
        "table": "customers",
        "customer_ids": customers_df.iloc[idx]["customer_id"].tolist(),
    }

    idx = customer_pools["negative_financial_value"]
    customers_df.iloc[idx, customers_df.columns.get_loc("annual_income")] = (
        -customers_df.iloc[idx]["annual_income"].to_numpy()
    )
    manifest["negative_financial_value"] = {
        "count": int(len(idx)),
        "table": "customers",
        "customer_ids": customers_df.iloc[idx]["customer_id"].tolist(),
    }

    idx = customer_pools["inconsistent_income"]
    factors = rng.choice([0.1, 0.15, 3.0, 4.0, 5.0], size=len(idx))
    customers_df.iloc[idx, customers_df.columns.get_loc("monthly_income")] = (
        customers_df.iloc[idx]["monthly_income"].to_numpy() * factors
    )
    manifest["inconsistent_income"] = {
        "count": int(len(idx)),
        "table": "customers",
        "customer_ids": customers_df.iloc[idx]["customer_id"].tolist(),
        "note": "Semantic only; passes DB constraints. For Phase 3 validation rules.",
    }

    idx = customer_pools["duplicate_customer"]
    duplicates = customers_df.iloc[idx].copy()
    manifest["duplicate_customer"] = {
        "count": int(len(idx)),
        "table": "customers",
        "customer_ids": duplicates["customer_id"].tolist(),
    }
    customers_df = pd.concat([customers_df, duplicates], ignore_index=True)

    fp_pool = _disjoint_samples(
        rng,
        len(financial_profiles_df),
        {"missing_value": inj_cfg["missing_value_rate"]},
    )
    idx = fp_pool["missing_value"]
    financial_profiles_df.iloc[
        idx, financial_profiles_df.columns.get_loc("total_assets")
    ] = np.nan
    manifest["missing_value"] = {
        "count": int(len(idx)),
        "table": "financial_profiles",
        "column": "total_assets",
        "row_keys": [
            f"{c}|{d}"
            for c, d in zip(
                financial_profiles_df.iloc[idx]["customer_id"],
                financial_profiles_df.iloc[idx]["as_of_date"],
                strict=True,
            )
        ],
    }

    ch_pool = _disjoint_samples(
        rng,
        len(credit_history_df),
        {"impossible_utilization": inj_cfg["impossible_utilization_rate"]},
    )
    idx = ch_pool["impossible_utilization"]
    bad_values = rng.choice([-0.5, -0.1, 2.5, 3.0, 5.0], size=len(idx))
    credit_history_df.iloc[
        idx, credit_history_df.columns.get_loc("credit_utilization")
    ] = bad_values
    manifest["impossible_utilization"] = {
        "count": int(len(idx)),
        "table": "credit_history",
        "row_keys": [
            f"{c}|{d}"
            for c, d in zip(
                credit_history_df.iloc[idx]["customer_id"],
                credit_history_df.iloc[idx]["as_of_date"],
                strict=True,
            )
        ],
    }

    return customers_df, financial_profiles_df, credit_history_df, manifest


def generate_dataset(
    config: dict[str, Any], seed: int | None = None, n_customers: int | None = None
) -> GeneratedDataset:
    """Generate a full synthetic CreditGuard dataset from a config dict.

    `seed` and `n_customers` override the corresponding config values when given.
    """
    effective_config = copy.deepcopy(config)
    if seed is not None:
        effective_config["seed"] = seed
    if n_customers is not None:
        effective_config["volume"]["n_customers"] = n_customers
    cfg = effective_config

    rng = np.random.default_rng(cfg["seed"])
    n = cfg["volume"]["n_customers"]

    customers_df, risk_propensity = _generate_customers(rng, cfg, n)
    base_bureau_df = _generate_customer_base(rng, cfg, customers_df, risk_propensity)

    wide = _generate_loan_skeleton(rng, cfg, customers_df)
    wide = wide.merge(customers_df, on="customer_id", how="left")
    wide = wide.merge(base_bureau_df, on="customer_id", how="left")
    risk_series = pd.Series(
        risk_propensity, index=customers_df["customer_id"], name="risk_propensity"
    )
    wide = wide.merge(risk_series, left_on="customer_id", right_index=True, how="left")

    dates_cfg = cfg["dates"]
    lag_days = rng.integers(
        dates_cfg["snapshot_lag_days_min"],
        dates_cfg["snapshot_lag_days_max"] + 1,
        size=len(wide),
    )
    wide["as_of_date"] = wide["application_date"] - pd.to_timedelta(lag_days, unit="D")
    wide = _deduplicate_dates_per_customer(
        wide, "customer_id", "as_of_date", direction=-1
    )

    wide = _apply_snapshot_noise(rng, wide, cfg)
    wide = _generate_loan_terms(rng, wide, cfg)
    wide, b0 = _compute_label(rng, wide, cfg)

    loan_applications_df = pd.DataFrame(
        {
            "loan_id": wide["loan_id"],
            "customer_id": wide["customer_id"],
            "loan_type": wide["loan_type"],
            "loan_amount": wide["loan_amount"],
            "loan_tenure_months": wide["loan_tenure_months"],
            "interest_rate": wide["interest_rate"],
            "loan_purpose": wide["loan_purpose"],
            "application_date": wide["application_date"].dt.date,
            "decision_date": wide["decision_date"].dt.date,
            "status": wide["status"],
        }
    )
    financial_profiles_df = pd.DataFrame(
        {
            "customer_id": wide["customer_id"],
            "as_of_date": wide["as_of_date"].dt.date,
            "monthly_income": wide["monthly_income_snap"],
            "monthly_expenses": wide["monthly_expenses_snap"],
            "existing_loan_count": wide["existing_loan_count_snap"],
            "existing_loan_amount": wide["existing_loan_amount_snap"],
            "monthly_emi": wide["monthly_emi_snap"],
            "savings_balance": wide["savings_balance_snap"],
            "total_assets": wide["total_assets_snap"],
            "total_liabilities": wide["total_liabilities"],
        }
    )
    credit_history_df = pd.DataFrame(
        {
            "customer_id": wide["customer_id"],
            "as_of_date": wide["as_of_date"].dt.date,
            "credit_history_months": wide["credit_history_months_snap"],
            "num_credit_accounts": wide["num_credit_accounts_snap"],
            "total_credit_limit": wide["total_credit_limit_snap"],
            "total_outstanding": wide["total_outstanding"],
            "credit_utilization": wide["credit_utilization"],
            "previous_defaults": wide["previous_defaults_snap"],
            "late_payments_12m": wide["late_payments_12m_snap"],
            "missed_payments_12m": wide["missed_payments_12m_snap"],
            "active_loans": wide["active_loans_snap"],
            "closed_loans": wide["closed_loans_snap"],
        }
    )
    loan_outcomes_df = pd.DataFrame(
        {
            "loan_id": wide["loan_id"],
            "default_12m": wide["default_12m"],
            "outcome_observed_date": wide["outcome_observed_date"].dt.date,
        }
    )

    customers_df, financial_profiles_df, credit_history_df, injection_manifest = (
        _inject_data_quality_issues(
            rng, cfg, customers_df, financial_profiles_df, credit_history_df
        )
    )

    achieved_default_rate = float(wide["default_12m"].mean())
    metadata = {
        "seed": cfg["seed"],
        "n_customers": n,
        "config": cfg,
        "label_coefficients": cfg["label"]["coefficients"],
        "label_interaction": cfg["label"]["interaction"],
        "calibrated_intercept_b0": float(b0),
        "achieved_default_rate": achieved_default_rate,
        "target_default_rate_band": [
            cfg["label"]["target_default_rate_min"],
            cfg["label"]["target_default_rate_max"],
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }

    return GeneratedDataset(
        customers=customers_df,
        loan_applications=loan_applications_df,
        financial_profiles=financial_profiles_df,
        credit_history=credit_history_df,
        loan_outcomes=loan_outcomes_df,
        metadata=metadata,
        injection_manifest=injection_manifest,
    )


def main() -> None:
    """CLI entrypoint: generate a dataset and write a versioned parquet snapshot."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CreditGuard dataset."
    )
    parser.add_argument("--config", default="config/data_generation.yaml")
    parser.add_argument("--n-customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = generate_dataset(config, seed=args.seed, n_customers=args.n_customers)

    settings = get_settings()
    output_root = (
        Path(args.output_dir) if args.output_dir else settings.data_dir / "raw"
    )
    dataset_version = make_dataset_version(dataset.metadata["config"])
    out_dir = write_dataset(dataset, dataset_version, output_root)

    row_counts = {
        "customers": len(dataset.customers),
        "loan_applications": len(dataset.loan_applications),
        "financial_profiles": len(dataset.financial_profiles),
        "credit_history": len(dataset.credit_history),
        "loan_outcomes": len(dataset.loan_outcomes),
    }
    print(f"Wrote dataset {dataset_version} to {out_dir}")
    for name, count in row_counts.items():
        print(f"  {name}: {count}")
    print(f"Realised default rate: {dataset.metadata['achieved_default_rate']:.4f}")


if __name__ == "__main__":
    main()

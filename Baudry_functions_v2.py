#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def simulate_central_scenario(seed=1234):
    np.random.seed(seed)

    # --- Policy Data ---
    date_range = pd.date_range(start="2016-01-01", end="2017-12-31", freq="D")
    dt_policydates = pd.DataFrame({'date_UW': date_range})

    policycount = np.random.poisson(700, size=len(dt_policydates))
    dt_policydates['policycount'] = policycount
    dt_policydates['date_lapse'] = dt_policydates['date_UW'].apply(lambda d: d + relativedelta(years=1))
    dt_policydates['expodays'] = (dt_policydates['date_lapse'] - dt_policydates['date_UW']).dt.days
    dt_policydates['pol_prefix'] = dt_policydates['date_UW'].dt.year * 10000 + \
                                   dt_policydates['date_UW'].dt.month * 100 + \
                                   dt_policydates['date_UW'].dt.day

    # Cover breakdowns
    dt_policydates['Cover_B'] = (dt_policydates['policycount'] * 0.25).round().astype(int)
    dt_policydates['Cover_BO'] = (dt_policydates['policycount'] * 0.45).round().astype(int)
    dt_policydates['Cover_BOT'] = dt_policydates['policycount'] - dt_policydates['Cover_B'] - dt_policydates['Cover_BO']

    # Repeat by policy count
    dt_policy = dt_policydates.loc[dt_policydates.index.repeat(dt_policydates['policycount'])].copy()
    dt_policy['pol_seq'] = dt_policy.groupby('pol_prefix').cumcount() + 1
    dt_policy['pol_number'] = (dt_policy['pol_prefix'] * 10000 + dt_policy['pol_seq']).astype(str)

    # Assign cover type
    dt_policy['Cover'] = 'BO'
    dt_policy['tmp_index'] = dt_policy.groupby('pol_prefix').cumcount()

    dt_policy.loc[dt_policy['tmp_index'] < (dt_policy['policycount'] - dt_policy['Cover_BO']), 'Cover'] = 'BOT'
    dt_policy.loc[dt_policy['tmp_index'] < dt_policy['Cover_B'], 'Cover'] = 'B'

    # Clean up
    dt_policy.drop(columns=['pol_prefix', 'policycount', 'pol_seq', 'Cover_B', 'Cover_BO', 'Cover_BOT', 'tmp_index'], inplace=True)

    # Assign Brand and Base_Price
    brand_cycle = np.tile(np.repeat([1, 2, 3, 4], [9, 6, 3, 2]), int(np.ceil(len(dt_policy)/20)))[:len(dt_policy)]
    base_price_map = {1: 600, 2: 550, 3: 300, 4: 150}
    dt_policy['Brand'] = brand_cycle
    dt_policy['Base_Price'] = [base_price_map[b] for b in dt_policy['Brand']]

    model_cycle = np.repeat([3, 2, 1, 0], [10, 7, 2, 1])
    model_mult_map = {3: 1.15**3, 2: 1.15**2, 1: 1.15**1, 0: 1.0}
    
    model_list = []
    model_mult_list = []
    for brand in dt_policy['Brand'].unique():
        brand_mask = dt_policy['Brand'] == brand
        count = brand_mask.sum()
        repeated_models = np.tile(model_cycle, int(np.ceil(count / len(model_cycle))))[:count]
        model_list.extend(repeated_models)
        model_mult_list.extend([model_mult_map[m] for m in repeated_models])
    
    dt_policy['Model'] = model_list
    dt_policy['Model_mult'] = model_mult_list
    dt_policy['Price'] = np.ceil(dt_policy['Base_Price'] * dt_policy['Model_mult']).astype(int)

    dt_policy = dt_policy[['pol_number', 'date_UW', 'date_lapse', 'Cover', 'Brand', 'Model', 'Price']]

    # --- Claims Data ---
    dt_policy = dt_policy.reset_index(drop=True)

    total_policies = len(dt_policy)
    claim_indices = np.random.choice(dt_policy.index, size=int(0.15 * total_policies), replace=False)

    print(f"Length of claim_indices: {len(claim_indices)}")
    print(f"Length of pol_number array: {len(dt_policy.loc[claim_indices, 'pol_number'].values)}")

    dt_claim = pd.DataFrame({
        'pol_number': dt_policy.loc[claim_indices, 'pol_number'].values,
        'claim_type': ['B'] * len(claim_indices),
        'claim_count': [1] * len(claim_indices),
        'claim_sev': np.random.beta(2, 5, size=len(claim_indices))
    })



    cov = dt_policy.index[dt_policy['Cover'] != 'B']
    claim_indices = np.random.choice(cov, size=int(0.05 * len(cov)), replace=False)
    oxidation_claims = pd.DataFrame({
        'pol_number': dt_policy.loc[claim_indices, 'pol_number'].values,
        'claim_type': 'O',
        'claim_count': 1,
        'claim_sev': np.random.beta(5, 3, size=len(claim_indices))
    })
    dt_claim = pd.concat([dt_claim, oxidation_claims], ignore_index=True)

    for model in range(4):
        model_cov = dt_policy[(dt_policy['Cover'] == 'BOT') & (dt_policy['Model'] == model)].index
        claim_count = int(0.05 * (1 + model) * len(model_cov))
        if claim_count > 0:
            theft_claims = pd.DataFrame({
                'pol_number': dt_policy.loc[np.random.choice(model_cov, size=claim_count, replace=False), 'pol_number'].values,
                'claim_type': 'T',
                'claim_count': 1,
                'claim_sev': np.random.beta(5, 0.5, size=claim_count)
            })
            dt_claim = pd.concat([dt_claim, theft_claims], ignore_index=True)

    dt_claim = dt_claim.merge(dt_policy[['pol_number', 'date_UW', 'Price', 'Brand']], on='pol_number', how='left')
    dt_claim['date_lapse'] = dt_claim['date_UW'] + pd.to_timedelta(365, unit='D')
    dt_claim['expodays'] = (dt_claim['date_lapse'] - dt_claim['date_UW']).dt.days
    dt_claim['occ_delay_days'] = (dt_claim['expodays'] * np.random.uniform(size=len(dt_claim))).astype(int)
    dt_claim['delay_report'] = np.floor(365 * np.random.beta(0.4, 10, size=len(dt_claim))).astype(int)
    dt_claim['delay_pay'] = np.floor(10 + 40 * np.random.beta(7, 7, size=len(dt_claim))).astype(int)

    dt_claim['date_occur'] = dt_claim['date_UW'] + pd.to_timedelta(dt_claim['occ_delay_days'], unit='D')
    dt_claim['date_report'] = dt_claim['date_occur'] + pd.to_timedelta(dt_claim['delay_report'], unit='D')
    dt_claim['date_pay'] = dt_claim['date_report'] + pd.to_timedelta(dt_claim['delay_pay'], unit='D')
    dt_claim['claim_cost'] = np.round(dt_claim['Price'] * dt_claim['claim_sev']).astype(int)

    dt_claim['clm_prefix'] = dt_claim['date_report'].dt.year * 10000 + \
                             dt_claim['date_report'].dt.month * 100 + \
                             dt_claim['date_report'].dt.day
    dt_claim['clm_seq'] = dt_claim.groupby('clm_prefix').cumcount() + 1
    dt_claim['clm_number'] = (dt_claim['clm_prefix'] * 10000 + dt_claim['clm_seq']).astype(str)

    dt_claim['polclm_seq'] = dt_claim.groupby('pol_number').cumcount() + 1
    dt_claim = dt_claim[dt_claim['polclm_seq'] == 1]

    dt_claim = dt_claim[['clm_number', 'pol_number', 'claim_type', 'claim_count', 'claim_sev',
                         'date_occur', 'date_report', 'date_pay', 'claim_cost']]

    return {
        'dt_policy': dt_policy.reset_index(drop=True),
        'dt_claim': dt_claim.reset_index(drop=True)
    }


# In[ ]:





# In[16]:


print(f"Length of claim_indices: {len(claim_indices)}")
print(f"Length of pol_number array: {len(dt_policy.loc[claim_indices, 'pol_number'].values)}")
print(f"Length of claim_type list: {len(['B'] * len(claim_indices))}")
print(f"Length of claim_count list: {len([1] * len(claim_indices))}")
print(f"Length of claim_sev array: {len(np.random.beta(2, 5, size=len(claim_indices)))}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
from datetime import datetime

def join_policy_claim(dt_PhoneData,
                      date_pol_start="date_UW",
                      date_pol_end="date_lapse",
                      date_occur="date_occur"):
    # Extract policy and claim data
    dt_policy = dt_PhoneData['dt_policy'].copy()
    dt_claim = dt_PhoneData['dt_claim'].copy()

    # Rename policy columns to generic names for matching
    dt_policy = dt_policy.rename(columns={date_pol_start: "date_pol_start", date_pol_end: "date_pol_end"})

    # Floor dates to the second
    dt_policy["date_pol_start"] = pd.to_datetime(dt_policy["date_pol_start"]).dt.floor("S")
    dt_policy["date_pol_end"] = pd.to_datetime(dt_policy["date_pol_end"]).dt.floor("S") - pd.Timedelta(seconds=1)

    dt_claim["date_occur"] = pd.to_datetime(dt_claim[date_occur]).dt.floor("S")
    dt_claim["date_occur_end"] = dt_claim["date_occur"]
    dt_claim["date_report"] = pd.to_datetime(dt_claim["date_report"]).dt.floor("S")
    dt_claim["date_pay"] = pd.to_datetime(dt_claim["date_pay"]).dt.floor("S")

    # Create policy period interval and occurrence interval
    dt_policy["policy_interval"] = pd.IntervalIndex.from_arrays(dt_policy["date_pol_start"],
                                                                dt_policy["date_pol_end"],
                                                                closed='both')
    dt_claim["occur_interval"] = pd.IntervalIndex.from_arrays(dt_claim["date_occur"],
                                                              dt_claim["date_occur_end"],
                                                              closed='both')

    # Perform overlap join using row-wise comparison
    dt_claim = dt_claim.set_index('pol_number')
    dt_policy = dt_policy.set_index('pol_number')

    def match_claims(row):
        if row.name not in dt_claim.index:
            return pd.Series([np.nan]*len(dt_claim.columns), index=dt_claim.columns)
        possible_claims = dt_claim.loc[[row.name]]
        matched = possible_claims[possible_claims["occur_interval"].overlaps(row["policy_interval"])]
        return matched.iloc[0] if not matched.empty else pd.Series([np.nan]*len(dt_claim.columns), index=dt_claim.columns)

    claim_data = dt_policy.apply(match_claims, axis=1)

    # Combine policy and claim
    dt_polclaim = dt_policy.reset_index().join(claim_data.reset_index(drop=True))

    # Fill date NAs with distant future for safety
    date_fields = [col for col in dt_polclaim.columns if "date" in col]
    for field in date_fields:
        dt_polclaim[field] = dt_polclaim[field].fillna(pd.to_datetime("2199-12-31 23:59:59"))

    # Set claim count/severity/cost NAs to zero
    for field in ["claim_count", "claim_sev", "claim_cost"]:
        if field in dt_polclaim.columns:
            dt_polclaim[field] = dt_polclaim[field].fillna(0)

    # Calculate exposure days
    dt_polclaim["ExpoDays"] = np.ceil(
        (dt_polclaim["date_pol_end"] - dt_polclaim["date_pol_start"]) / np.timedelta64(1, 'D')
    ).astype(int)

    # Remove zero-exposure entries
    dt_polclaim = dt_polclaim[dt_polclaim["ExpoDays"] > 0].reset_index(drop=True)

    return dt_polclaim


# In[4]:


def time_slice_polclaim(dt_polclaim, lst_Date_slice=None):
    """
    Time slice Policy & Claims data.
    Adds one column per slice named 'P_t_YYYYMMDD', showing claim cost if paid by that date, else 0.
    """

    # Default time slices: every 30 days from Jan 1, 2016 to June 30, 2019
    if lst_Date_slice is None:
        lst_Date_slice = pd.date_range(start="2016-01-01", end="2019-06-30", freq="30D").floor("S")

    dt_polclaim = dt_polclaim.copy()

    for date_slice in lst_Date_slice:
        col_name = f'P_t_{date_slice.strftime("%Y%m%d")}'
        dt_polclaim[col_name] = np.where(
            dt_polclaim['date_pay'] <= date_slice,
            dt_polclaim['claim_cost'],
            0
        )

    # Sort by policy number
    dt_polclaim = dt_polclaim.sort_values(by='pol_number').reset_index(drop=True)

    return dt_polclaim


# In[5]:


def RBNS_Train_ijk(dt_policy_claim, date_i, j_dev_period, k, reserving_dates, model_vars):
    date_i = pd.to_datetime(date_i)
    
    date_k = reserving_dates[reserving_dates.index(date_i) - k + 1]
    date_j = reserving_dates[reserving_dates.index(date_k) - j_dev_period]
    date_lookup = reserving_dates[reserving_dates.index(date_i) - j_dev_period - k + 1]
    
    target_lookup = reserving_dates[reserving_dates.index(date_i) - k]
    target_lookup_next = reserving_dates[reserving_dates.index(date_i) - k + 1]
    
    # Filter: reported but not paid as at date_lookup
    df = dt_policy_claim[
        (dt_policy_claim['date_report'] <= date_lookup) &
        (dt_policy_claim['date_pay'] > date_lookup)
    ].copy()

    df['date_lookup'] = date_lookup
    df['delay_train'] = (date_lookup - df['date_pol_start']).dt.days
    df['j'] = j_dev_period
    df['k'] = k
    df['target'] = np.where(
        df['date_pay'] <= target_lookup, 0,
        np.where(df['date_pay'] <= target_lookup_next, df['claim_cost'], 0)
    )

    return df[model_vars]


# In[6]:


def RBNS_Test_ijk(dt_policy_claim, date_i, j_dev_period, k, reserving_dates, model_vars):
    date_i = pd.to_datetime(date_i)

    date_lookup = reserving_dates[reserving_dates.index(date_i)]
    target_lookup = reserving_dates[reserving_dates.index(date_i) + j_dev_period - 1]
    target_lookup_next = reserving_dates[reserving_dates.index(date_i) + j_dev_period]

    # Filter: reported but not paid as at date_lookup
    df = dt_policy_claim[
        (dt_policy_claim['date_report'] <= date_lookup) &
        (date_lookup < dt_policy_claim['date_pay'])
    ].copy()

    df['date_lookup'] = date_lookup
    df['delay_train'] = (date_lookup - df['date_pol_start']).dt.days
    df['j'] = j_dev_period
    df['k'] = k
    df['target'] = np.where(
        df['date_pay'] <= target_lookup, 0,
        np.where(df['date_pay'] <= target_lookup_next, df['claim_cost'], 0)
    )

    return df[model_vars]


# In[7]:


def RBNS_Train(dt_policy_claim, date_i, i, k_max, reserving_dates, model_vars):
    dt_train = []

    for k in range(1, k_max + 1):
        for j in range(1, i - k + 2):  # inclusive loop in R is 1:(i - k + 1)
            df_part = RBNS_Train_ijk(dt_policy_claim, date_i, j, k, reserving_dates, model_vars)
            dt_train.append(df_part)

    return pd.concat(dt_train, ignore_index=True) if dt_train else pd.DataFrame(columns=model_vars)


# In[8]:


def RBNS_Test(dt_policy_claim, date_i, delta, k_max, reserving_dates, model_vars):
    dt_test = []

    for k in range(1, k_max + 1):
        for j in range(1, delta - k + 2):  # inclusive loop in R is 1:(delta - k + 1)
            df_part = RBNS_Test_ijk(dt_policy_claim, date_i, j, k, reserving_dates, model_vars)
            dt_test.append(df_part)

    return pd.concat(dt_test, ignore_index=True) if dt_test else pd.DataFrame(columns=model_vars)


# In[9]:


def IBNR_Freq_Train_ijk(dt_policy_claim, date_i, j_dev_period, k, reserving_dates, model_vars, verbose=False):
    import numpy as np
    import pandas as pd

    date_i = pd.to_datetime(date_i)
    
    idx_i = reserving_dates.index(date_i)
    date_k = reserving_dates[idx_i - k + 1]
    date_j = reserving_dates[reserving_dates.index(date_k) - j_dev_period]
    date_lookup = reserving_dates[idx_i - j_dev_period - k + 1]
    target_lookup = reserving_dates[idx_i - k]
    target_lookup_next = reserving_dates[idx_i - k + 1]

    if verbose:
        print(f"Valn date {date_i.date()}, j = {j_dev_period}, k = {k}")

    # Define IBNR population: policies not yet reported as of date_lookup
    df = dt_policy_claim[
        (dt_policy_claim['date_pol_start'] < date_lookup) &
        (dt_policy_claim['date_report'] > date_lookup)
    ].copy()

    # Add engineered features
    df['date_lookup'] = date_lookup
    df['delay_train'] = (df['date_lookup'] - df['date_pol_start']).dt.days

    df['j'] = j_dev_period
    df['k'] = k

    # Compute exposure in years
    min_date = pd.to_datetime(date_i).floor('s')
    df['exposure'] = ((np.minimum(df['date_pol_end'], min_date) - df['date_pol_start']).dt.total_seconds() / (365 * 24 * 60 * 60)).round(3)

    # Target is 1 if claim occurred and was paid in the target window
    df['target'] = np.where(
        (df['date_pay'] >= target_lookup) &
        (df['date_pay'] < target_lookup_next) &
        (df['date_occur'] <= date_lookup),
        1, 0
    )

    # Aggregate exposure across model vars excluding exposure
    group_vars = [var for var in model_vars if var != 'exposure']
    df_agg = df.groupby(group_vars, as_index=False)['exposure'].sum()

    # Ensure all model_vars are present in the final dataframe
    return df_agg[model_vars]


# In[10]:


def IBNR_Loss_Train_ijk(dt_policy_claim, date_i, j_dev_period, k, reserving_dates, model_vars, verbose=False):
    import numpy as np
    import pandas as pd

    date_i = pd.to_datetime(date_i)

    idx_i = reserving_dates.index(date_i)
    date_k = reserving_dates[idx_i - k + 1]
    date_j = reserving_dates[reserving_dates.index(date_k) - j_dev_period]
    date_lookup = reserving_dates[idx_i - j_dev_period - k + 1]
    target_lookup = reserving_dates[idx_i - k]
    target_lookup_next = reserving_dates[idx_i - k + 1]

    if verbose:
        print(f"Valn date {date_i.date()}, j = {j_dev_period}, k = {k}")

    df = dt_policy_claim[
        (dt_policy_claim['date_report'] > date_lookup) &
        (dt_policy_claim['date_occur'] < date_lookup) &
        (dt_policy_claim['date_pay'] >= target_lookup) &
        (dt_policy_claim['date_pay'] < target_lookup_next)
    ].copy()

    df['date_lookup'] = date_lookup
    df['delay_train'] = (date_lookup - df['date_pol_start']).dt.days
    df['j'] = j_dev_period
    df['k'] = k
    df['exposure'] = 1  # all claims treated equal

    df['target'] = np.where(
        (df['date_pay'] >= target_lookup) & (df['date_pay'] < target_lookup_next),
        df['claim_cost'],
        0
    )

    return df[model_vars]


# In[11]:


def IBNR_Test_ijk(dt_policy_claim, date_i, j_dev_period, k, reserving_dates, model_vars, verbose=False):
    import numpy as np
    import pandas as pd

    date_i = pd.to_datetime(date_i)

    idx_i = reserving_dates.index(date_i)
    date_lookup = reserving_dates[idx_i]
    target_lookup = reserving_dates[idx_i + j_dev_period - 1]
    target_lookup_next = reserving_dates[idx_i + j_dev_period]

    if verbose:
        print(f"Valn date {date_i.date()}, j = {j_dev_period}, k = {k}")

    df = dt_policy_claim[
        (dt_policy_claim['date_pol_start'] <= date_lookup) &
        (dt_policy_claim['date_report'] > date_lookup)
    ].copy()

    df['date_lookup'] = date_lookup
    df['delay_train'] = (date_lookup - df['date_pol_start']).dt.days
    df['j'] = j_dev_period
    df['k'] = k

    min_date = pd.to_datetime(date_i).floor('s')
    df['exposure'] = ((np.minimum(df['date_pol_end'], min_date) - df['date_pol_start']).dt.total_seconds() / (365 * 24 * 60 * 60)).round(3)

    df['target'] = np.where(
        (df['date_pay'] >= target_lookup) &
        (df['date_pay'] < target_lookup_next) &
        (df['date_occur'] <= date_lookup),
        df['claim_cost'],
        0
    )

    group_vars = [var for var in model_vars if var != 'exposure']
    df_agg = df.groupby(group_vars, as_index=False)['exposure'].sum()

    return df_agg[model_vars]


# In[12]:


def IBNR_Train(dt_policy_claim, date_i, i, k, reserving_dates, model_vars, verbose=False):
    dt_train_Freq = pd.DataFrame()
    dt_train_Loss = pd.DataFrame()

    for kk in range(1, k + 1):
        for j in range(1, i - kk + 2):
            df_freq = IBNR_Freq_Train_ijk(dt_policy_claim, date_i, j, kk, reserving_dates, model_vars, verbose)
            df_loss = IBNR_Loss_Train_ijk(dt_policy_claim, date_i, j, kk, reserving_dates, model_vars, verbose)

            dt_train_Freq = pd.concat([dt_train_Freq, df_freq], ignore_index=True)
            dt_train_Loss = pd.concat([dt_train_Loss, df_loss], ignore_index=True)

    return {'Freq': dt_train_Freq, 'Loss': dt_train_Loss}


# In[13]:


def IBNR_Test(dt_policy_claim, date_i, delta, k, reserving_dates, model_vars, verbose=False):
    dt_test = pd.DataFrame()

    for kk in range(1, k + 1):
        for j in range(1, delta - kk + 2):
            df = IBNR_Test_ijk(dt_policy_claim, date_i, j, kk, reserving_dates, model_vars, verbose)
            dt_test = pd.concat([dt_test, df], ignore_index=True)

    return dt_test


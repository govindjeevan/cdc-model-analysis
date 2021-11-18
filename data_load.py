import pandas as pd
import glob
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime, timedelta
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def get_cdc_dataframe_old():
    path = 'CDC-Cases-Forecast'
    all_files = glob.glob(path + "/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    cdc_frames = pd.concat(li, axis=0, ignore_index=True)
    cdc_frames.point =cdc_frames.point/7
    return cdc_frames

def get_cdc_dataframe():
    path = 'processed_data/cdc-inc-cases.csv'
    cdc_frames = pd.read_csv(path)
    allowed_forecast_dates = cdc_frames[cdc_frames.Model=="COVIDhub-baseline"].forecast_date.unique()
    new_allowed_forecast_dates = []
    # add 1 day to every baseline forecast day to account for submissions on both sunday and monday
    for date in allowed_forecast_dates:
        new_allowed_forecast_dates.append(date)
        new_allowed_forecast_dates.append(add_days(date,-1))

    allowed_target_dates = cdc_frames[cdc_frames.Model=="COVIDhub-baseline"].target_end_date.unique()
    cdc_frames = cdc_frames[cdc_frames['forecast_date'].isin(new_allowed_forecast_dates)]
    cdc_frames = cdc_frames[cdc_frames['target_end_date'].isin(allowed_target_dates)]
    cdc_frames.point =cdc_frames.point/7
    return cdc_frames

    
def get_jhu_dataframe():
    jhu = pd.read_csv("processed_data/jhu-us.csv")
    jhu.Date=pd.to_datetime(jhu.Date)
    jhu = jhu.set_index("Date", drop=True)
    return jhu.sort_index(ascending=True)


def get_ensemble_eligibility_dataframe():
    path = 'ensemble-metadata'
    all_files = glob.glob(path + "/*inc_case-model-eligibility.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        date = "-".join(filename.split("/")[-1].split("-")[:3])
        df = df[(df.location=="US") & (df.overall_eligibility=="eligible")][["model", "overall_eligibility"]]
        df["forecast_date"]=date
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)

def add_days(date, days):
    date_1 = datetime.strptime(date, "%Y-%m-%d")
    end_date = date_1 + timedelta(days=days)
    return end_date.strftime('%Y-%m-%d')

def get_jhu_dataframe_weekly():
    jhu_weekly={}
    cdc_frames = get_cdc_dataframe()
    jhu_frames = get_jhu_dataframe()
    for target_date in cdc_frames.target_end_date.unique():
        if not target_date in jhu_frames.index:
            continue
        if target_date in jhu_weekly:
            continue
        weeklySum = jhu_frames.loc[target_date].NewCases
        for i in range(6):
            pastDate = add_days(target_date, -(i+1))
            weeklySum += jhu_frames.loc[pastDate].NewCases
        jhu_weekly[target_date] = weeklySum/7
    jhu_weekly_df = pd.DataFrame.from_dict(jhu_weekly,orient='index', columns=['WeeklyCases']).sort_index(ascending=True)
    jhu_weekly_df = jhu_weekly_df.reset_index()
    jhu_weekly_df = jhu_weekly_df.rename(columns={"index": "target_end_date"})
    jhu_weekly_df.target_end_date=pd.to_datetime(jhu_weekly_df.target_end_date)
    jhu_weekly_df = (jhu_weekly_df.drop_duplicates(subset='target_end_date', keep='last'))
    jhu_weekly_df = jhu_weekly_df.set_index("target_end_date", drop=True).sort_index(ascending=True)
    return jhu_weekly_df


def get_linear_baseline():
    jhu_weekly = get_jhu_dataframe_weekly ()
    baseline2 = {}
    for i in jhu_weekly.index:
        date = i.date().strftime('%Y-%m-%d')
        prevPrevDate = add_days(date, -(7*5))
        prevDate = add_days(date, -(7*4))

        if prevPrevDate in jhu_weekly.index and prevDate in jhu_weekly.index:
            baseline2[date] = jhu_weekly.loc[prevDate, "WeeklyCases"] + 4*(jhu_weekly.loc[prevDate, "WeeklyCases"] - jhu_weekly.loc[prevPrevDate, "WeeklyCases"])
    baseline =  pd.DataFrame.from_dict(baseline2, orient='index', columns=['point']).sort_index(ascending=True)
    baseline = baseline.reset_index()
    baseline = baseline.rename(columns={"index": "target_end_date"})
    baseline.target_end_date=pd.to_datetime(baseline.target_end_date)
    baseline = baseline.set_index("target_end_date", drop=True).sort_index(ascending=True)
    baseline.point = pd.to_numeric(baseline.point, downcast="float")
    return baseline


def get_model_by_date_range(model, cdc_frames, horizon, start=None, end = None):
    model_frame = cdc_frames[(cdc_frames["Model"]==model) & (cdc_frames["target"] == str(horizon) + " wk ahead inc case")][["target_end_date", "point"]]
    model_frame.target_end_date=pd.to_datetime(model_frame.target_end_date)
    model_frame = (model_frame.drop_duplicates(subset='target_end_date', keep='last'))
    model_frame = model_frame.set_index("target_end_date", drop=True)
    model_frame = model_frame.sort_index()
    if start is not None and end is not None:
        model_target_df = model_frame[(model_frame.index >= start) & (model_frame.index <= end)]
        return model_target_df
    return model_frame

def get_mae(model_frame,jhu_weekly_df, start, end):
    model_target_df = model_frame[(model_frame.index >= start) & (model_frame.index <= end)]
    target_dates = jhu_weekly_df.index.intersection(set(model_target_df.index.unique()))
    if len(target_dates) == 0:
        return 0,0
    jnu_target_df = jhu_weekly_df.loc[target_dates]
    
    jnu_target_df["error"] = 100*abs(model_target_df.point - jnu_target_df.WeeklyCases)/jnu_target_df.WeeklyCases
    return round(jnu_target_df.error.mean(),2), len(jnu_target_df)



MODEL_TYPE = {
    "epi": ["TTU Squider", "JHU-IDD", "IowaStateLW-STEM", "Bpagano-RtDriven", "UCLA-SuEIR", "COVID19Sim-Simulator", "CovidAnalytics-DELPHI", "Columbia_UNC-SurvCon"],
    "ml": ["USC-SI_kJalpha", "QJHong-Encounter"],
    "ensemble": ["COVIDhub-ensemble", "UCF-AEM", "LNQ-ens1", "UVA-Ensemble", "Caltech CS156", "MIT-Cassandra"],
    "hybrid": [],
    "others": ["IBF-TimeSeries", "RobertWalraven-ESG", "JHUAPL-Bucky", "Umich-RidgeTfReg"],
    "baseline": ["Baseline II", "Baseline I"]
}

def get_model_type(model):
    for k, v in MODEL_TYPE.items():
        if model in v:
            return k
    
def get_model_type_color(model):
    COLOR_CODES = {
        "epi" : "#a157db",
        "ml" : "#57db80",
        "ensemble": "#db57b2",
        "others":  "#e8d264",
        "baseline": "#8f9194"
    }
    m_type = get_model_type(model)
    if m_type in COLOR_CODES.keys():
        return COLOR_CODES[m_type]
    else:
        return "#57d3db"

def check_ensemble_eligiblity(eligibility_frame, model, forecast_date):
    model_frame = eligibility_frame[(eligibility_frame.model==model) & (eligibility_frame.forecast_date==forecast_date) & (eligibility_frame.overall_eligibility
=="eligible")]
    if len(model_frame)>0:
        return True
    else:
        return False
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import requests
from thefuzz import fuzz, process
import numpy as np
import os
import re
import time
start = time.time()

app = FastAPI()
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "n8n-Python service is running. POST your file to /process"
    }

EXCEL_FILE_URL = "https://github.com/js-1881/n8n_repo/raw/main/turbine_types_id_enervis.xlsx"

# Anemos credentials
EMAIL    = "amani@flex-power.energy"
PASSWORD = "ypq_CZE2wpg*jgu7hfk"

# --- Anemos helper functions ---
def get_token():
    url = "https://keycloak.anemosgmbh.com/auth/realms/awis/protocol/openid-connect/token"
    data = {
        'client_id': 'webtool_vue',
        'grant_type': 'password',
        'username': EMAIL,
        'password': PASSWORD
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()['access_token']

def get_historical_product_id(token):
    url = "https://api.anemosgmbh.com/products_mva"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    for p in r.json():
        if "hist-ondemand" in p["mva_product_type"]["name"].lower():
            return p["id"]
    raise RuntimeError("No 'hist-ondemand' product found")

def start_historical_job_from_df(token, product_id, df_input):
    url = "https://api.anemosgmbh.com/jobs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    parkinfo = []
    for _, row in df_input.iterrows():
        if pd.notna(row["Matched_Turbine_ID"]):
            parkinfo.append({
                "id":             int(row["malo"]),
                "lat":            str(row["latitude"]),
                "lon":            str(row["longitude"]),
                "turbine_type_id":int(row["Matched_Turbine_ID"]),
                "hub_height":     int(row["hub_height_m"])
            })
    if not parkinfo:
        raise RuntimeError("No valid parkinfo entries")
    payload = {
        "mva_product_id": product_id,
        "parameters":     {"parkinfo": parkinfo}
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["uuid"]

def wait_for_job_completion(token, job_uuid, poll_interval=10):
    url = f"https://api.anemosgmbh.com/jobs/{job_uuid}"
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        r = requests.get(url, headers=headers)
        # handle token expiry
        if r.status_code == 401:
            token = get_token()
            headers["Authorization"] = f"Bearer {token}"
            r = requests.get(url, headers=headers)
        r.raise_for_status()
        info = r.json()
        if isinstance(info, list): 
            info = info[0]
        status = info.get("status","")
        if status.lower() in ("done","completed"):
            return info
        if status.lower() in ("failed","canceled"):
            raise RuntimeError(f"Job ended with status {status}")
        time.sleep(poll_interval)

def download_and_load_csv(url, token):
    r = requests.get(url, headers={"Authorization":f"Bearer {token}"})
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def download_result_files(job_info, token):
    files = job_info.get("files")
    if files:
        return [download_and_load_csv(f["url"], token) for f in files]
    # fallback to 'info'→'results'
    results = job_info.get("info",{}).get("results",[])
    dfs = []
    for res in results:
        year_data = res.get("Marktwertdifferenzen",{})
        df = pd.DataFrame.from_dict(year_data, orient="index", columns=["Marktwertdifferenz"])
        df.index.name = "Year"
        df = df.reset_index().assign(id=res["id"])
        dfs.append(df)
    return dfs if dfs else None



@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        # Step 1: Load user-uploaded Excel
        contents = await file.read()
        df_excel1 = pd.read_excel(io.BytesIO(contents), sheet_name='Übersicht BEE Portfolio', dtype={'malo': str})
        df_excel2 = pd.read_excel(io.BytesIO(contents), sheet_name='Koordinaten (2)')

        df_excel1.columns = df_excel1.columns.str.strip()
        df_excel2.columns = df_excel2.columns.str.strip()
        df_excel1['Gesellschaft'] = df_excel1['Gesellschaft'].astype(str).str.strip()
        df_excel2['Gesellschaft'] = df_excel2['Gesellschaft'].astype(str).str.strip()

        df_excel = df_excel1.merge(df_excel2[['Gesellschaft', 'unit_mastr_id']], on='Gesellschaft', how='left')
        df_excel.dropna(subset=["malo"], axis=0, inplace=True)

        df = df_excel
        df['malo'] = df['malo'].astype(str).str.strip()

        # Step 2: Filter for 'SEE' IDs
        id_list = df['unit_mastr_id'].dropna()
        valid_ids = [
            str(id_).strip().upper()
            for id_ in id_list
            if str(id_).strip().lower().startswith("see")
        ]
        unique_see = set(valid_ids)

        print("✅ Valid IDs to fetch:", valid_ids)

        # Step 3: Fetch token
        auth_response = requests.post(
            'https://api.blindleister.de/api/v1/authentication/get-access-token',
            headers={'accept': 'text/plain', 'Content-Type': 'application/json'},
            json={'email': 'lfritsch@flex-power.energy', 'password': 'Ceciistlieb123.'}
        )
        token = auth_response.text.strip('"')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJvcHNAZmxleC1wb3dlci5lbmVyZ3kifQ.Q1cDDds4fzzYFbW59UuZ4362FnmvBUQ8FY4UNhWp2a0'
        }

        # Fetch blindleister price
        print("🍚🍚")
        # === Years to fetch ===
        years = [2021, 2023, 2024]
        records = []
        
        # === Loop through each ID and fetch data for each year ===
        for site_id in valid_ids:
            print(f"Processing: {site_id}")
        
            for year in years:
                payload = {
                    'ids': [site_id],
                    'year': year
                }
        
                response = requests.post(
                    'https://api.blindleister.de/api/v1/market-price-atlas-api/get-market-price',
                    headers = headers,
                    json=payload
                )
        
                if response.status_code != 200:
                    print(f"  Year {year}: Failed ({response.status_code}) - {response.text}")
                    continue
        
                try:
                    result = response.json()
                    for entry in result:
                        entry['year'] = year
                        records.append(entry)
                except Exception as e:
                    print(f"  Year {year}: Error parsing response - {e}")
                    continue
        
        df_flat = pd.DataFrame(records)
        df_flat = pd.json_normalize(
            records,
            record_path="months",
            meta=[
                "year",
                "unit_mastr_id",
                "gross_power_kw",
                "energy_source",
                "annual_generated_energy_mwh",
                "benchmark_market_price_eur_mwh",
            ],
            errors="ignore"  # in case some records lack "months"
        )
        
        cols = [
            "year",
            "unit_mastr_id",
            "gross_power_kw",
            "energy_source",
            "annual_generated_energy_mwh",
            "benchmark_market_price_eur_mwh",
            "month",
            "monthly_generated_energy_mwh",
            "monthly_energy_contribution_percent",
            "monthly_market_price_eur_mwh",
            "monthly_reference_market_price_eur_mwh",
        ]
        df_flat = df_flat[cols]
        df_all_flat = df_flat.copy()
        
        df_all_flat['weighted_per_mwh_monthly'] = (
            ((df_all_flat['monthly_generated_energy_mwh'] * df_all_flat['monthly_market_price_eur_mwh']) -
             (df_all_flat['monthly_generated_energy_mwh'] * df_all_flat['monthly_reference_market_price_eur_mwh'])) /
            df_all_flat['monthly_generated_energy_mwh'] *
            df_all_flat['monthly_energy_contribution_percent'] / 100 * 12
        )

        print("🥨🥨🥨")
        year_agg_per_unit = df_all_flat.groupby(['year', 'unit_mastr_id'])['weighted_per_mwh_monthly'].mean().reset_index(name='weighted_year_agg_per_unit_eur_mwh')
        df_year_agg_per_unit = pd.DataFrame(year_agg_per_unit)

        weighted_years_pivot = df_year_agg_per_unit.pivot(
            index='unit_mastr_id',
            columns='year',
            values='weighted_year_agg_per_unit_eur_mwh'
        ).reset_index()
        
        
        # Rename columns for clarity
        weighted_years_pivot.columns.name = None  # remove the axis name
        weighted_years_pivot = weighted_years_pivot.rename(columns={
            2021: 'weighted_2021_eur_mwh_blindleister',
            2023: 'weighted_2023_eur_mwh_blindleister',
            2024: 'weighted_2024_eur_mwh_blindleister'
        })
        
        # Add a column to average the available yearly weighted values
        weighted_years_pivot['average_weighted_eur_mwh_blindleister'] = weighted_years_pivot[
            ['weighted_2021_eur_mwh_blindleister', 'weighted_2023_eur_mwh_blindleister', 'weighted_2024_eur_mwh_blindleister']
        ].mean(axis=1, skipna=True)
        
        # Show only the desired columns
        final_weighted_blindleister = weighted_years_pivot[[
            'unit_mastr_id',
            'weighted_2021_eur_mwh_blindleister',
            'weighted_2023_eur_mwh_blindleister',
            'weighted_2024_eur_mwh_blindleister',
            'average_weighted_eur_mwh_blindleister'
        ]]

        

        # Step 4: Fetch generator details
        all_records = []
        for site_id in valid_ids:
            response = requests.post(
                'https://api.blindleister.de/api/v1/mastr-api/get-generator-details',
                headers=headers,
                json={'ids': [site_id], 'year': 2024}
            )
            if response.status_code == 200:
                result = response.json()
                for entry in result:
                    entry['year'] = 2024
                    all_records.append(entry)

        df_blind_fetch = pd.DataFrame(all_records)
        if df_blind_fetch.empty:
            return {"error": "No valid records returned from external API."}

        # Step 5: Process data
        df_blind_fetch = df_blind_fetch[df_blind_fetch["energy_source"] == 'wind']
        df_blind_fetch['net_power_mw'] = df_blind_fetch['net_power_kw'] / 1000

        def clean_name(name):
            name = str(name).lower()
            for word in ['gmbh', 'se', 'deutschland', 'central europe', 'energy', 'gmbh & co. kg']:
                name = name.replace(word, '')
            return name.strip()

        df_blind_fetch['clean_manufacturer'] = df_blind_fetch['manufacturer'].apply(clean_name)

        def clean_turbine(name):
            if not isinstance(name, str): return ''
            for word in ['senvion', 'enercon', 'mit Serrations', 'Vensys']:
                name = name.replace(word, '')
            return name.strip()

        df_blind_fetch['turbine_model_clean'] = df_blind_fetch['turbine_model'].apply(clean_turbine)
        df_blind_fetch['add_turbine'] = df_blind_fetch['manufacturer']

        df_blind_fetch.loc[
            df_blind_fetch['manufacturer'].isin(['Vestas Deutschland GmbH', 'Senvion Deutschland GmbH', 'ENERCON GmbH']),
            'add_turbine'
        ] = df_blind_fetch['turbine_model_clean'].str.strip() + ' ' + df_blind_fetch['net_power_mw'].round(3).astype(str) + 'MW'

        df_blind_fetch.loc[
            df_blind_fetch['manufacturer'] == 'Nordex Energy GmbH',
            'add_turbine'
        ] = df_blind_fetch['turbine_model_clean'].str.strip() + ' ' + df_blind_fetch['net_power_kw'].astype(str)

        print("📦📦 reading excel")

        # Step 6: Match turbine names from GitHub
        ref_response = requests.get(EXCEL_FILE_URL)
        ref_response.raise_for_status()
        df_ref = pd.read_excel(io.BytesIO(ref_response.content))

        hardcoded_map = {
            "V90 MK8 Gridstreamer": "V-90 2.0MW Gridstreamer",
            "V126-3.45MW": "V-126 3.45MW",
            "V-90": "V-90 2.0MW Gridstreamer",
            "V-112 2.0MW": "V-112 3.3MW",
            "V136-3.6MW": "V-136 3.6MW",
            "V112-3,45": "V-112 3.45MW"
        }

        def match_add_turbine(row, choices, threshold=85):
            name = row['turbine_model']
            add_turbine = row['add_turbine']
            if name in hardcoded_map: return hardcoded_map[name]
            if add_turbine in hardcoded_map: return hardcoded_map[add_turbine]
            match, score = process.extractOne(add_turbine, choices, scorer=fuzz.token_sort_ratio)
            return match if score >= threshold else None

        df_blind_fetch['Matched_Turbine_Name'] = df_blind_fetch.apply(
            lambda row: match_add_turbine(row, df_ref['name'].dropna().unique()),
            axis=1
        )

        name_to_id = df_ref.set_index('name')['id'].to_dict()
        df_blind_fetch['Matched_Turbine_ID'] = df_blind_fetch['Matched_Turbine_Name'].map(name_to_id)

        df_blind_fetch["hub_height_m"] = df_blind_fetch["hub_height_m"].fillna(0).astype(int).astype(str)
        df_blind_fetch["Matched_Turbine_ID"] = df_blind_fetch["Matched_Turbine_ID"].astype(str)

        df_fuzzy = df_blind_fetch[[
            "unit_mastr_id", "latitude", "longitude", "Matched_Turbine_ID", "hub_height_m"
        ]]

        print("🥕🥕🥕🥕🥕🥕🥕🥕") 

        df_final = pd.merge(df_excel, df_fuzzy, on="unit_mastr_id", how="left")
        df_final['hub_height_m_numeric'] = pd.to_numeric(df_final['hub_height_m'], errors='coerce')

        df_final['hub_height_m'] = (df_final['hub_height_m_numeric'].apply(lambda x: int(x) if pd.notna(x) else ""))
        df_final.drop(columns=['hub_height_m_numeric'], inplace=True)

        #df_final = df_final.dropna(subset=["latitude"])
        print("🍣🍣🍣")

        # --- 1. Authenticate & submit historical job ---
        token      = get_token()
        product_id = get_historical_product_id(token)
        job_uuid   = start_historical_job_from_df(token, product_id, df_final)
        job_info   = wait_for_job_completion(token, job_uuid)

        # --- 2. Download & concatenate results ---
        dfs = download_result_files(job_info, token)
        all_df = pd.concat(dfs, ignore_index=True)
        all_df["Year"] = all_df["Year"].astype(str)
        target_years = ["2021", "2023", "2024"]
        
        # Step 1: Filter to keep only the minimum Marktwertdifferenz per (id, Year)
        df_filtered = all_df.loc[
            all_df.groupby(["id", "Year"])["Marktwertdifferenz"].idxmin()
        ].copy()
        
        # Step 2: Pivot to wide format
        df_enervis_pivot = df_filtered.pivot(
            index="id",
            columns="Year",
            values="Marktwertdifferenz"
        ).rename_axis(None, axis=1).reset_index()
        
        # Step 3: Ensure all year columns are present
        for year in target_years:
            if year not in df_enervis_pivot.columns:
                df_enervis_pivot[year] = np.nan
        
        # Step 4: Compute row-wise average over existing target years
        df_enervis_pivot["avg_enervis"] = df_enervis_pivot[target_years].mean(axis=1, skipna=True)
        columns_to_keep = ["id"] + target_years + ["avg_enervis"]
        df_enervis_pivot_filter = df_enervis_pivot[columns_to_keep]

        columnskeep = ["Projekt", "Gesellschaft", "tech", "malo", "unit_mastr_id", "Nennleistung [MW]"]

        df_final = df_final[columnskeep]
        df_excel_agg = df_final.groupby("malo").agg({
            'unit_mastr_id': 'first',
            'Projekt': 'first', 
            'Gesellschaft': 'first', 
            'tech': 'first',
            'Nennleistung [MW]': 'first'
        }).reset_index()
        
        merge_a1 = pd.merge(
            df_excel_agg, 
            final_weighted_blindleister, 
            on = ('unit_mastr_id'),
            how='left'
        )
        
        df_enervis_pivot_filter['id'] = df_enervis_pivot_filter['id'].astype(str)
        
        merge_a2 = pd.merge(
            merge_a1, 
            df_enervis_pivot_filter, 
            left_on = ('malo'),
            right_on = ('id'),
            how='left'
        )

        merge_a2 = merge_a2.drop(columns=['id'])

        print("✅ Excel file generated and response returned.")
        print("🥕🥕") 
        end = time.time()

        # Export to Excel and return as response
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            merge_a2.to_excel(writer,    sheet_name="Processed Data",   index=False)
            df_enervis_pivot_filter.to_excel(writer, sheet_name="Historical Results", index=False)
            final_weighted_blindleister.to_excel(writer, sheet_name="final_weighted_blindleister", index=False)
            df_excel_agg.to_excel(writer, sheet_name="df_excel_agg", index=False)
        output.seek(0)

        print(f"🕒 Finished in {time.time()-start:.2f}s")
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition":"attachment; filename=full_results.xlsx"}
        )
        

    except Exception as e:
        return {"error": str(e)}
        

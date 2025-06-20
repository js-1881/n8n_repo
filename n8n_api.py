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

app = FastAPI()

EXCEL_FILE_URL = "https://github.com/js-1881/n8n_repo/raw/main/turbine_types_id_enervis.xlsx"

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

        df_final = pd.merge(df_excel, df_fuzzy, on="unit_mastr_id", how="left")
        df_final["hub_height_m"] = df_final["hub_height_m"].apply(lambda x: int(x) if x != "0" else "")

        df_final = df_final.dropna(subset=["latitude"])

        # Step 7: Export to Excel and return as response
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='Processed Data', index=False)
            df_ref.to_excel(writer, sheet_name='Reference Data', index=False)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={"Content-Disposition": "attachment; filename=processed_results.xlsx"}
        )

    except Exception as e:
        return {"error": str(e)}

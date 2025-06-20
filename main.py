from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import requests
from pathlib import Path

app = FastAPI()

# URL to your Excel file in GitHub
EXCEL_FILE_URL = "https://github.com/js-1881/n8n_repo/raw/main/turbine_types_id_enervis.xlsx"

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        # Read uploaded file to DataFrame
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        # Add 'processed' column = sum of numeric columns per row
        df['processed'] = df['Nennleistung [MW]'] * 1000

        # Download and read the Excel file from GitHub
        response = requests.get(EXCEL_FILE_URL)
        response.raise_for_status()  # Raise an error for bad status codes
        df2 = pd.read_excel(io.BytesIO(response.content))

        # Save DataFrame to Excel in-memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Processed Data', index=False)
            df2.to_excel(writer, sheet_name='Reference Data', index=False)

        output.seek(0)

        # Return the Excel file as downloadable attachment
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                "Content-Disposition": "attachment; filename=processed_results.xlsx"
            }
        )

    except Exception as e:
        return {"error": str(e)}

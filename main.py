from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io

app = FastAPI()

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    # Read uploaded file to DataFrame
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))

    # Add 'processed' column = sum of numeric columns per row
    df['processed'] = df['Nennleistung [MW]'] * 1000

    # Save DataFrame to Excel in-memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    # Return the Excel file as downloadable attachment
    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            "Content-Disposition": f"attachment; filename=processed.xlsx"
        }
    )

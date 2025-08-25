from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil, os, uuid, base64
from process_mcq_sheet import process_sheet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/grade")
async def grade_mcq(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are accepted.")

    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = process_sheet(temp_path)
        return results

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answers": []}  # <-- ensures valid JSON
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class Base64Image(BaseModel):
    filename: str
    content: str


@app.post("/grade_base64")
async def grade_base64_image(data: Base64Image):
    temp_path = None
    try:
        base64_data = data.content
        if base64_data.startswith("data:image"):
            base64_data = base64_data.split(";base64,")[-1]

        image_bytes = base64.b64decode(base64_data)
        temp_path = f"temp_{uuid.uuid4().hex}_{data.filename}"

        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        results = process_sheet(temp_path)
        return results

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answers": []}  # <-- always JSON
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# OMR Backend (FastAPI)

This is the backend for the Optical Mark Recognition (OMR) project.
It uses a YOLOv8 model to detect answers and a registration number model.

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload

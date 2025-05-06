import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from json_maker import analyze_player_performance
from start_detection import start_detection
from upload_to_cloudinary import upload_json_to_cloudinary
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the upload directory and filename
UPLOAD_DIR = Path("uploads/videos")
TARGET_FILENAME = "uploaded_video.mp4"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Validate file type (only allow common video formats)
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    try:
        # Define the target path
        target_path = UPLOAD_DIR / TARGET_FILENAME

        # Save the file
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        start_detection()

        players_file = "C:/Users/modyy/Grad Project/Ai/AiGit/AiGit/players_data.csv"
        ball_file = "C:/Users/modyy/Grad Project/Ai/AiGit/AiGit/output/datasets/ball_data.csv"
        players_list = ["player1", "player2", "player3", "player4"]
        result = analyze_player_performance(players_file, ball_file, players_list)

        # Example usage
        file_path = "player_analysis.json"  # Replace with actual file path
        url = upload_json_to_cloudinary(file_path)

        return {
            "message": "Video uploaded successfully",
            "filepath": url,
            "path": str(target_path)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading video: {str(e)}"
        )
    finally:
        file.file.close()


# Optional: Endpoint to check if video exists
@app.get("/check-video/")
async def check_video():
    target_path = UPLOAD_DIR / TARGET_FILENAME
    if target_path.exists():
        return {
            "filename": TARGET_FILENAME,
            "exists": True,
            "path": str(target_path)
        }
    return {
        "filename": TARGET_FILENAME,
        "exists": False
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
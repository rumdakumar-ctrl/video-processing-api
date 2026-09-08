import os
import uuid
import shutil
import subprocess
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


app = FastAPI(title="Video Processing API")


# Blogger se API request allow karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = "/tmp/video_processing"

os.makedirs(BASE_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Video Processing API is running"
    }


@app.post("/process")
async def process_video(video: UploadFile = File(...)):

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file provided."
        )

    allowed_extensions = (
        ".mp4",
        ".mov",
        ".webm",
        ".mkv"
    )

    filename_lower = video.filename.lower()

    if not filename_lower.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )


    job_id = str(uuid.uuid4())

    job_dir = os.path.join(
        BASE_DIR,
        job_id
    )

    os.makedirs(job_dir, exist_ok=True)


    input_video = os.path.join(
        job_dir,
        "input_video.mp4"
    )

    silent_video = os.path.join(
        job_dir,
        "processed_silent.mp4"
    )

    final_video = os.path.join(
        job_dir,
        "final_video.mp4"
    )


    try:

        # Save uploaded video
        with open(input_video, "wb") as buffer:

            shutil.copyfileobj(
                video.file,
                buffer
            )


        # Open video
        cap = cv2.VideoCapture(
            input_video
        )

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video."
            )


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if fps <= 0:
            fps = 24


        # H.264 output
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            silent_video,
            fourcc,
            fps,
            (width, height)
        )


        # Confirmed mask position
        #
        # These coordinates are based on
        # the test video used during development.
        center_x = 605
        center_y = 1165

        axes_x = 42
        axes_y = 40


        frame_number = 0


        while True:

            success, frame = cap.read()

            if not success:
                break


            # Create small elliptical mask
            mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )


            # Keep coordinates inside video
            safe_x = min(
                max(center_x, 0),
                width - 1
            )

            safe_y = min(
                max(center_y, 0),
                height - 1
            )


            cv2.ellipse(
                mask,
                (
                    safe_x,
                    safe_y
                ),
                (
                    axes_x,
                    axes_y
                ),
                0,
                0,
                360,
                255,
                -1
            )


            # Slight feathering
            mask = cv2.GaussianBlur(
                mask,
                (15, 15),
                0
            )


            # Blur only the selected area
            blurred = cv2.GaussianBlur(
                frame,
                (31, 31),
                0
            )


            mask_float = (
                mask.astype(np.float32)
                / 255.0
            )


            for channel in range(3):

                frame[:, :, channel] = (
                    frame[:, :, channel]
                    * (1.0 - mask_float)
                    +
                    blurred[:, :, channel]
                    * mask_float
                ).astype(np.uint8)


            out.write(frame)


            frame_number += 1


        cap.release()
        out.release()


        # Add original audio back
        ffmpeg_command = [
            "ffmpeg",
            "-y",

            "-i",
            silent_video,

            "-i",
            input_video,

            "-map",
            "0:v:0",

            "-map",
            "1:a:0?",

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "20",

            "-c:a",
            "aac",

            "-b:a",
            "128k",

            "-shortest",

            final_video
        ]


        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True
        )


        if result.returncode != 0:

            raise HTTPException(
                status_code=500,
                detail="FFmpeg processing failed."
            )


        if not os.path.exists(
            final_video
        ):

            raise HTTPException(
                status_code=500,
                detail="Output video was not created."
            )


        return FileResponse(
            final_video,
            media_type="video/mp4",
            filename="processed-video.mp4"
        )


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Temporary cleanup is intentionally
        # omitted here because FileResponse
        # needs the output file to remain available.
        pass

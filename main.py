import os
import uuid
import shutil
import subprocess
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


app = FastAPI(title="Video Processing API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = "/tmp/video_processing"

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB

BASE_CENTER_X = 605
BASE_CENTER_Y = 1165

BASE_AXES_X = 50
BASE_AXES_Y = 48


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Video Processing API is running"
    }


def cleanup_job(job_dir):
    try:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
    except Exception:
        pass


@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    mode: str = Form("reconstruction")
):

    mode = mode.lower().strip()

    if mode not in ["blur", "reconstruction"]:
        raise HTTPException(
            status_code=400,
            detail="Mode must be blur or reconstruction"
        )

    allowed_extensions = [
        ".mp4",
        ".mov",
        ".webm",
        ".mkv"
    ]

    filename = video.filename or "video.mp4"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    job_id = str(uuid.uuid4())

    job_dir = os.path.join(
        BASE_DIR,
        job_id
    )

    os.makedirs(
        job_dir,
        exist_ok=True
    )

    input_path = os.path.join(
        job_dir,
        "input_video.mp4"
    )

    silent_path = os.path.join(
        job_dir,
        "processed_silent.mp4"
    )

    final_path = os.path.join(
        job_dir,
        "final_video.mp4"
    )

    try:

        # Save upload in small chunks
        total_size = 0

        with open(
            input_path,
            "wb"
        ) as buffer:

            while True:

                chunk = await video.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:

                    cleanup_job(job_dir)

                    raise HTTPException(
                        status_code=413,
                        detail="Video is too large. Maximum allowed size is 100 MB."
                    )

                buffer.write(chunk)

        await video.close()

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            cleanup_job(job_dir)

            raise HTTPException(
                status_code=400,
                detail="Could not open video"
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

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 24

        scale_x = width / 720.0
        scale_y = height / 1280.0

        center_x = int(
            BASE_CENTER_X * scale_x
        )

        center_y = int(
            BASE_CENTER_Y * scale_y
        )

        axes_x = int(
            BASE_AXES_X * scale_x
        )

        axes_y = int(
            BASE_AXES_Y * scale_y
        )

        # Watermark mask
        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.ellipse(
            mask,
            (center_x, center_y),
            (axes_x, axes_y),
            0,
            0,
            360,
            255,
            -1
        )

        # Small ROI around watermark
        padding = 25

        x1 = max(
            center_x - axes_x - padding,
            0
        )

        y1 = max(
            center_y - axes_y - padding,
            0
        )

        x2 = min(
            center_x + axes_x + padding,
            width
        )

        y2 = min(
            center_y + axes_y + padding,
            height
        )

        roi_mask = mask[
            y1:y2,
            x1:x2
        ]

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            silent_path,
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():

            cap.release()
            cleanup_job(job_dir)

            raise HTTPException(
                status_code=500,
                detail="Could not create output video"
            )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            roi = frame[
                y1:y2,
                x1:x2
            ]

            if mode == "reconstruction":

                repaired_roi = cv2.inpaint(
                    roi,
                    roi_mask,
                    5,
                    cv2.INPAINT_TELEA
                )

                frame[
                    y1:y2,
                    x1:x2
                ] = repaired_roi

            else:

                soft_mask = cv2.GaussianBlur(
                    roi_mask,
                    (15, 15),
                    0
                )

                blurred_roi = cv2.GaussianBlur(
                    roi,
                    (31, 31),
                    0
                )

                mask_float = (
                    soft_mask.astype(
                        np.float32
                    ) / 255.0
                )

                mask_float = mask_float[
                    :,
                    :,
                    np.newaxis
                ]

                repaired_roi = (
                    roi.astype(np.float32)
                    * (1 - mask_float)
                    +
                    blurred_roi.astype(np.float32)
                    * mask_float
                ).astype(np.uint8)

                frame[
                    y1:y2,
                    x1:x2
                ] = repaired_roi

            out.write(frame)

        cap.release()
        out.release()

        # Add original audio
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            silent_path,
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            final_path
        ]

        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            cleanup_job(job_dir)

            raise HTTPException(
                status_code=500,
                detail="FFmpeg processing failed"
            )

        cleanup_task = BackgroundTask(
            cleanup_job,
            job_dir
        )

        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="processed_video.mp4",
            background=cleanup_task
        )

    except HTTPException:
        raise

    except Exception as e:

        cleanup_job(job_dir)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

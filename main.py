import os
import uuid
import shutil
import subprocess
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


app = FastAPI(title="Video Processing API")


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = "/tmp/video_processing"

os.makedirs(BASE_DIR, exist_ok=True)


# Gemini visible logo position
# Tested with the current 720x1280 videos.

BASE_CENTER_X = 605
BASE_CENTER_Y = 1165

BASE_AXES_X = 50
BASE_AXES_Y = 48


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Video Processing API is running"
    }


# =====================================================
# PROCESS VIDEO
# =====================================================

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    mode: str = Form("reconstruction")
):

    if not video.filename:

        raise HTTPException(
            status_code=400,
            detail="No video file provided."
        )


    # -------------------------------------------------
    # Allowed modes
    # -------------------------------------------------

    mode = mode.lower().strip()

    if mode not in [
        "blur",
        "reconstruction"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Invalid processing mode. Use blur or reconstruction."
        )


    # -------------------------------------------------
    # Allowed video extensions
    # -------------------------------------------------

    allowed_extensions = (
        ".mp4",
        ".mov",
        ".webm",
        ".mkv"
    )

    filename_lower = video.filename.lower()

    if not filename_lower.endswith(
        allowed_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )


    # -------------------------------------------------
    # Create job directory
    # -------------------------------------------------

    job_id = str(
        uuid.uuid4()
    )

    job_dir = os.path.join(
        BASE_DIR,
        job_id
    )

    os.makedirs(
        job_dir,
        exist_ok=True
    )


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

        # =================================================
        # SAVE UPLOADED VIDEO
        # =================================================

        with open(
            input_video,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                video.file,
                buffer
            )


        # =================================================
        # OPEN VIDEO
        # =================================================

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


        if width <= 0 or height <= 0:

            cap.release()

            raise HTTPException(
                status_code=400,
                detail="Invalid video dimensions."
            )


        # =================================================
        # SCALE MASK FOR VIDEO SIZE
        # =================================================

        # Our tested coordinates are based on
        # a 720x1280 video.

        scale_x = width / 720.0
        scale_y = height / 1280.0


        center_x = int(
            BASE_CENTER_X * scale_x
        )

        center_y = int(
            BASE_CENTER_Y * scale_y
        )

        axes_x = max(
            10,
            int(BASE_AXES_X * scale_x)
        )

        axes_y = max(
            10,
            int(BASE_AXES_Y * scale_y)
        )


        # Keep coordinates inside video

        center_x = min(
            max(center_x, 0),
            width - 1
        )

        center_y = min(
            max(center_y, 0),
            height - 1
        )


        # =================================================
        # CREATE MASK ONCE
        # =================================================

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )


        cv2.ellipse(
            mask,
            (
                center_x,
                center_y
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


        # =================================================
        # VIDEO WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            silent_video,
            fourcc,
            fps,
            (width, height)
        )


        if not out.isOpened():

            cap.release()

            raise HTTPException(
                status_code=500,
                detail="Could not create output video."
            )


        # =================================================
        # PROCESS FRAMES
        # =================================================

        frame_number = 0


        while True:

            success, frame = cap.read()

            if not success:

                break


            # =================================================
            # MODE 1: BLUR
            # =================================================

            if mode == "blur":

                # Soft mask edge
                soft_mask = cv2.GaussianBlur(
                    mask,
                    (15, 15),
                    0
                )


                blurred = cv2.GaussianBlur(
                    frame,
                    (31, 31),
                    0
                )


                mask_float = (
                    soft_mask.astype(
                        np.float32
                    ) / 255.0
                )


                mask_float = mask_float[
                    :, :, np.newaxis
                ]


                result = (
                    frame.astype(
                        np.float32
                    )
                    * (1.0 - mask_float)
                    +
                    blurred.astype(
                        np.float32
                    )
                    * mask_float
                ).astype(
                    np.uint8
                )


            # =================================================
            # MODE 2: RECONSTRUCTION / INPAINTING
            # =================================================

            else:

                result = cv2.inpaint(
                    frame,
                    mask,
                    5,
                    cv2.INPAINT_TELEA
                )


            # =================================================
            # WRITE FRAME
            # =================================================

            out.write(
                result
            )


            frame_number += 1


        cap.release()
        out.release()


        # =================================================
        # CHECK SILENT VIDEO
        # =================================================

        if not os.path.exists(
            silent_video
        ):

            raise HTTPException(
                status_code=500,
                detail="Processed video was not created."
            )


        # =================================================
        # ADD ORIGINAL AUDIO
        # =================================================

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
            "18",

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


        # =================================================
        # CHECK FINAL VIDEO
        # =================================================

        if not os.path.exists(
            final_video
        ):

            raise HTTPException(
                status_code=500,
                detail="Final video was not created."
            )


        # =================================================
        # RETURN VIDEO
        # =================================================

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

        # FileResponse needs the final file
        # to remain available while downloading.

        pass

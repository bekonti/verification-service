import base64
import time
import os
from fastapi import HTTPException

import cv2
import numpy as np
from deepface import DeepFace

global model_name
model_name = 'Facenet512'


async def verify_doc_with_liveness(img, vid):
    try:
        img_path = await get_path(img)

        # Generate a unique temporary filename
        temp_filename = f"test.mp4"

        # Save the uploaded video file to disk
        with open(temp_filename, "wb") as temp_file:
            content = await vid.read()
            temp_file.write(content)

        cam = cv2.VideoCapture(temp_filename)

        frames_base64 = []
        while True:
            # Reading from frame
            ret, frame = cam.read()

            if not ret:
                break

            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)

            # Convert to base64 encoding and decode to string
            base64_str = base64.b64encode(buffer).decode('utf-8')

            # Append to list of base64 frames
            frames_base64.append(base64_str)

        # Release the video capture object and close windows
        cam.release()
        cv2.destroyAllWindows()

        array = []
        start_time = time.time()
        for iFrame in frames_base64:
            image_data_from_frame_base64 = f"data:image/jpg;base64,{iFrame}"
            array.append(
                DeepFace.verify(
                    img_path,
                    image_data_from_frame_base64,
                    model_name=model_name
                ).get("verified"))

        end_time = time.time()
        count_true = sum(filter(lambda x: x, array))
        percentage_true = (count_true / len(array)) * 100
        elapsed_time = end_time - start_time
        return {
            "score": percentage_true,
            "scored_time_in_seconds": elapsed_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


async def verify(img1, img2):
    return DeepFace.verify(await get_path(img1), await get_path(img2), model_name=model_name)


async def get_represent(img1):
    return DeepFace.represent(await get_path(img1), model_name=model_name)


async def get_path(img1):
    # Read the file streams
    file1_content = await img1.read()

    # Convert the file content to a numpy array
    np_arr1 = np.frombuffer(file1_content, np.uint8)

    # Decode the numpy array into an image
    return cv2.imdecode(np_arr1, cv2.IMREAD_COLOR)

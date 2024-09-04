import base64
import json
import time
import os
from fastapi import HTTPException

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from scipy.spatial.distance import cosine


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
            imageDataFromFrameBase64 = f"data:image/jpg;base64,{iFrame}"
            array.append(
                DeepFace.verify(
                    img_path,
                    imageDataFromFrameBase64
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
    img_path_1 = await get_path(img1)
    img_path_2 = await get_path(img2)
    return DeepFace.verify(img_path_1, img_path_2)


async def get_represent(img1):
    img_path = await get_path(img1)
    return DeepFace.represent(img_path)


async def get_path(img1):
    # Read the file streams
    file1_content = await img1.read()

    # Convert the file content to a numpy array
    np_arr1 = np.frombuffer(file1_content, np.uint8)

    # Decode the numpy array into an image
    return cv2.imdecode(np_arr1, cv2.IMREAD_COLOR)


def default():
    # Compare two images directly
    result = DeepFace.verify(
        "pics/yernur3.png",
        "pics/yernur.png"
        # "yernur4.png",
        # "yernurDoc.png"
    )

    imageFile = "pics/yernurDoc"
    facialArea = DeepFace.extract_faces("%s.png" % imageFile)
    print(facialArea)
    xy = facialArea.__getitem__(0).get("facial_area")
    x = xy.get("x") - 50
    y = xy.get("y") - 50
    w = xy.get("w") + 100
    h = xy.get("h") + 100
    crop_area = (x, y, x + w, y + h)
    image = Image.open("%s.png" % imageFile)
    cropped_image = image.crop(crop_area)
    cropped_image.save("%s_cropped.png" % imageFile)
    # print(DeepFace.extract_faces("pics/yernur.png"))

    print(result.get("verified"))
    print(json.dumps(result, indent=2))

    image_path = "pics/yernur3_1_2.png"
    image_path2 = "pics/yernur3_1.png"

    # Extract face embedding (represent) using a specific model
    embedding = DeepFace.represent(img_path=image_path, model_name='VGG-Face')[0]['embedding']
    embedding2 = DeepFace.represent(img_path=image_path2, model_name='VGG-Face')[0]['embedding']

    # Store the embedding in a database or a file (e.g., JSON, CSV, database table)
    # For this example, we will just print it
    print("Face Embedding:", embedding)
    similarity = 1 - cosine(embedding2, embedding)
    # Define a threshold for similarity (e.g., 0.7)
    threshold = 0.5
    if similarity > threshold:
        print("Face is recognized (match found).")
    else:
        print("Face is not recognized (no match).")

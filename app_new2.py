# import face_recognition
import os
import json
from flask import Flask,request#,render_template, flash, redirect, url_forr
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import requests
import random
import numpy as np
from threading import Thread
from threading import Thread
import cv2
from PIL import Image, ImageStat#,ImageDraw,ImageFont
import uuid #----
import pickle
from flask import Flask, request, jsonify
from PIL import Image, ImageStat
import requests
import json
from flask import request, jsonify
from PIL import Image as Img
from PIL import ImageStat
from Logger import Logs
import traceback
import sentry_sdk
import os
from os import listdir
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
import time
import datetime
import mediapipe as mp
from lib.embeddings import compute_encodings
from lib.auth_token import get_valid_token
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()


print('....1')
MyFaceNet = FaceNet()
print('....2')



from sentry_sdk.integrations.flask import FlaskIntegration
'''//////////////////'''
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FlaskIntegration(),
    ],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
   #  profiles_sample_rate=1.0,
)

class FaceRecognitionLogger:
    def __init__(self, debug=False):
        self.debug = debug

    def log_step(self, message, show_separator=False):
        """Log a step message"""
        if self.debug:
            if show_separator:
                print("\n" + "=" * 50)
            print(message)

    def plot_images(self, images, titles=None, figsize=(15, 5)):
        """Plot one or more images"""
        if not self.debug:
            return

        if not isinstance(images, list):
            images = [images]
        if not titles:
            titles = [f"Image {i+1}" for i in range(len(images))]

        plt.figure(figsize=figsize)
        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis("off")
        plt.show()

    def plot_face_detection(self, image, faces, title="Detected Faces"):
        """Plot image with detected face boxes"""
        if not self.debug:
            return

        debug_image = image.copy()
        for idx, face in enumerate(faces):
            x1, y1, width, height = face["box"]
            cv2.rectangle(
                debug_image, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2
            )
            cv2.putText(
                debug_image,
                f"Face {idx+1} ({face['confidence']:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def plot_face_crop(self, face_crop, index):
        """Plot a cropped face"""
        if not self.debug:
            return

        plt.figure(figsize=(5, 5))
        plt.imshow(face_crop)
        plt.title(f"Cropped Face {index+1}")
        plt.axis("off")
        plt.show()

    def log_comparison_result(self, face1_idx, face2_idx, similarity_score, threshold):
        """Log face comparison result"""
        if self.debug:
            print(f"Face {face1_idx+1} (Image 1) vs Face {face2_idx+1} (Image 2):")
            print(f"  Similarity Score: {similarity_score:.2f}%")
            print(f"  Match: {'✅' if similarity_score > threshold else '❌'}")

    def log_final_result(self, result):
        """Log final comparison result"""
        if self.debug:
            print("\n=== Final Results ===")
            print(f"Match Found: {'✅' if result['match'] else '❌'}")
            print(f"Best Match Score: {result['best_match_score']:.2f}%")
            print("=" * 50)

logger=FaceRecognitionLogger()
# Disable/enable logging
# logger.disable_logs()
# logger.enable_logs()
logger_ = Logs()



app = Flask(__name__)#, template_folder='template')
csrf = CSRFProtect()
csrf.init_app(app) # Compliant
UPLOAD_FOLDER = 'temp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)



def mediapipe_detect_faces(image, logger=logger, threshold=0):
    """Detects faces in an image using MediaPipe Face Detection."""
    mp_face_detection = mp.solutions.face_detection
    faces = []
    
    
    # Avoid redundant color conversion if image is already RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = image if image[0,0,0] <= image[0,0,2] else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-calculate image dimensions
    h, w = image.shape[:2]

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=threshold
    ) as face_detection:
        # plot image
        logger.plot_images([rgb_image], ["Sending to face detection"])
        results = face_detection.process(rgb_image)
        print("results", results)

        if results.detections:
            # Pre-allocate faces list with known size
            faces = [None] * len(results.detections)
            
            if logger is not None:
                logger.log_step(f"Found {len(results.detections)} faces")

            # Use enumerate to avoid manual indexing
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                print("bboxC", bboxC)
                # Vectorized calculations
                faces[i] = {
                    "box": [
                        int(bboxC.xmin * w),
                        int(bboxC.ymin * h),
                        int(bboxC.width * w),
                        int(bboxC.height * h)
                    ],
                    "confidence": detection.score[0]
                }
        elif logger is not None:
            logger.log_step("No faces detected")

    if logger is not None:
        logger.plot_face_detection(image, faces)
    return faces

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def brighten_image(image, alpha=1.5, beta=30):
    # Alpha controls contrast (1.0 is no change, higher is more contrast)
    # Beta controls brightness (0 is no change, positive values increase brightness)
    brightened_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return brightened_image


@app.route("/get-color", methods=['POST'])
def get_color():
    data = request.json
    userid = data.get("userid", "")
    img_url = data.get("image", "")
    call_userid = data.get("call_userid", "")
    device_n = data.get("device_id", "")

    try:
        # Make a request to get the image from the given URL
        response = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1"})
        
        if response.status_code == 200:
            # Save the image locally
            with open('123.jpg', 'wb') as handler:
                handler.write(response.content)
            
            # Open the saved image
            image = Image.open('123.jpg')
            
            # Get the width and height of the image
            (w, h) = image.size
            
            # Calculate the ratios for the portions of the image to be cropped
            ratio_width = (10 / 100) * w
            ratio_height1 = (10 / 100) * h
            ratio_height2 = (20 / 100) * h
            
            # Calculate the top position for cropping
            top = h / 3
            
            # Crop the image into three portions using the calculated values
            first_image = image.crop((0 + ratio_width, 0 + ratio_height1, w - ratio_width, top))
            sec_image = image.crop((0 + ratio_width, top, w - ratio_width, (top * 2) - ratio_height2))
            third_image = image.crop((0 + ratio_width, (top * 2) + ratio_height2, w - ratio_width, (top * 3) - 10))
            
            # Create a list of the cropped images
            img_list = [first_image, sec_image, third_image]
            
            # Create a dictionary to store the colors of each portion
            message = {'topcolor': '', 'centercolor': '', 'bottomcolor': ''}
            
            # Create a list of keys for the dictionary
            temp = ['topcolor', 'centercolor', 'bottomcolor']
            
            # Iterate over the list of images and calculate the average color for each portion
            for index, img_portion in enumerate(img_list):
                stat = ImageStat.Stat(img_portion)
                avg_color = stat.mean
                
                # Calculate the brightness of the average color
                brightness = sum(avg_color) / len(avg_color)
                
                # Set the text color based on the brightness
                if brightness < 128:
                    text_color = '#FFFFFFFF'  # white
                else:
                    text_color = '#FF000000'  # black
                
                # Store the text color in the dictionary
                message[temp[index]] = text_color
                
                # Save the cropped image with the text color in the filename
                img_portion.save(f'results/new_color_{index}_{text_color}.jpg')
            
            # Create a response array with the success flag and the data dictionary
            response_data = {"success": True, "data": message}
            # logger.info(f'get-color response sent {response_data}', user_id=userid, session_id='', log_module='Face_Rec', call_userid=call_userid, device_name=device_n)
            
            # Convert the response array to JSON and return it
            return json.dumps(response_data)
    
    except Exception as ex:
        # Log the exception and create a response array with the success flag and the exception
        logger.error(f'Exception in get color main {ex}', user_id=userid, session_id='', log_module='Face_Rec', call_userid=call_userid, device_name=device_n)
        response_data = {"success": False, "data": str(ex)}
        
        # Convert the response array to JSON and return it
        return json.dumps(response_data)

def encoding_generator(image, faces, logger=None):
    """Generate face encodings for detected faces"""
    if not faces:
        return []

    # Pre-convert image to RGB once
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-allocate list for better memory efficiency
    encodings = [None] * len(faces)
    
    logger and logger.log_step(f"\nGenerating encodings for {len(faces)} faces...")

    # Process all faces in a single batch
    face_arrays = []
    for idx, face in enumerate(faces):
        # Extract face coordinates more efficiently
        box = face["box"]
        x1, y1 = abs(box[0]), abs(box[1])
        x2, y2 = x1 + box[2], y1 + box[3]

        # Crop face region
        face_crop = rgb_image[y1:y2, x1:x2]
        
        logger and logger.plot_face_crop(face_crop, idx)

        # Prepare face array more efficiently
        face_array = expand_dims(
            asarray(Img.fromarray(face_crop).resize((160, 160), reducing_gap=3.0)), 
            axis=0
        )
        face_arrays.append(face_array)

    # Generate encodings in batch
    face_arrays = np.vstack(face_arrays)
    batch_encodings = MyFaceNet.embeddings(face_arrays)

    # Store results
    for idx, encoding in enumerate(batch_encodings):
        logger and logger.log_step(f"Generated encoding for face {idx+1}, shape: {encoding.shape}")
        encodings[idx] = encoding

    return encodings



def dlib_fn(image, source, userid, logger=logger_):
    '''take img,source and userid return False 
    if it does not find face in an image and if 
    finds face then return its encodings'''

    # faces = MyFaceNet.mtcnn().detect_faces(image)
    faces = mediapipe_detect_faces(image, logger)
    if len(faces) == 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=brighten_image(image)
        # cv2.imwrite('bighten_img.jpg',image)
        # faces = MyFaceNet.mtcnn().detect_faces(image)
        faces = mediapipe_detect_faces(image, logger)
        logger.warning(
                    f'Number of faces {len(faces)}.....{faces}',
                    user_id=userid,
                    session_id='',
                    log_module='Face_Rec',
                    # call_userid=call_userid,
                    # device_name=device_name
                    )
    # encodings=compute_encodings(image,faces)
    # encoding_generator
    encodings=encoding_generator(image,faces)

    
    if (source != "facebook" and source != "instagram"):

        
        if len(faces) < 1:
            # If no face is found, return a JSON response indicating failure and reason
            message = 'Face is unclear'
            __array = {"success": False, "message": message}
            return json.dumps(__array)
        else:
            
                # Append the facial encodings to the 'knownEncodings' list in the user's dictionary
            for encoding in encodings:
                user_data_model[userid]['knownEncodings'].append(encoding)
    else:
        # Append the facial encodings to the 'knownEncodings' list in the user's dictionary
        for encoding in encodings:
            user_data_model[userid]['knownEncodings'].append(encoding)
            
    return None


def add_friend(filename, userid, call_userid, device_n, logger=logger_):
    '''load th existing model of user and check 
    if unique id is same then it will rename the friend and 
    if unique id does not exits then generate unique id and 
        save all the info like unique id, name, encodings
        and friend id and add this data to users model file'''
    # Retrieve user data from the global data structure
    user_dictionary = user_data_model[userid]

    if os.path.exists(filename):
        # Open the file in read binary mode
        f = open(filename, "rb")
        
        while True:
            try:
                # Load the pickled data from the file
                model_file_data = pickle.load(f)
                '''{'encodings': [array([]), array([])], 'names': 'Muhammad Asharf Khaskheli', 'friend_id': 6982, 'unique_id': 'd0cb198d-b7c7-4317-8992-bed629514f11'}'''
                if str(model_file_data["unique_id"]) == str(user_dictionary['unique_id']):
              
                    logger.info(f'Changing name from {model_file_data["names"]} to {user_dictionary["name"]}',user_id=userid,
                    session_id='',
                    log_module='Face_Rec',
                    call_userid=call_userid,
                    device_name=device_n
                    )
                    # Modify the name and friend ID in the loaded data
                    model_file_data["names"] = user_dictionary['name']
                    model_file_data["friend_id"] = user_dictionary['friend_id']
                
                # Append the modified data to the user's 'objs' list
                user_data_model[userid]['objs'].append(model_file_data)

            except Exception as ex:
                # Print any exception that occurred during editing/loading the file
         
                if str(ex) =='Ran out of input':

                    logger.warning(
                    'Data loop ended',
                    user_id=userid,
                    session_id='',
                    log_module='Face_Rec',
                    call_userid=call_userid,
                    device_name=device_n
                    )
                else:

                    logger.error(
                    f'Exception in add friend function {ex}',
                    user_id=userid,
                    session_id='',
                    log_module='Face_Rec',
                    call_userid=call_userid,
                    device_name=device_n
                    )
                break

        
        f.close()  # Close the file after loading and modifying the data

    if not user_dictionary['unique_id']:
        # Generate a unique ID for the user if it doesn't exist
        user_dictionary['unique_id'] = str(uuid.uuid4())
        
        # Append a new object to the user's 'objs' list with the unique ID, known encodings, name, and friend ID
        user_dictionary['objs'].append({"encodings": user_dictionary['knownEncodings'], "names": user_dictionary['name'], "friend_id": user_dictionary['friend_id'], "unique_id": user_dictionary['unique_id']})

    # Open the file in write binary mode
    f = open(filename, "wb")
    
    for objects in user_dictionary['objs']:
        # Write each object in the 'objs' list to the file using pickle
        f.write(pickle.dumps(objects))
    
    f.close()  # Close the file after writing the data
    
    # Return the friend ID and unique ID from the user's dictionary
    return user_dictionary['friend_id'], user_dictionary['unique_id']

user_data_model = {}

@app.route("/check-face", methods=['POST'])
def create_modal():
    '''take img, creates encodings and save it to the model file '''
    # Get JSON data from the request
    data = request.json
    logger = logger_
    
    # Extract relevant data from the JSON request
    userid = data.get("user_id", None)
    friend_id = data.get("friend_id", "")
    unique_id = data.get("unique_id", None)
    source = data.get("source", "")
    bulk_images = data["image"]
    name = data["name"]
    call_userid=data.get("callUserId","")
    device_n=data.get("device_id","")
    location_id=data.get("location_id", None)
    print("****************Recieved Request for Add Visitor*********************")
    if not location_id:
            print("!!!!!!!!!!!!!!!!!!!!!!! Missing required parameters: location_id!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "Missing required parameters:location_id"}, 400
    if not userid:
            print("!!!!!!!!!!!!!!!!!!!!!!! Missing required parameters: userid!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "Missing required parameters: userid"}, 400
    if not unique_id:
            print("!!!!!!!!!!!!!!!!!!!!!!! Missing required parameters: userid or location_id!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "Missing required parameters: unique_id"}, 400

    # print(
    #     f'Add Visitor with Request Params {data}',
    #     user_id=userid,
    #     session_id='',
    #     log_module='Face_Rec',
    #     call_userid=call_userid,
    #     device_name=device_n,
    #     location_id=location_id
    #     )

    print("Request params = ", {
        "user_id": userid,
        "session_id":'',
        "log_module":'Face_Rec',
        "call_userid":call_userid,
        "device_name":device_n,
        "location_id":location_id
            })
    
  
    logger.info(
        f'Create model request with data {data}',
        user_id=userid,
        session_id='',
        log_module='Face_Rec',
        call_userid=call_userid,
        device_name=device_n,
        location_id=location_id
        )
    # Initialize a global data structure (possibly a dictionary) for user data
    global user_data_model
    user_data_model[userid] = {
        "knownEncodings": [],
        "userid": userid,
        "objs": [],
        "friend_id": friend_id,
        "unique_id": unique_id,
        "source": source,
        "name": name,
        "bulk_images": bulk_images,
        "location_id": location_id
    }

    # Retrieve user data from the global data structure
    user_dictionary = user_data_model[userid]

    try:
        # Create a directory for the user if it doesn't exist
        directory = "user-model-for-id-{0}".format(user_dictionary['location_id'])
        if not os.path.exists(directory):
            os.mkdir(directory, 0o775)
        
        # Define the filename for the user model
        filename = directory + "/model-user-main.json"

        # Process bulk images
        for bulk in user_dictionary['bulk_images']:
            try:
                # Download the image from the URL
                response = requests.get(bulk, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1"})
                
                if response.status_code == 200:
                    content = response.content
                    # fromstring #there was a warning 'DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead'
                    nparr = np.frombuffer(content, np.uint8) 
                    image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

                    # Perform facial recognition using dlib_fn
                    result = dlib_fn(image, user_dictionary['source'], user_dictionary['userid'])

                    if result is not None:

                        return result
                else:
                    pass
                  
                    logger.info(
                        'Failed to fetch image',
                        user_id=userid,
                        session_id='',
                        log_module='Face_Rec',
                        call_userid=call_userid,
                        device_name=device_n
                        )
                    
            except Exception as ex:

                logger.error(
                        f'Fetching image exception {ex}',
                        user_id=userid,
                        session_id='',
                        log_module='Face_Rec',
                        call_userid=call_userid,
                        device_name=device_n
                        )
    except Exception as ex:
    
        logger.error(
            f'create model main code exception {ex}',
            user_id=userid,
            session_id='',
            log_module='Face_Rec',
            call_userid=call_userid,
            device_name=device_n
            )
    # Add friend and obtain unique ID
    friend_id, unique_id = add_friend(filename, user_dictionary['userid'], call_userid, device_n)
    
    # Prepare a response
    __array = {"success": True, "message": "Friend Added", "unique_id": unique_id}

    logger.info(
            f'create model response sent {__array}',
            user_id=userid,
            session_id='',
            log_module='Face_Rec',
            call_userid=call_userid,
            device_name=device_n
            )
    # Return the response as JSON
    return json.dumps(__array)

def are_all_values_same(non_zero_counts_dict):
    # Get the first value from the dictionary
    first_value = next(iter(non_zero_counts_dict.values()), None)
    
    # If the dictionary is empty or contains only one key, return True
    if first_value is None or len(non_zero_counts_dict) <= 1:
        return True
    
    # Check if all values are equal to the first value
    return all(value == first_value for value in non_zero_counts_dict.values())

def compare_with_L2_distance(predefine_encoding, encoding):
    return np.linalg.norm(predefine_encoding - encoding) * 100 < 50


def add_encoding_for_known_person(userid, call_userid, device_name, name, new_encodings, logger=logger_):
    """
    Appends new encoding to an existing known person's record in the pickle file.
    """
    global user_data

    filename = user_data[userid]["modelfile"]
    updated_records = []

    # Load existing records and update the correct object
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    if data["names"] == name:
                        # Append the new encodings to the correct record
                        data["encodings"].extend(new_encodings)
                    updated_records.append(data)
                except EOFError:
                    break

    # Rewrite the updated records back to the file
    with open(filename, "wb") as f:
        for record in updated_records:
            pickle.dump(record, f)

    logger.info(
        f"Added {len(new_encodings)} new encodings for known person '{name}'",
        user_id=userid,
        session_id="",
        log_module="Face_Rec",
        call_userid=call_userid,
        device_name=device_name,
    )


def compare_face(userid, call_userid, device_name,):
    """
    Compare face encodings and handle both known and unknown persons in the image.
    Updates or adds encodings for both categories accordingly.
    """
    global user_data
    print("user_data = ", user_data)
    print("userid = ", userid)

    try:
        # Load the model data from the pickle file
        print("user_data[userid]['modelfile'] = ", user_data[userid]["modelfile"])
        with open(user_data[userid]["modelfile"], "rb") as f:
            record = {}
            while True:
                try:
                    fulldata = pickle.load(f)
                    record_key = fulldata["names"] + "_" + str(fulldata["unique_id"])
                    record[record_key] = fulldata
                except EOFError:
                    break
    except FileNotFoundError:
        logger.error(f"Model file not found for user {userid}.")
        return {"known_faces": [], "unknown_faces": []}

    detected_faces = []
    unmatched_encodings = []
    matched_known_encodings = {}
    matched_unknown_encodings = {}

    # Compare each encoding
    for encoding in user_data[userid]["encodings"]:
        is_matched = False

        for record_key, fulldata in record.items():
            matches = [
                compare_with_L2_distance(predefine_encoding, encoding)
                for predefine_encoding in fulldata["encodings"]
            ]
            print("matches = ", matches)
            print("encoding = ", encoding)
            if any(matches):
                is_matched = True
                name = fulldata["names"]
                unique_id = str(fulldata["unique_id"])
                detected_faces.append({"name": name, "unique_id": unique_id})

                # Store matched encoding for the corresponding record
                if name == "unknown":
                    if unique_id not in matched_unknown_encodings:
                        matched_unknown_encodings[unique_id] = []
                    matched_unknown_encodings[unique_id].append(encoding)
                else:
                    if name not in matched_known_encodings:
                        matched_known_encodings[name] = []
                    matched_known_encodings[name].append(encoding)
                break

        if not is_matched:
            unmatched_encodings.append(encoding)

    # Append matched encodings to their respective known or unknown objects
    for unique_id, encodings in matched_unknown_encodings.items():
        add_encoding_for_unknown_exist_person(
            userid, call_userid, device_name, unique_id, encodings
        )

    for name, encodings in matched_known_encodings.items():
        add_encoding_for_known_person(userid, call_userid, device_name, name, encodings)

    # Register new unknown faces for unmatched encodings
    if unmatched_encodings:
        print("unmatched_encodings = ", unmatched_encodings)
        user_data[userid][
            "encodings"
        ] = unmatched_encodings  # Temporarily store unmatched

        friend_ids, unique_ids = add_unknown(
            user_data[userid]["image"],
            user_data[userid]["modelfile"],
            "unknown",
            userid,
            call_userid,
            device_name,
        )
        for unique_id in unique_ids:
            detected_faces.append({"name": "unknown", "unique_id": str(unique_id)})

    # Collect and return detected face information
    unique_known_faces = {
        face["unique_id"] for face in detected_faces if face["name"] != "unknown"
    }
    unique_unknown_faces = {
        face["unique_id"] for face in detected_faces if face["name"] == "unknown"
    }

    print(f"Known faces: {unique_known_faces}")
    print(f"Unknown faces: {unique_unknown_faces}")

    # Logging results
    # logger.info(
    #     f"Known faces detected: {unique_known_faces}",
    #     user_id=userid,
    #     session_id="",
    #     log_module="Face_Rec",
    #     call_userid=call_userid,
    #     device_name=device_name,
    # )
    # logger.info(
    #     f"Unknown faces detected: {unique_unknown_faces}",
    #     user_id=userid,
    #     session_id="",
    #     log_module="Face_Rec",
    #     call_userid=call_userid,
    #     device_name=device_name,
    # )

    return {
        "known_faces": list(unique_known_faces),
        "unknown_faces": list(unique_unknown_faces),
    }

def add_encoding_for_unknown_exist_person(userid, call_userid, device_name, unique_id, new_encodings, logger=logger_):
    """
    Appends new encoding to an existing unknown person's record in the pickle file.
    """
    global user_data

    filename = user_data[userid]["modelfile"]
    updated_records = []

    # Load existing records and update the correct object
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    if data["unique_id"] == unique_id:
                        # Append the new encodings to the correct record
                        data["encodings"].extend(new_encodings)
                    updated_records.append(data)
                except EOFError:
                    break

    # Rewrite the updated records back to the file
    with open(filename, "wb") as f:
        for record in updated_records:
            pickle.dump(record, f)

    logger.info(
        f"Added {len(new_encodings)} new encodings for unknown person with unique_id '{unique_id}'",
        userid=userid,
        session_id="",
        log_module="Face_Rec",
        call_userid=call_userid,
        device_name=device_name,
    )


def add_unknown(image, model_filename, name, userid, call_userid, device_name, logger=logger_):
    """
    Registers new unknown faces with unique IDs and stores them in the pickle file.
    """
    print("**************add_unknown****************************")
    global user_data

    new_encodings = user_data[userid]["encodings"]
    existing_records = []

    # Load existing records
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            while True:
                try:
                    existing_records.append(pickle.load(f))
                except EOFError:
                    break

    # Generate and store new records for each unknown face
    unique_ids = []
    friend_ids = []

    for encoding in new_encodings:
        unique_id = str(uuid.uuid4())  # Generate a new UUID
        friend_id = ""  # No friend ID for unknown faces

        # Create new record
        new_record = {
            "encodings": [encoding],  # Store each encoding as a list
            "names": name,
            "friend_id": friend_id,
            "unique_id": unique_id,
        }

        existing_records.append(new_record)
        unique_ids.append(unique_id)
        friend_ids.append(friend_id)

    # Write all records back to the file
    with open(model_filename, "wb") as f:
        for record in existing_records:
            pickle.dump(record, f)

    logger.info(
        f"Added {len(new_encodings)} new unknown faces to the model file with unique IDs: {unique_ids}",
        user_id=userid,
        session_id="",
        log_module="Face_Rec",
        call_userid=call_userid,
        device_name=device_name,
    )

    return friend_ids, unique_ids

user_data = {}

@app.route("/uploader", methods=["POST"])
def multi_face_rec_main():
    """
    Handle image upload, detect faces, generate encodings, compare faces, 
    and return the names of known and unknown faces with notifications.
    """
    start_time = time.time()
    print("*******************uploader request received***********************")

    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            print("!!!!!!!!!!!!!!!!!!!!!!! NO File Part!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "No file part"}, 400

        # Get the uploaded file
        file = request.files['file']

        # Get user data from the form
        userid = request.form.get("userid")
        device_name = request.form.get("device_name", None)
        call_userid = request.form.get("call_user_id", None)
        location_id = request.form.get("location_id", None)

        if not userid or not location_id:
            print("!!!!!!!!!!!!!!!!!!!!!!! Missing required parameters: userid or location_id!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "Missing required parameters: userid or location_id"}, 400

        print("Request params = ", {
                "userid": userid,
                "device_name": device_name,
                "call_userid": call_userid,
                "location_id": location_id
            })

        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Load the image using OpenCV
        print("file_path = ", file_path)
        image = cv2.imread(file_path)
        if image is None:
            print("!!!!!!!!!!!!!!!!!!!!!!! Invalid image file!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return {"success": False, "message": "Invalid image file"}, 400

        # # Check original dimensions
        # original_height, original_width = image.shape[:2]
        # print(f"Original dimensions (Height x Width): {original_height} x {original_width}")

        # # Rotate only if the image is in landscape orientation
        # if original_height < original_width:
        #     print("Image is in landscape mode, rotating to portrait...")
        #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # else:
        #     print("Image is already in portrait mode, no rotation needed.")
        #     image = image

        # Detect faces in the image
        # faces = MyFaceNet.mtcnn().detect_faces(image)
        faces = mediapipe_detect_faces(image)
        print("faces = ", faces)
        if not faces:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image=brighten_image(image)
                # cv2.imwrite('bighten_img.jpg',image)
                # faces = MyFaceNet.mtcnn().detect_faces(image)
                faces = mediapipe_detect_faces(image)
                if not faces:
                    print("!!!!!!!!!!!!!!!!!!!!!!! No faces detected in the image !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    return {"success": False, "message": "No faces detected in the image"}, 400

        # Generate face encodings
        encodings = encoding_generator(image, faces, logger)

        # Prepare the user model directory
        model_directory = f"user-model-for-id-{location_id}"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory, mode=0o775)

        # Prepare the user data dictionary
        modelfile = os.path.join(model_directory, "model-user-main.json")
        user_data[userid] = {
            "device_name": device_name,
            "call_userid": call_userid,
            "userid": userid,
            "filename": filename,
            "image": image,
            "encodings": encodings,
            "faces": faces,
            "modelfile": modelfile,
        }

        # Ensure the model file exists
        if not os.path.exists(modelfile):
            with open(modelfile, "wb") as f:
                pass

        # Perform face comparison
        result = compare_face(userid, call_userid, device_name)

        # Extract known and unknown data
        known_faces = result.get("known_faces", [])
        unknown_faces = result.get("unknown_faces", [])
        unknown_faces_count = len(unknown_faces)

        print("**************************************known_faces = ", known_faces)
        print("**************************************unknown_faces = ", unknown_faces)

        # Generate notification message
        notification_message = generate_notification(known_faces, unknown_faces_count, device_name)

        # Send notification
        response = send_notification(
            filename=file_path,
            userid=userid,
            device_id=device_name,
            call_userid=call_userid,
            slug=notification_message,  # Pass the generated notification message directly
            device_name=device_name,
            known_faces=known_faces,
            unknown_faces=unknown_faces,
        )

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Return the comparison results
        __array  = {
            "success": True,
            "data": response
        }
        return json.dumps(__array)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        traceback.print_exc()
        response = {
        "success": False,
        "message": error_message}
    
        return json.dumps(response), 500


def send_notification(filename, userid, call_userid, slug, device_name, device_id, known_faces, unknown_faces, logger=logger_):
    """
    Send a notification based on the detected faces.
    
    Args:
        filename (str): Path to the image file.
        userid (str): User ID of the device owner.
        call_userid (str): Call user ID (optional).
        slug (str): Notification slug.
        device_name (str): Name of the device.
        device_id (str): Device ID.
        known_faces (list): List of known face IDs.
        unknown_faces (list): List of unknown face IDs.

    Returns:
        str: API response or error message.
    """
    url = os.getenv("SEND_NOTIFICATION_URL")
    print("SEND_NOTIFICATION URL= ", url)
    # print("Image filepath = ", filename)
    # if not myurl:
    #     return "SEND_NOTIFICATION environment variable is not set."
        # Get a valid token from the auth_token module
    auth_token = get_valid_token(os.getenv("EMAIL"), os.getenv("PASSWORD"))

    headers = {
        'Authorization': f'Bearer {auth_token}'
    }

    try:
        with open(filename, 'rb') as image_file:
            files = {'image': (os.path.basename(filename), image_file, 'image/jpeg')}

            slug="visitor_recognition"

            # Create body similar to your example
            payload = {
                'slug': slug,
                'userid': userid,
                'device_id': device_id,
                'external_cam': '0',
            }
            payload[f'unique_id[]'] = known_faces + unknown_faces

            # # Add each unique_id[] to the body
            # for idx, face_id in enumerate(known_faces + unknown_faces):
            #     payload[f'unique_id[]'] = face_id

            print("send notification body = ", payload)
            # print("header = ", headers)

            response = requests.request("POST", url, headers=headers, data=payload, files=files)

            logger.info(
                f'Notification sent with message: "{slug}" and data: {payload}',
                extra={
                    "user_id": userid,
                    "session_id": "",
                    "log_module": "Face_Rec",
                    "call_userid": call_userid,
                    "device_name": device_name,
                }
            )
            print("send notification api = ", response.text)
            logger.info(f'Response of send/notification API response: {response}')
            logger.info(f'Response of send/notification API response.text: {response.text}')

            logger.info(f'Response of send/notification API response.status_code: {response.status_code}')

            return response.text if response.status_code == 200 else f"Notification failed: {response.status_code} {response.text}"

    except Exception as e:
        logger.error(f"Error in send_notification: {str(e)}", exc_info=True)
        return "Notification failed due to an exception."


def generate_notification(known_faces, unknown_count, device_name):
    """
    Generate a notification message based on known and unknown faces detected.

    Args:
        known_faces (list): List of names of known faces.
        unknown_count (int): Number of unknown faces detected.
        device_name (str): Name of the doorbell/device.

    Returns:
        str: Notification message.
    """
    # Ensure device_name is meaningful
    device_name = device_name or "the doorbell"

    # Count the number of known faces
    known_count = len(known_faces)

    # Generate notifications based on scenarios
    if known_count == 1 and unknown_count == 0:
        return f"{known_faces[0]} is at the door {device_name}."
    elif known_count == 0 and unknown_count == 1:
        return f"A new visitor is at the door {device_name}."
    elif known_count > 1 and unknown_count == 0:
        return f"{' & '.join(known_faces)} are at the door {device_name}."
    elif known_count == 0 and unknown_count > 1:
        return f"New visitors are at the door {device_name}."
    elif known_count == 1 and unknown_count == 1:
        return f"{known_faces[0]} along with a new visitor is at the door {device_name}."
    elif known_count == 1 and unknown_count > 1:
        return f"{known_faces[0]} along with some new visitors is at the door {device_name}."
    
    # Fallback for unexpected cases
    return f"The iamge is not clear {device_name}."

# def send_notification(filename, userid,call_userid, person_name, friend_id, unique_id, device_name):
#     '''takes the message and send this to irvinei db'''
#     # Send a notification with the specified data
#     print("send notification")
#     myurl = os.getenv("SEND_NOTIFICATION")#
#     # myurl = "https://beta9.irvinei.com/api/send-notification"
#     # myurl = "https://beta.irvinei.com/api/api/send-notification"
#     # myurl = "https://beta9.irvinei.com/api/v1/send-notification"
#     files = {'image': open(filename, 'rb')}
#     body = {
#         "userid": userid,
#         "message": person_name + " is at the door",
#         "friend_id": friend_id,
#         "unique_id": unique_id,
#         "device_name": device_name
#     }

#     print("body = ", body)

#     print("sending request to send_request")
   
#     getdata = requests.post(myurl, data=body, files=files)
#     print("get response")

#     logger.info(
#                 f'Notification sent with data {body}',
#                 user_id=userid,
#                 session_id='',
#                 log_module='Face_Rec',
#                 call_userid=call_userid,
#                 device_name=device_name
#                 )
#     api_response=getdata.text
#     # print('....',getdata.text)
#     print("status code = ", getdata.status_code)
#     if getdata.status_code == 200:
#         api_response=api_response.strip()
#         print('inside')
#     return api_response


if __name__ == '__main__':
   
    app.run(host='0.0.0.0', port=1000)

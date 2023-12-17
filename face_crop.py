import random as rd
import cv2
import os

count = 0
def detect_and_save_faces(name,image_path, output_folder):
    global count
    image = cv2.imread(image_path)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)0
        )

        print(f"[INFO] Found {len(faces)} Faces in {image_path} ")

        for (x, y, w, h) in faces:
            face_roi = image[y:y + h, x:x + w]
            cv2.imwrite(output_folder+"/"+str(rd.randint(0,10000))+f"_{name}.jpeg", face_roi)
            count += 1
            cv2.imshow("Detected Faces", face_roi)
            cv2.waitKey(3)
            cv2.destroyAllWindows()
    except:
        print("Error")

if __name__ == "__main__":
    name = "pakhi"
    dataset_path = "./dataset/pakhi"
    output_path = f"./output_faces/{name}"

    os.makedirs(output_path, exist_ok=True)

    image_files = os.listdir(dataset_path)
    for count,img_file in enumerate(image_files):
        img_path = os.path.join(dataset_path, img_file)
        detect_and_save_faces(name,img_path, output_path)

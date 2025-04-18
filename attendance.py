import tkinter as tk
import csv
import cv2 # type: ignore
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

window = tk.Tk()
window.title("STUDENT ATTENDANCE USING FACE RECOGNITION SYSTEM")
window.geometry('800x500')

dialog_title = 'QUIT'
dialog_text = "are you sure?"
window.configure(background='green')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def clear():
    std_name.delete(0, 'end')
    res = ""
    label4.configure(text=res)


def clear2():
    std_number.delete(0, 'end')
    res = ""
    label4.configure(text=res)


def takeImage():
    name = (std_name.get())
    Id = (std_number.get())
    if name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                # store each student picture with its name and id
                cv2.imwrite("TrainingImages\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + h])
                cv2.imshow('FACE RECOGNIZER', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # stop the camera when the number of picture exceed 50 pictures for each student
            if sampleNum > 50:
                break

        cam.release()
        cv2.destroyAllWindows()
        # print the student name and id after a successful face capturing
        res = 'Student details saved with: \n Matric number : ' + Id + ' and  Full Name: ' + name

        row = [Id, name]

        with open('studentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        label4.configure(text=res)
    else:

        if name.isalpha():
            res = "Enter correct Matric Number"
            label4.configure(text=res)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        if not imagePath.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def trainImage():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("Trainner.yml")
    res = "Image Trained"
    label4.configure(text=res)


def trackImage():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    try:
        df = pd.read_csv("studentDetails.csv")
    except Exception as e:
        label4.configure(text=f"Error reading student details: {e}")
        return

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = pd.DataFrame(columns=["Id", "Name", "Date", "Time"])
    recorded_ids = set()

    while True:
        ret, img = cam.read()
        if not ret:
            label4.configure(text="Failed to access camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print(f"Detected ID: {Id}, Confidence: {conf}")

            if conf < 70:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M')

                name_row = df.loc[df['ID'] == Id]['NAME'].values
                name = name_row[0] if len(name_row) > 0 else "Unknown"

                if Id not in recorded_ids:
                    print(f"Recording attendance for: {Id}, {name}")
                    recorded_ids.add(Id)
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

                    try:
                        with open('AttendanceFile.csv', 'a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([Id, name, date, timeStamp])
                        label4.configure(text=f"Attendance Recorded for {name}")
                    except Exception as e:
                        label4.configure(text=f"File write error: {e}")
            else:
                print("Unknown face detected")
                label4.configure(text="Face not recognized")

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, str(Id), (x, y-10), font, 1, (255, 255, 255), 2)

        cv2.imshow('Tracking...', img)
        if cv2.waitKey(1) == ord('q') or len(recorded_ids) > 0:
            break

    cam.release()
    cv2.destroyAllWindows()





label1 = tk.Label(window, background="green", fg="black", text="Name :", width=10, height=1,
                  font=('Helvetica', 16))
label1.place(x=83, y=40)
std_name = tk.Entry(window, background="yellow", fg="black", width=25, font=('Helvetica', 14))
std_name.place(x=280, y=41)
label2 = tk.Label(window, background="green", fg="black", text="Matric Number :", width=14, height=1,
                  font=('Helvetica', 16))
label2.place(x=100, y=90)
std_number = tk.Entry(window, background="yellow", fg="black", width=25, font=('Helvetica', 14))
std_number.place(x=280, y=91)

clearBtn1 = tk.Button(window, background="red", command=clear, fg="white", text="CLEAR", width=8, height=1,
                      activebackground="red", font=('Helvetica', 10))
clearBtn1.place(x=580, y=42)
clearBtn2 = tk.Button(window, background="red", command=clear2, fg="white", text="CLEAR", width=8,
                      activebackground="red", height=1, font=('Helvetica', 10))
clearBtn2.place(x=580, y=92)

label3 = tk.Label(window, background="green", fg="red", text="Notification", width=10, height=1,
                  font=('Helvetica', 20, 'underline'))
label3.place(x=320, y=155)
label4 = tk.Label(window, background="yellow", fg="black", width=55, height=4, font=('Helvetica', 14, 'italic'))
label4.place(x=95, y=205)

takeImageBtn = tk.Button(window, command=takeImage, background="yellow", fg="black", text="CAPTURE IMAGE",
                         activebackground="red",
                         width=15, height=3, font=('Helvetica', 12))
takeImageBtn.place(x=130, y=360)
trainImageBtn = tk.Button(window, command=trainImage, background="yellow", fg="black", text="TRAINED IMAGE",
                          activebackground="red",
                          width=15, height=3, font=('Helvetica', 12))
trainImageBtn.place(x=340, y=360)
trackImageBtn = tk.Button(window, command=trackImage, background="yellow", fg="black", text="TRACK IMAGE", width=12,
                          activebackground="red", height=3, font=('Helvetica', 12))
trackImageBtn.place(x=550, y=360)

window.mainloop()

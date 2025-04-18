<h2 align="center"> STUDENT ATTENDANCE USING FACIAL RECOGNITION SYSTEM </h2>

<h4 align="center"><i>This is an automatic student attendance system using face recognition. The aim is to automate the process of attendance maintenance.</i></h4><br>


## 👩 FACE RECOGNITION 

Face recognition is a biometric recognition technique.
Biometric recognition is an information system that allows the identification of a person based on some of its main physiological and behavioral characteristics.
Face recognition is a broad problem of identifying or verifying people in photographs and videos, a process comprised of detection, alignment, feature extraction, and a recognition task
It has 4 steps which are :
1. **Face Detection** 
2. **Data Gathering**
3. **Data Comparison**
4. **Face Recognition** 


## ⚠️ TECHONLOGY USED

* [OPENCV-CONTRIB](https://opencv.org/about/)

* [TKINTER](https://docs.python.org/3/library/tkinter.html)

* [HAAR-CASCADE CLASSIFIER](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

* [LocalBinaryPatternHistogram (LBPH) recognizer](https://docs.opencv.org/master/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html)

* [trainner.yml](https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html)

* [PIL](https://pillow.readthedocs.io/en/stable/)


## ⚙️ HOW THE SYSTEM WORKS?

This system works accordingly with a series of step explained below:

1. **DATA COLLECTION**:
<br>

![capture](https://github.com/brahmjyot/Student-attendance-system-using-facial-recognition-Python-OpenCV-/blob/2b9d289871765713fa3dc758f4a6c4e562ccf7fa/Screenshot%202025-04-18%20095953.png)




The student interact with the system through the Graphical User Interface (GUI) above.
The first step the student has to enter his details(Name and ID) this details will be stored in a csv file **'StudentDetailss.csv'**, the ID is Matric Number on the GUI.
second step, the student will click on the **CAPTURE IMAGE** button to capture his faces, here 50pictures of the student will be taken and stored in the **TrainingImages** Folder.  The **haar-cascadeclassifier** file to detect faces through the video stream while the student face is being captured.\
The notification board will print out the student details after a succesfull data collection.


<br>


2. **IMAGE TRAINED**

The student has to click on the  **TRAIN IMAGE** button which will link his details, face features to the **LBPHrecognizer** to ease further face recognition,
the recognizer will save the face features in the **trainner.yml** and "IMAGE TRAINED" will be printed on the GUI notification board after a successfull linkage.

![capture3](https://github.com/brahmjyot/Student-attendance-system-using-facial-recognition-Python-OpenCV-/blob/2b9d289871765713fa3dc758f4a6c4e562ccf7fa/Screenshot%202025-04-17%20225733.png)

3. **FACE TRACKING**

The student has to click on the **TRACK IMAGE** button to allow the face recognizer to track his face through a video stream, when trhe system successfully recognize the student face, his details will show and "ATTENDANCE UPDATED" will be printed out otherwise , the ID will be Unkown and "ID UNKOWN, ATTENDANCE NOT UPDATED" will be printed out.
Simustenously, a csv file **AttendanceFile.csv'** will be updated with the ID,NAME of the student and DATE Aand TIME at which his face has recognized.
the Unkown face captured will be store in the **UnkownImages** folder.<br>


![capture3](https://github.com/brahmjyot/Student-attendance-system-using-facial-recognition-Python-OpenCV-/blob/2b9d289871765713fa3dc758f4a6c4e562ccf7fa/Screenshot%202025-04-18%20100103.png)


## 🔑 PEREQUISITES

All the dependencies and required libraries are included in the file `requirements.txt` 


## 🚀 INSTALLATION

Clone the repo\
```$ git clone https://github.com/memudualimatou/STUDENT-ATTENDANCE-USING-FACIAL-RECOGNITION-SYSTEM-OPENCV.git```


Change your directory to the cloned repo and create a Python virtual environment named 'test'

```$ python -m venv test```

Now you have made a virtual environment to run your code now you just have to activate the virtual environment 

```$ source venv/Scripts/activate```

Now you should see your shell change to something like ```(venv) user@PC MINGW64 ~/your-project-folder``` 

Now you have to first install the setup tools or if already present then you can upgrade it using this 

```$ pip install --upgrade pip setuptools wheel``` 


Now, run the following command in your Terminal/Command Prompt to install the libraries required

```$ pip install -r requirements.txt```


When you will run, it will create a face recognition model with the name **Trainner.yml** 

## 👏 And it's done!
Feel free to mail me for any doubts/query ✉️ brahm140803@gmail.com

##  🤝 Contribution
Feel free to file a new issue with a respective title and description on the the Student attendance repository. If you already found a solution to your problem, I would love to review your pull request!

## ❤️ Owner
Made with ❤️  by Brahmjyot Singh

import customtkinter as ct
from tkinter import ttk
from tkinter import *
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression




class LOADING:
    def __init__(self):

        self.root = Tk()
        image = PhotoImage(file="C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\f2.png")
        height = 530
        width = 730
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        self.label = Label(self.root, image=image, text="Welcome to Machine Learning!")
        self.label.pack()

        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate', maximum=100,
                                        style='Custom.Horizontal.TProgressbar')
        self.progress.place(x=(width - 300) // 2, y=height - 50)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("Custom.Horizontal.TProgressbar", troughcolor='#5e5a66',
                        background='#1a1625')

        self.loading_label = Label(self.root, text="Loading... 0%", fg="white", bg="#1a1525")
        self.loading_label.place(x=(width - 100) // 2, y=height - 80)

        self.root.overrideredirect(True)

        self.root.resizable(False, False)

        self.increase_progress()

        self.root.mainloop()

    def increase_progress(self):
        progress_value = self.progress["value"]
        if progress_value < 100:
            progress_value += 1
            self.progress["value"] = progress_value
            self.loading_label.config(text=f"Loading... {progress_value}%", fg="white")
            self.root.after(5, self.increase_progress)
        else:
            self.root.destroy()
            GUI()


class GUI:

    def __init__(self):

        ct.set_appearance_mode("system")
        self.root = ct.CTk()
        self.root.title("Machine Learning Diabetic Prediction")
        self.root.geometry("2000x1200+0+0")
        self.root.configure(fg_color="#1a1625")
        self.root.iconbitmap("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\loader.ico")
        self.root.resizable()
        self.helloLabel = ct.CTkLabel(self.root,
                                      text="Welcome to \n \n Machine Learning Project\n \n \n \n Ahmed Islam \n \n "
                                           "Abdalla Basem \n \n Belal Fathy",
                                      fg_color="#1a1625", font=("Arial", 30, "bold"))
        self.helloLabel.place(x=700, y=200)

        self.models()

        self.frame2 = None
        self.frame1 = None
        self.frame = None

        self.leftLabel = ct.CTkLabel(self.root, text="\n \n \nSettings", font=("Montserrat", 30, "bold"), anchor="n",
                                     fg_color="#5e5a66", width=250, height=1400, padx=0, pady=0)
        self.leftLabel.pack(side="left", fill="y")

        self.predictButoon = ct.CTkButton(self.leftLabel, text="Predict", fg_color=self.leftLabel.cget("fg_color"),
                                          bg_color=self.leftLabel.cget("fg_color"), hover_color="#908d96", width=230,
                                          height=50, command=self.homeFrame, corner_radius=15)
        self.predictButoon.place(x=10, y=300)

        self.staticsButton = ct.CTkButton(self.leftLabel, text="static", fg_color=self.leftLabel.cget("fg_color"),
                                          bg_color=self.leftLabel.cget("fg_color"), hover_color="#908d96", width=230,
                                          height=50, command=self.frame1Statics, corner_radius=15)
        self.staticsButton.place(x=10, y=400)

        self.dataButton = ct.CTkButton(self.leftLabel, text="Data", fg_color=self.leftLabel.cget("fg_color"),
                                       bg_color=self.leftLabel.cget("fg_color"), hover_color="#908d96", width=230,
                                       height=50, corner_radius=15, command=self.frame2Data)
        self.dataButton.place(x=10, y=500)


        self.count = 0
        self.count1 = 0
        self.text = ''
        self.text50 = ''
        self.id = 2779

        self.root.mainloop()

    # all alone no any help from any website
    def saveAction(self):
        self.models()  # to update the accuracy and add the new entry to my csv

        # I use those for slider the text in reportLabel and StatusLabel
        self.count = 0
        self.text = ""
        self.count1 = 0
        self.text50 = ""

        new_data = pd.DataFrame({
            "Pregnancies": [self.Pregnancies.get()],
            "Glucose": [self.Glucose.get()],
            "BloodPressure": [self.BloodPressure.get()],
            "SkinThickness": [self.SkinThickness.get()],
            "Insulin": [self.Insulin.get()],
            "BMI": [self.BMI.get()],
            "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction.get()],
            "Age": [self.Age.get()]

        })
        for column in new_data.columns:
            if new_data[column].dtype == '':
                new_data[column] = 0






        self.predicationKNN = self.knn_classifier.predict(new_data)
        self.predicationNV = self.nb_classifier.predict(new_data)
        self.predicationLOG = self.logistic_regression.predict(new_data)

        if self.predicationKNN == 0:
            knnText = "KNN Result : you are healthy"
        else:
            knnText = "KNN Result : you are sick"
        if self.predicationNV == 0:
            NVText = "NAIVE BIASE Result : you are healthy"
        else:
            NVText = "NAIVE BIASE Result : you are sick"
        if self.predicationLOG == 0:
            LOGText = "LOGISTIC REGRESSION Result : you are healthy"
        else:
            LOGText = "LOGISTIC REGRESSION Result : you are sick"

        # choosing the best accuracy to apply bcs it's a medical app
        self.maxi = max(self.accuracy_logistic, self.accuracy_nb, self.accuracy_knn)

        if self.maxi == self.accuracy_knn:
            self.predication = self.knn_classifier.predict(new_data)

        elif self.maxi == self.accuracy_nb:
            self.predication = self.nb_classifier.predict(new_data)

        elif self.maxi == self.accuracy_logistic:
            self.predication = self.logistic_regression.predict(new_data)

        new_data['Outcome'] = self.predication
        print(self.predication)
        print(self.maxi)
        # append the new data from entry
        self.df = self.df._append(new_data, ignore_index=True)
        self.df.to_csv("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\Healthcare-Diabetes.csv", index=False)

        # slider the yext in mentioned labels
        self.text3 = "best accuracy Result : you are healthy"
        self.text4 = "best accuracy Result : you are diabetic"
        self.slider0(knnText+"\n\n"+NVText+"\n\n"+LOGText)
        if self.predication == 1:
            self.slider1(self.text4)

        elif self.predication == 0:
            self.slider1(self.text3)

    def changeOnHover(self, e):
        # this makes me so happy when I make it , instead of reinitialize all entries I make it in one line it takes
        # around 4h to bring this idea, but I think it deserves it , customtkinter decontamination is not available a
        # lot so i type e.widget.configure and chatgpt add for me master I wasn't;t know it
        e.widget.master.configure(border_width=2, border_color="#382bf0")

    def changeOnHover2(self, e):
        e.widget.master.configure(border_width=0)

    # sliders , it was a good idea bcs it add a pretty to app , some videos on YouTube and geekforgeeks
    def slider0(self, text):


        if self.count >= len(text):
            return
        else:
            self.text = self.text + text[self.count]
            self.reportLabel.configure(text=self.text)
        self.count += 1
        self.reportLabel.after(100, lambda: self.slider0(text))

    def slider1(self, text):

        if self.count1 >= len(text):
            return
        else:
            self.text50 = self.text50 + text[self.count1]
            self.statusLabel.configure(text=self.text50)
        self.count1 += 1
        self.statusLabel.after(100, lambda: self.slider1(text))

    # this frame to show the accuracy of models and graph
    def frame1Statics(self):
        self.staticsButton.configure(fg_color="#908d96")
        self.predictButoon.configure(fg_color="#5e5a66")
        self.dataButton.configure(fg_color="#5e5a66")
        if self.helloLabel:
            self.helloLabel.destroy()
        if self.frame is not None:
            self.frame.destroy()
        if self.frame2 is not None:
            self.frame2.destroy()
        self.frame1 = ct.CTkFrame(self.root, fg_color=self.root.cget("fg_color"), width=2000, height=1400,
                                  corner_radius=0)
        self.frame1.pack()
        self.knnAcLabel = ct.CTkLabel(self.frame1,
                                      text="KNN ACCURACY \n \n" + str(round((self.accuracy_knn * 100), 3)) + " %",
                                      font=("arial", 15), fg_color="#5e5a66", width=200, height=100, corner_radius=30)
        self.knnAcLabel.place(x=150, y=100)
        image1 = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\knnScatter.png")
        ct_image1 = ct.CTkImage(image1, size=(230, 230))
        self.knnPLLabel = ct.CTkLabel(self.frame1, text="", image=ct_image1, font=("arial", 15), fg_color="#5e5a66", width=300,
                                      height=400, corner_radius=30)
        self.knnPLLabel.place(x=100, y=250)

        self.nvAcLabel = ct.CTkLabel(self.frame1, text="Naive Biase ACCURACY \n \n" + str(
            round((self.accuracy_nb * 100), 3)) + " %", font=("arial", 15), fg_color="#5e5a66", width=200, height=100,
                                     corner_radius=30)
        self.nvAcLabel.place(x=525, y=100)
        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\nvScatter.png")
        ct_image = ct.CTkImage(image,size=(230, 230))
        self.nvPLLabel = ct.CTkLabel(self.frame1, image=ct_image, text="", font=("arial", 15), fg_color="#5e5a66", width=300,
                                     height=400, corner_radius=30)
        self.nvPLLabel.place(x=500, y=250)

        self.logAcLabel = ct.CTkLabel(self.frame1, text="Logistic ACCURACY \n \n" + str(
            round((self.accuracy_logistic * 100), 3)) + " %", font=("arial", 15), fg_color="#5e5a66", width=200,
                                      height=100, corner_radius=30)
        self.logAcLabel.place(x=950, y=100)
        image2 = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\logScatter.png")
        ct_image2 = ct.CTkImage(image2, size=(230, 230))
        self.logPLLabel = ct.CTkLabel(self.frame1, text="", image=ct_image2, font=("arial", 15), fg_color="#5e5a66", width=300,
                                      height=400, corner_radius=30)
        self.logPLLabel.place(x=900, y=250)

    # to show csv file in table
    def frame2Data(self):

        self.dataButton.configure(fg_color="#908d96")
        self.staticsButton.configure(fg_color="#5e5a66")
        self.predictButoon.configure(fg_color="#5e5a66")
        if self.helloLabel:
            self.helloLabel.destroy()
        if self.frame is not None:
            self.frame.destroy()
        if self.frame1 is not None:
            self.frame1.destroy()
        self.frame2 = ct.CTkFrame(self.root, fg_color=self.root.cget("fg_color"), width=2000, height=1400,
                                  corner_radius=0)
        self.frame2.pack()
        self.tableLabel = ct.CTkLabel(self.frame2, text="Table", font=("arial", 15), fg_color="#5e5a66", width=1000,
                                      height=600, corner_radius=30)
        self.tableLabel.place(x=150, y=100)

        # from youtube videos
        df = pd.read_csv("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\Healthcare-Diabetes.csv")

        # from GitHub and I apply some edit like font and change the color
        style = ttk.Style()

        style.theme_use("default")

        style.configure("Treeview",
                        background="#2f2b3a",
                        foreground="#908d96",
                        rowheight=25,
                        fieldbackground="#343638",
                        bordercolor="#908d96",
                        borderwidth=0,
                        font=("Montserrat", 10))
        style.map('Treeview', background=[('selected', '#908d96')])

        style.configure("Treeview.Heading",
                        background="#908d96",
                        foreground="#e1e1e1",
                        relief="flat",
                        font=("Montserrat", 12))
        style.map("Treeview.Heading",
                  background=[('active', '#908d96')])

        # read in website about it
        tree = ttk.Treeview(self.tableLabel, style="Custom.Treeview", height=25)

        columns = list(df.columns)
        tree["columns"] = columns

        for column in columns:
            tree.heading(column, text=column)

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        for col in columns:
            tree.column(col, width=90)

        tree.place(x=70, y=60)

    def homeFrame(self):

        if self.helloLabel:
            self.helloLabel.destroy()

        self.predictButoon.configure(fg_color="#908d96")
        self.dataButton.configure(fg_color="#5e5a66")
        self.staticsButton.configure(fg_color="#5e5a66")

        if self.frame1 is not None:
            self.frame1.destroy()

        if self.frame2 is not None:
            self.frame2.destroy()

        self.frame = ct.CTkFrame(self.root, fg_color=self.root.cget("fg_color"), width=2000, height=1400,
                                 corner_radius=0)
        self.frame.pack()

        self.reportLabel = ct.CTkLabel(self.frame, text="", font=("Arial", 20, "bold"), text_color="white",
                                       fg_color="#5e5a66", width=500, height=350, corner_radius=30, pady=10,
                                       justify="center")
        self.reportLabel.place(x=100, y=100)

        self.statusLabel = ct.CTkLabel(self.frame, text="status", fg_color="#5e5a66", width=320, height=170,
                                       corner_radius=30)
        self.statusLabel.place(x=180, y=500)

        self.button = ct.CTkButton(self.frame, text="Save", fg_color="#382bf0", width=100, height=50, corner_radius=15,
                                   hover_color="#7a5af5", command=self.saveAction)
        self.button.place(x=1150, y=710)

        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\refresh-page-option.png")
        image = image.resize((60, 60))
        ct_image = ct.CTkImage(image)
        self.button2 = ct.CTkButton(self.frame, image=ct_image, text="", corner_radius=80, width=10, height=10,
                                    command=self.test1, fg_color=self.frame.cget("fg_color"))
        self.button2.place(x=1200, y=40)
        self.button2.bind("<Enter>", self.on_enter)
        self.button2.bind("<Leave>", self.on_leave)

        self.Pregnancies = ct.CTkEntry(self.frame, placeholder_text="Pregnancies", font=("Arial", 15, "bold"),
                                       width=300, height=40, corner_radius=15, fg_color="#5e5a66", text_color="white",
                                       border_width=0)
        self.Pregnancies.place(x=750, y=70)
        self.Pregnancies.bind("<Enter>", self.changeOnHover)
        self.Pregnancies.bind("<Leave>", self.changeOnHover2)

        self.Glucose = ct.CTkEntry(self.frame, placeholder_text="Glucose", font=("Arial", 15, "bold"), width=300,
                                   height=40, corner_radius=15, fg_color="#5e5a66", text_color="white", border_width=0)
        self.Glucose.place(x=850, y=150)
        self.Glucose.bind("<Enter>", self.changeOnHover)
        self.Glucose.bind("<Leave>", self.changeOnHover2)

        self.BloodPressure = ct.CTkEntry(self.frame, placeholder_text="Blood Pressure", font=("Arial", 15, "bold"),
                                         width=300, height=40, corner_radius=15, fg_color="#5e5a66", text_color="white",
                                         border_width=0)
        self.BloodPressure.place(x=750, y=230)
        self.BloodPressure.bind("<Enter>", self.changeOnHover)
        self.BloodPressure.bind("<Leave>", self.changeOnHover2)

        self.SkinThickness = ct.CTkEntry(self.frame, placeholder_text="Skin Thickness", font=("Arial", 15, "bold"),
                                         width=300, height=40, corner_radius=15, fg_color="#5e5a66", text_color="white",
                                         border_width=0)
        self.SkinThickness.place(x=850, y=310)
        self.SkinThickness.bind("<Enter>", self.changeOnHover)
        self.SkinThickness.bind("<Leave>", self.changeOnHover2)

        self.Insulin = ct.CTkEntry(self.frame, placeholder_text="Insulin", font=("Arial", 15, "bold"), width=300,
                                   height=40, corner_radius=15, fg_color="#5e5a66", text_color="white", border_width=0)
        self.Insulin.place(x=750, y=390)
        self.Insulin.bind("<Enter>", self.changeOnHover)
        self.Insulin.bind("<Leave>", self.changeOnHover2)

        self.BMI = ct.CTkEntry(self.frame, placeholder_text="BMI", font=("Arial", 15, "bold"), width=300, height=40,
                               corner_radius=15, fg_color="#5e5a66", text_color="white", border_width=0)
        self.BMI.place(x=850, y=470)
        self.BMI.bind("<Enter>", self.changeOnHover)
        self.BMI.bind("<Leave>", self.changeOnHover2)

        self.DiabetesPedigreeFunction = ct.CTkEntry(self.frame, placeholder_text="Diabetes Pedigree Function",
                                                    font=("Arial", 15, "bold"), width=300, height=40, corner_radius=15,
                                                    fg_color="#5e5a66", text_color="white", border_width=0)
        self.DiabetesPedigreeFunction.place(x=750, y=550)
        self.DiabetesPedigreeFunction.bind("<Enter>", self.changeOnHover)
        self.DiabetesPedigreeFunction.bind("<Leave>", self.changeOnHover2)

        self.Age = ct.CTkEntry(self.frame, placeholder_text="Age", font=("Arial", 15, "bold"), width=300, height=40,
                               corner_radius=15, fg_color="#5e5a66", text_color="white", border_width=0)
        self.Age.place(x=850, y=630)
        self.Age.bind("<Enter>", self.changeOnHover)
        self.Age.bind("<Leave>", self.changeOnHover2)

    def on_enter(self, e):
        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\refresh-page-option (1).png")
        image = image.resize((60, 60))
        ct_image = ct.CTkImage(image)
        self.button2.configure(image=ct_image, fg_color=self.frame.cget("fg_color"))

    def on_leave(self, e):
        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\refresh-page-option.png")
        image = image.resize((60, 60))
        ct_image = ct.CTkImage(image)
        self.button2.configure(image=ct_image, fg_color=self.frame.cget("fg_color"))

    def test1(self):
        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\time-left.png")
        image = image.resize((60, 60))
        ct_image = ct.CTkImage(image)
        self.button2.configure(image=ct_image)
        self.button2.after(300, self.test2)

    def test2(self):
        image = Image.open("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\refresh-page-option.png")
        image = image.resize((60, 60))
        ct_image = ct.CTkImage(image)
        self.button2.configure(image=ct_image)
        self.Pregnancies.delete(0, "end")
        self.BMI.delete(0, "end")
        self.BloodPressure.delete(0, "end")
        self.DiabetesPedigreeFunction.delete(0, "end")
        self.Glucose.delete(0, "end")
        self.Age.delete(0, "end")
        self.Insulin.delete(0, "end")
        self.SkinThickness.delete(0, "end")

    # abdalla basem not me
    def models(self):
        self.df = pd.read_csv("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject2\\Healthcare-Diabetes.csv")
        print(self.df.isnull().sum())

        x = self.df.drop(['Outcome', 'Id'], axis=1)  # Features
        y = self.df['Outcome']  # Target variable

        x.replace(0, np.nan, inplace=True)
        column_means = x.mean()
        x.fillna(column_means, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Naive Bayes
        self.nb_classifier = GaussianNB()
        self.nb_classifier.fit(X_train, y_train)
        self.y_pred_nb = self.nb_classifier.predict(X_test)
        self.accuracy_nb = accuracy_score(y_test, self.y_pred_nb)
        print("Naive Bayes Accuracy:", self.accuracy_nb)

        plt.figure()  # Create a new figure
        plt.scatter(X_test.Glucose, self.y_pred_nb, color='blue')
        plt.xlabel('Features')
        plt.ylabel('Predictions')
        plt.title('Naive Bayes Scatter Plot')
        plt.savefig("nvScatter.png")

        # KNN
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.knn_classifier.fit(X_train, y_train)
        self.y_pred_knn = self.knn_classifier.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, self.y_pred_knn)
        print("KNN Accuracy:", self.accuracy_knn)

        plt.figure()
        plt.scatter(X_test.Glucose, self.y_pred_knn, color='red')
        plt.xlabel('Features')
        plt.ylabel('Predictions')
        plt.title('KNN Scatter Plot')
        plt.savefig("knnScatter.png")




        self.logistic_regression = LogisticRegression()
        self.logistic_regression.fit(X_train, y_train)
        self.y_pred_logistic = self.logistic_regression.predict(X_test)
        self.accuracy_logistic = accuracy_score(y_test, self.y_pred_logistic)
        print("Logistic Regression Accuracy:", self.accuracy_logistic)

        plt.figure()
        plt.scatter(X_test.Glucose, self.y_pred_logistic, color='green')
        plt.xlabel('Features')
        plt.ylabel('Predictions')
        plt.title('Logistic Regression Scatter Plot')
        plt.savefig("logScatter.png")

LOADING()
print("hello world")
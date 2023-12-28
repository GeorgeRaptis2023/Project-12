import face_recognition,tkinter,tkinter.filedialog,time,cv2,os,os.path
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image,ImageTk
class App():
    def __init__(self, root):
        self.root=root
        self.root.title("Ai Image Recognition")
        root.geometry("500x500")

        self.frame=tkinter.Frame(root)
        self.frame.grid(row=0,column=0)

        self.title=tkinter.Label(self.frame,text="Compare")
        self.title.grid(row=0,column=0)
        
        self.button=tkinter.Button(self.frame,text="click",command=self.compare)
        self.button.grid(row=0,column=1)
        
        self.button_photo = tkinter.Button(self.frame, text="Capture Image", command=self.capture_image)
        self.button_photo.grid(row=0, column=2)
        self.name_counter=1
        self.vid = cv2.VideoCapture(0)

        self.comparison=tkinter.Label(self.frame,text="")
        self.comparison.grid(row=12,column=0,columnspan=8)

        self.result1=tkinter.Label(self.frame,text="")
        self.result1.grid(row=3,column=0,columnspan=4)

        self.face1=tkinter.Label(self.frame)
        self.face1.grid(row=4,column=0,columnspan=2,rowspan=2)

        self.face2=tkinter.Label(self.frame)
        self.face2.grid(row=4,column=2,columnspan=2,rowspan=2)

        self.tolerance1 = tkinter.Scale(self.frame,from_=1, to=100, tickinterval=5, orient="horizontal")
        self.tolerance1.grid(row=0,column=3)

        self.option_numjitters1 = tkinter.Scale(self.frame,from_=1, to=20, tickinterval=5,orient="horizontal")
        self.option_numjitters1.grid(row=0,column=4)
        
        self.option_size1 = tkinter.Listbox(self.frame,height=3,exportselection=False) 
        self.option_size1.grid(row=2,column=1)
        self.option_size1.insert(1,"large")
        self.option_size1.insert(2,"small")
        self.option_type1 = tkinter.Listbox(self.frame,heigh=3,exportselection=False) 
        self.option_type1.grid(row=2,column=2)
        self.option_type1.insert(1,"hog")
        self.option_type1.insert(2,"cnn")

        self.result2=tkinter.Label(self.frame,text="")
        self.result2.grid(row=9,column=0,columnspan=4)

        self.face3=tkinter.Label(self.frame)
        self.face3.grid(row=10,column=0,columnspan=2,rowspan=2)
        
        self.face4=tkinter.Label(self.frame)
        self.face4.grid(row=10,column=2,columnspan=2,rowspan=2)


        self.tolerance2 = tkinter.Scale(self.frame,from_=1, to=100, tickinterval=5, orient="horizontal")
        self.tolerance2.grid(row=6,column=2)

        self.option_numjitters2 = tkinter.Scale(self.frame,from_=1, to=20, tickinterval=5,orient="horizontal")
        self.option_numjitters2.grid(row=6,column=3)
        
        self.option_size2 = tkinter.Listbox(self.frame,height=3,exportselection=False) 
        self.option_size2.grid(row=8,column=1)
        self.option_size2.insert(1,"large")
        self.option_size2.insert(2,"small")
        
        self.option_type2 = tkinter.Listbox(self.frame,heigh=3,exportselection=False) 
        self.option_type2.grid(row=8,column=2)
        self.option_type2.insert(1,"hog")
        self.option_type2.insert(2,"cnn")
    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            file_location = f"captured_image{self.name_counter}.jpg"
            cv2.imwrite(file_location, frame)
            self.vid,self.name_counter = cv2.VideoCapture(0),self.name_counter+1
    def recognize(self,encoding_size,model_type,tolerance,numjitters,filelocation1,filelocation2,text):
            if encoding_size:encoding_size="small"
            else:encoding_size="large"
            if model_type:model_type="cnn"
            else:model_type="hog"
            t_start=time.process_time()

            first_image = face_recognition.load_image_file(filelocation1) 
            second_image = face_recognition.load_image_file(filelocation2)

            face_locations1=face_recognition.face_locations(first_image,model=model_type) 
            face_locations2=face_recognition.face_locations(second_image,model=model_type)

            first_encoding = face_recognition.face_encodings(first_image,model=encoding_size,known_face_locations=face_locations1,num_jitters=numjitters)[0] 
            second_encoding = face_recognition.face_encodings(second_image,model=encoding_size,known_face_locations=face_locations2,num_jitters=numjitters)[0] 

            result=face_recognition.compare_faces([first_encoding], second_encoding,tolerance=tolerance) 
            distance=face_recognition.api.face_distance([first_encoding], second_encoding) 
           
            t_process_ms=1000*(time.process_time()-t_start)       
            if result[0]:
                answer="Recognized"
            else:
                answer="Failed"
            text.configure(text=f"Result: {answer} in {t_process_ms:.2f} ms with distance {distance[0]:.2f}")        
            top, right, bottom, left = face_locations1[0]
            face_image = first_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image=pil_image.resize((75,75))       
            top, right, bottom, left = face_locations2[0]
            face_image2 = second_image[top:bottom, left:right]
            pil_image2 = Image.fromarray(face_image2)
            pil_image2=pil_image2.resize((75,75))    
            return ImageTk.PhotoImage(pil_image),ImageTk.PhotoImage(pil_image2),t_process_ms,distance
    
    def find_knn(self,train_dir ,img_predict_path,distance_threshold=0.6 ,n_neighbors=1, knn_algo='ball_tree'):
        X,y = [],[]
        for class_dir in os.listdir(train_dir):
            folderlocation=os.path.join(train_dir, class_dir)
            if not os.path.isdir(folderlocation):continue
            for img_path in image_files_in_folder(folderlocation):
                image = face_recognition.load_image_file(img_path)
                X.append(face_recognition.face_encodings(image)[0])
                y.append(class_dir)
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)
        faces_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_predict_path))
        if(knn_clf.kneighbors(faces_encodings, n_neighbors)[0][0][0] <= distance_threshold):return knn_clf.predict(faces_encodings)[0]
        else:return 'Failed'

    def compare(self):
        try:
            filelocation1 = tkinter.filedialog.askopenfilename()
            filelocation2 = tkinter.filedialog.askopenfilename()
            encoding_size=self.option_size1.curselection()[0]
            model_type=self.option_type1.curselection()[0]
            tolerance=self.tolerance1.get()/100
            numjitters=self.option_numjitters1.get()
        except:
            print("Wrong Selection")
            return
       

        self.image1,self.image2,time1,distance1=self.recognize(encoding_size,model_type,tolerance,numjitters,filelocation1,filelocation2,self.result1)
        self.face1['image'],self.face2['image']=self.image1,self.image2
        
        
        try:
            filelocation1 = tkinter.filedialog.askopenfilename()
            filelocation2 = tkinter.filedialog.askopenfilename()
            encoding_size=self.option_size2.curselection()[0]
            model_type=self.option_type2.curselection()[0]
            tolerance=self.tolerance2.get()/100
            numjitters=self.option_numjitters2.get()
        except:
            print("Wrong Selection")
            return
        try:
            self.image3,self.image4,time2,distance2=self.recognize(encoding_size,model_type,tolerance,numjitters,filelocation1,filelocation2,self.result2)
            self.face3['image'],self.face4['image']=self.image3,self.image4
        except:
            print('Failed')
            return
        
        try:

            filelocation = tkinter.filedialog.askopenfilename()
            result_knn=self.find_knn('train',filelocation)
        except:
            print('Failed')
            return            
        comptext=""
        if(abs(time1-time2)<0.01):comptext+=f"Both processes took about the same time ."
        elif(time1>time2):comptext+=f"The second process was faster by {time1-time2} ms ."
        else:comptext+=f"The first process was faster by {time2-time1} ms ."
        if(abs(distance1-distance2)<0.01):comptext+=f"Both processes found about the same distance"
        elif(distance2>distance1):comptext+=f"The first process found a smaller distance by {distance2[0]-distance1[0]} "
        else:comptext+=f"The second process  process found a smaller distance by {distance1[0]-distance2[0]}"
        if(result_knn=="Failed"):comptext+='Also K-neighbour failed.'
        else:comptext+=f'.Also K-neighbour recognized the face as {result_knn}.'
        self.comparison.configure(text=comptext)

         
root = tkinter.Tk()
App(root)
root.mainloop()

import face_recognition,tkinter,tkinter.filedialog,time   
from PIL import Image,ImageTk
class App():
    def __init__(self, root):
        self.root=root
        self.root.title("Ai Image Recognition")
        root.geometry("450x450")
        self.frame=tkinter.Frame(root)
        self.frame.grid(row=0,column=0)
        self.title=tkinter.Label(self.frame,text="Compare")
        self.title.grid(row=0,column=0)
        self.resultl=tkinter.Label(self.frame,text="result:")
        self.resultl.grid(row=2,column=0)
        self.result=tkinter.Label(self.frame,text="")
        self.result.grid(row=2,column=1)
        self.face=tkinter.Label(self.frame)
        self.face.grid(row=2,column=2)
        self.button=tkinter.Button(self.frame,text="click",command=self.compare)
        self.button.grid(row=0,column=1)
        self.tolerance = tkinter.Scale(self.frame,from_=1, to=100, tickinterval=5, orient="horizontal")
        self.tolerance.grid(row=0,column=2)
        self.numj = tkinter.Scale(self.frame,from_=1, to=100, tickinterval=5, orient="horizontal")
        self.numj.grid(row=0,column=3)
        
        self.listbox1 = tkinter.Listbox(self.frame,height=3,exportselection=False) 
        self.listbox1.grid(row=1,column=2)
        self.listbox1.insert(1,"large")
        self.listbox1.insert(2,"small")
        self.listbox2 = tkinter.Listbox(self.frame,heigh=3,exportselection=False) 
        self.listbox2.grid(row=1,column=3)
        self.listbox2.insert(1,"hog")
        self.listbox2.insert(2,"cnn")
        
    def compare(self):
        t_start=time.process_time()
        try:
            model1=self.listbox1.curselection()[0]
            model2=self.listbox2.curselection()[0]
            tolerancecomp=self.tolerance.get()/100
            numjitters=self.numj.get()
            
            if model1:model1="small"
            else:model1="large"
            if model2:model2="cnn"
            else:model2="hog"
            print(model1,model2,numjitters,tolerancecomp)
            filelocation = tkinter.filedialog.askopenfilename()
            known_image = face_recognition.load_image_file(filelocation) 
            filelocation = tkinter.filedialog.askopenfilename()
            unknown_image = face_recognition.load_image_file(filelocation)
            me = face_recognition.face_encodings(known_image,model=model1,num_jitters=numjitters)[0] 
            face_locations=face_recognition.face_locations(unknown_image,model=model2) 
            unknown_encoding = face_recognition.face_encodings(unknown_image,model=model1,num_jitters=numjitters)[0] 
            result=face_recognition.compare_faces([me], unknown_encoding,tolerance=tolerancecomp) 
            distance=face_recognition.api.face_distance([me], unknown_encoding) 
            t_process_ms=1000*(time.process_time()-t_start)       
            if result[0]:
                answer="Known"
            else:
                answer="Unknown"
            self.result.configure(text=f"{answer} in {t_process_ms:.2f} ms with distance {distance[0]:.2f}")        
            top, right, bottom, left = face_locations[0]
            face_image = unknown_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image=pil_image.resize((80,80))    
            self.image_tk=ImageTk.PhotoImage(pil_image)        
            self.face['image']=self.image_tk
        except:
            print("Wrong selection")
            return
        
    
        
        
        
   
root = tkinter.Tk()
App(root)
root.mainloop()
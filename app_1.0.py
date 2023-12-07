import face_recognition,tkinter

class App():

    def __init__(self, root):
        self.root=root
        self.root.title("Ai Image Recognition")
        root.geometry("400x400")
        self.frame=tkinter.Frame(root)
        self.frame.grid(row=0,column=0)
        self.title=tkinter.Label(self.frame,text="Compare")
        self.title.grid(row=0,column=0)
        self.locationInput=tkinter.Text(self.frame,height=1,width=25)
        self.locationInput.grid(row=0,column=2)
        self.resultl=tkinter.Label(self.frame,text="result:")
        self.resultl.grid(row=2,column=0)
        self.result=tkinter.Label(self.frame,text="")
        self.result.grid(row=2,column=1)
        self.button=tkinter.Button(self.frame,text="click",command=self.compare)
        self.button.grid(row=0,column=1)
    def compare(self):
        filelocation=self.locationInput.get("1.0","end-1c")
        
        known_image = face_recognition.load_image_file("me.jpg")
        
        unknown_image = face_recognition.load_image_file(filelocation)
        me = face_recognition.face_encodings(known_image)[0]

        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        result=face_recognition.compare_faces([me], unknown_encoding)
        self.result.configure(text=result)



        
        
       


      
root = tkinter.Tk()
App(root)
root.mainloop()
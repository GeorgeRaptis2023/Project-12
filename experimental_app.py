import tkinter as tk
from tkinter import filedialog
import face_recognition
import cv2
from PIL import Image, ImageTk

class FacialRecognitionApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition App")
        root.geometry("600x400")
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0)

        self.title = tk.Label(self.frame, text="Facial Recognition")
        self.title.grid(row=0, column=0)

        self.video_source = 0  # use webcam
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(self.frame, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=1, column=0, columnspan=3)

        self.location_input = tk.Entry(self.frame, width=25)
        self.location_input.grid(row=2, column=1)

        self.result_label = tk.Label(self.frame, text="Result:")
        self.result_label.grid(row=3, column=0)

        self.result = tk.Label(self.frame, text="")
        self.result.grid(row=3, column=1)

        self.capture_button = tk.Button(self.frame, text="Capture Image", command=self.capture_image)
        self.capture_button.grid(row=2, column=0)

        self.compare_button = tk.Button(self.frame, text="Compare", command=self.compare)
        self.compare_button.grid(row=2, column=2)

        self.update()

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            file_location = "captured_image.jpg"
            cv2.imwrite(file_location, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.location_input.delete(0, tk.END)
            self.location_input.insert(0, file_location)

    def compare(self):
        file_location = self.location_input.get()

        try:
            known_image = face_recognition.load_image_file("me.jpg")
            unknown_image = face_recognition.load_image_file(file_location)

            me_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

            result = face_recognition.compare_faces([me_encoding], unknown_encoding)
            self.result.configure(text="Match" if result[0] else "No Match")

        except Exception as e:
            self.result.configure(text=f"Error: {str(e)}")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)

class FileSelectionApp():
    def __init__(self, root, facial_recognition_app):
        self.root = root
        self.root.title("File Selection App")
        root.geometry("400x150")
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0)

        self.title = tk.Label(self.frame, text="File Selection")
        self.title.grid(row=0, column=0)

        self.location_input = tk.Entry(self.frame, width=25)
        self.location_input.grid(row=0, column=2)

        self.browse_button = tk.Button(self.frame, text="Select File", command=self.select_file)
        self.browse_button.grid(row=0, column=1)

        self.confirm_button = tk.Button(self.frame, text="Confirm", command=self.confirm_selection)
        self.confirm_button.grid(row=1, column=1)

        # ref to facialrec app
        self.facial_recognition_app = facial_recognition_app

    def select_file(self):
        file_location = filedialog.askopenfilename(initialdir="/", title="Select a File")
        if file_location:
            self.location_input.delete(0, tk.END)
            self.location_input.insert(0, file_location)

    def confirm_selection(self):
        selected_file_location = self.location_input.get()
        # send the file to the app
        self.facial_recognition_app.location_input.delete(0, tk.END)
        self.facial_recognition_app.location_input.insert(0, selected_file_location)
        # switch to app
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()

    # both apps
    facial_recognition_app = FacialRecognitionApp(root)
    file_selection_app = FileSelectionApp(tk.Toplevel(root), facial_recognition_app)

    root.mainloop()

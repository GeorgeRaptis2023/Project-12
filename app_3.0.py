import tkinter as tk
from tkinter import filedialog
import face_recognition

class FacialRecognitionApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition App")
        root.geometry("400x400")
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0)

        self.title = tk.Label(self.frame, text="Facial Recognition")
        self.title.grid(row=0, column=0)

        self.location_input = tk.Entry(self.frame, width=25)
        self.location_input.grid(row=0, column=2)

        self.result_label = tk.Label(self.frame, text="Result:")
        self.result_label.grid(row=2, column=0)

        self.result = tk.Label(self.frame, text="")
        self.result.grid(row=2, column=1)

        self.browse_button = tk.Button(self.frame, text="Select Image", command=self.select_image)
        self.browse_button.grid(row=0, column=1)

        self.compare_button = tk.Button(self.frame, text="Compare", command=self.compare)
        self.compare_button.grid(row=1, column=1)

    def select_image(self):
        file_location = filedialog.askopenfilename(initialdir="/", title="Select an Image")
        if file_location:
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

        # Reference to the FacialRecognitionApp instance
        self.facial_recognition_app = facial_recognition_app

    def select_file(self):
        file_location = filedialog.askopenfilename(initialdir="/", title="Select a File")
        if file_location:
            self.location_input.delete(0, tk.END)
            self.location_input.insert(0, file_location)

    def confirm_selection(self):
        selected_file_location = self.location_input.get()
        # Pass the selected file location to the FacialRecognitionApp
        self.facial_recognition_app.location_input.delete(0, tk.END)
        self.facial_recognition_app.location_input.insert(0, selected_file_location)
        # Switch to the FacialRecognitionApp
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    # Create instances of both apps
    facial_recognition_app = FacialRecognitionApp(root)
    file_selection_app = FileSelectionApp(tk.Toplevel(root), facial_recognition_app)

    root.mainloop()

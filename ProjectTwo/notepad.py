import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
import os

def createWidgets():
    
    global textArea
    textArea = Text(root)
    textArea.grid(sticky= N+E+S+W)

    menuBar = Menu(root)
    root.config(menu=menuBar)
    fileMenu= Menu(menuBar, tearoff=0)
    fileMenu.add_command(label="New",command=newFile)
    fileMenu.add_command(label="Open",command=openFile)
    fileMenu.add_command(label="Save",command=saveFile)
    fileMenu.add_separator()
    fileMenu.add_command(label="Exit",command=exit)
    menuBar.add_cascade(label="File", menu=fileMenu)

    editMenu= Menu(menuBar, tearoff=0)
    editMenu.add_command(label="Cut",command=cut)
    editMenu.add_command(label="Paste",command=paste)
    editMenu.add_command(label="Copy",command=copy)
    menuBar.add_cascade(label="Edit", menu=editMenu)

    helpMenu= Menu(menuBar, tearoff=0)
    helpMenu.add_command(label="About Notepad",command=help)
    menuBar.add_cascade(label="Help", menu=helpMenu)

def newFile():
    global  textArea
    root.title("Untitled - Notepad")
    file = None
    textArea.delete(1.0,END)
    
def openFile():
    global textArea
    file = filedialog.askopenfile(defaultextension=".txt", filetypes=[("All files", "*.*"),("Text Doocument", "*.txt")])
    file = file.name

    if file == "":
        file=None
    else:
        root.title(os.path.basename(file)+ " - Notepad")
        textArea.delete(1.0,END)
        file = open(file, "rb")
        textArea.insert(1.0, file.read())
        file.close()

def saveFile():
    global textArea, file
    if file==None:
        file=filedialog.asksaveasfilename(initialfile="Untitled.txt", defaultextension=".txt", filetypes=[("All files", "*.*"),("Text Doocument", "*.txt")])
        if file == None:
            file-None
        else:
            file = open(file, "w")
            file.write(textArea.get(1.0,END))
            file.close()
            file = file.name
            root.title(os.path.basename(file)+ " - Notepad")
    else:
        file = open(file, "w")
        file.write(textArea.get(1.0,END))
        file.close()

def exit():
    root.destroy

def cut():
    global textArea
    textArea.event_generate("<<Cut>>")

def copy():
    global textArea
    textArea.event_generate("<<Copy>>")
    
def paste():
    global textArea
    textArea.event_generate("<<Paste>>")

def help():
    messagebox.showinfo("Notepad", "This is a simple notepad")

root = tk.Tk()
root.title("Untitled - Notepad")
file = None

createWidgets()

root.mainloop()
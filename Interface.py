import tkinter as tk
from TranslationServices import translate

target_lang = "en"

oldText = ""

window = tk.Tk()


window.title("Translation")
window.attributes("-topmost", True)
window.geometry("300x300+950+200")
text = tk.Text(window)
text.pack()
text.insert(tk.END,translate("",target_lang))
def updateWindow():
    file = open("Communication.txt","r",encoding="utf-8")
    newText = file.read()
    file.close()
    global oldText
    if(newText!=oldText):
        text.delete("1.0",tk.END)
        text.insert(tk.END,newText)
        oldText = newText
    window.after(2000,updateWindow)
  
window.after(2000,updateWindow)
window.mainloop()
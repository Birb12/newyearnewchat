from tkinter import *
import talk
import train
import threading
import time
from multiprocessing.pool import Pool
from redis import Redis
from rq import Queue
import customtkinter
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import re
import smsnotify
import initlogin

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

bubbles = []

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self.window.resizable(height=True, width=True)
        self._setup_main_window()
        self.window.title = "New Year New Chat"
        self.window.wm_iconbitmap(bitmap = "icon.ico")        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=550)
        
        
        # head label
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=-0.2, relheight=0.012)
        self.font = customtkinter.CTkFont(family="Sans-serif", size=15)
        
        self.text_widget = customtkinter.CTkTextbox(self.window, width=20, height=2, 
                                 padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        frame = customtkinter.CTkFrame(self.window, width=700,height=100,corner_radius=10, fg_color="#5865F2")
        frame.place(relx=-0.1, rely=-0.1)

        self.framea = customtkinter.CTkFrame(self.window, width=200,height=75,corner_radius=10, fg_color="#5865F2")
        self.framea.place(relx=0.5, rely=0.6)

        self.canvascontainerframe = customtkinter.CTkFrame(self.window, width=530, height=460, corner_radius=10, fg_color="#000000")
        self.canvascontainerframe.place(relx=0.0, rely=0.1)
        self.canvas = customtkinter.CTkCanvas(self.canvascontainerframe, width=530, height=460,bg="white")
        scrollbar = customtkinter.CTkScrollbar(self.canvascontainerframe, command=self.canvas.yview)
        scrollbar.pack(side=RIGHT, fill = Y)
        self.canvas.pack()
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        
        bottom_label = customtkinter.CTkLabel(self.window, height=100, text=" ")
        bottom_label.place(relwidth=1, rely=0.825)
        
        self.msg_entry = customtkinter.CTkEntry(bottom_label, font=self.font)
        self.msg_entry.place(relwidth=0.74, relheight=0.4, rely=0.3, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        self.loadtextimage = customtkinter.CTkImage(light_image=Image.open("chat1.png"))
        send_button = customtkinter.CTkButton(bottom_label, width=20,command=lambda: self._on_enter_pressed(None), image=self.loadtextimage, border_width=0, fg_color= "#5865F2", text="Send", font=self.font)
        send_button.place(relx=0.77, rely=0.3, relheight=0.4, relwidth=0.22)

        self.commandstack = []
        self.userstack = []
        self.userresolutions = ["dont throw errors at me"]
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self.msg_entry.delete(0, 'end')
        self.make_bubble(msg)


        commands = ["/save", "new years resolution", "new years resolutions", "add", "remove", "notify", "remember", "login", "signup"]

        if '+1' in msg and self.commandstack[-1] == "notify":
            self.make_bubble_robot("You'll be getting a message soon! These messages are sent monthly.")
            smsnotify.run_notify(self.userresolutions[0], msg)
            return
        
        if '@' in msg and self.commandstack[-1] == "login":
            list = msg.split()

            if len(list) < 2 or len(list) > 2:
                self.make_bubble_robot("Incorrect number of arguments, please try again") ; return
            else:
                email = list[0]
                pwd = list[1]
                success, name, user = initlogin.login(email, pwd)
                self.userstack.append(user)
                if success: self.make_bubble_robot("Login Successful. Welcome " + name + "!") ; return
                else: self.make_bubble_robot("Login unsuccessful :(") ; return

        elif '@' in msg and self.commandstack[-1] == "signup":
            list = msg.split()
            if len(list) < 3 or len(list) > 3:
                self.make_bubble_robot("Incorrect number of arguments, please try again") ; return
            else:
                email = list[0]
                pwd = list[1]
                name = list[2]
                success, name = initlogin.signup(email, pwd, name)

                if success: self.make_bubble_robot("Sign-up Successful. Welcome " + name + "!") ; return
                else: self.make_bubble_robot("Sign-up unsuccessful, please try again") ; return
        else:
            with open('userdataset.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                format = [msg]
                writer.writerow(format)



        for command in commands:
            if command in msg:
                if command == "/save": talk.get_model().save_pretrained("output-medium1") ; self.make_bubble_robot("Successfully saved current model!") ; return
                elif command == "login": self.make_bubble_robot("Please give your email and password.") ; self.commandstack.append("login"); return
                elif command == "signup": self.make_bubble_robot("Please give your email, password and name.") ; self.commandstack.append("signup"); return
                elif command == "new years resolution" or command == "new years resolutions": self.make_bubble_robot("Would you like to make a list? Use add to add new years resolutions, and remove to remove new years resolutions!"); self.make_bubble_robot("Keep in mind you need to be logged in for this feature."); return
                elif command == "add":
                    a = msg.replace('add ','')
                    allres = initlogin.locate_resolutions(self.userstack[0], "add", a)
                    self.make_bubble_robot("Done! Here are your new resolutions: ")
                    self.userresolutions[0] = allres

                    for i in allres:
                        if i:
                            self.make_bubble_robot("-" + i)
                    return
                elif command == "remove":
                    a = msg.replace('remove ', '')
                    allres = initlogin.locate_resolutions(self.userstack[0], "remove", a)
                    self.make_bubble_robot("Done! Here are your new resolutions: ")
                    self.userresolutions[0] = allres

                    for i in allres:
                        if i:
                            self.make_bubble_robot("-" + i)
                    return
                elif command == "notify": self.make_bubble_robot("What is your phone number?: ") ; self.commandstack.append("notify"); return



        get_response = talk.talk(msg)
        self.make_bubble_robot(get_response)
        self.window.update()
        time.sleep(2)
        train.beginrealtime(talk.get_tokenizer(), talk.get_model())
            
    bubbles = []

    def make_bubble(self, message):
        if bubbles:
            self.canvas.move(ALL, 0, -65)

        self.frame2 = customtkinter.CTkFrame(self.canvas,fg_color="#5865F2", height=60, width=50)
        label = customtkinter.CTkLabel(self.frame2, width=250, text=message,font=self.font, wraplength=250, text_color="white", justify=RIGHT, anchor="e").grid(row=1, column=0,sticky="w",padx=5,pady=3)
        self.frame2.update_idletasks()

        self.i = self.canvas.create_window(350,360, window=self.frame2)
        self.canvas.create_window(40,40, window=label)
        self.window.update_idletasks()
        bubbles.append(1)
        
    def make_bubble_robot(self, message):
        if bubbles:
            self.canvas.move(ALL, 0, -65)

        self.frame2 = customtkinter.CTkFrame(self.canvas,fg_color="#D3D3D3", height=60, width=50)
        label = customtkinter.CTkLabel(self.frame2, width=250, text=message,font=self.font, wraplength=250, justify=LEFT, anchor="w", text_color="white").grid(row=1, column=0,sticky="w",padx=5,pady=3)
        self.frame2.update_idletasks()

        self.i = self.canvas.create_window(150,360, window=self.frame2)
        self.canvas.create_window(40,40, window=label)
        self.window.update_idletasks()
        bubbles.append(1)




                     
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()

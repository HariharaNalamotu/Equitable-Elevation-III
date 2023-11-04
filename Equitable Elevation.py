import instaloader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import cv2
import dlib
import numpy as np
from collections import Counter
from shutil import copy2
import tkinter as tk
from PIL import Image, ImageTk
import glob

class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc = nn.Linear(64, 2)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)  
            x = self.dropout2(x)
            x = self.fc(x)
            output = F.log_softmax(x, dim=1)
            return output

def calculate():
    L = instaloader.Instaloader()

    target_username = input_text.get()
    print(target_username)
    profile = instaloader.Profile.from_username(L.context, target_username)
    
    
    for post in profile.get_posts():
        try:
                L.download_post(post, target=target_username)
        except Exception as o:
                print(o)

    
    input_directory = target_username
    output_directory = target_username+"_cropped"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_directory, filename)
            crop_most_common_face(image_path, output_directory)
            print(f"Processed: {filename}")

    image_files = glob.glob(os.path.join(output_directory, "*.jpg"))

    first_image_path = image_files[-3] if image_files else None

    new_image = Image.open(first_image_path)
    new_image = new_image.resize((new_image.width*5, new_image.height*5))
    new_photo = ImageTk.PhotoImage(new_image)
    image_label.config(image=new_photo)
    image_label.image = new_photo

    print("Face cropping complete.")
    
    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    
    net.eval()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    image_dir = output_directory
    
    probabilities = []
    for image_name in os.listdir(image_dir):
        
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = net(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            adult_prob = probs[0][0].item()
            child_prob = probs[0][1].item()
    
        probabilities.append([adult_prob*100, child_prob*100])
    
    avg = 0
    counter = 0
    
    for probability in probabilities:
        counter += 1
        avg += probability[0]
        avg -= probability[1]
    
    avg = avg/counter
    
    if avg < 0:
        x = 100+avg
        x = x/2
        output_text.set("Child \nProbability:"+str(x-avg))
    else:
        x = 100-avg
        x = x/2
        output_text.set("Adult \nProbability:"+str(x+avg))

    shutil.rmtree(target_username)
    shutil.rmtree(output_directory)

def detect_faces(image_path):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces


def crop_most_common_face(image_path, output_dir):
    faces = detect_faces(image_path)

    faces = [tuple((face.left(), face.top(), face.width(), face.height())) for face in faces]

    face_counts = Counter(faces)
    
    most_common_face = face_counts.most_common(1)[0][0]

    image = cv2.imread(image_path)
    x, y, w, h = most_common_face
    cropped_image = image[y:y+h, x:x+w]

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    try: 
        cv2.imwrite(output_path, cropped_image)
    except Exception as o:
        print(o)

root = tk.Tk()
root.attributes('-fullscreen', True)
root.title("InstaVerify")

input_text = tk.StringVar()
output_text = tk.StringVar()

input_entry = tk.Entry(root, textvariable=input_text)
input_entry.pack()

calculate_button = tk.Button(root, text="Calculate", command=calculate)
calculate_button.pack()

output_label = tk.Label(root, textvariable=output_text)
output_label.pack()

image_label = tk.Label(root)
image_label.pack()

root.mainloop()

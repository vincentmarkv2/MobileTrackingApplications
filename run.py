import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

root = tk.Tk()
root.title("Mobile Price Predictor")

features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
            'touch_screen', 'wifi']

entries = {}
for i, feat in enumerate(features):
    tk.Label(root, text=feat).grid(row=i, column=0, sticky='e')
    e = tk.Entry(root, width=10)
    e.grid(row=i, column=1)
    entries[feat] = e

def predict():
    try:
        values = [float(entries[f].get()) for f in features]
        input_df = pd.DataFrame([values], columns=features)
        pred = model.predict(input_df)[0]
        labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
        messagebox.showinfo("Prediction", f"Price Range: {pred} ({labels[pred]})")
    except ValueError:
        messagebox.showerror("Error", "Enter valid numbers")

tk.Button(root, text="Predict", command=predict).grid(row=len(features), column=0, columnspan=2)

root.mainloop()
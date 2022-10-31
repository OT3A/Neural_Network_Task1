import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from main import train, main

features =['bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g']

screen = tk.Tk()

# config the screen window
screen.geometry('600x350')
screen.resizable(False, False)
screen.title('Task 1')

# label
label_select_feature = ttk.Label(text="Select two features",font=('calibre',10, 'bold'))
label_select_feature.pack(fill=tk.X, padx=5, pady=5)

# create a combobox feature 1
selected_feature1 = tk.StringVar()
feature1_cb = ttk.Combobox(screen, textvariable=selected_feature1)

feature1_cb['values'] = [i for i in features]

feature1_cb['state'] = 'readonly'

feature1_cb.pack(fill=tk.X, padx=5, pady=5)

# create a combobox feature 2
selected_feature2 = tk.StringVar()
feature2_cb = ttk.Combobox(screen, textvariable=selected_feature2)

feature2_cb['values'] = [i for i in features]

feature2_cb['state'] = 'readonly'

feature2_cb.pack(fill=tk.X, padx=5, pady=5)

# label 
label_select_class = ttk.Label(text="Select two class",font=('calibre',10, 'bold'))
label_select_class.pack(fill=tk.X, padx=5, pady=5)

# create a combobox
selected_class = tk.StringVar()
class_cb = ttk.Combobox(screen, textvariable=selected_class)

class_cb['values'] = ['Adelie & Gentoo','Adelie & Chinstrap','Gentoo & Chinstrap']

class_cb['state'] = 'readonly'

class_cb.pack(fill=tk.X, padx=5, pady=5)



eta_var=tk.StringVar()
eat_label = ttk.Label(screen, text = 'Enter learning rate', font=('calibre',10, 'bold'))
eat_label.pack(fill=tk.X,padx=5, pady=5)


name_entry = tk.Entry(screen,textvariable = eta_var, font=('calibre',10,'normal'))
name_entry.pack(fill=tk.X,padx=5, pady=5)

epochs_var=tk.StringVar()
epochs_label = ttk.Label(screen, text = 'Enter number of epochs', font=('calibre',10, 'bold'))
epochs_label.pack(fill=tk.X,padx=5, pady=5)


epochs_entry = tk.Entry(screen,textvariable = epochs_var, font=('calibre',10,'normal'))
epochs_entry.pack(fill=tk.X,padx=5, pady=5)


Checkbutton1 = tk.IntVar()  
checkbox = ttk.Checkbutton(screen, text = "Bias", 
                      variable = Checkbutton1,
                      onvalue = 1,
                      offvalue = 0,
                      )
  
def solve():
    if selected_feature1.get() == selected_feature2.get() or selected_feature1.get() == '' or selected_feature2.get() == '' or selected_class.get() == '' or epochs_var.get() == '' or eta_var.get() == '':
        showinfo(
            title='error',
            message='error! please reassign the form'
        )
    else:
        c1 = selected_class.get().split(' & ')[0]
        c2 = selected_class.get().split(' & ')[1]
        feature1 = selected_feature1.get()
        feature2 = selected_feature2.get()
        epochs = int(epochs_var.get())
        eta =float(eta_var.get())
        bias = Checkbutton1.get()
        accuracy = main(c1, c2, feature1, feature2, epochs, eta, bias)
        showinfo(
            title = 'accuracy',
            message = f'Accuracy of classes {c1}, {c2} by feature {feature1}, {feature2} = {accuracy}' 
        )

checkbox.pack(padx=5, pady=5)

B = ttk.Button(screen,text='solve',command=solve)
B.pack()
screen.mainloop()
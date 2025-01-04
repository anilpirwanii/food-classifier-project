# 🌟 Machine Learning-Based Snack Scanner

Welcome to the **Snack Scanner**, your ultimate food recognition companion powered by machine learning! Upload a food image, and the app will predict its category with accuracy.

---

## 🌐 Access the App Online

You can try the app without any setup by visiting the live link:  
👉 [Snack Scanner on Streamlit](https://food-classifier-project.streamlit.app/)

If you'd like to run the app locally, follow the steps below.

---

## 📂 Food Categories

Here are the 15 categories the app currently supports:

- 🥗 Caesar Salad
- 🍗 Chicken Wings
- 🍫 Chocolate Cake
- 🐟 Fish and Chips
- 🍟 French Fries
- 🌭 Hot Dog
- 🍦 Ice Cream
- 🍕 Pizza
- 🍁🍟 Poutine
- 🍜 Ramen
- 🥟 Samosa
- 🥩 Steak
- 🍣 Sushi
- 🌮 Tacos
- 🧇 Waffles

---

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anilpirwanii/food-classifier-project.git
cd food-classifier-project
```

### 2. Set up the Environment 

Create a Virtual Environment
```bash
python -m venv food-env
source food-env/bin/activate  # On Windows: food-env\Scripts\activate
```

Install the Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Model and Labels
Ensure the following files are in the notebooks directory:

```bash
food_classifier_model.keras
class_labels.json
```

### 4. Run the App

```bash
streamlit run notebooks/app.py
```
The app will start, and you can access it at http://localhost:8501.


### 🖼️ Example Usage
Launch the app and upload an image of food.  
The app will display the predicted category along with an accuracy score.  
If the app cannot confidently classify the image, it will suggest focusing on a single food item by cropping the image.  
### 🔧 Troubleshooting
Missing Model File: Ensure food_classifier_model.keras is in the correct directory. If not, download it and place it in notebooks/.  
Dependency Issues: Run pip install -r requirements.txt to ensure all dependencies are installed.  
Permission Issues: Ensure you have the necessary permissions for file uploads.  
### 📄 Technology Stack
Framework: Streamlit  
Backend: TensorFlow with MobileNetV2  
Frontend: Interactive UI with modern design elements  

### 📧 Contact
If you have questions, feedback, or ideas, feel free to reach out:

📧 Email: aka158@sfu.ca  
🌐 GitHub  
🌐 LinkedIn  


🌟 Demo  
Check out the live app here: Snack Scanner


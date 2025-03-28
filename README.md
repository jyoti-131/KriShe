# **🌾 कृShe** 🚜  

🔍 **An AI-powered web application designed to assist farmers in making informed decisions about crop selection, fertilizer use, and accessing government schemes based on real-time soil, environmental, and policy data.**  

---

## **📎 Project Structure**
```
├── instance/
│   ├── users.db                      # Database for user authentication
├── static/
│   ├── about1.jpg
│   ├── background.jpg
│   ├── image/
│   ├── img.jpg
│   ├── new.jpg
├── templates/
│   ├── 404.html
│   ├── about.html
│   ├── chatbot.html
│   ├── fertilizer_form.html
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── farmer_schemes.html            # Page displaying government farming schemes
├── Crop_recommendation.csv            # Dataset for crop recommendation
├── README.md                          # Documentation (You're reading this!)
├── app.py                              # Main Flask application
├── fertilizer_model.pkl                # Fertilizer recommendation model
├── minmaxscaler.pkl                    # Scaler for preprocessing
├── model.pkl                           # Crop classification model
├── train_fertilizer_model.py           # Training script for fertilizer model
├── timescaler.pkl                      # Time-based scaling for predictions
```

---

## **⚡ Features**
✅ **🌱 Crop Recommendation** – Predicts the best crop based on environmental conditions.  
✅ **🧪 Fertilizer Suggestion** – Provides the most suitable fertilizer recommendation.  
✅ **🌍 Bilingual Support** – Available in both Hindi & English for accessibility.  
✅ **💬 AI Chatbot** – Integrated chatbot for answering farming-related queries.  
✅ **🔒 User Authentication** – Secure login and registration system.  
✅ **🌐 Government Schemes** – Displays farming-related government schemes for farmers.  

---

## **🔧 Installation & Setup**
### **1️⃣ Create & Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Flask Application**
```bash
python app.py
```
🌟 The app will run on **`http://127.0.0.1:5000/`**  

---

## **📌 How It Works**
1️⃣ **User Logs In/Register** 🔑  
2️⃣ **Selects Crop Recommendation, Fertilizer Suggestion, or Government Schemes** 🌾  
3️⃣ **Provides Soil & Climate Inputs (for crop/fertilizer suggestions)** 🌍  
4️⃣ **Gets AI-Based Recommendations** 🤖  
5️⃣ **Chatbot Assistance for Queries** 💬  
6️⃣ **Accesses Government Farming Schemes for Additional Benefits** 🌐  

---

## **📞 Contact**  
📩 For queries, feel free to contact.


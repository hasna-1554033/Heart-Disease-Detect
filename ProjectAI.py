import streamlit as st
import base64
import pandas as pd

file_path = 'Dataset Project.csv'
dataset = pd.read_csv(file_path)

for column in dataset.columns:
    dataset[column] = pd.to_numeric(dataset[column], errors='coerce').fillna(0)

dataset["Num"] = dataset["Num"].apply(lambda x: 1 if x > 1 else x)

X = dataset.drop(columns=["Num"]).values.tolist()
y = dataset["Num"].values.tolist()

n_train = int(0.7 * len(X))

X_train = X[:n_train] 
X_test = X[n_train:]  
y_train = y[:n_train]  
y_test = y[n_train:]  

def calculate_prior(y_train):

    total = len(y_train)
    prior = {}
    for cls in set(y_train):
        prior[cls] = y_train.count(cls) / total
    return prior

def calculate_mean_variance(X_train, y_train):

    classes = set(y_train)
    mean_variance = {}

    for cls in classes:
        indices = [i for i, y in enumerate(y_train) if y == cls]
        X_cls = [X_train[i] for i in indices]
        
        mean_variance[cls] = {}
        for j in range(len(X_cls[0])):  
            feature_values = [row[j] for row in X_cls]
            mean = sum(feature_values) / len(feature_values)
            var = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
            mean_variance[cls][j] = (mean, var + 1e-6)  # Tambahkan epsilon untuk stabilitas
    return mean_variance

def calculate_likelihood(x, mean, var):

    coeff = 1.0 / ((2.0 * 3.141592653589793 * var) ** 0.5)  # π ≈ 3.14159
    exponent = 2.718281828459045 ** (-((x - mean) ** 2) / (2 * var))  # e ≈ 2.71828
    return coeff * exponent

def calculate_posterior(row, prior, mean_variance):

    posteriors = {}
    for cls, cls_prior in prior.items():
        posterior = cls_prior 
        for j, value in enumerate(row):
            mean, var = mean_variance[cls][j]
            posterior *= calculate_likelihood(value, mean, var)
        posteriors[cls] = posterior
    return posteriors

def naive_bayes_predict(X_test, prior, mean_variance):

    predictions = []
    for row in X_test:
        posteriors = calculate_posterior(row, prior, mean_variance)
        predictions.append(max(posteriors, key=posteriors.get))
    return predictions


st.set_page_config(page_title="Heart Disease Prediction", page_icon="🩺", layout="centered")
def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .block-container {{
        padding: 0 !important;
    }}
    header {{
        display: none;
    }}
    body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
    }}
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        width: 100vw;
        margin: 0;
        padding: 0;
    }} 
    header, footer {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = 1
if "has_disease" not in st.session_state:
    st.session_state.has_disease = False

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# Page 1
if st.session_state.page == 1:
    add_background("Page 1.png")
    st.markdown(
        """
        <div style="position: relative; left: -240px; margin-top: 30%; font-family: 'Arial', sans-serif;">
            <h1 style="color: #E44F4F; font-size: 3em; font-weight: bold;">Heart Disease Prediction</h1>
            <p style="color: #E44F4F; font-size: 1.4em; font-weight: normal;">
                Find out if you have any indication of heart disease<br>
                based on the information provided.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: left; position: relative; left: -200px; margin-top: -55%;">
            <h1 style="color: #E44F4F; font-size: 2.0em; font-weight: bold;">Heartify</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Go to Next Page"):
        next_page()

# Page 2
elif st.session_state.page == 2:
    add_background("Page 2.png")  
    st.header("Enter Your Details Below")

    st.markdown(
        """
        <style>
        .scrollable-area {
            max-height: 600px;  /* Batasi tinggi untuk scroll */
            overflow-y: auto;   /* Izinkan scroll vertikal */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="scrollable-area">', unsafe_allow_html=True)

        # input user yang panjang
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("CP (Chest pain type)", options=[1, 2, 3, 4], format_func=lambda x: f"Type {x-1}")
        trestbps = st.number_input("Trestbps (Resting blood pressure in mm Hg)", min_value=50, max_value=200, value=120)
        chol = st.number_input("Chol (Serum cholesterol in mg/dl)", min_value=100, max_value=600, value=220)
        fbs = st.selectbox("Fbs (Fasting blood sugar > 120 mg/dl)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Restecg (Resting electrocardiographic results)", options=[0, 1, 2])
        thalach = st.number_input("Thalach (Maximum heart rate achieved)", min_value=50, max_value=220, value=150)
        exang = st.selectbox("Exang (Exercise induced angina)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-2.0, max_value=5.0, value=0.0)
        slop = st.selectbox("Slop (Slope of the peak exercise ST segment)", options=[0, 1, 2])
        ca = st.selectbox("Ca (Number of major vessels colored by fluoroscopy)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thal (Thalassemia)", options=[3, 6, 7], format_func=lambda x: "Normal" if x == 3 else ("Fixed Defect" if x == 6 else "Reversible Defect"))

        st.markdown('</div>', unsafe_allow_html=True)  

    if st.button("Predict"):
        data = []
        data.append(int(age))
        data.append(int(sex))
        data.append(int(cp))
        data.append(int(trestbps))
        data.append(int(chol))
        data.append(int(fbs))
        data.append(int(restecg))
        data.append(int(thalach))
        data.append(int(exang))
        data.append(float(oldpeak))
        data.append(int(slop))
        data.append(int(ca))
        data.append(int(thal))

        x_data = [[float(i) for i in data]]

        prior = calculate_prior(y_train)
        mean_variance = calculate_mean_variance(X_train, y_train)

        y_pred = naive_bayes_predict(x_data, prior, mean_variance)
        st.session_state.has_disease = y_pred[0]  # simpan hasil prediksi dalam key yang tepat
        print(data)
        print(y_pred)
        next_page()

# Page 3
elif st.session_state.page == 3:
    # ada indikasi penyakit jantung
    if st.session_state.has_disease == 1:
        add_background("Page 3 (has disease).png")
        st.markdown(
            """
            <div style="position: relative; left: 315px; margin-top: 20%; font-family: 'Arial', sans-serif;">
                <h1 style="color: #E44F4F; font-size: 3em; font-weight: bold;">Heart Disease Prediction</h1>
                <p style="color: #E44F4F; font-size: 1.4em; font-weight: normal;">
                    The prediction indicates that you might have heart disease.
                </p>
                <h3 style="color: #E44F4F; font-size: 1.8em;">Recommendations for Better Heart Health:</h3>
                <ul style="color: #333; font-size: 1.2em;">
                    <li><strong>Adopt a heart-healthy diet:</strong> Eat more fruits, vegetables, and whole grains. Limit salt, sugar, and unhealthy fats.</li>
                    <li><strong>Regular physical activity:</strong> At least 30 minutes of moderate exercise most days of the week.</li>
                    <li><strong>Maintain a healthy weight:</strong> Monitor BMI and stay in a healthy range.</li>
                    <li><strong>Avoid smoking and limit alcohol:</strong> Quit smoking and moderate alcohol intake.</li>
                    <li><strong>Manage stress levels:</strong> Practice mindfulness or yoga to reduce stress.</li>
                    <li><strong>Visit a healthcare professional:</strong> Regular check-ups for blood pressure, cholesterol, and sugar levels.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # gada indikasi penyakit jantung
        add_background("Page 3 (hasn't disease).png")
        st.markdown(
            """
            <div style="position: relative; left: -240px; margin-top: 18%; font-family: 'Arial', sans-serif;">
                <h1 style="color: #E44F4F; font-size: 3em; font-weight: bold;">Heart Disease Prediction</h1>
                <p style="color: #4CAF50; font-size: 1.4em; font-weight: normal;">
                    You are unlikely to have heart disease!
                </p>
                <h3 style="color: #E44F4F; font-size: 1.8em;">Recommendations to Prevent Heart Disease:</h3>
                <ul style="color: #333; font-size: 1.2em;">
                    <li><strong>Maintain a balanced diet:</strong> Continue eating heart-healthy foods like fruits and vegetables.</li>
                    <li><strong>Exercise regularly:</strong> Engage in physical activities such as walking or jogging.</li>
                    <li><strong>Monitor your health:</strong> Periodically check blood pressure, cholesterol, and sugar levels.</li>
                    <li><strong>Avoid harmful habits:</strong> Refrain from smoking and limit alcohol consumption.</li>
                    <li><strong>Stay active and stress-free:</strong> Engage in hobbies and socialize to reduce stress.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 2rem;
        font-weight: bold;
        width: 100%;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .info-text {
        color: #2C3E50;
        font-size: 0.9rem;
    }
    .diabetes-positive {
        background-color: #FFEBEE;
        color: #C62828;
        border: 2px solid #EF5350;
    }
    .diabetes-negative {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 2px solid #66BB6A;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
        color: white;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
    .feature-name {
        font-weight: bold;
        color: #34495E;
    }
    .feature-value {
        font-weight: bold;
        color: #E74C3C;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)


# Load and preprocess data
@st.cache_data
def load_data():
    diabetes_df = pd.read_csv('latest_diabetes.csv')
    return diabetes_df


@st.cache_resource
def build_model(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale the input variables using StandardScaler
    scaler = StandardScaler()
    standard_data = scaler.fit_transform(X)
    X = standard_data

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    return model, scaler


# Creating plots for analysis
@st.cache_data
def create_distribution_plot(df):
    fig = plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, palette=['green', 'red'])
    plt.title('Distribution of Glucose by Outcome', fontsize=15)
    plt.xlabel('Glucose Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    return fig


@st.cache_data
def create_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    mask = np.triu(correlation_matrix)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Features', fontsize=15)
    return fig


@st.cache_data
def create_feature_comparison(df):
    fig = px.scatter(df, x='BMI', y='Glucose', color='Outcome',
                     size='Age', hover_data=['Pregnancies', 'BloodPressure', 'Insulin'],
                     color_discrete_sequence=['green', 'red'],
                     labels={'Outcome': 'Diabetes Status', 'BMI': 'Body Mass Index', 'Glucose': 'Glucose Level'},
                     title='BMI vs Glucose by Diabetes Status')
    return fig


def create_radar_chart(input_values, feature_names):
    # Add first point at the end to close the circle
    input_values_closed = np.append(input_values, input_values[0])
    feature_names_closed = np.append(feature_names, feature_names[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=input_values_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Patient Values',
        line_color='#E74C3C',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Patient Feature Profile'
    )

    return fig


def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 14}},
            'bar': {'color': "#E74C3C"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#27AE60'},
                {'range': [30, 70], 'color': '#F39C12'},
                {'range': [70, 100], 'color': '#C0392B'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'font': {'size': 24}}
    ))

    fig.update_layout(
        height=400,  # Increased height
        margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
        autosize=True,
    )

    return fig


def normalize_input_data(input_data, df):
    # Get min and max values for each feature for normalization
    min_values = df.drop('Outcome', axis=1).min()
    max_values = df.drop('Outcome', axis=1).max()

    # Normalize each input value between 0 and 1
    normalized_data = [(input_data[i] - min_values[i]) / (max_values[i] - min_values[i])
                       for i in range(len(input_data))]

    return normalized_data


def app():
    # Load data and build model
    diabetes_df = load_data()
    model, scaler = build_model(diabetes_df)

    # Sidebar for user inputs
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.title('Patient Information')
        st.markdown('Adjust the sliders to input patient data.')

        # Feature inputs in a more organized way
        preg = st.slider('Pregnancy', 0, 17, 4)
        glucose = st.slider('Glucose', 0, 199, 121)
        bp = st.slider('BloodPressure', 0, 122, 72)
        skinthickness = st.slider('SkinThickness', 0, 99, 23)
        insulin = st.slider('Insulin', 0, 846, 30)
        bmi = st.slider('BMI', 0.0, 67.1, 32.0)
        dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
        age = st.slider('Age', 21, 81, 29)

        # Add a predict button
        predict_button = st.button('Predict Diabetes Status', key='predict')
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content - removed columns as we no longer need them for the image

    # Center the title and description without image
    st.markdown('<h1 class="main-header" style="text-align: center;">Diabetes Disease Prediction</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="info-text" style="text-align: center;">This application predicts whether a patient has diabetes based on medical and demographic features.</p>',
        unsafe_allow_html=True)

    # Make a prediction
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    scaled_data = scaler.transform(reshaped_input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]  # Get probability of positive class

    # Display prediction result
    if prediction[0] == 1:
        prediction_text = "This person has Diabetes"
        prediction_class = "diabetes-positive"
    else:
        prediction_text = "This person does not have Diabetes"
        prediction_class = "diabetes-negative"

    # Create dashboard layout with centered prediction box
    st.markdown(
        f'<div class="prediction-box {prediction_class}" style="max-width: 600px; margin: 0 auto;">{prediction_text}</div>',
        unsafe_allow_html=True)

    # Display key metrics in cards
    st.markdown('<h2 class="sub-header" style="text-align: center;">Patient Metrics</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Glucose Level</div>
        </div>
        '''.format(glucose), unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">BMI</div>
        </div>
        '''.format(bmi), unsafe_allow_html=True)

    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Blood Pressure</div>
        </div>
        '''.format(bp), unsafe_allow_html=True)

    with col4:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Age</div>
        </div>
        '''.format(age), unsafe_allow_html=True)

    # Create tabs for visualizations
    st.markdown('<h2 class="sub-header" style="text-align: center;">Analysis & Visualizations</h2>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Patient Profile", "Risk Analysis"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Feature Values")
            feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                             "Insulin", "BMI", "DPF", "Age"]

            for i, feature in enumerate(feature_names):
                st.markdown(
                    f'<span class="feature-name">{feature}:</span> <span class="feature-value">{input_data[i]}</span>',
                    unsafe_allow_html=True)

        with col2:
            # Create radar chart
            normalized_data = normalize_input_data(input_data, diabetes_df)
            radar_fig = create_radar_chart(normalized_data, feature_names)
            st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display risk gauge with more space
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            gauge_fig = create_risk_gauge(probability)
            st.plotly_chart(gauge_fig, use_container_width=True, height=450)

            st.markdown(f"""
            <div style="text-align: center; margin-top: -20px;">
                <p>The diabetes risk score is based on the model's prediction probability.</p>
                <p><strong>Score: {probability * 100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("High Risk Factors")
            # Identify high-risk factors (normalized values above 0.7)
            high_risk_factors = [feature_names[i] for i, val in enumerate(normalized_data) if val > 0.7]

            if high_risk_factors:
                for factor in high_risk_factors:
                    st.markdown(f"• <span style='color: #C0392B; font-weight: bold;'>{factor}</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("No significantly elevated risk factors identified.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Recommendations")

            if prediction[0] == 1:
                st.markdown("""
                • Consult with a healthcare professional
                • Monitor blood glucose regularly
                • Maintain a healthy diet and exercise routine
                • Follow medication guidelines as prescribed
                """)
            else:
                st.markdown("""
                • Continue regular health check-ups
                • Maintain a balanced diet
                • Exercise regularly
                • Monitor blood glucose periodically
                """)

        st.markdown('</div>', unsafe_allow_html=True)

    # Removed Data Insights tab as requested

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;">
        <p>Diabetes Prediction Tool - For educational purposes only. Not for medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()
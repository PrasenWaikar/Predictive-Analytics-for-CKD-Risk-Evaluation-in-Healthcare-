import pickle
import streamlit as st
import pandas as pd

with open("kidney.pkl", "rb") as pickle_in:
    vote_model = pickle.load(pickle_in)

df = pd.read_csv("clean_dataset.csv")  
feature_columns = df.drop('Unnamed: 0',axis=1).columns.tolist()  
st.set_page_config(
    page_title='CKD prediction',
    page_icon=':female-doctor:',
    layout='wide',
    initial_sidebar_state='expanded'

    )
# with st.container():
#     st.title('Chronic Kidney Disease Predictor')
#     st.write('Please connect this app to your clinical lab to help diagnose chronic kidney disease (CKD) from patient test results. This app predicts, using a machine learning model, whether a patient is at risk of CKD based on the measurements it receives from your lab. You can also update the measurements manually using the sliders in the sidebar.')

# st.title('Chronic Kidney Disease Predictor')

# Custom styled header using markdown
st.markdown(
    """
    <div style="background-color:GhostWhite;padding:13px">
        <h1 style="color:black;text-align:center;">Chronic Kidney Disease Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Apply background color to the entire page using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: Lavender;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write('Please connect this app to your clinical lab to help diagnose chronic kidney disease (CKD) from patient test results. This app predicts, using a machine learning model, whether a patient is at risk of CKD based on the measurements it receives from your lab. You can also update the measurements manually by giving inputs in the sidebar.')


def main():
  pass  
    # st.markdown(
    #     """
    #     <div style="background-color:cyan;padding:13px">
    #     <h1 style="color:black;text-align:center;">Kidney Disease Prediction</h1>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )



@st.cache_data
def prediction(user_input):

    input_df = pd.DataFrame([user_input], columns=feature_columns)


    prediction = vote_model.predict(input_df)

    return "Kidney Disease found" if prediction == 1 else "Kidney Disease not detected"

# Streamlit UI


#     # Create input fields dynamically for all features
# user_input = {}

# for col in feature_columns:
#     if df[col].dtype == "object":  # Categorical values
#         user_input[col] = st.sidebar.radio(f"{col}", [0, 1])  # Only allows 0 or 1
#     else:  # Numerical values
#         user_input[col] = st.sidebar.number_input(
#             f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].median())
#         )


user_input = {}

# Loop through feature columns
for col in feature_columns:
    if df[col].dtype == "object":  # Categorical values
        user_input[col] = st.sidebar.radio(
            f"{col}", [0, 1], key=f"{col}_radio"  # Unique key for radio button
        )
    else:  # Numerical values
        user_input[col] = st.sidebar.number_input(
            f"{col}", 
            min_value=float(df[col].min()), 
            max_value=float(df[col].max()), 
            value=float(df[col].median()),
            key=f"{col}_number"  # Unique key for number input
        )

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=feature_columns)

st.write("User Input Preview:", input_df)


# Prediction button
if st.button("Predict"):
    result = prediction(user_input)
    st.success(f"Report Result: {result}")



if __name__ == '__main__':
    main()

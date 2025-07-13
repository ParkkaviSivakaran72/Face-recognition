import streamlit as st
from streamlit_autorefresh import st_autorefresh   
import pandas as pd 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDENCE_DIR = os.path.join(BASE_DIR, "attendence")
attendance_path = os.path.join(ATTENDENCE_DIR, "attendance.csv") 

count = st_autorefresh(interval=1000, limit=100, key="dataframe_autorefresh")
if count == 0:
    st.write("Welcome to the Face Attendance System")
elif count%5 == 0:
    st.write("Refreshing attendance data...")
else:
    st.write("Attendance data is being updated...")

df = pd.read_csv(attendance_path)

st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from src.health_model_tf import get_health_model_tf  # Physical health model
from src.mental_model_tf import get_mental_model_tf  # Mental health model
from src.hospital_finder import find_hospitals_by_coords
import pandas as pd
import requests

# Set page config
st.set_page_config(page_title="AI Health Advisor", page_icon="🩺", layout="centered")

st.title("🩺 AI Health Advisor & Smart Hospital Recommender")
st.write(
    "This app estimates your lifestyle-related health risk using a machine learning model "
    "trained on a Health & Lifestyle dataset, and recommends nearby hospitals based on "
    "your current location."
)

# Initialize persistent session keys
if "geolocation" not in st.session_state:
    st.session_state["geolocation"] = None
if "health_result" not in st.session_state:
    st.session_state["health_result"] = None
if "mental_result" not in st.session_state:
    st.session_state["mental_result"] = None

# ----------------------------------------
# 1️⃣ Physical Health Risk Assessment
# ----------------------------------------

st.header("1️⃣ Physical Health Risk Assessment")

col_ph_left, col_ph_right = st.columns([1, 1], gap="medium")

with col_ph_left:
    st.subheader("Your Details")
    # Lifestyle-related features (user selects)
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    height_cm = st.number_input("Height (cm)", min_value=130.0, max_value=220.0, value=170.0)
    weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=70.0)
    
    st.subheader("Lifestyle Habits")
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    exercise_freq = st.selectbox(
        "Exercise frequency", ["None", "1-2 times/week", "3-5 times/week", "Daily"]
    )
    diet_quality = st.selectbox("Diet quality", ["Poor", "Good", "Excellent"])
    alcohol = st.selectbox("Alcohol consumption", ["None", "Low", "Moderate", "High"])
    chronic = st.selectbox("Chronic disease", ["No", "Yes"])
    
    st.subheader("Other Factors")
    stress_level = st.slider("Stress level (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep hours per night", 3.0, 12.0, 7.0, step=0.5)
    
    # BMI calculation
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.caption(f"Calculated BMI: **{bmi:.1f}**")
    
    # Button to analyze risk
    analyze_btn = st.button("Analyze my health risk", key="health_analyze")

with col_ph_right:
    # Collect user inputs
    user_input = {
        "Age": age,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Other": 1 if gender == "Other" else 0,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "BMI": bmi,
        "Smoker_Yes": 1 if smoker == "Yes" else 0,
        "Exercise_Freq_1-2 times/week": 1 if exercise_freq == "1-2 times/week" else 0,
        "Exercise_Freq_3-5 times/week": 1 if exercise_freq == "3-5 times/week" else 0,
        "Exercise_Freq_Daily": 1 if exercise_freq == "Daily" else 0,
        "Diet_Quality_Good": 1 if diet_quality == "Good" else 0,
        "Diet_Quality_Excellent": 1 if diet_quality == "Excellent" else 0,
        "Alcohol_Consumption_Low": 1 if alcohol == "Low" else 0,
        "Alcohol_Consumption_Moderate": 1 if alcohol == "Moderate" else 0,
        "Alcohol_Consumption_High": 1 if alcohol == "High" else 0,
        "Chronic_Disease_Yes": 1 if chronic == "Yes" else 0,
        "Stress_Level": stress_level,
        "Sleep_Hours": sleep_hours,
    }
    
    if analyze_btn:
        model = get_health_model_tf()
        result = model.predict(user_input)
        st.session_state["health_result"] = result
    
    # Display health result if available
    st.subheader("Your Health Risk")
    if st.session_state.get("health_result"):
        result = st.session_state["health_result"]
        st.metric("Risk Level", result['risk_label'])
        st.write("**Probability Breakdown:**")
        for label, prob in result["risk_proba"].items():
            st.write(f"• {label}: {prob * 100:.1f}%")
        
        st.divider()
        if result["risk_index"] == 2:
            st.warning("High health risk detected. Immediate action is required.")
            st.write("**Advice:** Seek medical help right away.")
        elif result["risk_index"] == 1:
            st.info("Moderate health risk. Consider lifestyle changes.")
            st.write("**Advice:** Improve exercise, diet, and stress management.")
        else:
            st.success("Low health risk. Keep up the good work!")
            st.write("**Advice:** Continue your healthy lifestyle.")
    else:
        st.write("👈 Fill your details and click 'Analyze my health risk' to see results.")

# ----------------------------------------
# 2️⃣ Mental Wellbeing Assessment
# ----------------------------------------

st.header("2️⃣ Mental Wellbeing Assessment")

col_mh_left, col_mh_right = st.columns([1, 1], gap="medium")

with col_mh_left:
    st.subheader("Your Mental Health")
    # Inputs for mental health (user selects)
    stress_level_m = st.slider("Stress level (1 = low, 10 = high)", 1, 10, 5, key="stress_level_m")
    social_support = st.slider("Social support level (1 = none, 10 = strong)", 1, 10, 5)
    loneliness = st.slider("Loneliness level (1 = low, 10 = high)", 1, 10, 5)
    
    st.subheader("History & Habits")
    history_anxiety = st.selectbox("History of Anxiety", ["No", "Yes"])
    history_depression = st.selectbox("History of Depression", ["No", "Yes"])
    history_panic = st.selectbox("History of Panic Attacks", ["No", "Yes"])
    use_therapy = st.selectbox("Do you currently use therapy?", ["No", "Yes"])
    work_pressure = st.slider("Work/School pressure (1 = low, 10 = high)", 1, 10, 5)
    
    # Button to analyze
    analyze_mental_btn = st.button("Analyze my mental wellbeing", key="mental_health_button")

with col_mh_right:
    st.subheader("Your Mental Wellbeing Results")
    
    # Prepare inputs for prediction
    mental_input = {
        "Stress_Level": stress_level_m,
        "Social_Support": social_support,
        "Loneliness": loneliness,
        "History_Anxiety": 1 if history_anxiety == "Yes" else 0,
        "History_Depression": 1 if history_depression == "Yes" else 0,
        "History_Panic_Attacks": 1 if history_panic == "Yes" else 0,
        "Use_Therapy": 1 if use_therapy == "Yes" else 0,
        "Work_School_Pressure": work_pressure,
    }
    
    # Compute Risk_Score the same way as in the training notebook so the model gets the expected feature
    try:
        risk_score = 0
        if mental_input.get("Stress_Level", 0) >= 7:
            risk_score += 3
        if mental_input.get("Social_Support", 0) <= 4:
            risk_score += 3
        if mental_input.get("Loneliness", 0) >= 7:
            risk_score += 2
        # Reduce the weight of historical diagnosis flags so they don't dominate the score
        if mental_input.get("History_Anxiety", 0) == 1:
            risk_score += 2
        if mental_input.get("History_Depression", 0) == 1:
            risk_score += 2
        if mental_input.get("History_Panic_Attacks", 0) == 1:
            risk_score += 1
        if mental_input.get("Use_Therapy", 0) == 1:
            risk_score -= 1
    except Exception:
        risk_score = 0
    
    # Attach Risk_Score to the feature dict (model expects this column)
    mental_input["Risk_Score"] = risk_score
    
    if analyze_mental_btn:
        mental_model = get_mental_model_tf()
        mental_result = mental_model.predict(mental_input)
        # Simple balanced probability adjustment
        # Start at 100% Low, then add small equal weights for each risky input factor
        # Each "risky" condition adds ~3% to the risk pool, distributed to Moderate and High
        
        # Count risk factors (user's "yes" answers and slider deviations from neutral)
        risk_factors = 0.0
        
        # History flags: each "Yes" adds 3 risk factors (increased weight for more visible impact)
        if mental_input.get("History_Anxiety", 0) == 1:
            risk_factors += 3.0
        if mental_input.get("History_Depression", 0) == 1:
            risk_factors += 3.0
        if mental_input.get("History_Panic_Attacks", 0) == 1:
            risk_factors += 3.0
        
        # Slider contributions: deviation from neutral/safe baseline
        # Stress: neutral=5, each unit above 5 adds 0.9 factor (3x larger)
        stress = float(mental_input.get("Stress_Level", 0))
        risk_factors += max(0.0, (stress - 5.0) * 0.9)
        
        # Loneliness: neutral=5, each unit above 5 adds 0.9 factor (3x larger)
        loneliness_v = float(mental_input.get("Loneliness", 0))
        risk_factors += max(0.0, (loneliness_v - 5.0) * 0.9)
        
        # Social support: neutral=5, each unit below 5 adds 0.9 factor (lower is worse, 3x larger)
        social = float(mental_input.get("Social_Support", 0))
        risk_factors += max(0.0, (5.0 - social) * 0.9)
        
        # Work/School pressure: neutral=5, each unit above 5 adds 0.6 factor (3x larger)
        wp = float(mental_input.get("Work_School_Pressure", 0))
        risk_factors += max(0.0, (wp - 5.0) * 0.6)
        
        # Use Therapy: "Yes" reduces risk by 3 factors (protective, 3x larger)
        if mental_input.get("Use_Therapy", 0) == 1:
            risk_factors -= 3.0
        risk_factors = max(0.0, risk_factors)  # Never go below 0
        
        # Convert risk_factors to probabilities
        # Each factor unit = ~6% of total probability mass shift from Low to Moderate+High (3x larger)
        prob_to_shift = min(0.95, risk_factors * 0.06)  # Max 95% shifted from Low
        p_low = 1.0 - prob_to_shift
        # Distribute shifted probability: 40% Moderate, 60% High (balanced but favoring high)
        p_moderate = prob_to_shift * 0.4
        p_high = prob_to_shift * 0.6
        
        adjusted = {"Low": float(p_low), "Moderate": float(p_moderate), "High": float(p_high)}
        mental_result["risk_proba"] = adjusted
        # Recompute label and index based on highest probability
        labels = ["Low", "Moderate", "High"]
        probs_list = [adjusted[l] for l in labels]
        max_i = probs_list.index(max(probs_list))
        mental_result["risk_index"] = int(max_i)
        mental_result["risk_label"] = labels[max_i]
        
        st.session_state["mental_result"] = mental_result
    
    # Display mental result if available
    if st.session_state.get("mental_result"):
        mental_result = st.session_state["mental_result"]
        st.metric("Risk Level", mental_result['risk_label'])
        st.write("**Probability Breakdown:**")
        for label, prob in mental_result["risk_proba"].items():
            st.write(f"• {label}: {prob * 100:.1f}%")
        
        st.divider()
        if mental_result["risk_index"] == 2:
            st.warning("High mental wellbeing risk. Please consider seeking professional help immediately.")
            st.write("**Advice:** Seek therapy, talk to family/friends, consider a professional counselor.")
        elif mental_result["risk_index"] == 1:
            st.info("Moderate mental wellbeing risk. Consider taking steps to improve mental health.")
            st.write("**Advice:** Take time for stress-reducing activities. Seek support if necessary.")
        else:
            st.success("Low mental wellbeing risk. Keep up the good work!")
            st.write("**Advice:** Continue engaging in positive activities and maintaining strong connections.")
    else:
        st.write("👈 Fill your details and click 'Analyze my mental wellbeing' to see results.")

# ----------------------------------------
# 3️⃣ Nearby Hospital Recommendations
# ----------------------------------------
st.header("3️⃣ Nearby Hospital Recommendations")

col_hosp_left, col_hosp_right = st.columns([1, 1], gap="medium")

with col_hosp_left:
    st.subheader("Location Settings")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        detect_and_save = st.button("📍 Detect & Save", key="detect_location")
    with col_btn2:
        clear_location = st.button("❌ Clear", key="clear_location")
    
    # Detect and save location (only when user requests)
    if detect_and_save:
        st.info("🔄 Requesting location access...")
        geo = streamlit_geolocation()
        # Save the raw geolocation result for diagnostics (even if fields are None)
        st.session_state["geolocation_raw"] = geo
        if geo and (geo.get("latitude") is not None and geo.get("longitude") is not None):
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            st.session_state["geolocation"] = {"lat": lat, "lon": lon}
            st.success(f"✓ Saved: {lat:.2f}, {lon:.2f}")
        else:
            st.warning("Browser location unavailable. Use manual input below.")
    
    # Clear saved location
    if clear_location:
        st.session_state["geolocation"] = None
        st.success("Location cleared.")
    
    # Manual location input as fallback
    st.divider()
    st.caption("**Or enter manually:**")
    # Quick approximate location using IP lookup (server-side) as an extra fallback
    if st.button("📡 Use approximate location (IP-based)", key="ip_location"):
        st.info("Looking up approximate location from your IP address...")
        try:
            r = requests.get("http://ip-api.com/json/", timeout=6)
            ip_data = r.json()
        except Exception as e:
            ip_data = {"status": "REQUEST_EXCEPTION", "error_message": str(e)}
        if ip_data.get("status") == "success":
            lat = ip_data.get("lat")
            lon = ip_data.get("lon")
            st.session_state["geolocation"] = {"lat": lat, "lon": lon}
            st.success(f"✓ Saved approximate location: {lat:.4f}, {lon:.4f}")
        else:
            st.warning("Unable to determine approximate location from IP.")
    col_lat, col_lon = st.columns([1, 1])
    with col_lat:
        manual_lat = st.number_input("Latitude", value=0.0, format="%.6f", key="manual_lat")
    with col_lon:
        manual_lon = st.number_input("Longitude", value=0.0, format="%.6f", key="manual_lon")
    
    if st.button("✅ Save Manual Location", key="save_manual"):
        if manual_lat != 0.0 or manual_lon != 0.0:
            st.session_state["geolocation"] = {"lat": manual_lat, "lon": manual_lon}
            st.success(f"✓ Saved: {manual_lat:.2f}, {manual_lon:.2f}")
        else:
            st.warning("Please enter valid coordinates.")
    
    # Show current saved location
    saved_geo = st.session_state.get("geolocation")
    if saved_geo:
        st.info(f"✓ Using saved: {saved_geo['lat']:.2f}, {saved_geo['lon']:.2f}")
    else:
        st.caption("No location saved yet.")

with col_hosp_right:
    st.subheader("Nearby Hospitals")
    
    # Get combined risk index from both assessments
    health_result = st.session_state.get("health_result")
    mental_result = st.session_state.get("mental_result")

    risk_index = health_result["risk_index"] if health_result else None
    mental_risk_index = mental_result["risk_index"] if mental_result else None

    combined_risk_index = max(
        risk_index if risk_index is not None else 0,
        mental_risk_index if mental_risk_index is not None else 0,
    )

    if combined_risk_index == 2:
        hospital_type = "emergency hospital"
    elif combined_risk_index == 1:
        hospital_type = "general clinic"
    else:
        hospital_type = "family medicine clinic"

    # Check for saved location and search for hospitals
    saved_geo = st.session_state.get("geolocation")
    if saved_geo:
        lat = saved_geo.get("lat")
        lon = saved_geo.get("lon")

        # Be explicit: allow negative coords and non-zero floats; only reject None
        if lat is not None and lon is not None:
            with st.spinner("🔍 Searching..."):
                try:
                    hospitals, raw = find_hospitals_by_coords(lat, lon, radius=5000)
                except Exception as e:
                    st.error(f"Error searching hospitals: {e}")
                    hospitals, raw = [], {"status": "EXCEPTION", "error_message": str(e)}

                if not hospitals:
                    st.warning("No hospitals found nearby.")
                else:
                    # Show top hospital
                    h = hospitals[0]
                    st.success(f"🏥 **{h['name']}**")
                    st.caption(f"Address: {h['address']}")
                    st.caption(f"Rating: {h.get('rating', 'N/A')} ⭐")
                    st.markdown(f"[View on Maps →](https://www.google.com/maps/search/?api=1&query={h['location']['lat']},{h['location']['lng']})")

                    # Show other hospitals in a compact way
                    if len(hospitals) > 1:
                        st.divider()
                        st.caption(f"**Other {len(hospitals)-1} hospitals nearby:**")
                        for i, hospital in enumerate(hospitals[1:], 1):
                            st.markdown(f"**{i}. {hospital['name']}**")
                            st.caption(f"{hospital['address']} | ⭐ {hospital.get('rating', 'N/A')}")
    else:
        st.info("👈 Save your location to see nearby hospitals.")

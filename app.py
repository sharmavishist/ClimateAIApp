import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import joblib
import warnings
import os
import gdown
import requests
import base64

warnings.filterwarnings("ignore")

# page config MUST be first streamlit command
st.set_page_config(
    page_title="Climate Change AI Agent",
    #page_icon="🌍",
    layout="wide")

# after set_page_config
def add_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# call function
add_bg_image("project_bg.jpg")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_GlobalLandTemperaturesByCountry.csv")
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_model():
    if not os.path.exists("climate_model.pkl"):
        with st.spinner("Loading trained ML model..."):
            # download climate model from Google Drive
            gdown.download(
                id="1M17J26z3pVdmkB3hygwToyCz-UYAdMsB", #Gdrive file ID for climate_model.pkl
                output="climate_model.pkl",
                quiet=False
            )
            # download label encoder from Google Drive
            gdown.download(
                id="1I9ZzkS8LhhOJmt9f51LWJl-HVtyONMGz", #Gdrive file ID for label_encoder.pkl
                output="label_encoder.pkl",
                quiet=False
            )
    
    # load our actual Colab trained model
    model = joblib.load("climate_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

df = load_data()
model, le = load_model()


# setup Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────

st.sidebar.title("Climate AI Agent")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["Dashboard", "Climate Chatbot", "Temperature Predictor", "Country Report"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info:**")
st.sidebar.markdown(f"Countries: {df['Country'].nunique()}")
st.sidebar.markdown(f"Years: {int(df['Year'].min())} - {int(df['Year'].max())}")
st.sidebar.markdown(f"Records: {len(df):,}")

# ─────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────

if page == "Dashboard":

    st.title("Climate Change Dashboard")
    st.markdown("Explore 270 years of real climate data across 204 countries")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Countries", df['Country'].nunique())
    col2.metric("Global Avg Temp", f"{df['AverageTemperature'].mean():.2f}°C")
    col3.metric("Data Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
    col4.metric("Total Records", f"{len(df):,}") 
    

    st.markdown("---")

    st.subheader("Global Temperature Trend Over Years")
    yearly_avg = df.groupby("Year")["AverageTemperature"].mean().reset_index()
    fig1 = px.line(
        yearly_avg, x="Year", y="AverageTemperature",
        title="Global Average Temperature (1743-2013)",
        color_discrete_sequence=["firebrick"]
    )
    fig1.update_layout(xaxis_title="Year", yaxis_title="Average Temperature (°C)")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Hottest Countries")
        hottest = df.groupby("Country")["AverageTemperature"].mean().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(
            hottest, x="AverageTemperature", y="Country",
            orientation="h",
            color="AverageTemperature",
            color_continuous_scale="Reds"
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Top 10 Coldest Countries")
        coldest = df.groupby("Country")["AverageTemperature"].mean().sort_values().head(10).reset_index()
        fig3 = px.bar(
            coldest, x="AverageTemperature", y="Country",
            orientation="h",
            color="AverageTemperature",
            color_continuous_scale="Blues_r"
        )
        fig3.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature by Season")
        season_avg = df.groupby("Season")["AverageTemperature"].mean().reset_index()
        fig4 = px.bar(
            season_avg, x="Season", y="AverageTemperature",
            color="Season",
            color_discrete_map={
                "Summer": "red",
                "Spring": "green",
                "Autumn": "orange",
                "Winter": "blue"
            }
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.subheader("Monthly Temperature Pattern")
        monthly_avg = df.groupby("Month")["AverageTemperature"].mean().reset_index()
        monthly_avg["Month Name"] = monthly_avg["Month"].map({
            1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",
            5:"May", 6:"Jun", 7:"Jul", 8:"Aug",
            9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        })
        fig5 = px.line(
            monthly_avg, x="Month Name", y="AverageTemperature",
            markers=True,
            color_discrete_sequence=["darkorange"]
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    st.subheader("🗺️ Average Temperature by Country")
    country_avg = df.groupby("Country")["AverageTemperature"].mean().reset_index()
    fig6 = px.choropleth(
        country_avg,
        locations="Country",
        locationmode="country names",
        color="AverageTemperature",
        color_continuous_scale="RdYlBu_r",
        title="Global Temperature Map"
    )
    st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — CLIMATE CHATBOT
# ─────────────────────────────────────────

elif page == "Climate Chatbot":

    st.title("Climate Chatbot")
    st.markdown("Ask any general question about the climate dataset!")

    st.info("""
    **Example questions you can ask:**
    - Which is the hottest country?
    - Which is the coldest country?
    - What is the global average temperature?
    - What time period does the data cover?
    - How many countries are in the dataset?
    - What is the overall warming trend?
    """)

    question = st.text_input("Ask your climate question:", placeholder="e.g. Which is the hottest country?")

    if st.button("Ask AI"):
        if question:
            with st.spinner("AI is thinking..."):

                top5_hot = df.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False).head(5)
                top5_cold = df.groupby('Country')['AverageTemperature'].mean().sort_values().head(5)
                yearly_trend = df.groupby('Year')['AverageTemperature'].mean()

                data_summary = f"""
                You are a climate data expert assistant.
                You have access to the following real climate dataset:
                - Total countries: {df['Country'].nunique()}
                - Data range: {df['Year'].min()} to {df['Year'].max()}
                - Global average temperature: {df['AverageTemperature'].mean():.2f}°C
                Top 5 hottest countries:
                {top5_hot.to_string()}
                Top 5 coldest countries:
                {top5_cold.to_string()}
                Temperature in earliest year: {yearly_trend.iloc[0]:.2f}°C
                Temperature in latest year: {yearly_trend.iloc[-1]:.2f}°C
                Overall warming: {yearly_trend.iloc[-1] - yearly_trend.iloc[0]:.2f}°C
                Answer only questions based on this data.
                For prediction questions tell user to use Temperature Predictor page.
                """

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": data_summary + "\n\nUser question: " + question}],
                    max_tokens=500
                )

                st.success("**AI Answer:**")
                st.write(response.choices[0].message.content)
        else:
            st.warning("Please enter a question!")

# ─────────────────────────────────────────
# PAGE 3 — TEMPERATURE PREDICTOR
# ─────────────────────────────────────────

elif page == "Temperature Predictor":

    st.title("Temperature Predictor")
    st.markdown("Predict temperature for any country using our ML model!")

    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox("Select Country:", sorted(df['Country'].unique()))

    with col2:
        year = st.number_input("Enter Year:", min_value=1743, max_value=2100, value=2025)

    with col3:
        month = st.selectbox("Select Month:", [
            (1, "January"), (2, "February"), (3, "March"),
            (4, "April"), (5, "May"), (6, "June"),
            (7, "July"), (8, "August"), (9, "September"),
            (10, "October"), (11, "November"), (12, "December")
        ], format_func=lambda x: x[1])

    if st.button("Predict Temperature"):
        with st.spinner("Processing..."):

            month_num = month[0]

            # check if year is in dataset range
            if year <= 2013:
                # fetch ACTUAL data from dataset
                actual_data = df[
                    (df['Country'] == country) &
                    (df['Month'] == month_num) &
                    (df['Year'] == year)
                ]['AverageTemperature']

                if len(actual_data) > 0:
                    # actual data exists for this year
                    temperature = actual_data.values[0]
                    data_type = "Actual Historical Data"
                    delta_label = "From dataset"
                else:
                    # year <= 2013 but no data for this country/month/year
                    # fall back to average
                    temperature = df[
                        (df['Country'] == country) &
                        (df['Month'] == month_num)
                    ]['AverageTemperature'].mean()
                    data_type = "Historical Average"
                    delta_label = "No exact record found"
            else:
                # year > 2013 — use ML model to predict
                country_encoded = le.transform([country])[0]
                input_data = pd.DataFrame({
                    'Country_Encoded': [country_encoded],
                    'Year': [year],
                    'Month': [month_num]
                })
                temperature = model.predict(input_data)[0]
                data_type = "ML Prediction"
                delta_label = "Future prediction"

            # get historical average for comparison
            historical = df[
                (df['Country'] == country) &
                (df['Month'] == month_num)
            ]['AverageTemperature'].mean()

            # show results
            st.markdown(f"### Data Type: `{data_type}`")
            col1, col2 = st.columns(2)
            col1.metric(
                "Temperature",
                f"{temperature:.1f}°C",
                delta=delta_label
            )
            col2.metric(
                "Historical Average",
                f"{historical:.1f}°C",
                delta=f"{temperature - historical:.1f}°C vs average"
            )

            # get AI explanation
            with st.spinner("Getting AI explanation..."):
                explanation_prompt = f"""
                You are a climate expert. Explain this temperature data:
                Country: {country}
                Month: {month[1]}
                Year: {year}
                Temperature: {temperature:.1f}°C
                Data type: {data_type}
                Historical average for this month: {historical:.1f}°C

                If this is actual historical data, explain what was
                happening climatically in {country} in {year}.
                If this is a future ML prediction, explain what
                factors might influence this temperature.
                Give a brief 3-4 sentence explanation.
                """

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": explanation_prompt}],
                    max_tokens=300
                )

                st.info("**AI Explanation:**")
                st.write(response.choices[0].message.content)

            # show historical trend chart
            st.subheader(f"📈 Historical Temperature Trend for {country}")
            country_yearly = df[
                df['Country'] == country
            ].groupby('Year')['AverageTemperature'].mean().reset_index()

            fig = px.line(
                country_yearly, x="Year", y="AverageTemperature",
                color_discrete_sequence=["steelblue"]
            )

            # add marker for selected year
            fig.add_vline(
                x=year,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{year} ({temperature:.1f}°C)"
            )

            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — COUNTRY REPORT
# ─────────────────────────────────────────

elif page == "Country Report":

    st.title("📄 Country Climate Report")
    st.markdown("Generate a full AI climate report for any country!")

    # country selection
    country = st.selectbox("Select Country:", sorted(df['Country'].unique()))

    # year range slider
    st.markdown("**Select Year Range:**")
    year_range = st.slider(
        "Year Range",
        min_value=int(df['Year'].min()),   # 1743
        max_value=int(df['Year'].max()),   # 2013
        value=(int(df['Year'].min()), int(df['Year'].max())),  # default full range
        step=1
    )

    # season dropdown
    season = st.selectbox(
        "Select Season (optional):",
        ["All Seasons", "Summer", "Spring", "Autumn", "Winter"]
    )

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):

            # filter data for selected country
            country_data = df[df['Country'] == country]

            # filter by year range
            country_data = country_data[
                (country_data['Year'] >= year_range[0]) &
                (country_data['Year'] <= year_range[1])
            ]

            # filter by season if selected
            if season != "All Seasons":
                country_data = country_data[country_data['Season'] == season]

            # check if data exists
            if len(country_data) == 0:
                st.warning("No data found for selected filters!")
                st.stop()

            # calculate statistics
            avg_temp = country_data['AverageTemperature'].mean()
            max_temp = country_data['AverageTemperature'].max()
            min_temp = country_data['AverageTemperature'].min()

            # find when max and min occurred
            max_temp_row = country_data[country_data['AverageTemperature'] == max_temp].iloc[0]
            min_temp_row = country_data[country_data['AverageTemperature'] == min_temp].iloc[0]

            max_temp_year = int(max_temp_row['Year'])
            max_temp_month = int(max_temp_row['Month'])
            min_temp_year = int(min_temp_row['Year'])
            min_temp_month = int(min_temp_row['Month'])

            # month names
            month_names = {
                1:"January", 2:"February", 3:"March",
                4:"April", 5:"May", 6:"June",
                7:"July", 8:"August", 9:"September",
                10:"October", 11:"November", 12:"December"
            }

            # hottest and coldest months
            hottest_month_num = int(country_data.groupby('Month')['AverageTemperature'].mean().idxmax())
            coldest_month_num = int(country_data.groupby('Month')['AverageTemperature'].mean().idxmin())
            hottest_month = month_names[hottest_month_num]
            coldest_month = month_names[coldest_month_num]

            # warming trend
            mid_year = (year_range[0] + year_range[1]) // 2
            early_avg = country_data[country_data['Year'] <= mid_year]['AverageTemperature'].mean()
            recent_avg = country_data[country_data['Year'] > mid_year]['AverageTemperature'].mean()
            warming = recent_avg - early_avg
            warming_text = f"{warming:.2f}°C"

            # ── KPI CARDS ──
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**🌡️ Avg Temperature**")
                st.markdown(f"### {avg_temp:.1f}°C")

            with col2:
                st.markdown("**🔥 Hottest Month**")
                st.markdown(f"### {hottest_month}")

            with col3:
                st.markdown("**❄️ Coldest Month**")
                st.markdown(f"### {coldest_month}")

            with col4:
                st.markdown("**📈 Warming Trend**")
                st.markdown(f"#### {warming_text}")

            st.markdown("---")

            # ── LINE CHART ──
            st.subheader(f"📈 Temperature Trend for {country} ({year_range[0]} - {year_range[1]})")

            # group by year for line chart
            country_yearly = country_data.groupby('Year')['AverageTemperature'].mean().reset_index()

            fig = px.line(
                country_yearly,
                x="Year",
                y="AverageTemperature",
                color_discrete_sequence=["steelblue"],
                title=f"{country} Temperature Trend ({season})"
            )

            # add max temp marker
            fig.add_scatter(
                x=[max_temp_year],
                y=[max_temp],
                mode="markers+text",
                marker=dict(color="red", size=10),
                text=[f"Max: {max_temp:.1f}°C"],
                textposition="top center",
                name="Max Temp"
            )

            # add min temp marker
            fig.add_scatter(
                x=[min_temp_year],
                y=[min_temp],
                mode="markers+text",
                marker=dict(color="blue", size=10),
                text=[f"Min: {min_temp:.1f}°C"],
                textposition="bottom center",
                name="Min Temp"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── MIN MAX INFO ──
            st.markdown("---")
            st.subheader("🌡️ Temperature Extremes")

            col1, col2 = st.columns(2)

            with col1:
                st.error(f"""
                **🔥 Highest Temperature**
                - Temperature: **{max_temp:.1f}°C**
                - Year: **{max_temp_year}**
                - Month: **{month_names[max_temp_month]}**
                """)

            with col2:
                st.info(f"""
                **❄️ Lowest Temperature**
                - Temperature: **{min_temp:.1f}°C**
                - Year: **{min_temp_year}**
                - Month: **{month_names[min_temp_month]}**
                """)

            st.markdown("---")

            # ── AI REPORT ──
            st.subheader("🤖 AI Climate Analysis")

            report_prompt = f"""
            Generate a professional climate analysis report for {country}.

            Filters applied:
            - Year range: {year_range[0]} to {year_range[1]}
            - Season filter: {season}

            Statistics:
            - Average temperature: {avg_temp:.1f}°C
            - Highest recorded: {max_temp:.1f}°C in {month_names[max_temp_month]} {max_temp_year}
            - Lowest recorded: {min_temp:.1f}°C in {month_names[min_temp_month]} {min_temp_year}
            - Hottest month: {hottest_month}
            - Coldest month: {coldest_month}
            - Warming trend: {warming_text} change from first half to second half of selected period
            - Data range: {year_range[0]} to {year_range[1]}

            Write a 5-6 sentence professional climate report covering:
            1. Overall climate summary for selected period
            2. Seasonal patterns
            3. Notable temperature extremes and when they occurred
            4. Warming trend analysis
            5. Climate outlook
            """

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": report_prompt}],
                max_tokens=500
            )

            st.success("**AI Generated Report:**")
            st.write(response.choices[0].message.content)

            # ── MONTHLY PATTERN CHART ──
            st.markdown("---")
            st.subheader(f"📅 Monthly Pattern for {country}")
            country_monthly = country_data.groupby('Month')['AverageTemperature'].mean().reset_index()
            country_monthly['Month Name'] = country_monthly['Month'].map(month_names)
            fig2 = px.bar(
                country_monthly,
                x="Month Name",
                y="AverageTemperature",
                color="AverageTemperature",
                color_continuous_scale="RdYlBu_r",
                title=f"{country} Monthly Temperature Pattern ({year_range[0]}-{year_range[1]})"
            )
            st.plotly_chart(fig2, use_container_width=True)
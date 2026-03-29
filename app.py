import streamlit as st

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from groq import Groq
    import warnings
    warnings.filterwarnings("ignore")
    st.success("Libraries imported! ✅")
except Exception as e:
    st.error(f"Library import error: {e}")
    st.stop()

try:
    df = pd.read_csv("cleaned_GlobalLandTemperaturesByCountry.csv")
    df.dropna(inplace=True)
    st.success(f"Data loaded! {df.shape} ✅")
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()
    
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import warnings
warnings.filterwarnings("ignore")

# page configuration
st.set_page_config(
    page_title="Climate Change AI Agent",
    page_icon="🌍",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA AND TRAIN MODEL (runs once)
# ─────────────────────────────────────────

@st.cache_data  # cache data so it loads only once
def load_data():
    df = pd.read_csv("cleaned_GlobalLandTemperaturesByCountry.csv")
    df.dropna(inplace=True)
    return df

@st.cache_resource  # cache model so it trains only once
def train_model(df):
    le = LabelEncoder()
    le.fit(df["Country"])
    X = df[["Country_Encoded", "Year", "Month"]]
    y = df["AverageTemperature"]
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, le

# load data and train model
df = load_data()
model, le = train_model(df)

# setup Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────

st.sidebar.title("🌍 Climate AI Agent")
st.sidebar.markdown("---")

# navigation menu
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
    
    st.title("🌍 Climate Change Dashboard")
    st.markdown("Explore 270 years of real climate data across 204 countries")
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Countries", df['Country'].nunique())
    col2.metric("Global Avg Temp", f"{df['AverageTemperature'].mean():.2f}°C")
    col3.metric("Data Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
    col4.metric("Total Records", f"{len(df):,}")
    
    st.markdown("---")
    
    # chart 1 — global temperature trend
    st.subheader("🌡️ Global Temperature Trend Over Years")
    yearly_avg = df.groupby("Year")["AverageTemperature"].mean().reset_index()
    fig1 = px.line(
        yearly_avg, x="Year", y="AverageTemperature",
        title="Global Average Temperature (1743-2013)",
        color_discrete_sequence=["firebrick"]
    )
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Temperature (°C)"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    # chart 2 — top 10 hottest and coldest countries
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Top 10 Hottest Countries")
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
        st.subheader("❄️ Top 10 Coldest Countries")
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
    
    # chart 3 — temperature by season
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍂 Temperature by Season")
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
        st.subheader("📅 Monthly Temperature Pattern")
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
    
    # chart 4 — world map
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
    
    st.title("🤖 Climate Chatbot")
    st.markdown("Ask any general question about the climate dataset!")
    
    # example questions
    st.info("""
    **Example questions you can ask:**
    - Which is the hottest country?
    - Which is the coldest country?
    - What is the global average temperature?
    - What time period does the data cover?
    - How many countries are in the dataset?
    - What is the overall warming trend?
    """)
    
    # chat input
    question = st.text_input("Ask your climate question:", placeholder="e.g. Which is the hottest country?")
    
    if st.button("Ask AI"):
        if question:
            with st.spinner("AI is thinking..."):
                
                # prepare dataset summary
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
                
                Temperature in earliest year ({df['Year'].min()}): {yearly_trend.iloc[0]:.2f}°C
                Temperature in latest year ({df['Year'].max()}): {yearly_trend.iloc[-1]:.2f}°C
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
    
    st.title("🌡️ Temperature Predictor")
    st.markdown("Predict temperature for any country using our ML model!")
    
    # input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # dropdown of all available countries
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
        with st.spinner("Predicting..."):
            
            # get month number
            month_num = month[0]
            
            # encode country
            country_encoded = le.transform([country])[0]
            
            # prepare input
            input_data = pd.DataFrame({
                'Country_Encoded': [country_encoded],
                'Year': [year],
                'Month': [month_num]
            })
            
            # predict
            predicted_temp = model.predict(input_data)[0]
            
            # get historical average
            historical = df[
                (df['Country'] == country) &
                (df['Month'] == month_num)
            ]['AverageTemperature'].mean()
            
            # show results
            col1, col2 = st.columns(2)
            col1.metric("Predicted Temperature", f"{predicted_temp:.1f}°C")
            col2.metric("Historical Average", f"{historical:.1f}°C",
                       delta=f"{predicted_temp - historical:.1f}°C")
            
            # get AI explanation
            with st.spinner("Getting AI explanation..."):
                explanation_prompt = f"""
                You are a climate expert. Explain this ML temperature prediction briefly:
                Country: {country}
                Month: {month[1]}
                Year: {year}
                Predicted Temperature: {predicted_temp:.1f}°C
                Historical Average: {historical:.1f}°C
                Give a brief 3-4 sentence explanation.
                """
                
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": explanation_prompt}],
                    max_tokens=300
                )
                
                st.info("**AI Explanation:**")
                st.write(response.choices[0].message.content)
            
            # show historical trend chart for selected country
            st.subheader(f"📈 Historical Temperature Trend for {country}")
            country_yearly = df[df['Country'] == country].groupby('Year')['AverageTemperature'].mean().reset_index()
            fig = px.line(
                country_yearly, x="Year", y="AverageTemperature",
                color_discrete_sequence=["steelblue"]
            )
            fig.add_hline(y=predicted_temp, line_dash="dash",
                         line_color="red", annotation_text="Predicted")
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — COUNTRY REPORT
# ─────────────────────────────────────────

elif page == "Country Report":
    
    st.title("📄 Country Climate Report")
    st.markdown("Generate a full AI climate report for any country!")
    
    # country selection
    country = st.selectbox("Select Country:", sorted(df['Country'].unique()))
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            
            # filter data
            country_data = df[df['Country'] == country]
            
            # calculate statistics
            avg_temp = country_data['AverageTemperature'].mean()
            max_temp = country_data['AverageTemperature'].max()
            min_temp = country_data['AverageTemperature'].min()
            
            month_names = {
                1:"January", 2:"February", 3:"March",
                4:"April", 5:"May", 6:"June",
                7:"July", 8:"August", 9:"September",
                10:"October", 11:"November", 12:"December"
            }
            
            hottest_month_num = int(country_data.groupby('Month')['AverageTemperature'].mean().idxmax())
            coldest_month_num = int(country_data.groupby('Month')['AverageTemperature'].mean().idxmin())
            hottest_month = month_names[hottest_month_num]
            coldest_month = month_names[coldest_month_num]
            
            early_data = country_data[country_data['Year'] < 1900]['AverageTemperature']
            recent_data = country_data[country_data['Year'] > 1980]['AverageTemperature']
            
            if len(early_data) > 0 and len(recent_data) > 0:
                warming = recent_data.mean() - early_data.mean()
                warming_text = f"{warming:.2f}°C since 1900"
            else:
                warming_text = "Not available (limited data)"
            
            # show KPI cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            col2.metric("Hottest Month", hottest_month)
            col3.metric("Coldest Month", coldest_month)
            col4.metric("Warming Trend", warming_text)
            
            # generate AI report
            report_prompt = f"""
            Generate a professional climate analysis report for {country}.
            
            Statistics:
            - Average temperature: {avg_temp:.1f}°C
            - Highest recorded: {max_temp:.1f}°C
            - Lowest recorded: {min_temp:.1f}°C
            - Hottest month: {hottest_month}
            - Coldest month: {coldest_month}
            - Warming trend: {warming_text}
            - Data range: {int(country_data['Year'].min())} to {int(country_data['Year'].max())}
            
            Write a 5-6 sentence professional climate report.
            """
            
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": report_prompt}],
                max_tokens=500
            )
            
            st.success("**AI Generated Report:**")
            st.write(response.choices[0].message.content)
            
            # show temperature chart
            st.subheader(f"📈 Temperature Trend for {country}")
            country_yearly = df[df['Country'] == country].groupby('Year')['AverageTemperature'].mean().reset_index()
            fig = px.line(
                country_yearly, x="Year", y="AverageTemperature",
                color_discrete_sequence=["steelblue"],
                title=f"{country} Temperature Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # show monthly pattern
            st.subheader(f"📅 Monthly Pattern for {country}")
            country_monthly = country_data.groupby('Month')['AverageTemperature'].mean().reset_index()
            country_monthly['Month Name'] = country_monthly['Month'].map(month_names)
            fig2 = px.bar(
                country_monthly, x="Month Name", y="AverageTemperature",
                color="AverageTemperature",
                color_continuous_scale="RdYlBu_r",
                title=f"{country} Monthly Temperature Pattern"
            )
            st.plotly_chart(fig2, use_container_width=True)

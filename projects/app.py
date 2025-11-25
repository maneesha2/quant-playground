import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fertility Analysis Nepal", layout="wide")
st.markdown("##### Determinants of Fertility in Nepal (2000–2022)")

# -----------------------------
# Project Overview
# -----------------------------
st.markdown("""
**Overview:**  
Nepal has experienced a significant **demographic and socioeconomic transition** over the past two decades. Fertility rates have declined, life expectancy has increased, and the sectors like education, female labor participation, and urban development has been improving.  

Understanding the factors behind these changes is crucial for **policymakers, healthcare providers, and social planners**.

**Objective:**  
Explore the relationship between fertility rates and key socioeconomic factors such as **education, labor participation, urbanization, and economic development**.  
Identify which variables have the **strongest impact** on fertility decline in Nepal from 2000 to 2022.

**Scope of Analysis:**  
- **Temporal Analysis:** Track fertility rates and socioeconomic changes over 2000–2022.  
- **Correlation Assessment:** Understand strength and direction of relationships between fertility and predictors.  
- **Regression Modeling:** Quantify influence of predictors and assess predictive accuracy.  
- **Elasticity Analysis:** Measure relative impact of each predictor on fertility.
""")

# -----------------------------
# Load and Prepare Data
# -----------------------------
df = pd.read_csv('datasets/Region_WB_sm.csv')
df_nepal = df[df['country'] == 'Nepal'].copy()

# Interpolate missing data
df_nepal['contraceptive'] = df_nepal['contraceptive'].interpolate()
df_nepal['sec_school_f'] = df_nepal['sec_school_f'].interpolate()
df_nepal['year'] = df_nepal['year'].astype(int)

# Rename columns for clarity
df_nepal.rename(columns={
    'fertility': 'fertility_rate',
    'adolescent_fert': 'adolescent_fert_rate',
    'contraceptive': 'contraceptive_use',
    'infant_mort': 'infant_mortality',
    'life_exp': 'life_expectancy',
    'gdp_pc': 'gdp_per_capita',
    'fem_labor': 'female_labor_participation',
    'sec_school_f': 'female_secondary_school'
}, inplace=True)

numeric_cols = ['fertility_rate','adolescent_fert_rate','contraceptive_use','infant_mortality', 
                'life_expectancy','gdp_per_capita','female_labor_participation',
                'female_secondary_school','urban']

st.markdown("------")
col1, col2 = st.columns(2)

with col1:
    fig = px.line(df_nepal, x='year', y='fertility_rate', markers=True,
                  title='Fertility Rate in Nepal (2000–2022)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Fertility declined rapidly from 2000–2010, followed by a steady decline until 2022.")

with col2:
    fig = px.line(df_nepal, x='year', 
                  y=['life_expectancy','gdp_per_capita','female_secondary_school','urban'], 
                  markers=True, 
                  title='Socioeconomic Indicators (2000–2022)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
- **GDP per capita** increased rapidly  
- **Female secondary school enrollment** increased steadily  
- **Life expectancy** increased minimally  
- **Urbanization** showed moderate growth
""")
st.markdown("------")
# -----------------------------
# Correlation Analysis (Key Insights)
# -----------------------------
st.markdown("Correlation Analysis : Relationship of fertility rate with (Socioeconomic Variables)")
st.markdown("""
- **Values close to +1 or –1 indicate very strong relationships.**
- **+1 indicates a strong positive relationship** (as one variable increases, the other also increases).
- **–1 indicates a strong inverse relationship** (as one variable increases, the other decreases).
- **Values near 0 suggest weak or no linear relationship between variables.**
""")

corr_matrix = df_nepal[numeric_cols].corr()
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Viridis',
    text=corr_matrix.values,
    texttemplate="%{text:.2f}"
))
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**Key Relationships:**

- **Fertility decreases with:** contraceptive use, life expectancy, GDP per capita, female labor participation, female secondary school enrollment and urbanization 
- **Fertility increases with:** Infant Mortality, Adolescent Fertility Rate  
""")
st.markdown("------")

# -----------------------------
# Regression Analysis
# -----------------------------
st.markdown("##### Fertility Rate Regression Analysis")

st.markdown("""
**Objective:**  
To understand how socioeconomic factors influence fertlity rate using OLS regression

**Model Overview:**  
- **Dependent Variable:** Fertility Rate  
- **Predictors:** Life Expectancy, GDP per Capita, Female Labor Participation, Female Secondary School Enrollment, Urbanization  
- **Method Used:** Ordinary Least Squares (OLS)

**Why OLS?**  
OLS is used because it’s simple, easy to interpret, and clearly shows how each factor is associated with changes in fertility. It’s a reliable method when the relationships between variables are mostly linear.
""")

X = df_nepal[['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']]
y = df_nepal['fertility_rate']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

with st.expander("Regression Summary"):
    st.text(model.summary())
    
st.markdown("""
**Key Findings:**  
- R² = 0.983 → 98% of variation explained  
- Significant negative predictors: Life Expectancy, Female Labor Participation, Urbanization  
- Female education: minimal direct effect  
- GDP per capita: minor positive effect (possible multicollinearity)
""")
st.markdown("------")
# -----------------------------
# Elasticity Analysis
# -----------------------------
st.markdown("##### Elasticity of Fertility Rate")
st.markdown("Elasticity provides a standardized measure to compare the strength of each predictor, highlighting which factors have the greatest practical impact on fertility rate.")
elasticity = {}
for col in ['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']:
    beta = model.params[col]
    elasticity[col] = (beta * df_nepal[col].mean()) / df_nepal['fertility_rate'].mean()
elasticity = dict(sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True))
# st.dataframe(pd.DataFrame(list(elasticity.items()), columns=['Variable', 'Elasticity']))

st.markdown("""
**Key Findings:**
- **Life Expectancy:** –4.41 → Life expectancy is most influential variable as its 1% increase reduces fertility by  ~4.4%  
- **Urbanization:** –1.18 → Increased urban living strongly reduces fertility as (~1.18% decline per 1% increase)
- **Female Labor Participation:** –1.13 → Higher female employability lowers fertility (~1.13% decline per 1% increase)  
- **GDP per Capita:** 0.83 → Minor positive effect  
- **Female Secondary School Enrollment:** –0.038 → Minimal direct effect
""")
st.markdown("------")
# -----------------------------
# Predicted vs Actual Fertility
# -----------------------------
# -----------------------------
# Predicted vs Actual Fertility (Card-like)
# -----------------------------
with st.container():
    st.subheader("Predicted vs Actual Fertility Rate")
    st.markdown("""
This chart compares **actual fertility rates** with **predicted values** from the regression model.  
A red dashed line represents the ideal 1:1 match; points close to the line indicate high prediction accuracy.
""")
    
    df_nepal['predicted_fertility'] = model.predict(X)
    fig = px.scatter(
        df_nepal, 
        x='fertility_rate', 
        y='predicted_fertility', 
        labels={'fertility_rate': 'Actual Fertility Rate', 
                'predicted_fertility': 'Predicted Fertility Rate'},
        color='year',
        hover_data=['year'],
        title='Actual vs Predicted Fertility Rate'
    )
    
    # Add 1:1 line
    fig.add_shape(
        type='line',
        x0=df_nepal['fertility_rate'].min(), 
        y0=df_nepal['fertility_rate'].min(),
        x1=df_nepal['fertility_rate'].max(), 
        y1=df_nepal['fertility_rate'].max(),
        line=dict(color='red', dash='dash'),
        name='Ideal Fit'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("The scatter plots shows high accuracy of the regression model in predicting fertility rates over the years. Predicted values closely followed the downward trend in fertility from 3.98 (2000) to 1.98 (2023). There is minor deviations in some years, but the overall predictive performance is strong with R² = 0.983.")



st.markdown("---")
# -----------------------------
# Regression Coefficients & Elasticity Ranking
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression Coefficients")
    coef_df = pd.DataFrame({'Variable': model.params.index[1:], 'Coefficient': model.params.values[1:]})
    fig = px.bar(coef_df, x='Coefficient', y='Variable', orientation='h', text='Coefficient',
                 color='Coefficient', color_continuous_scale='Viridis',
                 title='Regression Coefficients')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    ##### Regression Coefficient Summary

    - **Life expectancy, urbanization, and female labor participation** have the strongest negative effects on fertility.
    - **Female education** has minimal independent impact, likely due to overlap with other variables.
    - **GDP per capita** shows a small positive effect.
    """)




with col2:
    st.subheader("Elasticity Ranking")
    elasticity_df = pd.DataFrame({'Variable': list(elasticity.keys()), 'Elasticity': list(elasticity.values())})
    fig = px.bar(elasticity_df, x='Elasticity', y='Variable', orientation='h', text='Elasticity',
                 color='Elasticity', color_continuous_scale='Cividis',
                 title='Elasticity of Fertility Rate w.r.t Predictors')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
###### Elasticity Summary

- **Life expectancy** shows the strongest negative elasticity, making it the most influential driver of fertility.
- **Urbanization** and **female labor participation** also significantly reduce fertility.
- **GDP per capita** has a moderate positive elasticity.
- **Female secondary schooling** shows very low elasticity, indicating minimal direct impact.
""")


# -----------------------------
# Conclusions
# -----------------------------
st.markdown("""
### Conclusions
- Nepal’s fertility rate has steadily declined, driven by improvements in **life expectancy, female labor participation, and urbanization**.  
- **Female education** has a positive indirect role but shows minimal direct effect on fertility in this period.  
- **Economic development (GDP per capita)** shows a minor positive effect, possibly due to multicollinearity.  
- Elasticity analysis highlights **life expectancy** and **urbanization** as the most influential predictors.  
- Policy implications: Hence, focus on female empowerment, urban planning, and healthcare improvements is necessary to sustain fertility decline. As fertility rate of Nepal rapidly declines, the focus of demographic planning must be shift. 
  Future policy should focus on managing economic impact of a slowing population growth specifically by preparing for potential challenges like **labor shortages** and an **elderly dependency ratio**.
             
""")

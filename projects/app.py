import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fertility Analysis Nepal", layout="wide")
st.title("Determinants of Fertility in Nepal (2000–2022)")

# -----------------------------
# Project Overview
# -----------------------------
with st.expander("Project Overview", expanded=True):
    st.markdown("""
**Overview:**  
Nepal has experienced a significant **demographic and socioeconomic transition** over the past two decades. Fertility rates have declined, life expectancy has increased, and sectors like education, female labor participation, and urban development have improved.  

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

# Interpolate missing data with logging
interpolated_cols = ['contraceptive', 'sec_school_f']
for col in interpolated_cols:
    missing_before = df_nepal[col].isna().sum()
    df_nepal[col] = df_nepal[col].interpolate()
    missing_after = df_nepal[col].isna().sum()
    st.markdown(f"**{col}**: interpolated {missing_before - missing_after} missing values.")

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

numeric_cols = [
    'fertility_rate','adolescent_fert_rate','contraceptive_use','infant_mortality', 
    'life_expectancy','gdp_per_capita','female_labor_participation',
    'female_secondary_school','urban'
]

st.markdown("---")

# -----------------------------
# Temporal Trends
# -----------------------------
st.subheader("Temporal Trends")
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

st.markdown("---")

# -----------------------------
# Correlation Analysis
# -----------------------------
st.subheader("Correlation Analysis")
st.markdown("""
- **Values close to +1 or –1 indicate strong relationships.**  
- **+1 indicates positive relationship** (both variables increase together).  
- **–1 indicates inverse relationship** (one variable increases, the other decreases).  
- **Values near 0 suggest weak or no linear relationship.**
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
- Fertility decreases with: contraceptive use, life expectancy, GDP per capita, female labor participation, female secondary school enrollment, urbanization.  
- Fertility increases with: Infant Mortality, Adolescent Fertility Rate.  
""")

st.markdown("---")

# -----------------------------
# Regression Analysis
# -----------------------------
st.subheader("Fertility Rate Regression Analysis")
st.markdown("""
**Objective:**  
Understand how socioeconomic factors influence fertility rate using OLS regression.

**Model Overview:**  
- **Dependent Variable:** Fertility Rate  
- **Predictors:** Life Expectancy, GDP per Capita, Female Labor Participation, Female Secondary School Enrollment, Urbanization  
- **Method:** Ordinary Least Squares (OLS)

**Why OLS?**  
OLS is simple, interpretable, and shows how each factor is associated with fertility changes. Effective for mostly linear relationships.
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

st.markdown("---")

# -----------------------------
# Elasticity Analysis
# -----------------------------
st.subheader("Elasticity of Fertility Rate")
st.markdown("Elasticity provides a standardized measure to compare the strength of each predictor, highlighting which factors have the greatest practical impact on fertility rate.")

elasticity = {}
for col in ['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']:
    beta = model.params[col]
    elasticity[col] = (beta * df_nepal[col].mean()) / df_nepal['fertility_rate'].mean()
elasticity = dict(sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True))

st.markdown("""
**Key Findings:**
- **Life Expectancy:** –4.41 → Most influential; 1% increase reduces fertility by ~4.4%  
- **Urbanization:** –1.18 → 1% increase leads to ~1.18% decline in fertility  
- **Female Labor Participation:** –1.13 → Higher female employability lowers fertility (~1.13% decline per 1% increase)  
- **GDP per Capita:** 0.83 → Minor positive effect  
- **Female Secondary School Enrollment:** –0.038 → Minimal direct effect
""")

st.markdown("---")

# -----------------------------
# Predicted vs Actual Fertility
# -----------------------------
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
    hover_data=['year', 'life_expectancy', 'female_labor_participation'],
    title='Actual vs Predicted Fertility Rate'
)
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
st.markdown("Predicted values closely follow the downward trend from 3.98 (2000) to 1.98 (2023), showing strong predictive performance (R² = 0.983).")

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
**Summary:**  
- Life expectancy, urbanization, and female labor participation have the strongest negative effects.  
- Female education has minimal independent impact.  
- GDP per capita shows a small positive effect.
""")

with col2:
    st.subheader("Elasticity Ranking")
    elasticity_df = pd.DataFrame({'Variable': list(elasticity.keys()), 'Elasticity': list(elasticity.values())})
    fig = px.bar(elasticity_df, x='Elasticity', y='Variable', orientation='h', text='Elasticity',
                 color='Elasticity', color_continuous_scale='Cividis',
                 title='Elasticity of Fertility Rate w.r.t Predictors')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**Summary:**  
- Life expectancy has the strongest negative elasticity.  
- Urbanization and female labor participation also significantly reduce fertility.  
- GDP per capita has moderate positive elasticity.  
- Female secondary schooling shows minimal direct effect.
""")

# -----------------------------
# Conclusions
# -----------------------------
st.markdown("---")
st.subheader("Conclusions")
st.markdown("""
- Nepal’s fertility rate has steadily declined, driven by improvements in **life expectancy, female labor participation, and urbanization**.  
- **Female education** shows minimal direct effect but plays a positive indirect role.  
- **GDP per capita** shows a minor positive effect, possibly due to multicollinearity.  
- Elasticity analysis highlights **life expectancy** and **urbanization** as the most influential predictors.  
- **Policy implications:** Focus on female empowerment, urban planning, and healthcare improvements to sustain fertility decline.  
- Future policies of Nepal should be prepare for potential challenges of slowing population growth: **labor shortages** and increasing **elderly dependency ratio**.
""")

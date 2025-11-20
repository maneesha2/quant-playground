import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="Fertility Analysis Nepal", layout="wide")
st.title("📈 Determinants of Fertility in Nepal (2000–2022)")

# -----------------------------
# Project Overview
# -----------------------------
st.header("Project Overview")
st.markdown("""
**Objective:**  
Analyze and quantify how key socioeconomic and demographic variables have influenced Nepal’s fertility rate over time, including:
- Life expectancy  
- GDP per capita  
- Female education  
- Labor force participation  
- Urbanization  

**Goal:**  
- Use statistical and quantitative methods to identify the major drivers of fertility decline in Nepal.  
- Provide insights into the country's demographic and socioeconomic transition.
""")

# -----------------------------
# Load Data
# -----------------------------
st.header("Data")
df = pd.read_csv('datasets/Region_WB_sm.csv')
df_nepal = df[df['country'] == 'Nepal'].copy()
df_nepal['contraceptive'] = df_nepal['contraceptive'].interpolate(method='linear')
df_nepal['sec_school_f'] = df_nepal['sec_school_f'].interpolate(method='linear')
df_nepal['year'] = df_nepal['year'].astype(int)

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

st.dataframe(df_nepal.head())

numeric_cols = ['fertility_rate','adolescent_fert_rate','contraceptive_use','infant_mortality',
                'life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']

# -----------------------------
# Fertility Trend
# -----------------------------
st.header("Fertility Rate Trend")
fig = px.line(df_nepal, x='year', y='fertility_rate', markers=True,
              title='Fertility Rate Trend in Nepal (2000-2022)')
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Socioeconomic Trends
# -----------------------------
st.header("Socioeconomic Trends")
fig = px.line(df_nepal, x='year', y=['life_expectancy','gdp_per_capita','female_secondary_school','urban'],
              markers=True, title='Socioeconomic Trends in Nepal (2000-2022)')
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.header("Correlation Heatmap")
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

# -----------------------------
# Regression Analysis
# -----------------------------
st.header("Regression Analysis")
X = df_nepal[['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']]
y = df_nepal['fertility_rate']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.subheader("Regression Summary")
st.text(model.summary())

# Elasticity
st.subheader("Elasticity of Fertility Rate")
elasticity = {}
for col in ['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']:
    beta = model.params[col]
    elasticity[col] = (beta * df_nepal[col].mean()) / df_nepal['fertility_rate'].mean()
elasticity = dict(sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True))
st.dataframe(pd.DataFrame(list(elasticity.items()), columns=['Variable', 'Elasticity']))

# Predicted vs Actual
st.subheader("Predicted vs Actual Fertility Rate")
df_nepal['predicted_fertility'] = model.predict(X)
fig = px.scatter(df_nepal, x='fertility_rate', y='predicted_fertility',
                 title='Actual vs Predicted Fertility Rate')
fig.add_shape(type='line', x0=df_nepal['fertility_rate'].min(), y0=df_nepal['fertility_rate'].min(),
              x1=df_nepal['fertility_rate'].max(), y1=df_nepal['fertility_rate'].max(),
              line=dict(color='red', dash='dash'))
st.plotly_chart(fig, use_container_width=True)

# Regression Coefficients
st.subheader("Regression Coefficients")
coef_df = pd.DataFrame({
    'Variable': model.params.index[1:],  
    'Coefficient': model.params.values[1:]
})
fig = px.bar(coef_df, x='Coefficient', y='Variable', orientation='h',
             text='Coefficient', color='Coefficient', color_continuous_scale='Viridis',
             title='Regression Coefficients for Fertility Rate')
st.plotly_chart(fig, use_container_width=True)

# Elasticity Ranking Bar
st.subheader("Elasticity Ranking")
elasticity_df = pd.DataFrame({'Variable': list(elasticity.keys()), 'Elasticity': list(elasticity.values())})
fig = px.bar(elasticity_df, x='Elasticity', y='Variable', orientation='h',
             text='Elasticity', color='Elasticity', color_continuous_scale='Cividis',
             title='Elasticity of Fertility Rate w.r.t Predictors')
st.plotly_chart(fig, use_container_width=True)

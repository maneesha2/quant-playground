## Project Overview  
### 📈 Determinants of Fertility in Nepal (2000–2022)

- **Objective**  
  - Analyze and quantify how key socioeconomic and demographic variables have influenced Nepal’s fertility rate over time, including:  
    - Life expectancy  
    - GDP per capita  
    - Female education  
    - Labor force participation  
    - Urbanization  

- **Goal**  
  - Use statistical and quantitative methods to identify the major drivers of fertility decline in Nepal.  
  - Provide insights into the country's demographic and socioeconomic transition.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# Optional: plotly defaults
px.defaults.width = 900
px.defaults.height = 500
```


```python
# Loading the dataset
df = pd.read_csv('../../datasets/Region_WB_sm.csv')
# Filtering for Nepal
df_nepal = df[df['country'] == 'Nepal']
df_nepal.isnull().sum()
df_nepal.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>iso2c</th>
      <th>iso3c</th>
      <th>year</th>
      <th>fertility</th>
      <th>adolescent_fert</th>
      <th>contraceptive</th>
      <th>infant_mort</th>
      <th>life_exp</th>
      <th>gdp_pc</th>
      <th>fem_labor</th>
      <th>sec_school_f</th>
      <th>urban</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3096</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2000</td>
      <td>3.984</td>
      <td>118.816</td>
      <td>37.3</td>
      <td>61.8</td>
      <td>62.642</td>
      <td>547.221082</td>
      <td>21.360</td>
      <td>27.949949</td>
      <td>13.397</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3097</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2001</td>
      <td>3.793</td>
      <td>118.047</td>
      <td>39.3</td>
      <td>58.8</td>
      <td>63.288</td>
      <td>564.286229</td>
      <td>21.571</td>
      <td>30.452579</td>
      <td>13.947</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3098</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2002</td>
      <td>3.604</td>
      <td>115.000</td>
      <td>NaN</td>
      <td>55.9</td>
      <td>63.507</td>
      <td>556.438455</td>
      <td>21.795</td>
      <td>34.056541</td>
      <td>14.240</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3099</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2003</td>
      <td>3.439</td>
      <td>111.926</td>
      <td>NaN</td>
      <td>53.3</td>
      <td>64.229</td>
      <td>570.289674</td>
      <td>22.026</td>
      <td>35.716770</td>
      <td>14.538</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3100</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2004</td>
      <td>3.279</td>
      <td>106.216</td>
      <td>38.3</td>
      <td>50.9</td>
      <td>64.744</td>
      <td>589.469929</td>
      <td>22.285</td>
      <td>NaN</td>
      <td>14.841</td>
      <td>South Asia</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter Nepal data
df_nepal = df[df['country'] == 'Nepal'].copy()
# Handle missing values by interpolation
df_nepal['contraceptive'] = df_nepal['contraceptive'].interpolate(method='linear')
df_nepal['sec_school_f'] = df_nepal['sec_school_f'].interpolate(method='linear')
# Convert year to integer
df_nepal['year'] = df_nepal['year'].astype(int)
df_nepal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>iso2c</th>
      <th>iso3c</th>
      <th>year</th>
      <th>fertility</th>
      <th>adolescent_fert</th>
      <th>contraceptive</th>
      <th>infant_mort</th>
      <th>life_exp</th>
      <th>gdp_pc</th>
      <th>fem_labor</th>
      <th>sec_school_f</th>
      <th>urban</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3096</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2000</td>
      <td>3.984</td>
      <td>118.816</td>
      <td>37.300000</td>
      <td>61.8</td>
      <td>62.642</td>
      <td>547.221082</td>
      <td>21.360</td>
      <td>27.949949</td>
      <td>13.397</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3097</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2001</td>
      <td>3.793</td>
      <td>118.047</td>
      <td>39.300000</td>
      <td>58.8</td>
      <td>63.288</td>
      <td>564.286229</td>
      <td>21.571</td>
      <td>30.452579</td>
      <td>13.947</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3098</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2002</td>
      <td>3.604</td>
      <td>115.000</td>
      <td>38.966667</td>
      <td>55.9</td>
      <td>63.507</td>
      <td>556.438455</td>
      <td>21.795</td>
      <td>34.056541</td>
      <td>14.240</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3099</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2003</td>
      <td>3.439</td>
      <td>111.926</td>
      <td>38.633333</td>
      <td>53.3</td>
      <td>64.229</td>
      <td>570.289674</td>
      <td>22.026</td>
      <td>35.716770</td>
      <td>14.538</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3100</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2004</td>
      <td>3.279</td>
      <td>106.216</td>
      <td>38.300000</td>
      <td>50.9</td>
      <td>64.744</td>
      <td>589.469929</td>
      <td>22.285</td>
      <td>38.378639</td>
      <td>14.841</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3101</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2005</td>
      <td>3.115</td>
      <td>101.456</td>
      <td>43.150000</td>
      <td>48.8</td>
      <td>65.275</td>
      <td>603.190109</td>
      <td>22.575</td>
      <td>41.040508</td>
      <td>15.149</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3102</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2006</td>
      <td>2.967</td>
      <td>95.034</td>
      <td>48.000000</td>
      <td>46.9</td>
      <td>65.891</td>
      <td>617.477639</td>
      <td>22.897</td>
      <td>39.517818</td>
      <td>15.462</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3103</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2007</td>
      <td>2.857</td>
      <td>89.634</td>
      <td>48.340000</td>
      <td>45.1</td>
      <td>66.169</td>
      <td>633.226129</td>
      <td>23.253</td>
      <td>39.880692</td>
      <td>15.781</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3104</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2008</td>
      <td>2.737</td>
      <td>85.221</td>
      <td>48.680000</td>
      <td>43.4</td>
      <td>66.442</td>
      <td>666.865095</td>
      <td>23.676</td>
      <td>46.195919</td>
      <td>16.105</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3105</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2009</td>
      <td>2.630</td>
      <td>82.717</td>
      <td>49.020000</td>
      <td>41.8</td>
      <td>66.600</td>
      <td>692.400001</td>
      <td>24.169</td>
      <td>47.009979</td>
      <td>16.434</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3106</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2010</td>
      <td>2.541</td>
      <td>79.643</td>
      <td>49.360000</td>
      <td>40.2</td>
      <td>66.772</td>
      <td>721.265221</td>
      <td>24.711</td>
      <td>55.607441</td>
      <td>16.768</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3107</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2011</td>
      <td>2.462</td>
      <td>80.049</td>
      <td>49.700000</td>
      <td>38.7</td>
      <td>67.123</td>
      <td>742.617253</td>
      <td>25.272</td>
      <td>59.736408</td>
      <td>17.108</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3108</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2012</td>
      <td>2.420</td>
      <td>81.500</td>
      <td>49.675572</td>
      <td>37.1</td>
      <td>67.364</td>
      <td>775.310578</td>
      <td>25.770</td>
      <td>64.910698</td>
      <td>17.458</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3109</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2013</td>
      <td>2.360</td>
      <td>82.881</td>
      <td>49.651144</td>
      <td>35.6</td>
      <td>67.598</td>
      <td>801.039061</td>
      <td>26.188</td>
      <td>67.366982</td>
      <td>17.815</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3110</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2014</td>
      <td>2.319</td>
      <td>84.501</td>
      <td>49.626716</td>
      <td>34.1</td>
      <td>67.812</td>
      <td>846.665542</td>
      <td>26.591</td>
      <td>69.295212</td>
      <td>18.182</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3111</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2015</td>
      <td>2.273</td>
      <td>81.619</td>
      <td>51.113358</td>
      <td>32.6</td>
      <td>67.374</td>
      <td>875.543635</td>
      <td>27.015</td>
      <td>70.295563</td>
      <td>18.557</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3112</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2016</td>
      <td>2.226</td>
      <td>78.814</td>
      <td>52.600000</td>
      <td>31.2</td>
      <td>68.444</td>
      <td>875.188944</td>
      <td>27.350</td>
      <td>73.282471</td>
      <td>18.942</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3113</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2017</td>
      <td>2.173</td>
      <td>75.975</td>
      <td>50.622707</td>
      <td>29.9</td>
      <td>68.740</td>
      <td>951.857299</td>
      <td>27.530</td>
      <td>76.625252</td>
      <td>19.336</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3114</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2018</td>
      <td>2.120</td>
      <td>72.782</td>
      <td>48.645414</td>
      <td>28.5</td>
      <td>69.036</td>
      <td>1021.914922</td>
      <td>27.628</td>
      <td>79.592964</td>
      <td>19.740</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3115</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2019</td>
      <td>2.080</td>
      <td>71.667</td>
      <td>46.668122</td>
      <td>27.3</td>
      <td>69.299</td>
      <td>1077.117965</td>
      <td>27.565</td>
      <td>82.560677</td>
      <td>20.153</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3116</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2020</td>
      <td>2.051</td>
      <td>70.275</td>
      <td>50.178748</td>
      <td>26.1</td>
      <td>69.106</td>
      <td>1031.536188</td>
      <td>27.236</td>
      <td>86.425003</td>
      <td>20.576</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3117</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2021</td>
      <td>2.025</td>
      <td>68.777</td>
      <td>53.689374</td>
      <td>25.1</td>
      <td>68.385</td>
      <td>1062.788843</td>
      <td>27.268</td>
      <td>85.097144</td>
      <td>21.008</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2022</td>
      <td>2.002</td>
      <td>68.330</td>
      <td>57.200000</td>
      <td>24.2</td>
      <td>70.087</td>
      <td>1113.554623</td>
      <td>27.516</td>
      <td>84.348712</td>
      <td>21.451</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2023</td>
      <td>1.984</td>
      <td>67.169</td>
      <td>57.200000</td>
      <td>23.3</td>
      <td>70.354</td>
      <td>1136.427693</td>
      <td>27.554</td>
      <td>89.730981</td>
      <td>21.903</td>
      <td>South Asia</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


```python
# check data info
df_nepal.info()
df_nepal.head()

#urban categories
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 24 entries, 3096 to 3119
    Data columns (total 14 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   country                     24 non-null     object 
     1   iso2c                       24 non-null     object 
     2   iso3c                       24 non-null     object 
     3   year                        24 non-null     int64  
     4   fertility_rate              24 non-null     float64
     5   adolescent_fert_rate        24 non-null     float64
     6   contraceptive_use           24 non-null     float64
     7   infant_mortality            24 non-null     float64
     8   life_expectancy             24 non-null     float64
     9   gdp_per_capita              24 non-null     float64
     10  female_labor_participation  24 non-null     float64
     11  female_secondary_school     24 non-null     float64
     12  urban                       24 non-null     float64
     13  region                      24 non-null     object 
    dtypes: float64(9), int64(1), object(4)
    memory usage: 2.8+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>iso2c</th>
      <th>iso3c</th>
      <th>year</th>
      <th>fertility_rate</th>
      <th>adolescent_fert_rate</th>
      <th>contraceptive_use</th>
      <th>infant_mortality</th>
      <th>life_expectancy</th>
      <th>gdp_per_capita</th>
      <th>female_labor_participation</th>
      <th>female_secondary_school</th>
      <th>urban</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3096</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2000</td>
      <td>3.984</td>
      <td>118.816</td>
      <td>37.300000</td>
      <td>61.8</td>
      <td>62.642</td>
      <td>547.221082</td>
      <td>21.360</td>
      <td>27.949949</td>
      <td>13.397</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3097</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2001</td>
      <td>3.793</td>
      <td>118.047</td>
      <td>39.300000</td>
      <td>58.8</td>
      <td>63.288</td>
      <td>564.286229</td>
      <td>21.571</td>
      <td>30.452579</td>
      <td>13.947</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3098</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2002</td>
      <td>3.604</td>
      <td>115.000</td>
      <td>38.966667</td>
      <td>55.9</td>
      <td>63.507</td>
      <td>556.438455</td>
      <td>21.795</td>
      <td>34.056541</td>
      <td>14.240</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3099</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2003</td>
      <td>3.439</td>
      <td>111.926</td>
      <td>38.633333</td>
      <td>53.3</td>
      <td>64.229</td>
      <td>570.289674</td>
      <td>22.026</td>
      <td>35.716770</td>
      <td>14.538</td>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>3100</th>
      <td>Nepal</td>
      <td>NP</td>
      <td>NPL</td>
      <td>2004</td>
      <td>3.279</td>
      <td>106.216</td>
      <td>38.300000</td>
      <td>50.9</td>
      <td>64.744</td>
      <td>589.469929</td>
      <td>22.285</td>
      <td>38.378639</td>
      <td>14.841</td>
      <td>South Asia</td>
    </tr>
  </tbody>
</table>
</div>




```python
numeric_cols = ['fertility_rate','adolescent_fert_rate','contraceptive_use','infant_mortality',
                'life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']

# Fertility Trend (Interactive)
fig = px.line(df_nepal, x='year', y='fertility_rate', 
              title='Fertility Rate Trend in Nepal (2000-2022)',
              markers=True)
fig.update_layout(xaxis_title='Year', yaxis_title='Fertility Rate')
fig.show()

```




```python
# Socioeconomic Trends (Interactive)
fig = px.line(df_nepal, x='year', y=['life_expectancy','gdp_per_capita',
                                     'female_secondary_school','urban'],
              title='Socioeconomic Trends in Nepal (2000-2022)',
              markers=True)
fig.update_layout(xaxis_title='Year', yaxis_title='Value')
fig.show()
```




```python

# Correlation Heatmap (Interactive)
corr_matrix = df_nepal[numeric_cols].corr()
fig = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   colorscale='Viridis',
                   text=corr_matrix.values,
                   texttemplate="%{text:.2f}"
                ))
fig.update_layout(title='Correlation Heatmap of Nepal Socioeconomic Variables')

fig.write_image("correlation_heatmap.png")  # 🔥 this makes it appear in HTML

fig.show() 
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[26], line 13
          3 fig = go.Figure(data=go.Heatmap(
          4                    z=corr_matrix.values,
          5                    x=corr_matrix.columns,
       (...)      9                    texttemplate="%{text:.2f}"
         10                 ))
         11 fig.update_layout(title='Correlation Heatmap of Nepal Socioeconomic Variables')
    ---> 13 fig.write_image("correlation_heatmap.png")  # 🔥 this makes it appear in HTML
         15 fig.show() 


    File ~/Research/quant-playground/env/lib/python3.11/site-packages/plotly/basedatatypes.py:3895, in BaseFigure.write_image(self, *args, **kwargs)
       3891     if kwargs.get("engine", None):
       3892         warnings.warn(
       3893             ENGINE_PARAM_DEPRECATION_MSG, DeprecationWarning, stacklevel=2
       3894         )
    -> 3895 return pio.write_image(self, *args, **kwargs)


    File ~/Research/quant-playground/env/lib/python3.11/site-packages/plotly/io/_kaleido.py:528, in write_image(fig, file, format, scale, width, height, validate, engine)
        524 format = infer_format(path, format)
        526 # Request image
        527 # Do this first so we don't create a file if image conversion fails
    --> 528 img_data = to_image(
        529     fig,
        530     format=format,
        531     scale=scale,
        532     width=width,
        533     height=height,
        534     validate=validate,
        535     engine=engine,
        536 )
        538 # Open file
        539 if path is None:
        540     # We previously failed to make sense of `file` as a pathlib object.
        541     # Attempt to write to `file` as an open file descriptor.


    File ~/Research/quant-playground/env/lib/python3.11/site-packages/plotly/io/_kaleido.py:345, in to_image(fig, format, width, height, scale, validate, engine)
        343     # Raise informative error message if Kaleido is not installed
        344     if not kaleido_available():
    --> 345         raise ValueError(
        346             """
        347 Image export using the "kaleido" engine requires the Kaleido package,
        348 which can be installed using pip:
        349 
        350     $ pip install --upgrade kaleido
        351 """
        352         )
        354     # Convert figure to dict (and validate if requested)
        355     fig_dict = validate_coerce_fig_to_dict(fig, validate)


    ValueError: 
    Image export using the "kaleido" engine requires the Kaleido package,
    which can be installed using pip:
    
        $ pip install --upgrade kaleido




```python
# Regression Analysis
# ===============================
# Dependent Variable: fertility_rate
# Independent Variables: life_expectancy, gdp_per_capita, female_labor_participation, female_secondary_school, urban

X = df_nepal[['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']]
y = df_nepal['fertility_rate']

# Add constant for intercept
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()


# Regression Summary
print(model.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         fertility_rate   R-squared:                       0.983
    Model:                            OLS   Adj. R-squared:                  0.978
    Method:                 Least Squares   F-statistic:                     207.2
    Date:                Tue, 18 Nov 2025   Prob (F-statistic):           2.97e-15
    Time:                        08:15:59   Log-Likelihood:                 27.325
    No. Observations:                  24   AIC:                            -42.65
    Df Residuals:                      18   BIC:                            -35.58
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------
    const                         18.2974      1.945      9.407      0.000      14.211      22.384
    life_expectancy               -0.1742      0.044     -3.984      0.001      -0.266      -0.082
    gdp_per_capita                 0.0028      0.001      3.904      0.001       0.001       0.004
    female_labor_participation    -0.1191      0.050     -2.373      0.029      -0.224      -0.014
    female_secondary_school       -0.0017      0.009     -0.186      0.854      -0.021       0.018
    urban                         -0.1789      0.086     -2.068      0.053      -0.361       0.003
    ==============================================================================
    Omnibus:                        0.977   Durbin-Watson:                   1.478
    Prob(Omnibus):                  0.614   Jarque-Bera (JB):                0.277
    Skew:                           0.244   Prob(JB):                        0.871
    Kurtosis:                       3.197   Cond. No.                     8.77e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 8.77e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
# Elasticity Estimation
# ===============================
# Elasticity = (beta * mean(X)) / mean(Y)
elasticity = {}
for col in ['life_expectancy','gdp_per_capita','female_labor_participation','female_secondary_school','urban']:
    beta = model.params[col]
    elasticity[col] = (beta * df_nepal[col].mean()) / df_nepal['fertility_rate'].mean()

# Rank by absolute elasticity
elasticity = dict(sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True))
print("Elasticity of Fertility Rate with respect to predictors:")
for k,v in elasticity.items():
    print(f"{k}: {v:.3f}")

```

    Elasticity of Fertility Rate with respect to predictors:
    life_expectancy: -4.411
    urban: -1.181
    female_labor_participation: -1.128
    gdp_per_capita: 0.836
    female_secondary_school: -0.038



```python
# Visualization of Regression Results (Interactive)
# ===============================
# Predicted vs Actual Fertility
df_nepal['predicted_fertility'] = model.predict(X)
fig = px.scatter(df_nepal, x='fertility_rate', y='predicted_fertility',
                 title='Actual vs Predicted Fertility Rate')
fig.add_shape(
    type='line', x0=df_nepal['fertility_rate'].min(), y0=df_nepal['fertility_rate'].min(),
    x1=df_nepal['fertility_rate'].max(), y1=df_nepal['fertility_rate'].max(),
    line=dict(color='red', dash='dash')
)
fig.update_layout(xaxis_title='Actual Fertility', yaxis_title='Predicted Fertility')
fig.show()
```




```python
# Regression Coefficients
coef_df = pd.DataFrame({
    'Variable': model.params.index[1:],  # skip constant
    'Coefficient': model.params.values[1:]
})
fig = px.bar(coef_df, x='Coefficient', y='Variable', orientation='h',
             title='Regression Coefficients for Fertility Rate', text='Coefficient',
             color='Coefficient', color_continuous_scale='Viridis')
fig.show()
```




```python
# Elasticity Ranking
elasticity_df = pd.DataFrame({
    'Variable': list(elasticity.keys()),
    'Elasticity': list(elasticity.values())
})
fig = px.bar(elasticity_df, x='Elasticity', y='Variable', orientation='h',
             title='Elasticity of Fertility Rate with Respect to Predictors',
             text='Elasticity', color='Elasticity', color_continuous_scale='Cividis')
fig.show()
```



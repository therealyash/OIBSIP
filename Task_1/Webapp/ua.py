# Unemployment Analysis of India from May 2019 - June 2020

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
#%matplotlib inline


# creating a dataframe
df = pd.read_csv('Task_1/Webapp/Unemployment in India.csv') 
st.title('Unemployment Analysis of India from May 2019 - June 2020')

# get index number of the rows which have null values & give the output in a list
null_var = df.isnull()
num_null = null_var.index[null_var.any(axis=1)].tolist()

# dropping rows from index 359-372
df = df.drop(index=range(359,373))

# renaming the columns of the dataframe
df.set_axis(['State', 'Date', 'Frequency', 'Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)', 'Area' ], axis=1, inplace=True)

# converting date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Adding month column
df['Month_Num'] = pd.DatetimeIndex(df['Date']).month

# Adding Month Name
df['Month'] = pd.DatetimeIndex(df['Date']).month_name()

# Adding Year
df['Year'] = pd.DatetimeIndex(df['Date']).year

# Top 15 States with High Unemployment Rate
st.title("State with Highest Unemployment - Tripura")
df2 = df[['State','Estimated Unemployment Rate (%)']].groupby('State').sum().sort_values(by='Estimated Unemployment Rate (%)',ascending=False)
plot = px.bar(df2, x=df2.index, y="Estimated Unemployment Rate (%)", title = 'Top 15 States with Highest Unemployment Rate') 
st.plotly_chart(plot)


# Piechart of Statewise Unemployment
st.title("Piechart of Statewise Unemployment")
plot = px.pie(df2, values='Estimated Unemployment Rate (%)', names = df2.index, title='Estimated Unemployment Rate (%)',color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(plot)

# Months with Highest Employment
st.title("Month with Highest Unemployment - May")
df3 = df[["Month","Estimated Unemployment Rate (%)"]].groupby("Month").sum().sort_values(by="Estimated Unemployment Rate (%)", ascending = False)
plot = px.bar(df3, x=df3.index, y="Estimated Unemployment Rate (%)", title = 'Months with Highest Unemployment Rate') 
st.plotly_chart(plot)

# Employment Statewise
st.title("State with Maximum Employed - UP")
df4 = df[["State","Estimated Employed"]].groupby("State").sum().sort_values(by="Estimated Employed", ascending = False)
plot = px.bar(df4, x=df4.index, y="Estimated Employed", title = 'Estimated People Employed State Wise') 
st.plotly_chart(plot)

# Monthwise Estimation of People Employed
st.title("Month with Highest Employed - June")
df5 = df[["Month","Estimated Employed"]].groupby("Month").sum().sort_values(by="Estimated Employed", ascending = False)
plot = px.bar(df5, x=df5.index, y="Estimated Employed", title = 'Estimated People Employed Monthly') 
st.plotly_chart(plot)

# State with Highest Labour Participation 
st.title("State with Highest Labour Participation - Tripura")
df6 = df[["State","Estimated Labour Participation Rate (%)"]].groupby("State").mean().sort_values(by="Estimated Labour Participation Rate (%)", ascending = False)
plot = px.bar(df6, x=df6.index, y="Estimated Labour Participation Rate (%)", title = 'Estimated Labour Participation Statewise') 
st.plotly_chart(plot)

# Month with Highest Labour Participation
st.title("Month with Highest Labour Participation - June")
df7 = df[["Month","Estimated Labour Participation Rate (%)"]].groupby("Month").sum().sort_values(by="Estimated Labour Participation Rate (%)", ascending = False)
plot = px.bar(df7, x=df7.index, y="Estimated Labour Participation Rate (%)", title = 'Estimated Labour Participation Monthly') 
st.plotly_chart(plot)

# Unemployment Rate Over Time
st.title("Monthly Analysis of Unemployment")
df8 = df[["Month","Year","Estimated Unemployment Rate (%)"]].groupby(["Month","Year"]).mean().sort_values(by="Estimated Unemployment Rate (%)", ascending = True)
df8 = df[["Month","Year","Estimated Unemployment Rate (%)"]].groupby(["Month","Year"]).mean().reset_index()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df8['Month'] = pd.Categorical(df8['Month'], categories=months, ordered=True)
df8 = df8.sort_values(['Year', 'Month'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df8['Year'].astype(str) + "-" + df8['Month'].astype(str), y=df8['Estimated Unemployment Rate (%)'], mode='lines+markers',line=dict(color='Orange'),marker=dict(color='Orange')))
fig.update_layout(title="Estimated Unemployment Rate over time", xaxis_title="Year-Month", yaxis_title="Estimated Unemployment Rate (%)")
st.plotly_chart(fig)

# Labour Participation Rate Over Time
st.title("Monthly Analysis of Labour Participation")
df9 = df[["Month","Year","Estimated Labour Participation Rate (%)"]].groupby(["Month","Year"]).mean().sort_values(by="Estimated Labour Participation Rate (%)", ascending = True)
df9 = df[["Month","Year","Estimated Labour Participation Rate (%)"]].groupby(["Month","Year"]).mean().reset_index()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df9['Month'] = pd.Categorical(df9['Month'], categories=months, ordered=True)
df9 = df9.sort_values(['Year', 'Month'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df9['Year'].astype(str) + "-" + df9['Month'].astype(str), y=df9['Estimated Labour Participation Rate (%)'], mode='lines+markers',line=dict(color='Orange'),marker=dict(color='Orange')))
fig.update_layout(title="Estimated Labour Participation Rate (%) Over Time", xaxis_title="Year-Month", yaxis_title="Estimated Labour Participation Rate (%)")
st.plotly_chart(fig)

# People employed Monthly
st.title("Monthly Analysis of People Employed")
df10 = df[["Month","Year","Estimated Employed"]].groupby(["Month","Year"]).mean().sort_values(by="Estimated Employed", ascending = True)
df10 = df[["Month","Year","Estimated Employed"]].groupby(["Month","Year"]).mean().reset_index()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df10['Month'] = pd.Categorical(df10['Month'], categories=months, ordered=True)
df10 = df10.sort_values(['Year', 'Month'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df10['Year'].astype(str) + "-" + df10['Month'].astype(str), y=df10['Estimated Employed'], mode='lines+markers',line=dict(color='Orange'),marker=dict(color='Orange')))
fig.update_layout(title="Estimated People Employed Over Time", xaxis_title="Year-Month", yaxis_title="Estimated Employed (in Million)")
st.plotly_chart(fig)

# People Employed Statewise & Labour Participation Rate
st.title("People Employed Statewise & Labour Participation Rate")
df11 = df[["State","Estimated Employed","Estimated Labour Participation Rate (%)"]].groupby("State").mean().sort_values(by="State", ascending = False)
# Create a subplot grid with 2 rows and 1 column
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
# line chart 
fig.add_trace(
    go.Scatter(x=df11.index, y=df11['Estimated Labour Participation Rate (%)'], mode='lines+markers', name='Labour Participation Rate',marker=dict(color='Orange')),
    row=1, col=1
)
# bar chart 
fig.add_trace(
    go.Bar(x=df11.index, y=df11['Estimated Employed'], name='Employed',marker=dict(color='Orange')),
    row=2, col=1
)
# Update method
fig.update_layout(title='',
                  xaxis_title='State',
                  yaxis_title='Estimated Employed (in Million) / Estimated Labour Participation Rate (%)',
                  height=700,
                  width=900)

st.plotly_chart(fig)
st.markdown("The graph clearly shows that Tripura has the highest labor participation rate, but unfortunately, very few people are employed in the region. In contrast, Uttar Pradesh has a considerable number of employed people with a decent labor participation rate. Sikkim, on the other hand, has the lowest number of employed individuals and an average labor participation rate. Maharashtra ranks second after Uttar Pradesh with the highest number of employed individuals and an average labor participation rate. Meghalaya has the second-highest labor participation rate, but unfortunately, there are very few employed people in the region.")


# Unemployment Rate & Labour Participation Rate
st.title("Unemployment Rate & Labour Participation Rate Barplot")
df12 = df[["State","Estimated Unemployment Rate (%)","Estimated Labour Participation Rate (%)"]].groupby("State").mean().sort_values(by="State", ascending = False)

# Create a subplot grid with 2 rows and 1 column
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

# Add the bar chart to the first subplot
fig.add_trace(
    go.Bar(x=df12.index, y=df12['Estimated Unemployment Rate (%)'], name='Unemployment Rate'),
    row=1, col=1
)

# Add the bar chart to the second subplot
fig.add_trace(
    go.Bar(x=df12.index, y=df12['Estimated Labour Participation Rate (%)'], name='Labour Participation Rate'),
    row=2, col=1
)

# Update the layout of the figure
fig.update_layout(title='',
                  xaxis_title='State',
                  yaxis_title='Estimated Unemployment Rate (%) / Estimated Labour Participation Rate (%)',
                  height=700,
                  width=900)


st.plotly_chart(fig)

st.title("Unemployment Rate & Labour Participation Rate Linechart")
# Create a subplot grid with 1 row and 1 column
fig = make_subplots(rows=1, cols=1)

# Add the line chart to the subplot
fig.add_trace(
    go.Scatter(x=df12.index, y=df12['Estimated Unemployment Rate (%)'], mode='lines+markers', name='Unemployment Rate'),
)

# Add the line chart to the subplot
fig.add_trace(
    go.Scatter(x=df12.index, y=df12['Estimated Labour Participation Rate (%)'], mode='lines+markers', name='Labour Participation Rate'),
)

# Update the layout of the figure
fig.update_layout(title='',
                  xaxis_title='State',
                  yaxis_title='Estimated Unemployment Rate (%) / Estimated Labour Participation Rate (%)',
                  height=500,
                  width=1000)

st.plotly_chart(fig)

st.title("Unemployment Rate & Labour Participation Rate Areachart")
fig = go.Figure()

# Add the area chart for unemployment rate
fig.add_trace(go.Scatter(x=df12.index, y=df12['Estimated Unemployment Rate (%)'], mode='lines', fill='tozeroy', name='Unemployment Rate',line=dict(color='#008b8b')))

# Add the area chart for labor participation rate
fig.add_trace(go.Scatter(x=df12.index, y=df12['Estimated Labour Participation Rate (%)'], mode='lines', fill='tonexty', name='Labor Participation Rate',line=dict(color='#f4a261')))

# Update the layout of the figure
fig.update_layout(title='Average Unemployment and Labor Participation Rates by State',
                  xaxis_title='State',
                  yaxis_title='Rate (%)',
                  height=600,
                  width=1000)

st.plotly_chart(fig)

st.title("Unemployment Rate & Labour Participation Rate Heatmap")

fig = px.imshow(df12.T, x=df12.index, y=['Estimated Unemployment Rate (%)', 'Estimated Labour Participation Rate (%)'],
                color_continuous_scale='blues', title='Unemployment Rate and Labour Participation Rate Across States')
fig.update_layout(yaxis=dict(title=''),height=600,width=1000)

st.plotly_chart(fig)


# Area with Maximum Labor Participation
st.title("Area with Maximum Labour Participation - Rural")
df13 = df[["Area","Estimated Labour Participation Rate (%)"]].groupby("Area").mean().sort_values(by="Area", ascending = False)
df13 = df[["Area","Estimated Labour Participation Rate (%)"]].groupby("Area").mean().sort_values(by="Area", ascending = False)

fig = go.Figure()
fig.add_trace(go.Bar(x=df13.index, y=df13["Estimated Labour Participation Rate (%)"],
                marker=dict(color='rgba(0, 0, 255, 0.5)', line=dict(color='rgba(0, 0, 255, 0.6)', width=1))))

fig.update_layout(title="Mean Estimated Labour Participation Rate by Area",
                  xaxis_title="Area",
                  yaxis_title="Mean Estimated Labour Participation Rate (%)",
                  width=500, height=600)

st.plotly_chart(fig)


# Area with Maximum Unemployment Rate
st.title("Area with Maximum Unemployment Rate - Urban")
df14 = df[["Area","Estimated Unemployment Rate (%)"]].groupby("Area").mean().sort_values(by="Area", ascending = False)

df14 = df[["Area","Estimated Unemployment Rate (%)"]].groupby("Area").mean().sort_values(by="Area", ascending = False)

fig = go.Figure()
fig.add_trace(go.Bar(x=df14.index, y=df14["Estimated Unemployment Rate (%)"],
                marker=dict(color='rgba(255,69,0 0.5)', line=dict(color='rgba(255,69,0 0.6)', width=1))))

fig.update_layout(title="Mean Estimated Unemployment Rate by Area",
                  xaxis_title="Area",
                  yaxis_title="Mean Estimated Unemployment Rate (%)",
                  width=500, height=600)

st.plotly_chart(fig)


# Area with Maximum People Employed
st.title("Area with Maximum People Employed - Rural")
# line of code to remove e+ in pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df15 = df[["Area","Estimated Employed"]].groupby("Area").mean().sort_values(by="Area", ascending = False)

df15 = df[["Area","Estimated Employed"]].groupby("Area").mean().sort_values(by="Area", ascending = False)

fig = go.Figure()
fig.add_trace(go.Bar(x=df15.index, y=df15["Estimated Employed"],
                marker=dict(color='rgba(50,205,50 0.6)', line=dict(color='rgba(50,205,50 0.6)', width=1))))

fig.update_layout(title="Mean Estimated People Employed by Area",
                  xaxis_title="Area",
                  yaxis_title="Mean Estimated Employed",
                  width=500, height=600)

st.plotly_chart(fig)


# Estimated Labour Participation, Unemployment, People Employed
st.title("Estimated Labour Participation, Unemployment, People Employed")

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Grouped bar chart for Estimated Labour Participation Rate
axs[0].bar(df13.index, df13["Estimated Labour Participation Rate (%)"], color='b')
axs[0].set_title("Estimated Labour Participation Rate")
axs[0].set_xlabel("Area")
axs[0].set_ylabel("Estimated Labour Participation Rate (%)")

# Grouped bar chart for Estimated Unemployment Rate
axs[1].bar(df14.index, df14["Estimated Unemployment Rate (%)"], color='g')
axs[1].set_title("Estimated Unemployment Rate")
axs[1].set_xlabel("Area")
axs[1].set_ylabel("Estimated Unemployment Rate (%)")

# Grouped bar chart for Estimated Employed
axs[2].bar(df15.index, df15["Estimated Employed"], color='r')
axs[2].set_title("Estimated Employed")
axs[2].set_xlabel("Area")
axs[2].set_ylabel("Estimated Employed")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

st.pyplot(plt)

# Indian Estimated People Employed Statewise
st.title("Indian Estimated People Employed Statewise Histplot")
plt.figure(figsize=(12, 10))

df.columns= ['State',
             'Date',
             'Frequency',
             'Estimated Unemployment Rate (%)',
             'Estimated Employed',
             'Estimated Labour Participation Rate (%)',
             'Area',
             'Month_Num',
             'Month',
             'Year']
plt.title("Indian Estimated People Employed Statewise")
sns.histplot(x="Estimated Employed", hue="State", data=df)
st.pyplot(plt)

st.title("Indian Estimated People Employed Statewise Barplot")
plt.figure(figsize=(14, 6))

ax = sns.barplot(data=df, x ='State', y ='Estimated Employed',errorbar = None)

# Rotate the x-axis labels to improve readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Indian Estimated People Employed Statewise")
st.pyplot(plt)


# Indian Estimated Unemployment Statewise
st.title("Indian Estimated Unemployment Statewise Histplot")
plt.figure(figsize=(12, 10))
plt.title("Indian Estimated Unemployment Statewise")
sns.histplot(x="Estimated Unemployment Rate (%)", hue="State", data=df)
st.pyplot(plt)

st.title("Indian Estimated Unemployment Statewise Barplot")
plt.figure(figsize=(14, 6))

ax = sns.barplot(data=df, x='State', y='Estimated Unemployment Rate (%)',errorbar = None)

# Rotate the x-axis labels to improve readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Indian Estimated Unemployment Statewise")
# Show the plot
st.pyplot(plt)


# Indian Estimated Labour Participation Statewise
st.title("Indian Estimated Labour Participation Statewise Histplot")
plt.figure(figsize=(12, 10))
plt.title("Indian Estimated Labour Participation Statewise")
sns.histplot(x="Estimated Labour Participation Rate (%)", hue="State", data=df)
st.pyplot(plt)

st.title("Indian Estimated Labour Participation Statewise Barplot")
plt.figure(figsize=(14, 6))

ax = sns.barplot(data=df, x='State', y='Estimated Labour Participation Rate (%)',errorbar = None)

# Rotate the x-axis labels to improve readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Indian Estimated Labour Participation Statewise")
# Show the plot
st.pyplot(plt)






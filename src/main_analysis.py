import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nlp_analysis import perform_nlp_analysis

# Load and clean data
df = pd.read_csv('./data/MarvelVsDC.csv')

# Data cleaning
df['Year'] = pd.to_numeric(df['Year'].replace({'-': '0'}, regex=True), errors='coerce')
df['IMDB_Score'] = pd.to_numeric(df['IMDB_Score'], errors='coerce')
df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')
df['USA_Gross'] = df['USA_Gross'].replace('[\$,M]', '', regex=True).astype(float)
df['RunTime'] = df['RunTime'].str.extract('(\d+)').astype(float)

# Exploratory Data Analysis
plt.style.use('seaborn')

# Compare the number of Marvel vs DC productions
category_counts = df['Category'].value_counts()
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar')
plt.title('Number of Productions: Marvel vs DC')
plt.xlabel('Category')
plt.ylabel('Count')
plt.savefig('production_counts.png')
plt.close()

# Compare IMDB Scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='IMDB_Score', data=df)
plt.title('IMDB Scores: Marvel vs DC')
plt.savefig('imdb_scores.png')
plt.close()

# Analyze the trend of productions over time
df_yearly = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)
df_yearly.plot(kind='line', figsize=(12, 6))
plt.title('Number of Productions Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Productions')
plt.savefig('production_trend.png')
plt.close()

# Compare USA Gross earnings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='USA_Gross', data=df)
plt.title('USA Gross Earnings: Marvel vs DC')
plt.savefig('usa_gross.png')
plt.close()

# Perform NLP analysis
clusters, top_terms = perform_nlp_analysis(df)

# Print top terms for each cluster
for i, terms in enumerate(top_terms):
    print(f"Cluster {i}: {', '.join(terms)}")

print("Analysis complete. Visualizations saved as PNG files.")
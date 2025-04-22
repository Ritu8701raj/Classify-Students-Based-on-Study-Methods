# Classify-Students-Based-on-Study-Methods
# 📚 Classify Students Based on Study Methods

This project uses unsupervised machine learning to classify students based on their learning styles, such as **visual**, **auditory**, or **kinesthetic**, using questionnaire response data. It applies **K-Means Clustering** and **PCA** for grouping and visualization.

## 🧠 Objective
To help educators identify student learning preferences by analyzing questionnaire data and grouping students into meaningful clusters.

---

## 🛠️ Tools and Libraries Used
- Python
- Google Colab
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧪 Methodology

### 1. **Data Collection**
- The dataset contains student responses on learning preferences.

### 2. **Preprocessing**
- Handle missing values
- Feature scaling using `StandardScaler`

### 3. **Clustering**
- Determine optimal number of clusters using **Elbow Method**
- Apply **K-Means** clustering

### 4. **Visualization**
- Use **PCA** to reduce dimensions for 2D visualization
- Plot clusters using **Seaborn**

---

## 📈 Results
- Students are grouped into 3 clusters, each indicating a distinct learning style.
- Visual representation shows clear separation among groups.
- Results saved in a new CSV file `clustered_students.csv`.

---

## 💻 Run It Yourself

```python
# Load data
df = pd.read_csv('/content/student_methods.csv')
df = df.dropna()

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Elbow method
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# PCA + plot
pca = PCA(n_components=2)
df[['PCA1', 'PCA2']] = pca.fit_transform(scaled_data)
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster')

# Module 4 Glossary: Building Unsupervised Learning Models

## Unsupervised Learning Overview

**Unsupervised Learning**
A type of machine learning where models learn patterns from unlabeled data without explicit target outputs or supervision.

**Unlabeled Data**
Data that consists only of input features without corresponding output labels or targets.

**Pattern Discovery**
The process of identifying hidden structures, relationships, or regularities in data without predefined categories.

**Exploratory Data Analysis (EDA)**
Statistical and visual techniques used to understand the main characteristics and patterns in a dataset.

## Clustering Fundamentals

**Clustering**
The task of grouping similar data points together into clusters based on their features, without using predefined labels.

**Cluster**
A group of data points that are similar to each other and dissimilar to points in other clusters.

**Cluster Assignment**
The process of assigning each data point to a specific cluster.

**Cluster Center (Centroid)**
A representative point (often the mean) that characterizes the center of a cluster.

**Intra-cluster Distance**
The distance between points within the same cluster. Lower values indicate more cohesive clusters.

**Inter-cluster Distance**
The distance between points in different clusters. Higher values indicate better-separated clusters.

**Cluster Cohesion**
A measure of how closely related points within a cluster are to each other.

**Cluster Separation**
A measure of how distinct different clusters are from each other.

## K-Means Clustering

**K-Means**
A popular clustering algorithm that partitions data into k clusters by iteratively assigning points to nearest centroids and updating centroid positions.

**k (in K-Means)**
The hyperparameter specifying the number of clusters to create.

**Centroid Initialization**
The process of selecting initial positions for cluster centroids before running K-Means.

**Random Initialization**
Randomly selecting k data points as initial centroids.

**K-Means++ Initialization**
An improved initialization method that spreads out initial centroids to speed up convergence and improve results.

**Assignment Step**
In K-Means, the phase where each point is assigned to the nearest centroid.

**Update Step**
In K-Means, the phase where centroids are recalculated as the mean of all points assigned to each cluster.

**Convergence (K-Means)**
The state when centroid positions no longer change significantly between iterations.

**Inertia (Within-Cluster Sum of Squares)**
A metric measuring the sum of squared distances between points and their assigned centroids. Lower is better.

**Elbow Method**
A technique for selecting k by plotting inertia vs. number of clusters and looking for an "elbow" where improvement diminishes.

**Local Optimum**
A solution that is optimal within a neighborhood but not globally optimal. K-Means can get stuck in local optima.

**Global Optimum**
The best possible solution across all possibilities. K-Means doesn't guarantee finding this.

**Multiple Runs**
Running K-Means several times with different initializations to avoid poor local optima.

## Hierarchical Clustering

**Hierarchical Clustering**
A clustering approach that builds a hierarchy of clusters, either by merging small clusters (agglomerative) or splitting large clusters (divisive).

**Agglomerative Clustering**
A bottom-up hierarchical clustering approach that starts with each point as its own cluster and iteratively merges the closest clusters.

**Divisive Clustering**
A top-down hierarchical clustering approach that starts with all points in one cluster and iteratively splits clusters.

**Dendrogram**
A tree-like diagram showing the hierarchical relationship between clusters and the sequence of merges or splits.

**Linkage Criterion**
The method used to calculate distance between clusters in hierarchical clustering.

**Single Linkage (Minimum Linkage)**
Linkage criterion using the minimum distance between points in two clusters.

**Complete Linkage (Maximum Linkage)**
Linkage criterion using the maximum distance between points in two clusters.

**Average Linkage**
Linkage criterion using the average distance between all pairs of points in two clusters.

**Ward's Linkage**
Linkage criterion that minimizes the within-cluster variance when merging clusters.

**Cutting the Dendrogram**
Selecting a height on the dendrogram to determine the number of final clusters.

## Density-Based Clustering

**Density-Based Clustering**
Clustering methods that group points in high-density regions separated by low-density regions, capable of finding arbitrary-shaped clusters.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
A popular density-based algorithm that groups together points that are closely packed and marks points in low-density regions as outliers.

**Epsilon (ε)**
In DBSCAN, the radius defining the neighborhood around each point.

**MinPts (Minimum Points)**
In DBSCAN, the minimum number of points required within an ε-neighborhood to form a dense region.

**Core Point**
In DBSCAN, a point with at least MinPts neighbors within its ε-neighborhood.

**Border Point**
In DBSCAN, a point that is within the ε-neighborhood of a core point but doesn't have MinPts neighbors itself.

**Noise Point (Outlier)**
In DBSCAN, a point that is neither a core point nor a border point, considered an outlier.

**Density-Reachable**
In DBSCAN, a point is density-reachable from another if there is a chain of core points connecting them.

**Density-Connected**
In DBSCAN, two points are density-connected if both are density-reachable from a common core point.

**HDBSCAN (Hierarchical DBSCAN)**
An extension of DBSCAN that builds a hierarchy of clusters and automatically selects optimal clusters, more robust to varying densities.

**Minimum Cluster Size**
In HDBSCAN, the smallest number of points required to form a cluster.

**Cluster Stability**
In HDBSCAN, a measure of how persistent a cluster is across different density thresholds.

## Dimensionality Reduction Fundamentals

**Dimensionality Reduction**
The process of reducing the number of features in a dataset while preserving important information.

**High-Dimensional Data**
Data with a large number of features, which can be difficult to visualize and process (curse of dimensionality).

**Feature Space**
The multi-dimensional space defined by all features in the dataset.

**Latent Features**
Hidden or underlying features that are not directly observed but can be inferred from the data.

**Projection**
The process of mapping high-dimensional data to a lower-dimensional space.

**Embedding**
A lower-dimensional representation of high-dimensional data that preserves important properties.

**Manifold**
A lower-dimensional surface embedded in a higher-dimensional space that captures the underlying structure of the data.

**Manifold Learning**
Techniques for discovering and exploiting the manifold structure in high-dimensional data.

## Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)**
A linear dimensionality reduction technique that transforms data into a new coordinate system where the axes (principal components) capture maximum variance.

**Principal Component (PC)**
An axis in the transformed space that captures a direction of maximum variance in the data. PC1 has the most variance, PC2 the second most, etc.

**Eigenvalue**
A scalar indicating the amount of variance captured by a principal component. Larger eigenvalues mean more important components.

**Eigenvector**
The direction of a principal component in the original feature space.

**Loading**
The weight of an original feature in a principal component, indicating the feature's contribution.

**Variance Explained**
The proportion of total variance in the data captured by each principal component or set of components.

**Cumulative Explained Variance**
The total proportion of variance explained by the first n principal components.

**Scree Plot**
A plot showing the variance explained by each principal component, used to decide how many components to retain.

**Whitening**
In PCA, transforming data so components have unit variance, removing correlation structure.

**Dimensionality Reduction (via PCA)**
Keeping only the top k principal components to reduce the number of features while retaining most information.

**Reconstruction Error**
The difference between original data and data reconstructed from reduced principal components.

**Covariance Matrix**
A matrix describing the covariance between all pairs of features, used in PCA computation.

**Orthogonal**
Principal components are perpendicular (uncorrelated) to each other in the feature space.

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

**t-SNE**
A non-linear dimensionality reduction technique particularly well-suited for visualizing high-dimensional data in 2D or 3D by preserving local structure.

**Stochastic Neighbor Embedding (SNE)**
The family of techniques that t-SNE belongs to, which preserve local neighborhoods in lower dimensions.

**Probability Distribution (t-SNE)**
t-SNE models similarities between points as probability distributions in both high and low dimensions.

**Perplexity**
A hyperparameter in t-SNE related to the number of nearest neighbors considered, typically between 5 and 50. Balances local vs. global structure.

**Student's t-Distribution**
The distribution used in t-SNE's low-dimensional space to allow dissimilar points to be far apart while keeping similar points close.

**KL Divergence (Kullback-Leibler Divergence)**
The cost function minimized by t-SNE, measuring the difference between high-dimensional and low-dimensional probability distributions.

**Crowding Problem**
An issue in dimensionality reduction where moderate distances in high dimensions are hard to represent in low dimensions, addressed by t-SNE.

**Local Structure Preservation**
t-SNE's strength in maintaining the relationships between nearby points in the visualization.

**Non-Deterministic**
t-SNE produces different results with different random initializations, unlike PCA.

**Non-Invertible**
t-SNE cannot transform new data points or map back to original space, unlike PCA.

## UMAP (Uniform Manifold Approximation and Projection)

**UMAP**
A modern non-linear dimensionality reduction technique that preserves both local and global structure better than t-SNE while being faster.

**Manifold Approximation**
UMAP's approach of modeling data as lying on a manifold and finding a lower-dimensional equivalent.

**Riemannian Geometry**
The mathematical framework underlying UMAP, dealing with curved spaces.

**n_neighbors (UMAP)**
A hyperparameter controlling how many neighbors to consider, balancing local vs. global structure (similar to perplexity in t-SNE).

**min_dist (UMAP)**
A hyperparameter controlling how tightly points can be packed in the low-dimensional space.

**Fuzzy Simplicial Set**
The mathematical structure UMAP uses to represent topological relationships in the data.

**Global Structure Preservation**
UMAP's advantage over t-SNE in better maintaining overall data structure and relationships between clusters.

**Deterministic Mode**
UMAP can produce consistent results with fixed random seed, more reproducible than t-SNE.

**Supervised UMAP**
A variant that can incorporate label information to create more discriminative embeddings.

**Inverse Transform**
UMAP supports transforming new data points to the low-dimensional space and approximate reconstruction, unlike t-SNE.

## Feature Engineering Integration

**Feature Extraction**
Creating new features from existing ones, often through dimensionality reduction techniques.

**Feature Selection**
Choosing a subset of original features rather than creating transformed features.

**Preprocessing Pipeline**
A sequence of data transformation steps applied before modeling, often including dimensionality reduction.

**Variance Threshold**
A feature selection method that removes features with low variance.

**Curse of Dimensionality**
Problems that arise when working with high-dimensional data, including sparsity, increased computational cost, and difficulty in visualization.

## Evaluation Metrics for Clustering

**Silhouette Score**
A metric measuring how similar points are to their own cluster compared to other clusters, ranging from -1 to 1 (higher is better).

**Silhouette Coefficient**
The silhouette score for an individual data point.

**Davies-Bouldin Index**
A metric measuring the average similarity between each cluster and its most similar cluster. Lower values indicate better clustering.

**Calinski-Harabasz Index**
A metric measuring the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better clustering.

**Adjusted Rand Index (ARI)**
A metric comparing clustering results to ground truth labels (when available), adjusted for chance.

**Normalized Mutual Information (NMI)**
A metric measuring the agreement between two clusterings, normalized to account for chance.

**Internal Validation**
Evaluating clustering quality using only the data itself, without external labels.

**External Validation**
Evaluating clustering quality by comparing to known ground truth labels.

## Implementation Terms

**KMeans class**
The scikit-learn class implementing K-Means clustering.

**AgglomerativeClustering**
The scikit-learn class implementing hierarchical agglomerative clustering.

**DBSCAN class**
The scikit-learn class implementing DBSCAN clustering.

**PCA class**
The scikit-learn class implementing Principal Component Analysis.

**TSNE class**
The scikit-learn class implementing t-SNE.

**fit_transform() method**
A method that fits the model and transforms the data in one step, commonly used in dimensionality reduction.

**transform() method**
Applies a fitted transformation to new data (available for PCA and UMAP, not t-SNE).

**inverse_transform() method**
Maps transformed data back to the original feature space (available for PCA).

**n_components parameter**
Specifies the number of dimensions to reduce to in dimensionality reduction algorithms.

**n_clusters parameter**
Specifies the number of clusters in clustering algorithms like K-Means.

---

*Last updated: November 10, 2025*

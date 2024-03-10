# Install pylibfm using: pip install pylibfm
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

# Sample user-item interaction data
data = [{'user': 1, 'item': 1, 'rating': 5},
        {'user': 1, 'item': 2, 'rating': 4},
        {'user': 2, 'item': 1, 'rating': 4},
        {'user': 2, 'item': 2, 'rating': 5},
        {'user': 2, 'item': 3, 'rating': 3}]
v = DictVectorizer()
X = v.fit_transform(data)
y = np.array([d['rating'] for d in data])

# Train the FM model
fm = pylibfm.FM()
fm.fit(X, y)

# Generate recommendations for users
user_id = 1
items = [{'user': user_id, 'item': i} for i in range(1, 4)]
predictions = fm.predict(v.transform(items))
top_items = np.argsort(predictions)[::-1][:2] + 1  # +1 to account for 0-based indexing
print("Recommendations for user", user_id, ":", top_items)

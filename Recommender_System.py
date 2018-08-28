#Movie recommender system using the Gradient Descent Algorithm and taking data set from lightfm library

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(data['train'])
print(data['test'])

model = LightFM(loss='warp') #warp = Weighted Approximated Rank Pairwise This uses the gradient descent algorithm

#train model
model.fit(data['train'], epochs=30, num_threads=2)


def SampleRecommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape

    #generate recommendation for each user we input
    for user_id in user_ids:
        #moviese they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("Known positives")

        for x in known_positives[:3]:
            print("%s " % x)

        print("Recommended ")
        for x in top_items[:3]:
            print("%s " % x)

        SampleRecommendation(model, data, [3, 25, 450])
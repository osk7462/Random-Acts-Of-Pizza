import json
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def get_length(data):
    length = []
    for single in data:
        request_message = single['request_text_edit_aware']
        length.append(len(request_message))
    return length


def get_politeness(data):
    polite_words = [
    "please","thanks","thank you","think", "thought", "thinking", "almost",
    "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
    "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
    "claimed", "doubt", "doubtful", "essentially", "estimate",
    "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
    "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
    "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
    "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
    "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
    "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
    "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
    "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
    "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
    "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates"
    ]

    polite_len = []
    for single_data in data:
        count = 0
        request_message = single_data["request_text_edit_aware"]
        # request_message = request_message.split(" ")
        for polite in polite_words:
            if polite in request_message:
                count += 1
        polite_len.append(count)
    return polite_len


def get_karma(data):
    karma_length = []
    for single_data in data:
        try:
            karma = single_data['requester_upvotes_plus_downvotes_at_request']/single_data['requester_upvotes_plus_downvotes_at_retrieval']
            karma_length.append(karma)
        except ZeroDivisionError:
            karma_length.append(0)
    return karma_length


def get_age(data):
    age = []
    for single_data in data:
        age1 = single_data['requester_days_since_first_post_on_raop_at_request']
        age.append(age1)
    return age


def get_username(data):
    users = []
    for singale_data in data:
        users.append(singale_data['request_id'])
    return users


def get_requests_status(data):
    status = []
    for single_data in data:
        status.append(single_data['requester_received_pizza'])
    return status


def creating_training_csv(filename,header,text_length,politeness,karma,age,user,status):
    with open(filename, "w") as file:
        csv_writer = csv.writer(file, dialect='excel')
        csv_writer.writerow(header)
        for i in range(len(text_length)):
            row = [ user[i],text_length[i], politeness[i], karma[i], age[i], status[i]]
            csv_writer.writerow(row)


def linear_regressor(filename):
    data = pd.read_csv(filename)
    x = data.iloc[: ,1:5].values
    y = data.iloc[:, 5:]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    predicted_value = model.predict(x_test)
    cm = confusion_matrix(y_test,predicted_value)
    correct_values = cm[0][0] + cm[1][1]
    total = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    accuracy = (correct_values/total) * 100
    return accuracy


if __name__ == '__main__':
    with open("train.json", "r") as datafile:
        data = json.load(datafile)
    text_length = get_length(data)
    politeness = get_politeness(data)
    karma = get_karma(data)
    age = get_age(data)
    user = get_username(data)
    status = get_requests_status(data)
    header = ["user_id", "message length", "politeness", "karma", "requester_days_since_first_post_on_raop_at_request", "requester_received_pizza"]
    filename = "train.csv"
    creating_training_csv(filename,header,text_length,politeness,karma,age,user,status)
    filename = "test.csv"
    accuracy = linear_regressor("train.csv")
    print("accuracy = {}".format(accuracy)) #74.00990099009901